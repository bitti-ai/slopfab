"""Teacher-force a captured native acoustic trajectory through upstream models."""
import argparse
import ast
import json
from pathlib import Path
import sys
import types
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from moss_reference import rotate


class QuantLinear(nn.Module):
    def __init__(self, weight, group, bias=None):
        super().__init__()
        self.weight, self.group, self.bias = weight, group, bias

    def forward(self, x):
        return F.linear(rotate(x, self.group) if self.group else x, self.weight, self.bias)


class Checkpoint:
    def __init__(self, path):
        self.file = safe_open(path, framework="pt")
        self.dense = {}

    def get(self, key):
        return self.file.get_tensor(key).cuda().bfloat16()

    def linear(self, name, lo=0, hi=None):
        if name not in self.dense:
            weight = self.file.get_tensor(name + ".weight").cuda()
            group = 0
            if weight.dtype == torch.int8:
                weight = weight.float() * self.file.get_tensor(name + ".weight_scale").cuda()
                config = json.loads(bytes(self.file.get_tensor(name + ".comfy_quant").tolist()))
                group = config.get("convrot_groupsize", 256) if config.get("convrot") else 0
            self.dense[name] = weight.bfloat16(), group
        weight, group = self.dense[name]
        return QuantLinear(weight[lo:hi], group)


def report(a, b, name):
    a, b = a.float(), b.float()
    rel = ((a-b).square().mean()/b.square().mean()).sqrt().item()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(name, "relative RMS", rel, "cosine", cos, flush=True)
    return rel


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("model", "trace", "config", "source"):
        p.add_argument("--" + arg, required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = Checkpoint(a.model)
    trace = load_file(a.trace)
    tokens = trace["tokens"].long()
    codes = trace["codes"].long()
    native = trace["hidden"].cuda().bfloat16()
    config_data = json.loads(Path(a.config).read_text())
    config_data["rope_theta"] = 1000000
    config = Qwen3Config(**config_data)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3Model(config)
    model.embed_tokens = nn.Identity()
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            prefix = "model." + name
            slices = {"q_proj": (0,4096), "k_proj": (4096,5120), "v_proj": (5120,6144),
                      "gate_proj": (0,12288), "up_proj": (12288,24576)}
            leaf = name.rsplit(".",1)[-1]
            if leaf in slices:
                prefix = prefix.rsplit(".",1)[0] + (".qkv_proj" if leaf.endswith("_proj") and leaf[0] in "qkv" else ".gate_up_proj")
                replacement = checkpoint.linear(prefix, *slices[leaf])
            else:
                replacement = checkpoint.linear(prefix)
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr, replacement)
        elif hasattr(module, "weight"):
            module.weight = nn.Parameter(checkpoint.get("model." + name + ".weight"), requires_grad=False)
    # Qwen's rotary module creates its buffer on meta during construction.
    model.rotary_emb.inv_freq = 1 / (1000000 ** (torch.arange(0,128,2,device="cuda").float()/128))
    prefills = checkpoint.file.get_tensor("model.embed_tokens_prefill.weight")[tokens].cuda()
    first = checkpoint.file.get_tensor("model.embed_tokens_audio.weight")[codes[:-1,0]].cuda()
    extra = checkpoint.file.get_tensor("model.audio_extra_embedding.weight")[(codes[:-1,1:]+torch.arange(7)*1024)].cuda()
    feedback = (first + extra.sum(1)) * (8 ** -.5)
    inputs = torch.cat((prefills, feedback))[None].bfloat16()
    out = model(inputs_embeds=inputs, use_cache=False).last_hidden_state[0,len(tokens)-1:]
    error = report(native[:,:4096], out, "global teacher-forced sequence")
    for i in (0,1,10,100,299):
        if i<len(out): report(native[i,:4096], out[i], f"global frame {i}")
    assert error < .04

    # Execute the original RVQ module classes, with only infrastructure adapted.
    tree = ast.parse(Path(a.source).read_text())
    tree.body = [node for node in tree.body if isinstance(node,ast.ClassDef) and node.name.startswith("RVQ")]
    comfy = types.SimpleNamespace(ops=types.SimpleNamespace(cast_to_input=lambda value,x:value.to(x)))
    comfy.ops.linear_input_act = lambda layer,x,kind: layer(F.silu(x.chunk(2,-1)[0])*x.chunk(2,-1)[1])
    def attend(q,k,v,heads,mask=None,**kwargs):
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        return y.transpose(1,2).reshape(q.shape[0],q.shape[2],-1)
    scope = {"torch":torch,"nn":nn,"comfy":comfy,"optimized_attention_for_device":lambda *args,**kwargs:attend}
    exec(compile(tree,a.source,"exec"),scope)
    class Embedding(nn.Embedding):
        def forward(self,x,out_dtype=None):return super().forward(x).to(out_dtype)
    decoder = scope["RVQDepthDecoder"]({"hidden_size":4096,"audio_vocab_size":1024,"audio_num_codebooks":8,
        "decoder_num_layers":4,"decoder_num_heads":16,"decoder_intermediate_size":6144,
        "decoder_merged_qkv":True,"decoder_merged_mlp":True},torch.bfloat16,"meta",types.SimpleNamespace(Linear=nn.Linear,Embedding=Embedding))
    for name,module in list(decoder.named_modules()):
        if isinstance(module,nn.Linear):
            replacement=checkpoint.linear("model.audio_decoder."+name)
            parent,_,attr=name.rpartition(".")
            setattr(decoder.get_submodule(parent),attr,replacement)
        elif hasattr(module,"weight"):
            module.weight=nn.Parameter(checkpoint.get("model.audio_decoder."+name+".weight"),requires_grad=False)
    # Use native global hidden states to isolate the depth decoder from global rounding.
    selected=torch.tensor([0,1,10,100,299],device="cuda")
    c=codes[selected.cpu()]
    first=checkpoint.file.get_tensor("model.embed_tokens_audio.weight")[c[:,0]].cuda()
    extra=checkpoint.file.get_tensor("model.audio_extra_embedding.weight")[c[:,1:]+torch.arange(7)*1024].cuda()
    sequence=torch.cat((decoder.projection(native[selected,:4096])[:,None],decoder.projection(first)[:,None],decoder.projection(extra[:,:6])),1)
    depth=decoder(sequence)[:,1:].reshape(len(selected),-1)
    assert report(native[selected,4096:],depth,"depth teacher-forced sequence") < .04


if __name__ == "__main__":main()
