"""Compare native MOSS components with PyTorch equations and upstream modules.

Requires torch, safetensors, diffusers and tokenizers (development only).
Pass the downloaded upstream pipeline_moss_image.py using --source.
No model or code downloads are performed by this tool.
"""
import argparse
import functools
import json
from pathlib import Path
import subprocess
import types

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def rotate(x, group=256):
    if x.shape[-1] % group:
        return x
    y = x.float().reshape(*x.shape[:-1], -1, group)
    for stride in (4 ** i for i in range(4) if 4 ** i < group):
        shape = y.shape
        parts = y.reshape(*shape[:-1], -1, 4, stride).unbind(-2)
        a, b, c, d = parts
        y = torch.stack((a+b+c-d, a+b-c+d, a-b+c+d, -a+b+c+d), -2).reshape(shape)
    return (y / group**0.5).reshape_as(x).to(x.dtype)


class Weights:
    def __init__(self, path):
        self.file = safe_open(path, framework="pt", device="cpu")
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            self.cache[name] = self.file.get_tensor(name).cuda()
        return self.cache[name]

    def linear(self, x, name):
        weight = self.get(name + ".weight")
        if weight.dtype == torch.int8:
            config = json.loads(bytes(self.file.get_tensor(name + ".comfy_quant").tolist()))
            if config.get("convrot"):
                x = rotate(x, config.get("convrot_groupsize", 256))
            weight = (weight.float() * self.get(name + ".weight_scale")).bfloat16()
        bias = self.get(name + ".bias") if name + ".bias" in self.file.keys() else None
        return F.linear(x, weight.to(x.dtype), bias.to(x.dtype) if bias is not None else None)

    def norm(self, x, name, offset=False):
        w = self.get(name)
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return (y * (1+w.float())).to(x.dtype) if offset else y.to(x.dtype) * w


def qwen_reference(weights, tokens):
    w = Weights(weights)
    name = "model.language_model.embed_tokens"
    table = w.file.get_tensor(name + ".weight")[tokens].cuda()
    if table.dtype == torch.int8:
        table = table.float() * w.file.get_tensor(name + ".weight_scale")[tokens].cuda()
        config = json.loads(bytes(w.file.get_tensor(name + ".comfy_quant").tolist()))
        if config.get("convrot"):
            table = rotate(table, config.get("convrot_groupsize", 256))
    x = table.bfloat16().unsqueeze(0)
    length = len(tokens)
    phase = torch.arange(length, device="cuda").float()[:, None] / 1e7 ** (torch.arange(32, device="cuda").float()/32)
    cos, sin = phase.cos().bfloat16(), phase.sin().bfloat16()

    def rope(t):
        a, b = t[..., :32], t[..., 32:64]
        return torch.cat((a*cos[:,None]-b*sin[:,None], b*cos[:,None]+a*sin[:,None], t[...,64:]), -1)

    for layer in range(32):
        p = f"model.language_model.layers.{layer}."
        y = w.norm(x, p+"input_layernorm.weight", True)
        if layer % 4 == 3:
            a = p+"self_attn."
            q, gate = w.linear(y, a+"q_proj").reshape(1,length,16,512).chunk(2,-1)
            q = rope(w.norm(q, a+"q_norm.weight", True))
            k = rope(w.norm(w.linear(y,a+"k_proj").reshape(1,length,4,256), a+"k_norm.weight", True))
            v = w.linear(y,a+"v_proj").reshape(1,length,4,256)
            # Native conditioner materializes BF16 probabilities, matching H3's causal primitive.
            q, k, v = (t.transpose(1,2) for t in (q,k,v))
            k, v = (t.repeat_interleave(4,1) for t in (k,v))
            scores = (q.float() @ k.float().transpose(-1,-2)) / 16
            mask = torch.ones(length,length,device="cuda",dtype=torch.bool).triu(1)
            prob = scores.masked_fill(mask, -torch.inf).softmax(-1).bfloat16()
            y = (prob @ v).transpose(1,2).reshape(1,length,4096) * gate.reshape(1,length,4096).sigmoid()
            y = w.linear(y, a+"o_proj")
        else:
            a = p+"linear_attn."
            qkv = w.linear(y,a+"in_proj_qkv")
            # Match fused causal convolution: FP32 accumulate and activation, BF16 output.
            qkv = F.silu(F.conv1d(qkv.float().transpose(1,2), w.get(a+"conv1d.weight").float(), padding=3,groups=8192)[...,:length]).bfloat16().transpose(1,2)
            q, k, v = qkv.split((2048,2048,4096),-1)
            q, k = (t.reshape(1,length,16,128).float().repeat_interleave(2,2) for t in (q,k))
            q = q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)/128**.5
            k = k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
            v = v.reshape(1,length,32,128).float()
            beta = w.linear(y,a+"in_proj_b").sigmoid().float()
            decay = (-w.get(a+"A_log").float().exp()*F.softplus(w.linear(y,a+"in_proj_a").float()+w.get(a+"dt_bias"))).exp()
            state = torch.zeros(1,32,128,128,device="cuda")
            output = []
            for i in range(length):
                state = state*decay[:,i,:,None,None]
                delta = (v[:,i]-(state*k[:,i,:,:,None]).sum(-2))*beta[:,i,:,None]
                state = state+k[:,i,:,:,None]*delta[:,:,None,:]
                output.append((state*q[:,i,:,:,None]).sum(-2))
            z = w.linear(y,a+"in_proj_z").reshape(1,length,32,128)
            y = torch.stack(output,1).bfloat16()
            y = (w.norm(y,a+"norm.weight") * F.silu(z.float())).bfloat16().reshape(1,length,4096)
            y = w.linear(y,a+"out_proj")
        x = x+y
        y = w.norm(x,p+"post_attention_layernorm.weight",True)
        y = F.silu(w.linear(y,p+"mlp.gate_proj"))*w.linear(y,p+"mlp.up_proj")
        x = x+w.linear(y,p+"mlp.down_proj")
    return w.norm(x,"model.language_model.norm.weight",True)[0].float()


def transformer_reference(path, source, latent, embedding, layers=42):
    # Execute only model declarations from the locally supplied upstream source.
    # Pipeline/library loading and any remote-code facilities are not used.
    class ConfigMixin:
        pass

    def register(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            import inspect
            bound = inspect.signature(fn).bind(self,*args,**kwargs)
            bound.apply_defaults()
            self.config = types.SimpleNamespace(**{k:v for k,v in bound.arguments.items() if k!="self"})
            return fn(self,*args,**kwargs)
        return wrapped

    namespace = dict(torch=torch, nn=nn, F=F, ModelMixin=nn.Module, ConfigMixin=ConfigMixin,
                     register_to_config=register, Transformer2DModelOutput=lambda **kw: types.SimpleNamespace(**kw))
    code = Path(source).read_text(encoding="utf8")
    exec(code[code.index("class _RMSNorm"):code.index("class MOSSImageScheduler")], namespace)
    with torch.device("meta"):
        model = namespace["MOSSImageTransformer"]()
    weights = Weights(path)

    class Linear(nn.Module):
        def __init__(self, name, part=None):
            super().__init__()
            self.name, self.part = name, part
            self.weight = torch.empty(0,device="cuda",dtype=torch.bfloat16)

        def forward(self, x):
            y = weights.linear(x,self.name)
            return y if self.part is None else y.chunk(3,-1)[self.part]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            if name+".weight" in weights.file.keys():
                setattr(parent,attr,Linear(name))
            else:
                setattr(parent,attr,Linear(parent_name+".qkv",("q","k","v").index(attr)))
    for name, _ in list(model.named_parameters()):
        parent, attr = name.rsplit(".",1)
        setattr(model.get_submodule(parent),attr,nn.Parameter(weights.get(name).bfloat16(),requires_grad=False))
    model.rope.bands = model.rope._default_bands(torch.device("cuda"))
    return model(latent.permute(2,0,1)[None].cuda(),torch.tensor([500.],device="cuda"),embedding[None].cuda().bfloat16(),skip_layers=list(range(layers,42))).sample[0].permute(1,2,0).float()


def vae_reference(path, latent):
    from diffusers.models.autoencoders.vae import Decoder
    state = load_file(str(path), device="cuda")
    decoder = Decoder(in_channels=32, out_channels=3, up_block_types=("UpDecoderBlock2D",)*4,
                      block_out_channels=(96,192,384,384),layers_per_block=2,norm_num_groups=32,
                      act_fn="silu",mid_block_add_attention=True).cuda().float()
    decoder.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith("decoder.")})
    z = latent.permute(2,0,1)[None].cuda()
    z = z*(state["bn.running_var"]+1e-4).sqrt()[None,:,None,None]+state["bn.running_mean"][None,:,None,None]
    b,c,h,w = z.shape
    z = z.reshape(b,c//4,2,2,h,w).permute(0,1,4,2,5,3).reshape(b,c//4,2*h,2*w)
    z = F.conv2d(z,state["post_quant_conv.weight"],state["post_quant_conv.bias"])
    return (decoder(z).clamp(-1,1)[0]*.5+.5).float()


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("text","transformer","vae"))
    parser.add_argument("--probe",required=True)
    parser.add_argument("--source")
    parser.add_argument("--layers", type=int, default=42)
    parser.add_argument("--embedding", help="Existing [tokens,4096] output tensor for a representative prompt")
    parser.add_argument("--latent-size", type=int, help="Latent height (width is height+2); defaults to 32 for DiT, 4 for VAE")
    parser.add_argument("--weights",default="weights")
    parser.add_argument("--out",default="build/moss_reference")
    args=parser.parse_args()
    root,out=Path(args.weights),Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    size = args.latent_size or (32 if args.mode == "transformer" else 4)
    latent=torch.randn(size,size+2,128)
    embedding=load_file(args.embedding)["output"] if args.embedding else torch.randn(10,4096)
    data=dict(image_latents=latent,prompt_embedding=embedding,timestep=torch.tensor([500.]))
    if args.mode=="text":
        from tokenizers import Tokenizer
        tok=Tokenizer.from_file(str(root/"text_encoder/qwen3.5_tokenizer.json"))
        ids=tok.encode("<|im_start|>user\nA red apple<|im_end|>\n<|im_start|>assistant\n<think>\n").ids
        data["tokens"]=torch.tensor(ids,dtype=torch.int32)
        model=root/"text_encoder/qwen3.5_9b_int8_convrot.safetensors"
        expected=qwen_reference(model,ids).cpu()
    elif args.mode=="transformer":
        if not args.source:
            parser.error("transformer comparison requires --source pipeline_moss_image.py")
        if not args.embedding:
            from tokenizers import Tokenizer
            tok=Tokenizer.from_file(str(root/"text_encoder/qwen3.5_tokenizer.json"))
            ids=tok.encode("<|im_start|>user\nA red apple<|im_end|>\n<|im_start|>assistant\n<think>\n").ids
            embedding=qwen_reference(root/"text_encoder/qwen3.5_9b_int8_convrot.safetensors",ids).cpu()
            torch.cuda.empty_cache()
            data["prompt_embedding"]=embedding
        model=root/"transformer/moss_image_sft_int8_convrot.safetensors"
        expected=transformer_reference(model,args.source,latent,embedding,args.layers).cpu()
        data["skip_layers"] = torch.tensor(list(range(args.layers,42)),dtype=torch.int32)
    else:
        model=root/"vae/flux2-vae_small.safetensors"
        expected=vae_reference(model,latent).cpu()
    torch.cuda.empty_cache()
    fixture=out/(args.mode+"-input.safetensors")
    actual=out/(args.mode+"-native.safetensors")
    save_file(data,str(fixture))
    save_file({"output":expected.contiguous()},str(out/(args.mode+"-reference.safetensors")))
    subprocess.run([args.probe,args.mode,str(model),str(fixture),str(actual)],check=True)
    observed=load_file(str(actual))["output"]
    error=observed-expected
    relative_rms=float(error.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-8))
    print(args.mode,"max_abs",float(error.abs().max()),"rms",float(error.square().mean().sqrt()),"relative_rms",relative_rms)
    # Long BF16 graphs accumulate rounding differences; VAE remains FP32 throughout.
    limit=0.03 if args.mode!="vae" else 0.001
    if not torch.isfinite(observed).all() or relative_rms>limit:
        raise SystemExit(f"component comparison exceeds {limit} relative RMS")


if __name__=="__main__":
    main()
