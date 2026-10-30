"""Compare native Krea components with upstream PyTorch modules and local weights.

Development only: torch with CUDA, safetensors, diffusers, transformers, einops,
and a local checkout of https://github.com/krea-ai/krea-2. Never downloads models.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from moss_reference import Weights, rotate


def replace_linears(model, weights, prefix=""):
    class Linear(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, x):
            result = weights.linear(x, self.name)
            weights.cache = {k: v for k, v in weights.cache.items()
                             if not k.startswith(self.name + ".")}
            return result

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, Linear(prefix + name))
    for name, _ in list(model.named_parameters()):
        value = weights.file.get_tensor(prefix + name).cuda()
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, nn.Parameter(value, requires_grad=False))


def transformer_reference(args, data):
    spec = importlib.util.spec_from_file_location("krea_mmdit", Path(args.source) / "mmdit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The comparison uses eager equations; compilation is irrelevant to parity.
    original_compile = torch.compile
    torch.compile = lambda *a, **kw: (a[0] if a else lambda f: f)
    try:
        spec.loader.exec_module(module)
    finally:
        torch.compile = original_compile

    def attention(q, k, v, mask=None, scale=None, gqa=False):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale,
                                              enable_gqa=gqa).transpose(1, 2).flatten(2)

    module.attention = attention
    config = module.SingleMMDiTConfig(6144, 256, 2560, 48, 4, 28, 2, 16,
                                     kvheads=12, txtheads=20, txtkvheads=20, txtlayers=12)
    with torch.device("meta"):
        model = module.SingleStreamDiT(config)
    weights = Weights(args.model)
    replace_linears(model, weights)
    model.eval()
    context = data["prompt_embedding"][None].cuda().bfloat16()
    if args.mode == "fusion":
        return model.txtmlp(model.txtfusion(context))[0].float().cpu()
    latent = data["image_latents"]
    h, w = latent.shape[:2]
    n = context.shape[1]
    pos = torch.zeros(1, n + h*w, 3, device="cuda")
    pos[0, n:, 1:] = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w),
                                              indexing="ij"), -1).reshape(-1, 2).cuda()
    mask = torch.ones(1, n + h*w, device="cuda", dtype=torch.bool)
    return model(latent.reshape(1, -1, 64).cuda().bfloat16(), context,
                 data["sigma"].cuda().bfloat16(), pos, mask)[0].reshape(h, w, 64).float().cpu()


def vae_reference(args, data):
    from diffusers import AutoencoderKLQwenImage
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
    model = AutoencoderKLQwenImage(base_dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
                                   temperal_downsample=[False, True, True]).cuda().float().eval()
    model.load_state_dict(convert_wan_vae_to_diffusers(load_file(args.model)), strict=True)
    x = data["image_latents"].cuda()
    h, w = x.shape[:2]
    x = x.reshape(h, w, 16, 2, 2).permute(2, 0, 3, 1, 4).reshape(1, 16, 1, h*2, w*2)
    mean = torch.tensor(model.config.latents_mean, device="cuda").reshape(1, 16, 1, 1, 1)
    std = torch.tensor(model.config.latents_std, device="cuda").reshape(1, 16, 1, 1, 1)
    return (model.decode(x * std + mean).sample[0, :, 0].clamp(-1, 1)*.5+.5).cpu()


def text_reference(args):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding
    from tokenizers import Tokenizer
    config = Qwen3VLTextConfig(hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                              num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                              vocab_size=151936, rope_theta=5000000,
                              rope_scaling={"rope_type": "default", "mrope_interleaved": True,
                                            "mrope_section": [24, 20, 20]})
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3VLTextModel(config)
    weights = Weights(args.model)

    class Embedding(nn.Module):
        def forward(self, ids):
            name = "model.language_model.embed_tokens"
            value = weights.file.get_tensor(name + ".weight")[ids.cpu()]
            if value.dtype == torch.int8:
                value = rotate(value.float(), 256) * weights.file.get_tensor(name + ".weight_scale")[ids.cpu()]
            return value.cuda().bfloat16()

    model.embed_tokens = Embedding()
    replace_linears(model, weights, "model.language_model.")
    model = model.bfloat16().eval()
    model.rotary_emb = Qwen3VLTextRotaryEmbedding(config, device="cuda")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    prefix = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
              "quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
    ids = tokenizer.encode(prefix + args.prompt).ids[:541]
    # Keep upstream's masked padding, independently checking native compaction.
    mask = [1]*len(ids) + [0]*(541-len(ids)) + [1]*5
    ids += [151643]*(541-len(ids)) + tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n").ids
    mask = torch.tensor([mask], device="cuda")
    positions = mask.cumsum(-1)-1
    positions.masked_fill_(mask == 0, 0)
    out = model(input_ids=torch.tensor([ids], device="cuda"), attention_mask=mask,
                position_ids=positions[None].expand(3, -1, -1), output_hidden_states=True, use_cache=False)
    valid = mask[0, 34:].bool()
    return torch.stack([out.hidden_states[i][0, 34:][valid] for i in range(2, 36, 3)], 1).float().cpu()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", default="build/Release/slopfab_kreaprobe.exe")
    parser.add_argument("--source", default="ref/krea-2")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["fusion", "transformer", "vae", "text"], required=True)
    parser.add_argument("--tokenizer", default="ref/text_encoder/tokenizer.json")
    parser.add_argument("--prompt", default="A red apple on a wooden table")
    args = parser.parse_args()
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path("build/krea-support")
    root.mkdir(parents=True, exist_ok=True)
    data = {"prompt_embedding": torch.randn(9, 12, 2560),
            "image_latents": torch.randn(3, 4, 64), "sigma": torch.tensor([.75])}
    fixture = root / (args.mode + "-input.safetensors")
    actual = root / (args.mode + "-native.safetensors")
    save_file(data, str(fixture))
    if args.mode == "text":
        expected = text_reference(args)
    elif args.mode == "vae":
        expected = vae_reference(args, data)
    else:
        expected = transformer_reference(args, data)
    torch.cuda.empty_cache()
    save_file({"output": expected.contiguous()}, str(root / (args.mode + "-expected.safetensors")))
    subprocess.run([args.probe, args.mode, args.model,
                    args.prompt if args.mode == "text" else str(fixture), str(actual)], check=True)
    observed = load_file(str(actual))["output"]
    error = observed - expected
    relative = float(error.square().mean().sqrt() / expected.square().mean().sqrt())
    print(args.mode, "relative_rms", relative, "max_abs", float(error.abs().max()), flush=True)
    tolerance = .001 if args.mode == "vae" else .04
    if not torch.isfinite(observed).all() or relative > tolerance:
        raise SystemExit("Krea numerical comparison failed")


if __name__ == "__main__":
    main()
