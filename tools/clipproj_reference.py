"""Compare native ClipProj text conditioning against PyTorch Qwen equations.

Development-only: requires torch, tokenizers and safetensors. Uses local files;
never downloads weights. Supported reference encoder: INT8 ConvRot Qwen3-VL.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from moss_reference import Weights, rotate


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--prompt", default="A red ball on a wooden table.")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    weights = Weights(args.encoder)
    projection = load_file(args.projection)
    with safe_open(args.projection, framework="pt") as f:
        tap = int(f.metadata()["tap"])
    ids = Tokenizer.from_file(args.tokenizer).encode(args.prompt, add_special_tokens=False).ids
    prefix = "model.language_model."
    embed = prefix + "embed_tokens"
    config = json.loads(bytes(weights.file.get_tensor(embed + ".comfy_quant").tolist()))
    codes = torch.stack([weights.file.get_slice(embed + ".weight")[i] for i in ids]).float()
    scale = torch.stack([weights.file.get_slice(embed + ".weight_scale")[i] for i in ids])
    x = (rotate(codes, config.get("convrot_groupsize", 256)) * scale).cuda().bfloat16()
    width, length = x.shape[1], len(ids)
    theta = torch.arange(length, device="cuda").float()[:, None] * (5000000. ** (-torch.arange(64, device="cuda").float() / 64))
    cos, sin = theta.cos().bfloat16()[:, None], theta.sin().bfloat16()[:, None]

    def rotary(value):
        a, b = value.chunk(2, -1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)

    for i in range(tap):
        p = prefix + f"layers.{i}."
        y = weights.norm(x, p + "input_layernorm.weight")
        q = rotary(weights.norm(weights.linear(y, p + "self_attn.q_proj").reshape(length, 32, 128), p + "self_attn.q_norm.weight"))
        k = rotary(weights.norm(weights.linear(y, p + "self_attn.k_proj").reshape(length, 8, 128), p + "self_attn.k_norm.weight"))
        v = weights.linear(y, p + "self_attn.v_proj").reshape(length, 8, 128)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).squeeze(0).transpose(0, 1).reshape(length, 4096)
        x += weights.linear(y, p + "self_attn.o_proj")
        y = weights.norm(x, p + "post_attention_layernorm.weight")
        y = F.silu(weights.linear(y, p + "mlp.gate_proj")) * weights.linear(y, p + "mlp.up_proj")
        x += weights.linear(y, p + "mlp.down_proj")
        weights.cache.clear()
    x = x.float().cpu()
    xn = (x - projection["mean_in"].float()) / projection["std_in"].float()
    y = xn @ projection["W"].float() if "W" in projection else None
    layers = sorted({int(k.split('.')[1]) for k in projection if k.startswith('mlp.')})
    if layers:
        residual = xn.to(projection[f"mlp.{layers[0]}.weight"].dtype)
        for i, layer in enumerate(layers):
            residual = F.linear(residual, projection[f"mlp.{layer}.weight"], projection.get(f"mlp.{layer}.bias"))
            if i + 1 < len(layers):
                residual = F.gelu(residual)
        y = residual.float() if y is None else y + residual.float()
    y = y * projection["std_out"].float() + projection["mean_out"].float()
    if "sink_out" in projection:
        y[0] = projection["sink_out"].float()
    actual = load_file(args.native)["prompt_embedding"]
    # Exclude the fixed sink from the accuracy metric; it otherwise hides errors.
    expected_rows, actual_rows = y[1:], actual[1:]
    relative = float((expected_rows - actual_rows).square().mean().sqrt() / expected_rows.square().mean().sqrt())
    cosine = float(F.cosine_similarity(expected_rows.flatten(), actual_rows.flatten(), dim=0))
    print(json.dumps({"tokens": length, "tap": tap, "width": width, "relative_rms": relative, "cosine": cosine}), flush=True)
    save_file({"prompt_embedding": y.contiguous(), "student_hidden": x.contiguous()}, str(Path(args.native).with_suffix('.reference.safetensors')))
    if relative > .04 or cosine < .999:
        raise SystemExit("ClipProj reference comparison exceeds tolerance")


if __name__ == "__main__":
    main()
