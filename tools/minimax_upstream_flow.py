"""Run the unmodified upstream MiniMax flow module on captured native conditioning.

Uses local Comfy dit.py/dav.py and converted checkpoints. Only infrastructure
and checkpoint-backed operations are adapted; model forward methods are upstream.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import wave
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from moss_reference import rotate
from minimax_music_reference import vae_reference


def load_flow(path, source, dtype):
    comfy = types.ModuleType("comfy")
    for name in ("ops", "model_management", "quant_ops"):
        module = types.ModuleType("comfy." + name)
        setattr(comfy, name, module)
        sys.modules[module.__name__] = module
    comfy.ops.cast_to_input = lambda value, x: value.to(device=x.device, dtype=x.dtype)
    comfy.model_management.in_training = True
    sys.modules["comfy"] = comfy
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.AttentionTensorContainer = lambda x: x
    attention.ComfyAttention = nn.Identity
    def attend(q, k, v, heads, **kwargs):
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
    attention.optimized_attention_for_device = lambda device: attend
    sys.modules[attention.__name__] = attention
    spec = importlib.util.spec_from_file_location("upstream_music3_dit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Linear(nn.Linear):
        group = 0
        def forward(self, x):
            if self.group:
                x = rotate(x, self.group)
            return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

    ops = types.SimpleNamespace(Linear=Linear, Conv1d=nn.Conv1d)
    model = module.MiniMaxMusic3DiT(dtype=dtype, device="meta", operations=ops)
    with safe_open(path, framework="pt") as f:
        weights = {}
        for key in model.state_dict():
            value = f.get_tensor(key)
            if value.dtype == torch.int8:
                name = key.removesuffix(".weight")
                value = value.float() * f.get_tensor(name + ".weight_scale")
                config = json.loads(bytes(f.get_tensor(name + ".comfy_quant").tolist()))
                if config.get("convrot"):
                    model.get_submodule(name).group = config.get("convrot_groupsize", 256)
            # Keep the Fourier frequencies at checkpoint precision. The upstream
            # module casts them to the FP32 timestep before computing sin/cos.
            weights[key] = value.cuda().to(torch.float32 if "timestep_features.weight" in key else dtype)
        model.load_state_dict(weights, assign=True)
    return model.eval()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("dit", "source", "hidden", "probe", "out"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--vae")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--precise-positions", action="store_true")
    p.add_argument("--check-native", action="store_true", help="Compare a complete 30-step native window; use --dtype fp16")
    a = p.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hidden = load_file(a.hidden)["hidden"][:a.frames]
    fixture = out / "hidden.safetensors"
    save_file({"hidden": hidden}, str(fixture))
    subprocess.run([a.probe, "mm-noise", a.dit, str(fixture), str(out / "noise.safetensors")], check=True)
    noise = load_file(str(out / "noise.safetensors"))["output"].bfloat16()
    x = noise.T[None].cuda().to(dtype)
    model = load_flow(a.dit, a.source, dtype)
    if a.precise_positions:
        rotary = model.diffusion_transformer.transformer.rotary_pos_emb
        def positions(length, device, dtype):
            phase = torch.outer(torch.arange(length,device=device).float(),rotary.inv_freq.float())
            cs,sn = phase.cos().to(dtype),phase.sin().to(dtype)
            return torch.stack((cs,-sn,sn,cs),-1).reshape(1,1,length,-1,2,2)
        rotary.forward_from_seq_len = positions
    condition = model.aligned_condition(hidden[None].cuda().to(dtype))
    save_file({"condition": condition[0].T.float().cpu().contiguous()}, str(out / "condition.safetensors"))
    for step in range(a.steps):
        t = torch.tensor([step / a.steps], device="cuda", dtype=torch.float32)
        c = model.diffusion_transformer(x, t, condition)
        u = model.diffusion_transformer(x, t, torch.zeros_like(condition))
        velocity = u + (c - u) * 1.7
        x = (x.float() + velocity.float() / a.steps).to(dtype)
        print(step + 1, "latent RMS", x.float().square().mean().sqrt().item(), flush=True)
    state = x[0].T.float().cpu().contiguous()
    save_file({"audio_latents": state}, str(out / "latents.safetensors"))
    del model, x, condition, c, u, velocity
    torch.cuda.empty_cache()
    if a.check_native:
        if a.steps != 30 or a.frames != 200:
            raise ValueError("Native regression requires the full 200-frame, 30-step window")
        fixture = out / "sample-input.safetensors"
        save_file({"hidden":hidden,"audio_latents":noise.float()},str(fixture))
        native_path=out / "native-latents.safetensors"
        subprocess.run([a.probe,"mm-sample",a.dit,str(fixture),str(native_path)],check=True)
        native=load_file(str(native_path))["output"]
        relative=((native-state).square().mean()/state.square().mean()).sqrt().item()
        cosine=F.cosine_similarity(native.flatten(),state.flatten(),dim=0).item()
        metrics={"frames":a.frames,"latent_rows":len(state),"steps":a.steps,"reference_dtype":a.dtype,
                 "relative_rms":relative,"cosine":cosine}
        print("native full-window comparison",metrics,flush=True)
        (out / "metrics.json").write_text(json.dumps(metrics,indent=2))
        assert relative<.05 and cosine>.998,metrics
    if a.vae:
        audio = vae_reference(a.vae, Path(a.source).with_name("dav.py"), state).float().cpu()
        with wave.open(str(out / "upstream.wav"), "wb") as wav:
            wav.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
            wav.writeframes((audio.clamp(-1, 1).numpy() * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    main()
