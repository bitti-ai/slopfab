"""Produce FP32 golden outputs from the complete upstream 3D v1 node.

Requires PyTorch and safetensors (development only). Pass the upstream node file:
https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/nodes/minimax_h3_latent_upscaler_3d.py
Extracts the network and the node's execute method, including its normalization.
Only the ComfyUI registration/UI and model loader are replaced with local stubs.
"""
import argparse
import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("weights", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--chunked", action="store_true")
parser.add_argument("--packed-latents", type=Path, help="Optional generate --dump-latents archive")
parser.add_argument("--latent-resolution", default="32x32", help="Packed archive's latent WIDTHxHEIGHT")
args = parser.parse_args()
torch.set_num_threads(8)
torch.backends.mkldnn.enabled = True
names = {"normalization", "zero_module", "AttnBlock3D", "ResBlockEmb3D", "TemporalConv",
         "LatentResizer3D", "_make_norm_tensors", "_resolve_device", "_is_rocm_build", "UpscaleMode"}
tree = ast.parse(args.source.read_text(encoding="utf-8"))
selected = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
        selected.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id in {"LATENTS_MEAN", "LATENTS_STD", "VAE_DOWNSAMPLE"} for t in node.targets):
        selected.append(node)
    elif isinstance(node, ast.ClassDef) and node.name == "MinimaxH3LatentUpscaler3D":
        node.body = [method for method in node.body if isinstance(method, ast.FunctionDef)
                     and method.name == "execute"]
        selected.append(node)
tree.body = selected
scope = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "gc": __import__("gc"),
         "Enum": Enum, "UpscaleConfig": dict,
         "io": SimpleNamespace(ComfyNode=object, NodeOutput=lambda value: value)}
exec(compile(tree, str(args.source), "exec"), scope)
model = scope["LatentResizer3D"]().eval()
model.load_state_dict(load_file(str(args.weights)))
scope["load_model"] = lambda name, device, precision: model
cases = [("identity", 1, 2, 4, 1.0), ("still", 1, 2, 4, 2.0), ("video", 3, 2, 4, 1.5)]
if args.chunked:
    cases.append(("chunked", 33, 2, 2, 2.0))
    cases.append(("whole", 33, 2, 2, 2.0))
inputs = [(name, torch.sin(torch.arange(24 * t * h * w, dtype=torch.float32) * .13)
           .reshape(1, 24, t, h, w), scale, name != "whole") for name, t, h, w, scale in cases]
if args.packed_latents:
    w, h = map(int, args.latent_resolution.split("x"))
    rows = load_file(str(args.packed_latents))["video_rows"]
    if h <= 0 or w <= 0 or h % 2 or w % 2 or rows.ndim != 2 or rows.shape[1] != 96:
        parser.error("expected even latent dimensions and video_rows [N,96]")
    if rows.shape[0] % (h * w // 4):
        parser.error("video_rows do not fit the latent dimensions")
    t = rows.shape[0] // (h * w // 4)
    x = rows.reshape(t, h // 2, w // 2, 24, 2, 2).permute(3, 0, 1, 4, 2, 5)
    inputs.append(("production", x.reshape(1, 24, t, h, w).contiguous(), 2.0, True))
outputs = {}
with torch.inference_mode():
    for name, x, scale, chunking in inputs:
        print(name, tuple(x.shape), "scale", scale, flush=True)
        y = scope["MinimaxH3LatentUpscaler3D"].execute(
            {"samples": x}, model_name="local", mode={"mode": scope["UpscaleMode"].SCALE_BY, "scale": scale},
            align=32, enable_temporal_chunking=chunking, force_unload=False,
            device="cpu", precision="fp32")["samples"]
        outputs[name + ".input"] = x[0].contiguous()
        outputs[name + ".output"] = y[0].contiguous().clone()
        outputs[name + ".scale"] = torch.tensor([scale])
        outputs[name + ".chunking"] = torch.tensor([float(chunking)])
save_file(outputs, str(args.output), metadata={"reference": "comfy-node-execute-v1"})
print(args.output, flush=True)
