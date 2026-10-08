"""Produce FP32 golden outputs from the upstream 3D v1 network and local weights.

Requires PyTorch and safetensors (development only). Pass the upstream node file:
https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/nodes/minimax_h3_latent_upscaler_3d.py
Only the network definitions are executed; no ComfyUI node or model loader runs.
"""
import argparse
import ast
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("weights", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--chunked", action="store_true")
args = parser.parse_args()
torch.set_num_threads(8)
torch.backends.mkldnn.enabled = True
names = {"normalization", "zero_module", "AttnBlock3D", "ResBlockEmb3D", "TemporalConv", "LatentResizer3D"}
tree = ast.parse(args.source.read_text(encoding="utf-8"))
tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
scope = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "gc": __import__("gc")}
exec(compile(tree, str(args.source), "exec"), scope)
model = scope["LatentResizer3D"]().eval()
model.load_state_dict(load_file(str(args.weights)))
cases = [("still", 1, 2, 4, 4, 8, 2.0), ("video", 3, 2, 4, 4, 6, 1.5)]
if args.chunked:
    cases.append(("chunked", 33, 2, 2, 4, 4, 2.0))
outputs = {}
with torch.inference_mode():
    for name, t, h, w, oh, ow, scale in cases:
        x = torch.sin(torch.arange(24 * t * h * w, dtype=torch.float32) * .13).reshape(1, 24, t, h, w)
        print(name, tuple(x.shape), "->", (oh, ow), flush=True)
        y = model(x, scale=scale, target_size=(t, oh, ow))
        outputs[name + ".input"] = x[0].contiguous()
        outputs[name + ".output"] = y[0].contiguous()
        outputs[name + ".scale"] = torch.tensor([scale])
save_file(outputs, str(args.output))
print(args.output, flush=True)
