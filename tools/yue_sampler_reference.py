"""Print YuE2 flow golden trajectories using ComfyUI's actual DPM++ 2M sampler.

Development only: --source points to a ComfyUI checkout. Requires CPU PyTorch;
does not initialize ComfyUI or load model checkpoints.
"""
import argparse
import ast
import hashlib
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True, type=Path)
args = parser.parse_args()
source = (args.source / "comfy/k_diffusion/sampling.py").read_text(encoding="utf-8")
function = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "sample_dpmpp_2m")
scope = dict(torch=torch, trange=lambda n, **kwargs: range(n))
exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream_dpmpp_2m", "exec"), scope)
print("source_sha256", hashlib.sha256(source.encode()).hexdigest())
for dtype in (torch.float64, torch.float32):
    trajectory = []
    calls = []
    def model(x, sigma, **kwargs):
        calls.append(float(sigma[0]))
        velocity = .2 * x + .1 * sigma**2 * torch.arange(1, 4, dtype=dtype)
        return x - sigma * velocity
    result = scope["sample_dpmpp_2m"](
        model, torch.tensor([1., -.75, .25], dtype=dtype),
        torch.linspace(1, 0, 7, dtype=dtype),
        callback=lambda d: trajectory.append(d["x"].clone()))
    trajectory.append(result)
    assert len(calls) == 6
    print(dtype, "trajectory", torch.stack(trajectory[1:]).tolist())
