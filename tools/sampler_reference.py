"""Print CPU golden trajectories using ComfyUI ER-SDE and SciPy beta quantiles.

Development only: --source points to a ComfyUI checkout. Only the sampler
function is loaded, so no ComfyUI device initialization or model files are needed.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from scipy.stats import beta
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True, type=Path)
args = parser.parse_args()
source = (args.source / "comfy/k_diffusion/sampling.py").read_text(encoding="utf-8")
function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "sample_er_sde")


def offset(sigmas, model):
    sigmas = sigmas.clone()
    if sigmas[0] >= 1:
        t = 1 - 1e-4
        sigmas[0] = model.shift * t / (1 + (model.shift - 1) * t)
    return sigmas


scope = dict(torch=torch, trange=lambda n, **kwargs: range(n),
             offset_first_sigma_for_snr=offset,
             sigma_to_half_log_snr=lambda s, model: -s.logit())
exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream_er_sde", "exec"), scope)
print("source_sha256", hashlib.sha256(source.encode()).hexdigest())
for first in (1., .93):
    model_sampling = SimpleNamespace(shift=3., noise_scale=1.)
    class Model:
        inner_model = SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda _: model_sampling))
        def __call__(self, x, sigma, **kwargs):
            velocity = .2*x + torch.tensor([.1, .2, .3, .4]) + sigma*.05
            return x + sigma*velocity
    counter = [0]
    def noise(*_):
        counter[0] += 1
        return torch.tensor([.2, -.1, .7, -.3])*counter[0]
    trajectory = []
    out = scope["sample_er_sde"](Model(), torch.tensor([.1, -.2, .3, -.4]),
        torch.tensor([first, .8, .55, .3, .12, 0]), noise_sampler=noise,
        callback=lambda d: trajectory.append(d["x"].clone()))
    trajectory.append(out)
    print("er_sde", first, torch.stack(trajectory).tolist())
for count, alpha, shape in [(8,.6,.6),(9,.6,.6),(8,2,5),(8,1,1)]:
    indices=np.rint(beta.ppf(1-np.arange(count)/count,alpha,shape)*999).astype(int)
    print("beta_indices",count,alpha,shape,indices.tolist())
