"""Model-free ER-SDE/beta CLI selection, aliases and settings precedence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())

def run(*args, success=True):
    result = subprocess.run([exe, "generate", "--transformer", "missing-sampler.safetensors",
        "--steps", "9", "--dry-run", *map(str,args)], capture_output=True, text=True)
    assert (result.returncode == 0) == success, (args,result.stdout,result.stderr)
    return result.stdout + result.stderr

for backend in ("cuda", "vulkan"):
    for schedule in ("--schedule", "--scheduler"):
        output=run("--inference-backend",backend,"--sampler","er_sde",schedule,"beta")
        assert "er_sde" in output and "beta quantiles" in output and "9 model evaluations" in output
output=run("--model","krea2","--inference-backend","vulkan","--vulkan-arithmetic","portable",
           "--sampler","er_sde","--schedule","beta")
assert "er_sde updates" in output and "schedule    beta" in output
run("--sampler","er_sde","--skip-every","2",success=False)
run("--schedule","unknown",success=False)
with tempfile.TemporaryDirectory(prefix="slopfab-sampler-") as directory:
    path=Path(directory)/"sampling.json"
    path.write_text(json.dumps({"version":1,"sampler":"er_sde","scheduler":"beta",
                               "beta_alpha":.7,"beta_beta":.5}))
    output=run("--sampling-settings",path)
    assert "er_sde" in output and "beta quantiles" in output
    for args in (("--sampler","euler","--sampling-settings",path),
                 ("--sampling-settings",path,"--sampler","euler")):
        assert "sampler             euler" in run(*args)
print("ER-SDE and beta CLI checks passed")
