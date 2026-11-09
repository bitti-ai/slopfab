"""P-H3 planning and capability reporting without model or GPU requirements."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())
workspace = tempfile.TemporaryDirectory(prefix="slopfab-ph3-")
root = Path(workspace.name)
config = root / "config.json"


def run(*args, success=True):
    result = subprocess.run([exe, "--config", str(config), *args],
                            cwd=root, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == success, (args, result.stdout, result.stderr)
    return result.stdout + result.stderr


base = ("requests", "run", "--model", "ph3_image", "--dry-run")
text = run(*base)
assert "P-H3 Image (planning only)" in text
assert "2912 x 2912" in text and "8 Euler updates" in text
assert "2 latent positions" in text and "GPU generation is not implemented" in text
text = run(*base, "--resolution", "2048x2048", "--steps", "8")
assert "2048 x 2048" in text
for args in [
    ("--steps", "4"), ("--sampler", "ab2"), ("--sampler", "renoise"),
    ("--resolution", "1000x1024"), ("--frames", "22"),
    ("--guidance", "3"), ("--negative-prompt", "blur"),
]:
    run(*base, *args, success=False)

text = run("image", "generate", "--model", "ph3_image", "--prompt", "A red cube", success=False)
assert "not wired into generation yet" in text
capabilities = json.loads(run("system", "capabilities", "--agent"))
def find_model(value):
    if isinstance(value, dict):
        if value.get("family") == "ph3_image":
            return value
        for nested in value.values():
            found = find_model(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = find_model(nested)
            if found:
                return found
model = find_model(capabilities)
assert model and model["planning_only"] and model["backends"] == []

# Verify the downloaded transformer without copying or changing it.
checkpoint = os.environ.get("SLOPFAB_PH3_IMAGE_CHECKPOINT")
if checkpoint:
    text = run("requests", "run", "--transformer", checkpoint, "--dry-run")
    assert "P-H3 Image (planning only)" in text
    run("requests", "run", "--model", "h3", "--transformer", checkpoint,
        "--dry-run", success=False)

workspace.cleanup()
print("P-H3 Image CLI checks passed")
