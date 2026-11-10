"""MiniMax-H3 planning and capability reporting without model or GPU requirements."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())
workspace = tempfile.TemporaryDirectory(prefix="slopfab-h3-image-")
root = Path(workspace.name)
config = root / "config.json"


def run(*args, success=True):
    result = subprocess.run([exe, "--config", str(config), *args],
                            cwd=root, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == success, (args, result.stdout, result.stderr)
    return result.stdout + result.stderr


base = ("requests", "run", "--model", "h3_image", "--dry-run")
text = run(*base)
assert "MiniMax-H3-Image" in text
assert "4096 x 4096" in text and "6 Euler updates" in text
assert "audio       none" in text
text = run(*base, "--resolution", "2048x2048", "--steps", "6")
assert "2048 x 2048" in text
for args in [
    ("--steps", "4"), ("--sampler", "ab2"), ("--sampler", "renoise"),
    ("--resolution", "1000x1024"), ("--frames", "22"),
    ("--guidance", "3"), ("--negative-prompt", "blur"),
]:
    run(*base, *args, success=False)

text = run("image", "generate", "--model", "h3_image", "--attention", "exact", "--prompt", "A red cube", success=False)
assert "flash2" in text
capabilities = json.loads(run("system", "capabilities", "--agent"))
def find_model(value):
    if isinstance(value, dict):
        if value.get("family") == "h3_image":
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
assert model and not model["planning_only"]

# Verify the downloaded transformer without copying or changing it.
checkpoint = os.environ.get("SLOPFAB_H3_IMAGE_CHECKPOINT")
if checkpoint:
    text = run("requests", "run", "--transformer", checkpoint, "--dry-run")
    assert "MiniMax-H3-Image" in text
    run("requests", "run", "--model", "h3", "--transformer", checkpoint,
        "--dry-run", success=False)

workspace.cleanup()
print("MiniMax-H3-Image CLI checks passed")
