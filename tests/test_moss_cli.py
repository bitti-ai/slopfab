"""Device-free MOSS CLI routing, defaults and incompatible-option checks."""
from pathlib import Path
import subprocess
import sys

exe = str(Path(sys.argv[1]).resolve())


def run(*args, success=True):
    result = subprocess.run([exe, "generate", "--dry-run", *args], capture_output=True, text=True)
    assert (result.returncode == 0) == success, (args, result.stdout, result.stderr)
    return result.stdout + result.stderr


text = run("--model", "moss", "--seed", "42")
assert "MOSS Image" in text and "1024 x 1024" in text and "50 Euler updates" in text
assert ".ppm" in text
text = run("--model", "moss", "--resolution", "512x768", "--steps", "1", "--uncond-skip-layers", "none")
assert "512 x 768" in text and "1 Euler updates" in text
for args in [
    ("--resolution", "513x512"), ("--guidance", "nan"),
    ("--uncond-skip-layers", "42"), ("--uncond-skip-layers", "22,"),
    ("--audio-steps", "8"), ("--frames", "22"), ("--attention", "sage2"),
    ("--inference-backend", "vulkan"), ("--sampler", "ab2"), ("--out", "image.png"),
]:
    run("--model", "moss", *args, success=False)
run("--model", "unknown", success=False)
run("--model", "h3", "--negative-prompt", "bad", success=False)
print("MOSS CLI checks passed")
