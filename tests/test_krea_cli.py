"""Device-free Krea defaults, routing and validation."""
from pathlib import Path
import subprocess
import sys

exe = str(Path(sys.argv[1]).resolve())


def run(*args, success=True):
    result = subprocess.run([exe, 'requests', 'run', "--dry-run", "--vulkan-arithmetic", "portable", *args], capture_output=True, text=True)
    assert (result.returncode == 0) == success, (args, result.stdout, result.stderr)
    return result.stdout + result.stderr


text = run("--model", "krea2", "--seed", "42")
assert "Krea 2" in text and "1024 x 1024" in text and "8 Euler updates" in text
assert "guidance    0" in text and ".png" in text
text = run("--model", "krea2", "--resolution", "512x768", "--steps", "52", "--guidance", "3.5")
assert "512 x 768" in text and "52 Euler updates" in text and "guidance    3.5" in text
for args in [
    ("--resolution", "513x512"), ("--guidance", "nan"), ("--guidance", "-1"),
    ("--audio-steps", "8"), ("--frames", "22"), ("--attention", "sage2"),
    ("--inference-backend", "vulkan", "--vulkan-arithmetic", "exact"), ("--sampler", "ab2"), ("--out", "image.jpg"),
    ("--uncond-skip-layers", "1"),
]:
    run("--model", "krea2", *args, success=False)

run("--model", "krea2", "--inference-backend", "vulkan")
print("Krea 2 CLI checks passed")
