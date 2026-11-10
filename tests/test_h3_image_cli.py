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
assert model["max_image_references"] == 16

# Verify the downloaded transformer without copying or changing it.
checkpoint = os.environ.get("SLOPFAB_H3_IMAGE_CHECKPOINT")
if checkpoint:
    text = run("requests", "run", "--transformer", checkpoint, "--dry-run")
    assert "MiniMax-H3-Image" in text
    run("requests", "run", "--model", "h3", "--transformer", checkpoint,
        "--dry-run", success=False)

# Full-image editing uses Picture 1, with ordered additional references.
source = root / "source.ppm"
source.write_bytes(b"P6\n64 32\n255\n" + bytes([90, 40, 20]) * (64 * 32))
second = root / "second.ppm"
second.write_bytes(b"P6\n32 64\n255\n" + bytes([20, 40, 90]) * (32 * 64))
text = run(*base, "--reference-image", str(second), "--edit-image", str(source))
assert "2048 x 1024" in text and "image editing (2 references)" in text
text = run(*base, "--edit-image", str(source), "--prompt", "Make the teapot blue")
assert "2048 x 1024" in text and "image editing (1 references)" in text
text = run(*base, "--edit-image", str(source), "--reference-image", str(source),
           "--resolution", "512x512")
assert "512 x 512" in text and "image editing (2 references)" in text
for extra in [("--edit-strength", "0.5"), ("--edit-feather", "2")]:
    run(*base, "--edit-image", str(source), *extra, success=False)
references = ("--reference-image", str(source)) * 16
text = run(*base, "--resolution", "512x512", *references)
assert "16 references" in text
run(*base, "--resolution", "512x512", *references, "--reference-image", str(source), success=False)

text = run(*base, "--edit-image", str(source), "--outpaint", "32,16,32,16")
assert "128 x 64" in text and "outpainting" in text and "6 Euler updates" in text
text = run(*base, "--edit-image", str(source), "--outpaint", "1,0,0,0",
           "--outpaint-blend", "1", "--reference-image", str(second))
assert "96 x 32" in text and "outpainting" in text
assert "delivered   65 x 32" in text and "pictures    2" in text
run("image", "generate", "--model", "h3", "--dry-run", "--edit-image", str(source),
    "--outpaint", "32,16,32,16", "--out", str(root / "expanded.png"))
for extra in [("--outpaint", "0,0,0,0"), ("--outpaint", "-1,0,32,0"),
              ("--outpaint", "1,2,3"), ("--outpaint", "2147483647,0,0,0"),
              ("--outpaint-blend", "1"), ("--outpaint-langevin-steps", "0")]:
    run(*base, "--edit-image", str(source), *extra, success=False)
for extra in [("--outpaint-blend", "2"), ("--outpaint-langevin-steps", "1"),
              ("--resolution", "512x512"), ("--edit-box", "0,0,32,32"),
              ("--edit-strength", "0.5"), ("--aspect", "1:1"), references]:
    run(*base, "--edit-image", str(source), "--outpaint", "32,16,32,16", *extra, success=False)
run(*base, "--outpaint", "32,16,32,16", success=False)

text = run(*base, "--edit-image", str(source), "--edit-box", "0,0,32,32", "--edit-feather", "2")
assert "masked inpainting" in text and "64 x 32" in text and "6 Euler updates" in text
mask = root / "mask.ppm"
mask.write_bytes(b"P6\n64 32\n255\n" + bytes([0,0,0]) * 1024 + bytes([128,128,128]) * 1024)
text = run(*base, "--edit-image", str(source), "--edit-mask", str(mask), "--reference-image", str(second))
assert "pixel mask" in text and "pictures    2" in text and "delivered   64 x 32" in text
for extra in [("--edit-box", "0,0,32,32"), ("--edit-strength", "0.5"),
              ("--edit-feather", "2"), ("--resolution", "512x512"), ("--outpaint", "32,0,0,0"), references]:
    run(*base, "--edit-image", str(source), "--edit-mask", str(mask), *extra, success=False)
run(*base, "--edit-mask", str(mask), success=False)
run(*base, "--edit-image", str(source), "--edit-mask", str(second), success=False)
run(*base, "--edit-image", str(source), "--edit-mask", str(source), success=False) # Colored mask.
mask.write_bytes(b"P6\n64 32\n255\n" + bytes(64 * 32 * 3))
run(*base, "--edit-image", str(source), "--edit-mask", str(mask), success=False)

workspace.cleanup()
print("MiniMax-H3-Image CLI checks passed")
