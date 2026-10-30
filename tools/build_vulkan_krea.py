"""Rebuild Krea's shader and update its pinned source/binary hashes."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--glslang", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = root / "src/vulkan/tensor_krea.comp"
artifact = source.with_suffix(".comp.spv")
subprocess.run([args.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp",
                str(source), "-o", str(artifact)], check=True)
manifest = root / "cmake/ShaderManifest.cmake"
text = manifest.read_text()
for path in (source, artifact):
    data = path.read_bytes()
    if path == source:
        data = data.replace(b"\r\n", b"\n")
    name = path.relative_to(root).as_posix()
    text, count = re.subn(r'(slopfab_shader\("' + re.escape(name) + r'" ")[a-f0-9]+',
                          lambda m: m[1] + hashlib.sha256(data).hexdigest(), text)
    if count != 1:
        raise RuntimeError(f"Expected one manifest entry for {name}")
manifest.write_text(text, newline="\n")
