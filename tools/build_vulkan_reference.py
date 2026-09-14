"""Build the reference encoder shader and record normalized source/artifact hashes."""
import argparse
import hashlib
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--glslang", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / "src" / "vulkan"
source = root / "tensor_reference.comp"
artifact = root / "tensor_reference.comp.spv"
subprocess.run([args.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp",
                str(source), "-o", str(artifact)], check=True)
(root / "reference.sha256").write_text(
    hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest() + "\n" +
    hashlib.sha256(artifact.read_bytes()).hexdigest() + "\n", encoding="ascii")
