"""Check embedded float-control declarations and the shader patching tool."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def declarations(path):
    raw = path.read_bytes()
    words = struct.unpack(f"<{len(raw) // 4}I", raw)
    capabilities, modes = set(), []
    at = 5
    while at < len(words):
        count, opcode = words[at] >> 16, words[at] & 0xFFFF
        assert count and at + count <= len(words), path
        if opcode == 17:
            capabilities.add(words[at + 1])
        if opcode == 16:
            modes.append(tuple(words[at + 1:at + count]))
        at += count
    assert len(modes) == len(set(modes)), f"duplicate execution mode: {path}"
    required = {mode[1] + 5 for mode in modes if 4459 <= mode[1] <= 4463}
    assert required <= capabilities, f"missing float-control capability: {path}"
    return capabilities, modes


for shader in (ROOT / "src/vulkan").glob("*.spv"):
    declarations(shader)

with tempfile.TemporaryDirectory() as directory:
    normal, denorm = [Path(directory) / name for name in ("normal.spv", "denorm.spv")]
    tool = ROOT / "tools/add_spirv_float_controls.py"
    source = ROOT / "src/vulkan/tensor_gemm.comp.spv"
    subprocess.run([sys.executable, str(tool), str(source), str(normal), str(denorm)], check=True)
    original = normal.read_bytes(), denorm.read_bytes()
    # DXC can already emit preservation modes. Patching it twice must be safe.
    subprocess.run([sys.executable, str(tool), str(normal), str(normal), str(denorm)], check=True)
    assert original == (normal.read_bytes(), denorm.read_bytes())
    for output in (normal, denorm):
        declarations(output)
    assert any(mode[1:] == (4459, 32) for mode in declarations(denorm)[1])

print("SPIR-V float controls: declarations and idempotent patching passed")
