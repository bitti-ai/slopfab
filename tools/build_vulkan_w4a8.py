"""Regenerate W4A8 shaders with glslang 16.5.0 and update pinned hashes."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glslang", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "src/vulkan/tensor_w4a8.comp"
    controls = root / "tools/add_spirv_float_controls.py"
    artifacts = [(source, "SOURCE", "-")]
    with tempfile.TemporaryDirectory(prefix="slopfab-w4a8-") as temporary:
        for quant in (False, True):
            name = "tensor_w4a8_quant" if quant else "tensor_w4a8"
            raw = Path(temporary) / (name + ".spv")
            command = [args.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp"]
            if quant:
                command.append("-DW4A8_QUANTIZE=1")
            subprocess.run(command + [str(source), "-o", str(raw)], check=True)
            normal = source.parent / (name + ".comp.spv")
            if quant:
                # NVIDIA 610.88 crashes compiling division with explicit FP32
                # RTE controls. BF16 and integer rounding remain explicit in
                # the shader. The 1e-30 divisor floor makes subnormal inputs
                # quantize to zero, so a denormal variant is unnecessary.
                subprocess.run([sys.executable, str(controls), "--preserve-only",
                                str(raw), str(normal)], check=True)
                artifacts.append((normal, "BINARY", "SLOPFAB_TENSOR_W4A8_QUANT_SPV_BYTES"))
            else:
                denorm = source.parent / (name + "_denorm.comp.spv")
                subprocess.run([sys.executable, str(controls), str(raw), str(normal),
                                str(denorm)], check=True)
                artifacts.extend([(normal, "BINARY", "SLOPFAB_TENSOR_W4A8_SPV_BYTES"),
                                  (denorm, "BINARY", "SLOPFAB_TENSOR_W4A8_DENORM_SPV_BYTES")])
    manifest = root / "cmake/ShaderManifest.cmake"
    text = manifest.read_text()
    for path, kind, variable in artifacts:
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        if kind == "SOURCE":
            data = data.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(data).hexdigest()
        line = f'slopfab_shader("{relative}" "{digest}" {kind} {variable})'
        pattern = r'slopfab_shader\("' + re.escape(relative) + r'"[^\n]*'
        if re.search(pattern, text):
            text = re.sub(pattern, lambda _: line, text)
        else:
            text += "\n" + line + "\n"
    manifest.write_text(text)


if __name__ == "__main__":
    main()
