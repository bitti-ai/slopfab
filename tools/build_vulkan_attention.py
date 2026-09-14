"""Rebuild fast attention artifacts with glslang (16.5.0 or newer).

Usage: python tools/build_vulkan_attention.py --glslang /path/to/glslang
Artifacts deliberately omit the exact-mode SPIR-V float-control transform.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glslang", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "src" / "vulkan"
    files = ["tensor_attention_fast.comp", "tensor_attention_sage_prepare.comp"]
    variants = [
        ("flash", []), ("flash_banded", ["BANDED"]),
        ("sage", ["SAGE"]), ("sage_banded", ["SAGE", "BANDED"]),
        ("sage_prepare", []),
    ]
    for subgroup in (32, 64):
        for tile, qrows, vrows in (("", 32, 16), ("_full", 32, 64),
                                   ("_wide", 64, 16), ("_wide_full", 64, 64)):
            if subgroup == 32 and not tile:
                continue  # existing compact names
            suffix = tile + ("_sg64" if subgroup == 64 else "")
            for band in (False, True):
                variants.append(("sage" + suffix + ("_banded" if band else ""),
                                 ["SAGE", f"QUERY_ROWS={qrows}", f"VALUE_ROWS={vrows}",
                                  f"SUBGROUP_SIZE={subgroup}"] + (["BANDED"] if band else [])))
    for name, defines in variants:
        source = files[1] if name == "sage_prepare" else files[0]
        output = f"tensor_attention_{name}.comp.spv"
        subprocess.run([args.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp",
                        *[f"-D{d}" if "=" in d else f"-D{d}=1" for d in defines], str(root / source),
                        "-o", str(root / output)], check=True)
        files.append(output)
    lines = []
    for name in files:
        data = (root / name).read_bytes()
        if name.endswith(".comp"):
            data = data.replace(b"\r\n", b"\n")
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
    (root / "attention_fast.sha256").write_text("".join(lines), encoding="ascii")


if __name__ == "__main__":
    main()
