"""Model-free CLI checks for mixed RefMod packaging and option scoping."""
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


exe = str(Path(sys.argv[1]).resolve())


def run(*args, success=True):
    result = subprocess.run([exe, *map(str, args)], capture_output=True, text=True)
    if (result.returncode == 0) != success:
        raise AssertionError((args, result.returncode, result.stdout, result.stderr))
    return result.stdout + result.stderr


def write(path, shape, dtype, kind):
    count = math.prod(shape)
    data = (struct.pack("<f", 0.25) if dtype == "F32" else
            struct.pack("<e", 0.25) if dtype == "F16" else b"\x80\x3e") * count
    metadata = {"_format_version": 4, "kind": kind, "name": kind,
                "description": 'A "reference"\nwith text', "custom": {"keep": [1, True]}}
    header = {"latent": {"dtype": dtype, "shape": shape, "data_offsets": [0, len(data)]},
              "__metadata__": {"refmod_meta": json.dumps(metadata)}}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return data


def read(path):
    raw = path.read_bytes()
    length, = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8:8 + length]), raw[8 + length:]


with tempfile.TemporaryDirectory(prefix="slopfab-assets-cli-") as directory:
    root = Path(directory)
    inputs = []
    payloads = []
    for kind, shape, dtype in [("image", [1, 24, 1, 4, 4], "F16"),
                               ("video", [1, 24, 2, 6, 8], "BF16"),
                               ("audio", [1, 32, 2, 3], "F32")]:
        path = root / (kind + ".safetensors")
        inputs.append(path)
        payloads.append(write(path, shape, dtype, kind))
    bundle = root / "mixed.safetensors"
    input_args = [arg for path in inputs for arg in ("--refmod", path)]
    run("bundle-refmods", *input_args, "--output", bundle, "--name", "hero",
        "--description", "appearance and voice")
    header, data = read(bundle)
    meta = json.loads(header["__metadata__"]["refmod_meta"])
    assert meta["kind"] == "bundle" and meta["_format_version"] == 5
    assert meta["name"] == "hero" and meta["description"] == "appearance and voice"
    assert [m["kind"] for m in meta["members"]] == ["image", "video", "audio"]
    for i, expected in enumerate(payloads):
        assert meta["members"][i]["custom"] == {"keep": [1, True]}
        begin, end = header[f"ref_{i}"]["data_offsets"]
        assert data[begin:end] == expected, "repacking must preserve latent bytes"

    common = ["generate", "--dry-run", "--prompt", "test", "--resolution", "32x32",
              "--frames", "22", "--steps", "2"]
    combined = run(*common, "--refmod", bundle, "--refmod-strength", ".25", "--refmod-copies", "2")
    separate_args = [arg for path in inputs for arg in
                     ("--refmod", path, "--refmod-strength", ".25", "--refmod-copies", "2")]
    separate = run(*common, *separate_args)
    packed = lambda text: re.search(r"packed sequence\s+(\d+)", text).group(1)
    assert packed(combined) == packed(separate)
    assert len(re.findall(r"strength 0\.250*, copies 2", combined)) == 3
    scoped = run(*common, "--refmod", bundle, "--refmod-strength", ".25", "--refmod-copies", "2",
                 "--refmod", inputs[0], "--refmod-strength", ".8")
    assert len(re.findall(r"strength 0\.250*, copies 2", scoped)) == 3
    assert len(re.findall(r"strength 0\.80*, copies 1", scoped)) == 1
    disabled = run(*common, "--refmod", bundle, "--refmod-strength", "0")
    assert packed(disabled) == packed(run(*common))

    # Flatten an existing bundle and safely replace that same path.
    run("bundle-refmods", "--refmod", bundle, "--refmod", inputs[0], "--output", bundle)
    header, _ = read(bundle)
    assert len(json.loads(header["__metadata__"]["refmod_meta"])["members"]) == 4
    before = bundle.read_bytes()
    run("bundle-refmods", "--refmod", inputs[0], "--refmod", root / "missing.safetensors",
        "--output", bundle, success=False)
    assert bundle.read_bytes() == before
    run("bundle-refmods", "--output", bundle, success=False)
    run("bundle-refmods", "--refmod", success=False)
    run("encode-text", "--prompt", "test", "--prompt-file", inputs[0], "--output", bundle,
        success=False)
    run("encode-text", "--inference-backend", "invalid", success=False)
    run("encode-text", "--prompt", success=False)
    run("encode-refmod", "--output", bundle, success=False)
    run("encode-refmod", "--reference-image", success=False)
    run("encode-refmod", "--reference-image", "missing.png", "--output", bundle,
        "--short-edge", "33", success=False)
    run("encode-refmod", "--short-edge", "32junk", success=False)
    run("encode-refmod", "--inference-backend", "cpu", success=False)
    run("encode-refmod", "--vulkan-arithmetic", "invalid", success=False)
    assert bundle.read_bytes() == before
    assert "encode-text" in run("--help")
    assert "bundle-refmods" in run("--help")
    assert "encode-refmod" in run("--help")
    assert "--reference-audio" in run("encode-refmod", "--help")

print("PASS: mixed bundle format, byte preservation, flattening, option scoping and CLI failures")
