"""Model-free validation of continuation constraints and retained latent bridges."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


exe = str(Path(sys.argv[1]).resolve())


def run(*args, success=True):
    result = subprocess.run(
        [exe, "generate", "--prompt", "test", "--transformer", "missing-lock-fixture.safetensors",
         "--frames", "17", "--dry-run", *map(str, args)], capture_output=True, text=True)
    if (result.returncode == 0) != success:
        raise AssertionError((args, result.returncode, result.stdout, result.stderr))
    return result.stdout + result.stderr


run("--lock-overlap", success=False)
with tempfile.TemporaryDirectory(prefix="slopfab-lock-overlap-") as directory:
    archive = Path(directory) / "source.safetensors"
    vb, ab = 14 * 96 * 4, 74 * 32 * 4
    header = {"__metadata__": {"slopfab_latents": "h3-av-v1", "width": "64", "height": "32",
                               "frames": "22", "fps": "24", "sampled": "1"},
              "video_rows": {"dtype": "F32", "shape": [14, 96], "data_offsets": [0, vb]},
              "audio_rows": {"dtype": "F32", "shape": [74, 32], "data_offsets": [vb, vb + ab]}}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    archive.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(vb + ab))
    base = ("--continue-from", archive, "--overlap-frames", "22")
    assert "conditioning only" in run(*base)
    assert "locked video and audio" in run(*base, "--lock-overlap", "--audio-steps", "5")
    run(*base, "--lock-overlap", "--skip-every", "2", success=False)
    run(*base, "--lock-overlap", "--synthetic-latents", success=False)

    # Two sources on H3's grid, long enough for asymmetric editable margins.
    vb, ab = 44 * 96 * 4, 244 * 32 * 4
    header["__metadata__"]["frames"] = "73"
    header["video_rows"] = {"dtype": "F32", "shape": [44, 96], "data_offsets": [0, vb]}
    header["audio_rows"] = {"dtype": "F32", "shape": [244, 32], "data_offsets": [vb, vb + ab]}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    archive.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(vb + ab))
    bridge = ("--bridge-from", archive, "--bridge-to", archive)
    description = run(*bridge, "--reference-image", "original.png")
    assert "29 gap frames; editable margins 17/17" in description
    assert "107 frames; locked video/audio context on both sides" in description
    assert "editable margins 0/34" in run(*bridge, "--bridge-left-margin", "0", "--bridge-right-margin", "34")
    run(*bridge, "--bridge-context", "5", "--audio-steps", "5")
    for options in [("--bridge-left-margin", "1"), ("--bridge-right-margin", "-17"),
                    ("--bridge-context", "6"), ("--bridge-right-margin", "68"),
                    ("--bridge-context", "22x"), ("--lock-overlap",),
                    ("--continue-from", archive), ("--motion-cache",),
                    ("--synthetic-latents",), ("--resolution", "32x32")]:
        run(*bridge, *options, success=False)
    run("--bridge-from", archive, success=False)
    run("--bridge-to", archive, success=False)
    run("--bridge-left-margin", "17", success=False)
print("continuation CLI checks passed")
