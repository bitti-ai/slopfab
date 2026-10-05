"""Model-free validation of the opt-in continuation overlap constraint."""
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
print("continuation CLI checks passed")
