"""Model-free CLI contracts for optional H3 ClipProj conditioning."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="slopfab-clipproj-") as directory:
    root = Path(directory)
    config = root / "config.json"
    config.write_text(json.dumps({"schema_version": 1, "weight_paths": [], "models": {
        "small-h3": {"family": "h3", "text_encoder": "small.safetensors",
                     "clip_projection": "projection.safetensors"}}}))

    def run(*args, code=0, payload=None):
        p = subprocess.run([exe, "--config", str(config), "--agent", *args],
                           input=None if payload is None else json.dumps(payload), cwd=root,
                           text=True, encoding="utf-8", capture_output=True, timeout=30)
        assert p.returncode == code, (args, p.returncode, p.stdout, p.stderr)
        events = [json.loads(line) for line in p.stdout.splitlines()]
        assert len(events) == 1, events
        return events[0]

    plan = run("generate", "video", "--model", "small-h3", "--prompt", "red ball", "--dry-run")["plans"][0]
    assert Path(plan["clip_projection"]) == root / "projection.safetensors"
    assert Path(plan["text_encoder"]) == root / "small.safetensors"
    request = {"schema_version": 1, "task": "video", "prompt": "red ball",
               "options": {"clip-projection": "small-projection.safetensors",
                           "text-encoder": "encoder.safetensors"}}
    result = run("validate", "--request", "-", payload=request)
    assert result["plans"][0]["clip_projection"] == "small-projection.safetensors"
    for command in ("generate", "encode-text"):
        assert "--clip-projection" in run("help", command)["options"]
    assert run("generate", "image", "--prompt", "red", "--clip-projection", "p", "--dry-run", code=2)["error"]["code"] == "INVALID_REQUEST"
    assert run("generate", "video", "--prompt", "red", "--clip-projection", "p", "--prompt-embedding", "e", "--dry-run", code=2)["error"]["code"] == "INVALID_REQUEST"
    assert run("generate", "video", "--prompt", "red", "--clip-projection", "p", "--dry-run", "--check-ready", code=3)["error"]["code"] == "MODEL_MISSING"

    # Text export must not discover unrelated transformers or VAEs. A malformed
    # explicit encoder should report that encoder, despite ambiguous H3 files.
    weights = root / "weights"
    weights.mkdir()
    header = json.dumps({"dummy": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    data = struct.pack("<Q", len(header)) + header + struct.pack("<f", 0)
    for name in ("h3_fl2va_one", "h3_fl2va_two", "qwen3vl_32b_one"):
        (weights / (name + ".safetensors")).write_bytes(data)
    run("settings", "set", "weight_paths", str(weights))
    malformed = root / "bad.safetensors"
    malformed.write_text("invalid")
    failure = run("encode-text", "--text-encoder", str(malformed), "--clip-projection", "p",
                  "--prompt", "red ball", "--output", "out.safetensors", code=1)
    assert failure["error"]["code"] != "MODEL_AMBIGUOUS"

print("PASS: ClipProj flags, JSON requests, model aliases, validation and isolated text export")
