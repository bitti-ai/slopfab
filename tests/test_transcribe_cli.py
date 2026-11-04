"""Whisper CLI validation without weights, a GPU, or network access."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

exe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)

    def run(*args):
        return subprocess.run([exe, *args], cwd=root, capture_output=True,
                              encoding="utf-8", timeout=20)

    help_result = run("help", "transcribe")
    assert help_result.returncode == 0, help_result.stderr
    assert "--word-timestamps" in help_result.stdout
    assert "docs/" not in help_result.stdout
    with wave.open(str(root / "audio.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(3200))
    (root / "model.safetensors").write_bytes(b"invalid checkpoint")
    base = ["transcribe", "audio.wav", "--model", "model.safetensors"]
    for args, message in [
        (["transcribe"], "use transcribe"),
        (base + ["--beam-size", "0"], "beam size"),
        (base + ["--backend", "invalid"], "backend must"),
        (base + ["--format", "invalid"], "format must"),
        (base + ["--out", "audio.wav"], "must differ"),
        (["transcribe", "audio.wav", "--model", ".", "--out", "model.safetensors"], "must differ"),
        (base + ["--out"], "missing value"),
        (base + ["--unknown"], "unknown transcribe option"),
    ]:
        result = run(*args)
        assert result.returncode != 0 and message in result.stderr, (args, result.stdout, result.stderr)
    (root / "existing.txt").write_text("keep this", encoding="utf-8")
    result = run(*base, "--out", "existing.txt")
    assert result.returncode != 0
    assert (root / "existing.txt").read_text() == "keep this"
    result = run("--agent", "transcribe", "missing.wav", "--model", "missing.safetensors")
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode != 0 and events[-1]["type"] == "error", result.stdout
    result = run("--agent", "capabilities")
    assert result.returncode == 0 and '"transcription"' in result.stdout, result.stderr
print("Whisper CLI help, validation, output protection and agent errors passed")
