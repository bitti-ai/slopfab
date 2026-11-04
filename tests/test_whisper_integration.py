"""Optional native CLI regression test with local tiny weights and the JFK WAV.

Usage: python tests/test_whisper_integration.py EXE MODEL JFK.wav [cuda|vulkan]
No downloads or Python ML packages. Not registered in weight-free CTest runs.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

exe, model, sample = (str(Path(arg).resolve()) for arg in sys.argv[1:4])
backend = sys.argv[4] if len(sys.argv) > 4 else "cuda"
with wave.open(sample) as wav:
    assert wav.getnchannels() == 1 and wav.getframerate() == 16000 and wav.getsampwidth() == 2
    pcm = wav.readframes(wav.getnframes())

with tempfile.TemporaryDirectory(prefix="whisper-integration-") as directory:
    root = Path(directory)

    def audio(name, data):
        path = root / (name + ".wav")
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(data)
        return str(path)

    def transcribe(path, *extra):
        run = subprocess.run([exe, "--agent", "transcribe", path, "--model", model,
                              "--backend", backend, *extra], cwd=root, capture_output=True,
                             encoding="utf-8", timeout=300)
        assert run.returncode == 0, (run.stdout, run.stderr)
        events = [json.loads(line) for line in run.stdout.splitlines()]
        terminal = [event for event in events if event["type"] in ("result", "error")]
        assert len(terminal) == 1 and terminal[0]["type"] == "result", events
        return terminal[0]["data"]

    silence = transcribe(audio("silence", bytes(64000)))
    assert silence["text"] == "" and silence["segments"] == []
    long_audio = audio("repeated", pcm + bytes(160000) + pcm + bytes(160000) + pcm)
    result = transcribe(long_audio, "--word-timestamps", "--out", "transcript.json")
    assert result["language"] == "en" and result["text"].lower().count("fellow americans") == 3, result
    assert json.loads((root / "transcript.json").read_text()) == result
    previous = 0
    for segment in result["segments"]:
        assert previous <= segment["start"] <= segment["end"] <= result["duration"]
        assert segment["words"]
        word_end = segment["start"]
        for word in segment["words"]:
            assert word_end <= word["start"] <= word["end"] <= segment["end"]
            word_end = word["end"]
        previous = segment["end"]
    beam = transcribe(sample, "--beam-size", "3", "--language", "en", "--out", "transcript.srt")
    assert "fellow americans" in beam["text"].lower()
    assert " --> " in (root / "transcript.srt").read_text()
print("PASS: silence, multi-window repeats, word timing, beam search, subtitles and agent results")
