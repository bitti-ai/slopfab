"""Y4M/WAV attachment, request conversion and pinning validation without models."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave

exe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="slopfab-driving-audio-") as directory:
    root = Path(directory)
    config = root / "config.json"
    config.write_text('{"schema_version":1,"weight_paths":[]}', encoding="utf-8")
    movie = root / "clip & video.y4m"
    movie.write_bytes(b"YUV4MPEG2 W4 H4 F24:1 Ip A1:1 C420jpeg\n" +
                     (b"FRAME\n" + bytes([128])*24)*48)
    sound = root / "clip & sound.wav"

    def audio(seconds):
        with wave.open(str(sound), "wb") as out:
            out.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            out.writeframes(struct.pack("<h", 8192)*int(seconds*8000))

    def run(*args, ok=True, request=None):
        result = subprocess.run([exe, "--agent", "--config", str(config), *map(str, args)],
                                input=json.dumps(request) if request is not None else None,
                                cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=20)
        assert (result.returncode == 0) == ok, (args, result.stdout, result.stderr)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert events and events[-1]["type"] == ("result" if ok else "error"), events
        return events[-1]

    base = ['requests', 'run', "--prompt", "test", "--transformer", "missing-h3.safetensors",
            "--resolution", "32x32", "--frames", "56", "--dry-run"]
    video = ["--reference-video", movie]
    attached = [*video, "--driving-audio", sound]
    pinned = [*attached, "--preserve-driving-audio"]
    for seconds in (1, 2, 3):
        audio(seconds)
        plan = run(*base, *pinned)["plans"][0]
        assert plan["conditioning"] == {"pin_target_audio": True, "include_reference_audio": False}
        assert "pinned driving soundtrack" in plan["description"]
        info = plan["reference_attachments"][0]
        assert Path(info["audio_path"]) == sound and Path(info["video_path"]) == movie
        assert info["source_start_seconds"] == info["clip_start_seconds"] == 0
        assert info["source_end_seconds"] == min(seconds, 2)
        assert info["source_frames"] == seconds*8000 and info["attached_frames"] == 16000
        assert info["trimmed_frames"] == max(0, seconds-2)*8000
        assert info["padded_frames"] == max(0, 2-seconds)*8000
    audio(2)
    plan = run(*base, *attached)["plans"][0]
    assert plan["conditioning"] == {"pin_target_audio": False, "include_reference_audio": True}
    assert "1 (1 with audio)" in plan["description"]
    standalone = run(*base, *video, "--reference-audio", sound)["plans"][0]
    assert standalone["reference_attachments"] == []
    assert "1 (0 with audio)" in standalone["description"]
    for extras, message in [
        (["--driving-audio", sound], "exactly one --reference-video"),
        ([*pinned, *video], "exactly one --reference-video"),
        ([*pinned, "--driving-audio", sound], "only be supplied once"),
        ([*video, "--reference-audio", sound, "--preserve-driving-audio"], "one video with a soundtrack"),
        ([*pinned, "--reference-audio", sound], "one video with a soundtrack"),
        ([*video, "--preserve-driving-audio"], "one video with a soundtrack"),
        ([*video, "--driving-audio", ""], "requires a WAV path"),
        ([*video, "--driving-audio"], "missing value"),
    ]:
        result = run(*base, *extras, ok=False)
        assert message in result["error"]["message"], result
    for settings, message in [({"version": 1, "include_reference_audio": True}, "include_reference_audio=true"),
                              ({"version": 1, "pin_target_audio": False}, "pin_target_audio=false")]:
        policy = root / "conditioning.json"
        policy.write_text(json.dumps(settings))
        for extras in (["--conditioning-settings", policy, *pinned],
                       [*pinned, "--conditioning-settings", policy]):
            assert message in run(*base, *extras, ok=False)["error"]["message"]
    request = {"schema_version": 1, "task": "video", "prompt": "test",
               "references": [{"type": "video", "path": str(movie)}], "driving_audio": str(sound),
               "resolution": {"width": 32, "height": 32},
               "options": {"transformer": "missing-h3.safetensors", "frames": 56,
                           "preserve-driving-audio": True,
                           "conditioning-settings": {"version": 1, "include_reference_audio": False}}}
    result = run("requests", "validate", "--request", "-", request=request)
    assert result["plans"][0]["reference_attachments"][0]["source_end_seconds"] == 2
    request["options"]["conditioning-settings"]["include_reference_audio"] = True
    assert "include_reference_audio=true" in run("requests", "validate", "--request", "-", request=request, ok=False)["error"]["message"]
    request["options"].pop("conditioning-settings")
    request["options"]["driving-audio"] = request.pop("driving_audio")
    assert run("requests", "validate", "--request", "-", request=request)["plans"][0]["conditioning"]["pin_target_audio"]
    request["driving_audio"] = str(sound)
    run("requests", "validate", "--request", "-", request=request, ok=False)
    request["options"].pop("driving-audio")
    for bad in (None, [], 3, ""):
        run("requests", "validate", "--request", "-", request={**request, "driving_audio": bad}, ok=False)
    request["references"].append({"type": "video", "path": str(movie)})
    assert "exactly one" in run("requests", "validate", "--request", "-", request=request, ok=False)["error"]["message"]
    for invalid in (movie, root / "missing.wav"):
        run(*base, *video, "--driving-audio", invalid, ok=False)
    (root / "truncated.wav").write_bytes(b"RIFF")
    run(*base, *video, "--driving-audio", root / "truncated.wav", ok=False)
    caps = run("system", "capabilities")["data"]["driving_audio"]
    assert caps["video_references"] == 1 and caps["duration_policy"] == "trim_end_or_pad_silence"
    assert run("requests", "schema")["data"]["properties"]["driving_audio"]["type"] == "string"
    help_result = subprocess.run([exe, "help", "requests", "run"], cwd=root, capture_output=True, text=True)
    assert "--driving-audio" in help_result.stdout and "include_reference_audio=false" in help_result.stdout
    # Invoke the whole adapter via the embedded JavaScript API. A subsequent
    # request without attachment must not inherit provenance from the first.
    script = root / "attach.js"
    script.write_text("import sf from 'slopfab'; export default () => {\n" +
        "const first = sf.cli(" + json.dumps([*base, *map(str, pinned)]) + ");\n" +
        "const second = sf.cli(" + json.dumps([*base, *map(str, video)]) + ");\n" +
        "return {first: first.result.plans[0], second: second.result.plans[0]}; };", encoding="utf-8")
    result = run("run", script)["data"]
    assert len(result["first"]["reference_attachments"]) == 1 and result["second"]["reference_attachments"] == []
    assert result["first"]["conditioning"]["pin_target_audio"]

print("PASS: driving audio attachment, JSON/JS requests, pinning conflicts and provenance")
