"""Persistent settings, migration, concurrent updates and multi-path discovery."""
import concurrent.futures
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())

with tempfile.TemporaryDirectory(prefix="slopfab-settings-") as directory:
    root = Path(directory)
    config = root / "settings" / "config.json"
    work = root / "work"
    work.mkdir()

    def run(*args, code=0, selected=config):
        result = subprocess.run([exe, "--agent", "--config", str(selected), *map(str, args)],
                                capture_output=True, text=True, encoding="utf-8", cwd=work, timeout=30)
        assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(events) == 1, events
        return events[0]

    fresh = run("settings", "get")
    assert Path(fresh["path"]) == config
    assert fresh["settings"] == {"schema_version": 1, "weight_paths": []}
    assert json.loads(config.read_text()) == fresh["settings"]
    startup = run()
    assert startup["missing_dependency"] in ("cuda", "ffmpeg", "weight_paths")
    if startup["missing_dependency"] == "weight_paths":
        assert "Ask the user for a weight path" in startup["instructions"]
        assert "settings set weight_paths PATH" in startup["instructions"]

    # Paths are relative to the invocation, not the config location; values
    # survive a fresh process and duplicates collapse without losing order.
    first = work / "first weights"
    second = work / "second weights"
    first.mkdir()
    second.mkdir()
    changed = run("settings", "set", "weight_paths", "first weights", "second weights", "first weights/")
    assert changed["value"] == [str(first), str(second)]
    assert run("settings", "get", "weight_paths")["value"] == changed["value"]
    assert "weight_paths" not in (run().get("missing_dependency"),)
    assert run("settings", "remove", "weight_paths", "first weights")["value"] == [str(second)]
    assert run("settings", "remove", "weight_paths", "first weights")["removed"] is False
    assert run("settings", "set", "weight_paths", json.dumps([str(first), str(second)]))["value"] == [str(first), str(second)]

    def checkpoint(path):
        path.parent.mkdir(exist_ok=True, parents=True)
        header = json.dumps({"test": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        header += b" " * (-len(header) % 8)
        path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0))

    transformer = first / "transformer" / "h3_fl2va.safetensors"
    encoder = second / "text_encoder" / "qwen3vl_32b.safetensors"
    video = second / "vae" / "video.safetensors"
    audio = second / "vae" / "audio.safetensors"
    files = [transformer, encoder, video, audio]
    for path in files:
        checkpoint(path)

    def h3():
        return next(m for m in run("models", "list")["data"]["models"] if m["name"] == "h3")

    found = h3()
    assert found["files_present"], found
    assert Path(found["transformer"]) == transformer and Path(found["text_encoder"]) == encoder
    # Overlapping roots/direct file entries must not make the same file ambiguous.
    run("settings", "set", "weight_paths", str(first), str(second), str(transformer))
    assert h3()["files_present"]
    # Individual checkpoints can replace all directory roots.
    run("settings", "set", "weight_paths", *map(str, files))
    assert h3()["files_present"]
    duplicate = second / "h3_fl2va_other.safetensors"
    checkpoint(duplicate)
    run("settings", "set", "weight_paths", str(first), str(second))
    ambiguous = h3()
    assert not ambiguous["files_present"] and "multiple matching checkpoints" in ambiguous["reason"]

    run("settings", "set", "presets.small", '{"task":"image","model":"krea2"}')
    run("settings", "set", "models.local", '{"family":"h3"}')
    assert run("settings", "get", "presets.small.model")["value"] == "krea2"
    # Independent writers must preserve each other's keys.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: run("settings", "set", f"presets.test{i}", '{"task":"image"}'), range(8)))
    assert len(run("settings", "get", "presets")["value"]) == 9
    run("settings", "remove", "models.local")
    assert run("settings", "get", "models.local", code=3)["error"]["code"] == "SETTING_NOT_FOUND"

    before = config.read_bytes()
    for args in [("set", "weight_paths", '[1]'), ("set", "weight_paths", '['),
                 ("set", "weight_paths", ''), ("set", "models", '[]'),
                 ("set", "schema_version", '2'), ("remove", "schema_version"),
                 ("set", "unknown", 'true'), ("set", "presets..bad", '{}')]:
        assert run("settings", *args, code=2)["error"]["code"] == "INVALID_REQUEST"
        assert config.read_bytes() == before
    assert run("settings", "remove", "weight_paths")["value"] == []
    assert run("settings", "get", "presets.small")["value"]["model"] == "krea2"
    assert not h3()["files_present"], "empty settings must not discover repository weights"

    legacy = root / "legacy.json"
    legacy.write_text(json.dumps({"weights_directory": "work/first weights", "presets": {"keep": {"task": "image"}}}))
    migrated = run("settings", "get", selected=legacy)["settings"]
    assert migrated["weight_paths"] == [str(first)] and "weights_directory" not in migrated
    assert migrated["presets"]["keep"]["task"] == "image"
    assert json.loads(legacy.read_text()) == migrated

    broken = root / "broken.json"
    broken.write_text('{')
    run("settings", "get", selected=broken, code=2)
    assert broken.read_text() == '{', "invalid settings must never be replaced"

    # Default global storage is per user and independent of executable/CWD.
    env = os.environ.copy()
    env.pop("SLOPFAB_CONFIG", None)
    env["LOCALAPPDATA"] = str(root / "global")
    env["XDG_CONFIG_HOME"] = str(root / "global")
    result = subprocess.run([exe, "settings", "get", "--agent"], env=env, cwd=work,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    global_settings = json.loads(result.stdout)
    assert Path(global_settings["path"]) == root / "global" / "slopfab" / "config.json"
    assert global_settings["settings"]["weight_paths"] == []

print("PASS: global settings persistence, mutation, migration, concurrency and weight discovery")
