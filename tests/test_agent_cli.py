"""Agent protocol, request validation and durable queue contracts; no GPU work."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

exe = str(Path(sys.argv[1]).resolve())


with tempfile.TemporaryDirectory(prefix="slopfab-agent-") as temp:
    root = Path(temp)
    config = root / "config.json"
    config.write_text(json.dumps({"schema_version": 1, "weights_directory": "empty-weights",
        "presets": {"apple-v1": {"task": "image", "model": "krea2", "options": {"steps": 8}}}}))
    (root / "empty-weights").mkdir()
    common = ["--agent", "--config", str(config), "--jobs-dir", str(root / "jobs")]

    def run(*args, input=None, code=0):
        p = subprocess.run([exe, *common, *args], input=input, text=True,
                           encoding="utf-8", capture_output=True, cwd=root, timeout=30)
        assert p.returncode == code, (args, p.returncode, p.stdout, p.stderr)
        events = [json.loads(line) for line in p.stdout.splitlines()]
        assert events and all(e["schema_version"] == 1 for e in events), p.stdout
        assert sum(e["type"] in ("result", "error", "help") for e in events) == 1, events
        return events[-1]

    startup = run()
    if "missing_dependency" in startup:
        assert len(startup["instructions"].splitlines()) == 1
        assert "commands" not in startup
    else:
        assert len(startup["instructions"].splitlines()) == 4
    help_result = run("help")
    instructions = help_result["instructions"].splitlines()
    assert instructions[1] == "When running inside agent, add --agent parameter"
    assert "--agent" not in "\n".join(instructions[2:])
    for command in help_result["commands"]:
        command_help=run(command["command"], "--help")
        assert command_help["type"] == "help"
        assert "--agent" not in command_help["usage"]
    assert "png" in run("capabilities")["data"]["output_formats"]
    assert run("schema")["data"]["additionalProperties"] is False
    assert run("models", "list")["data"]["models"]
    assert run("doctor")["data"]["ffmpeg_required"] is False

    request = {"schema_version": 1, "task": "image", "model": "krea2",
        "prompt": 'Snow, "light", and music\n日本語', "seed": "18446744073709551615",
        "resolution": {"width": 512, "height": 512}, "output": {"path": "apple.png", "format": "png"}}
    def validate(value, code=0, *flags):
        return run("validate", "--request", "-", *flags, input=json.dumps(value), code=code)
    result = validate(request)
    assert result["plans"][0]["seed"] == request["seed"]
    assert result["plans"][0]["width"] == 512
    assert result["validation"] == "structural" and not result["execution_guaranteed"]
    assert validate(request, 3, "--check-ready")["error"]["code"] == "MODEL_MISSING"
    for changes in ({"unknown": 1}, {"schema_version": 2}, {"seed": 42},
                    {"duration_seconds": 5}, {"resolution": {"width": 512, "height": 512, "typo": 1}},
                    {"options": {"typo": True}}, {"options": {"typo": False}},
                    {"seed": "18446744073709551616"}, {"count": 1.5}):
        assert validate(dict(request, **changes), 2)["error"]["code"] == "INVALID_REQUEST"
    assert validate(dict(request, output={"path": "wrong.mp4"}), 2)["error"]["code"] == "INVALID_REQUEST"
    assert run("validate", "--request", "-", input="{", code=2)["type"] == "error"
    assert run("validate", "--request", "-", input='{"schema_version":1,"task":"image","task":"music"}', code=2)["error"]["code"] == "INVALID_REQUEST"
    for changes in ({"prompt": 3}, {"output": []}, {"references": [{"type": "image"}]},
                    {"resolution": {"width": "512", "height": 512}}, {"prompt": "a\0b"}):
        assert validate(dict(request, **changes), 2)["error"]["code"] == "INVALID_REQUEST"
    assert run("validate", "--request", "-", "--steps", "3", input=json.dumps(request), code=2)["type"] == "error"
    # Values which resemble options must remain data.
    for prompt in ("--agent", "--request", "help", "--help", "--idempotency-key"):
        assert run("validate", "image", "--prompt", prompt, "--seed", "42")["ok"]
    assert run("validate", "video", "--model", "krea2", "--prompt", "test", code=4)["error"]["code"] == "UNSUPPORTED_CAPABILITY"
    video = {"schema_version": 1, "task": "video", "prompt": "test", "duration_seconds": 2,
             "resolution": {"width": 32, "height": 32}}
    planned = validate(video)["plans"][0]
    assert planned["requested_duration_seconds"] == 2 and planned["duration_seconds"] >= 2
    music = {"schema_version": 1, "task": "music", "prompt": "piano", "duration_seconds": 4,
             "options": {"vulkan-arithmetic": "portable"}}
    assert validate(music)["plans"][0]["model"] == "yue2"

    # Persist stdin requests, start a worker, and inspect the durable result.
    submitted = run("submit", "--request", "-", "--dry-run", "--idempotency-key", "same",
                    input=json.dumps(request))["job"]
    job_id = submitted["job_id"]
    again = run("submit", "--request", "-", "--dry-run", "--idempotency-key", "same",
                input=json.dumps(request))["job"]
    assert again["job_id"] == job_id
    assert run("submit", "--request", "-", "--dry-run", "--idempotency-key", "same",
               input=json.dumps(dict(request, prompt="changed")), code=2)["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    done = run("wait", job_id, "--timeout", "15")["job"]
    assert done["state"] == "succeeded", done
    assert done["result"]["plans"][0]["seed"] == request["seed"]
    assert Path(done["events_path"]).exists()
    assert run("cancel", job_id)["job"]["state"] == "succeeded"
    assert run("status", "../bad", code=2)["type"] == "error"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        ids=list(pool.map(lambda _: run("submit", "--request", "-", "--dry-run",
            "--idempotency-key", "concurrent", input=json.dumps(request))["job"]["job_id"], range(4)))
    assert len(set(ids))==1
    assert run("wait",ids[0],"--timeout","15")["job"]["state"]=="succeeded"

    # A failed job must have a terminal structured error and not get retried.
    failed = run("submit", "--request", "-", "--dry-run", "--check-ready",
                 input=json.dumps(request))["job"]["job_id"]
    done = run("wait", failed, "--timeout", "15")["job"]
    assert done["state"] == "failed" and done["result"]["error"]["code"] == "MODEL_MISSING", done

    # Idle workers release their OS locks on exit. Wait on the actual lock on
    # POSIX: unlinking an open file succeeds there and does not release flock.
    # Tell this worker to stop through its dedicated store sentinel.
    (root / "jobs" / "stop").touch()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if os.name == "nt":
                (root / "jobs" / "worker.lock").unlink()
            else:
                import fcntl
                with (root / "jobs" / "worker.lock").open("r+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except (PermissionError, BlockingIOError):
            time.sleep(.1)
    else:
        raise AssertionError("worker did not release its lock after stopping")

    # Public manifests survive worker death. Status detects an abandoned run;
    # queued cancellation and timeout never invoke inference.
    interrupted_id="a"*32
    queued_id="b"*32
    for identity,state in ((interrupted_id,"running"),(queued_id,"queued")):
        directory=root/"jobs"/identity
        directory.mkdir()
        record=dict(submitted,job_id=identity,state=state,idempotency_key="")
        (directory/"job.json").write_text(json.dumps(record))
    assert run("status",interrupted_id)["job"]["state"]=="interrupted"
    timed=run("wait",queued_id,"--timeout","0",code=8)
    assert timed["timed_out"] and timed["job"]["state"]=="queued"
    assert run("cancel",queued_id)["job"]["state"]=="cancelled"
    assert run("worker","--idle-seconds","1")["worker_finished"]

print("PASS: agent help, discovery, JSON validation, protocol isolation and persistent jobs")
