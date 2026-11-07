"""SSH orchestration, binary transfer, recovery and project import without GPUs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

original = Path(sys.argv[1]).resolve()
ssh = str(Path(sys.argv[2]).resolve())
platform = "windows" if os.name == "nt" else "linux"
with tempfile.TemporaryDirectory(prefix="slopfab-remote-") as temporary:
    root = Path(temporary)
    remote = root / "remote project å & ' quoted"
    remote.mkdir()
    exe = original
    package = root / "package"
    bundled = package / platform / original.name
    bundled.parent.mkdir(parents=True)
    shutil.copy2(original, bundled)
    env = dict(os.environ, SLOPFAB_CONFIG=str(root / "config.json"))

    def run(*args, code=0, extra_env=None):
        process = subprocess.run([str(exe), "--agent", *args], cwd=root,
                                 env=dict(env, **(extra_env or {})), capture_output=True,
                                 text=True, encoding="utf-8", timeout=45)
        assert process.returncode == code, (args, process.returncode, process.stdout, process.stderr)
        events = [json.loads(line) for line in process.stdout.splitlines()]
        assert events, process.stderr
        run.events = events
        return events[-1]

    setup = ["remote", "setup", "gpu", "--host", "fixture", "--ssh", ssh,
             "--platform", platform, "--binary-dir", str(package), "--directory", str(remote)]
    run(*setup)
    digest = hashlib.sha256(bundled.read_bytes()).hexdigest()
    installed = remote / ".slopfab/bin" / digest / original.name
    assert installed.read_bytes() == bundled.read_bytes()
    assert "gpu" in run("remote", "list")["remotes"]
    run("remote", "update", "gpu")
    assert not any(e["type"] == "remote_update" for e in run.events)
    prompt = "Quotes ' \" $(echo bad) `bad` & | ; \u00e5\n--host"
    dry = run("generate", "--host", "gpu", "image", "--prompt", prompt, "--dry-run")
    assert dry["ok"] and dry["artifacts"] == []
    assert dry["arguments"][3] == prompt
    grouped = run("image", "generate", "--host", "gpu", "--prompt", prompt, "--dry-run")
    assert grouped["ok"] and grouped["arguments"][3] == prompt
    assert dry["remote_result"]["type"] == "result"
    version = run("version", "--remote", "gpu")
    assert version["remote_result"]["version"]
    assert run("help", "generate", "--host", "gpu")["remote_result"]["type"] == "help"
    assert "models" in run("capabilities", "--host", "gpu")["remote_result"]["data"]
    run("projects", "create", "remote-project", "--host", "gpu")
    assert run("projects", "show", "remote-project", "--host", "gpu")["remote_result"]["project"]["name"] == "remote-project"
    assert run("projects", "show", "remote-project", "--host", "gpu")["artifacts"] == []
    run("settings", "get", "--host", "gpu")
    job_options = ["--host", "gpu", "--jobs-dir", "test-jobs"]
    submitted = run("submit", "image", "--prompt", "test", "--dry-run", *job_options)
    job_id = submitted["job"]["job_id"]
    assert run("wait", job_id, "--timeout", "15", *job_options)["job"]["state"] == "succeeded"
    assert run("status", job_id, *job_options)["job"]["state"] == "succeeded"
    assert run("cancel", job_id, *job_options)["job"]["state"] == "succeeded"
    # Stop the detached worker and wait for its lock before removing the fixture.
    job_store = remote / "test-jobs"
    (job_store / "stop").touch()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if os.name == "nt":
                (job_store / "worker.lock").unlink()
            else:
                import fcntl
                with (job_store / "worker.lock").open("r+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except (PermissionError, BlockingIOError):
            time.sleep(.1)
    else:
        raise AssertionError("remote worker did not stop")
    run("version", "--host", "missing", code=3)
    run("remote", "generate", code=2)
    run("projects", "open", "remote-project", "--host", "gpu", code=2)
    run("remote", "setup", "bad", "--host", "-oProxyCommand=bad", "--directory", str(remote), code=2)
    run("remote", "setup", "bad", "--platform", "invalid", "--directory", str(remote), code=2)
    run("generate", "--host", "gpu", "--unknown-option", code=2)
    run("version", "--host", "gpu", code=255, extra_env={"SLOPFAB_TEST_SSH_FAIL": "1"})

    # A changed bundle is installed automatically, without replacing a running
    # version. Appending data preserves the native executable's loadability.
    with bundled.open("ab") as stream:
        stream.write(b"remote update fixture")
    run("version", "--host", "gpu")
    assert any(e["type"] == "remote_update" for e in run.events)
    updated_hash = hashlib.sha256(bundled.read_bytes()).hexdigest()
    updated = remote / ".slopfab/bin" / updated_hash / original.name
    assert updated.read_bytes() == bundled.read_bytes() and installed.exists()
    updated.write_bytes(b"corrupt executable")
    run("version", "--host", "gpu")
    assert updated.read_bytes() == bundled.read_bytes()

    # General commands run in the remote project; explicit result descriptors
    # download only outputs, including scripts and native frame extraction.
    script = "import * as sf from 'slopfab'; export default () => { sf.writeText('script-result.txt', 'remote data'); return {output: 'script-result.txt'}; };"
    # Use the filesystem API's established namespace.
    script = script.replace("sf.writeText", "sf.fs.writeText")
    scripted = run("run", "--host", "gpu", "--eval", script)
    assert (Path(scripted["receipt"]).parent / scripted["artifacts"][0]["local"]).read_text() == "remote data"
    assert Path(scripted["data"]["output"]).read_text() == "remote data"
    assert scripted["remote_result"]["data"]["output"] == "script-result.txt"
    assert run("run", "--host", "gpu", "--eval", "export default () => 'metadata only'")["artifacts"] == []
    explicit = run("version", "--host", "gpu", "--download", "script-result.txt")
    assert len(explicit["artifacts"]) == 1
    frame = remote / "frame.y4m"
    frame.write_bytes(b"YUV4MPEG2 W2 H2 F24:1 Ip A1:1 C420jpeg\nFRAME\n" + bytes([100]*4 + [128,128]))
    extracted = run("video", "frames", str(frame), "--out", "frame.png", "--host", "gpu")
    assert len(extracted["artifacts"]) == 1
    assert (Path(extracted["receipt"]).parent / extracted["artifacts"][0]["local"]).read_bytes().startswith(b"\x89PNG")
    options = ["generate", "--host", "gpu"]
    # Every possible byte, including CR/LF, NUL, Ctrl-Z and invalid UTF-8. The
    # actual native worker and Windows PowerShell launch must preserve them.
    video = remote / "scene å & '.y4m"
    music = remote / "scene å & '.wav"
    image = remote / "still.png"
    video.write_bytes(bytes(range(256)) * 1024)
    music.write_bytes(bytes(reversed(range(256))) * 511)
    image.write_bytes(b"synthetic image registration")
    artifacts = [{"path": str(path), "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "media_type": mime} for path, mime in
                 [(video, "video/x-yuv4mpeg"), (music, "audio/wav"), (image, "image/png")]]
    result = {"type": "result", "ok": True, "results": [
        {"index": 0, "ok": True, "plan": {"output": str(video)}, "artifacts": artifacts[:2]},
        {"index": 1, "ok": True, "plan": {"output": str(image)}, "artifacts": artifacts[2:]}]}
    reply = root / "reply.ndjson"

    def write_reply(files=artifacts):
        reply.write_text("\n".join(json.dumps(event) for event in [
            {"type": "progress", "stage": "denoising"}, result,
            {"type": "remote_manifest", "artifacts": files}]) + "\n", encoding="utf-8")

    write_reply()
    run("projects", "create", "demo")
    generated = run(*options, "--local-project", "demo", "video", "--prompt", prompt,
                    extra_env={"SLOPFAB_TEST_REPLY": str(reply),
                               "SLOPFAB_TEST_POWERSHELL_SHELL": "1"})
    receipt = Path(generated["receipt"])
    for artifact in generated["artifacts"]:
        local = receipt.parent / artifact["local"]
        assert local.read_bytes() == Path(artifact["remote"]).read_bytes()
        assert artifact["downloaded"]
    assert Path(generated["artifacts"][0]["local"]).parent == Path(generated["artifacts"][1]["local"]).parent
    assert len(generated["project_items"]) == 2
    assert Path(generated["index_path"]).is_file()
    assert len(json.loads(receipt.read_text())["project_items"]) == 2
    run("fetch", str(receipt))
    project = run("projects", "show", "demo")["project"]
    assert len(project["items"]) == 2  # Idempotent project registration.
    assert project["items"][0]["generation"]["arguments"][3] == prompt
    assert len(project["items"][0]["generation"]["artifacts"]) == 2
    assert len(project["items"][1]["generation"]["artifacts"]) == 1

    # Corrupted transport cannot publish a final file; retry only downloads.
    downloads = root / "failed downloads"
    run(*options, "--download-dir", str(downloads), "video", code=6,
        extra_env={"SLOPFAB_TEST_REPLY": str(reply), "SLOPFAB_TEST_TRUNCATE": "1"})
    failed_receipt = next(downloads.glob("*/remote.json"))
    assert not list(failed_receipt.parent.rglob("*.tmp"))
    assert not list(failed_receipt.parent.rglob("*.y4m"))
    resumed = run("fetch", str(failed_receipt))
    assert all(a["downloaded"] for a in resumed["artifacts"])
    local = failed_receipt.parent / resumed["artifacts"][0]["local"]
    local.write_bytes(b"x" * local.stat().st_size)
    run("fetch", str(failed_receipt), code=6)
    assert local.read_bytes().startswith(b"xxx")

    # A receipt is not allowed to write outside its run directory.
    malicious = json.loads(failed_receipt.read_text())
    malicious["artifacts"][0]["local"] = "../../outside.png"
    failed_receipt.write_text(json.dumps(malicious))
    run("fetch", str(failed_receipt), code=6)
    assert not (root / "outside.png").exists()
    write_reply([dict(artifacts[0], path="C:/output/NUL.png")])
    run(*options, "video", code=6, extra_env={"SLOPFAB_TEST_REPLY": str(reply)})
    reply.write_text("not json\n")
    run(*options, "video", code=6, extra_env={"SLOPFAB_TEST_REPLY": str(reply)})
    reply.write_text('{"type":"remote_response"}\n')
    run(*options, "video", code=6, extra_env={"SLOPFAB_TEST_REPLY": str(reply)})

    # Internal streaming errors never contaminate the binary channel with JSON.
    missing = subprocess.run([str(exe), "_ssh-fetch"],
                             input=json.dumps({"path": str(remote / "missing"), "bytes": 1}).encode(),
                             capture_output=True, timeout=15)
    assert missing.returncode != 0 and missing.stdout == b""
print("remote CLI checks passed")
