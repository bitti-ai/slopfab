"""SSH orchestration, binary transfer, recovery and project import without GPUs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

original = Path(sys.argv[1]).resolve()
ssh = str(Path(sys.argv[2]).resolve())
platform = "windows" if os.name == "nt" else "posix"
with tempfile.TemporaryDirectory(prefix="slopfab-remote-") as temporary:
    root = Path(temporary)
    remote = root / "remote project å & ' quoted"
    remote.mkdir()
    exe = remote / original.name
    shutil.copy2(original, exe)
    env = dict(os.environ, SLOPFAB_CONFIG=str(root / "config.json"))

    def run(*args, code=0, extra_env=None):
        process = subprocess.run([str(exe), "--agent", *args], cwd=root,
                                 env=dict(env, **(extra_env or {})), capture_output=True,
                                 text=True, encoding="utf-8", timeout=45)
        assert process.returncode == code, (args, process.returncode, process.stdout, process.stderr)
        events = [json.loads(line) for line in process.stdout.splitlines()]
        assert events, process.stderr
        return events[-1]

    options = ["remote", "generate", "--host", "fixture", "--ssh", ssh,
               "--platform", platform, "--executable", str(exe), "--directory", str(remote)]
    prompt = "Quotes ' \" $(echo bad) `bad` & | ; å\n--agent"
    dry = run(*options, "--", "image", "--prompt", prompt, "--dry-run")
    assert dry["ok"] and dry["artifacts"] == []
    assert dry["arguments"][2] == prompt
    assert dry["remote_result"]["type"] == "result"

    run("remote", "generate", "--host", "-oProxyCommand=bad", "--directory", str(remote),
        "--", "image", "--dry-run", code=2)
    run(*options, "--platform", "invalid", "--", "image", "--dry-run", code=2)
    run(*options, code=2)
    run(*options, "--", "--unknown-option", code=2)
    run(*options, "--", "image", "--dry-run", code=255,
        extra_env={"SLOPFAB_TEST_SSH_FAIL": "1"})

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
    generated = run(*options, "--local-project", "demo", "--", "video", "--prompt", prompt,
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
    run("remote", "fetch", str(receipt))
    project = run("projects", "show", "demo")["project"]
    assert len(project["items"]) == 2  # Idempotent project registration.
    assert project["items"][0]["generation"]["arguments"][2] == prompt
    assert len(project["items"][0]["generation"]["artifacts"]) == 2
    assert len(project["items"][1]["generation"]["artifacts"]) == 1

    # Corrupted transport cannot publish a final file; retry only downloads.
    downloads = root / "failed downloads"
    run(*options, "--local-dir", str(downloads), "--", "video", code=6,
        extra_env={"SLOPFAB_TEST_REPLY": str(reply), "SLOPFAB_TEST_TRUNCATE": "1"})
    failed_receipt = next(downloads.glob("*/remote.json"))
    assert not list(failed_receipt.parent.rglob("*.tmp"))
    assert not list(failed_receipt.parent.rglob("*.y4m"))
    resumed = run("remote", "fetch", str(failed_receipt))
    assert all(a["downloaded"] for a in resumed["artifacts"])
    local = failed_receipt.parent / resumed["artifacts"][0]["local"]
    local.write_bytes(b"x" * local.stat().st_size)
    run("remote", "fetch", str(failed_receipt), code=6)
    assert local.read_bytes().startswith(b"xxx")

    # A receipt is not allowed to write outside its run directory.
    malicious = json.loads(failed_receipt.read_text())
    malicious["artifacts"][0]["local"] = "../../outside.png"
    failed_receipt.write_text(json.dumps(malicious))
    run("remote", "fetch", str(failed_receipt), code=6)
    assert not (root / "outside.png").exists()
    write_reply([dict(artifacts[0], path="C:/output/NUL.png")])
    run(*options, "--", "video", code=6, extra_env={"SLOPFAB_TEST_REPLY": str(reply)})
    reply.write_text("not json\n")
    run(*options, "--", "video", code=6, extra_env={"SLOPFAB_TEST_REPLY": str(reply)})

    # Internal streaming errors never contaminate the binary channel with JSON.
    missing = subprocess.run([str(exe), "remote", "_fetch"],
                             input=json.dumps({"path": str(remote / "missing"), "bytes": 1}).encode(),
                             capture_output=True, timeout=15)
    assert missing.returncode != 0 and missing.stdout == b""
print("remote CLI checks passed")
