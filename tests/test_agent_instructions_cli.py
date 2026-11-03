"""One-time AGENTS.md creation and context delivery, without models or a GPU."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())
template = Path(__file__).resolve().parents[1] / "templates" / "AGENTS.md"
expected = template.read_text(encoding="utf-8")

with tempfile.TemporaryDirectory(prefix="slopfab-instructions-") as temporary:
    root = Path(temporary)
    config = root / "config.json"
    config.write_text('{"schema_version":1,"weight_paths":[]}', encoding="utf-8")

    def run(directory, *args, code=0):
        result = subprocess.run([exe, "--config", str(config), *args], cwd=directory,
                                text=True, encoding="utf-8", capture_output=True, timeout=30)
        assert result.returncode == code, (args, result.stdout, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines()]

    fresh = root / "fresh å"
    fresh.mkdir()
    run(fresh, "version")
    assert not (fresh / "AGENTS.md").exists(), "human-mode invocation wrote instructions"
    events = run(fresh, "--agent", "version")
    assert len(events) == 1 and events[0]["version"]
    context = events[0]["agent_context"]
    assert Path(context["path"]) == fresh / "AGENTS.md"
    assert context["instructions"] == expected
    assert (fresh / "AGENTS.md").read_bytes() == expected.encode("utf-8")
    original_time = (fresh / "AGENTS.md").stat().st_mtime_ns
    for command in ("version", "help", "capabilities"):
        assert all("agent_context" not in event for event in run(fresh, "--agent", command))
    assert (fresh / "AGENTS.md").stat().st_mtime_ns == original_time

    # Existing instructions, including intentionally empty ones, are preserved.
    for n, content in enumerate((b"User instructions\r\nKeep these.\r\n", b"")):
        existing = root / f"existing-{n}"
        existing.mkdir()
        (existing / "AGENTS.md").write_bytes(content)
        events = run(existing, "--agent", "help")
        assert all("agent_context" not in event for event in events)
        assert (existing / "AGENTS.md").read_bytes() == content

    # The first invocation may be a failing command or a generation dry run.
    failed = root / "failed"
    failed.mkdir()
    events = run(failed, "--agent", "unknown-command", code=2)
    assert events[0]["type"] == "error" and events[0]["agent_context"]["instructions"] == expected
    planned = root / "planned"
    planned.mkdir()
    events = run(planned, "--agent", "validate", "image", "--prompt", "test")
    assert sum("agent_context" in event for event in events) == 1
    assert events[-1]["ok"] and (planned / "AGENTS.md").exists()

    # The file belongs to the working directory, not the shared config folder.
    assert not (root / "AGENTS.md").exists()
    simultaneous = root / "concurrent"
    simultaneous.mkdir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda _: run(simultaneous, "--agent", "version"), range(6)))
    assert sum("agent_context" in event for events in responses for event in events) == 1
    assert (simultaneous / "AGENTS.md").read_bytes() == expected.encode("utf-8")

    # Do not follow even a dangling AGENTS.md symlink to create another file.
    linked = root / "linked"
    linked.mkdir()
    target = root / "not-created.md"
    try:
        (linked / "AGENTS.md").symlink_to(target)
    except OSError:
        pass
    else:
        assert all("agent_context" not in event for event in run(linked, "--agent", "version"))
        assert not target.exists()

print("PASS: one-time AGENTS.md context, existing-file preservation, per-folder setup and concurrent launches")
