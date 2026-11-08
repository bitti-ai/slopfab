"""Versioned AGENTS.md creation/upgrades and context delivery, without models or a GPU."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())
template = Path(__file__).resolve().parents[1] / "templates" / "AGENTS.md"
expected = template.read_text(encoding="utf-8")
version = int(expected.splitlines()[0])
assert version > 0 and expected[0].isdigit()
assert "`slopfab projects open NAME`" in expected and "OUTPUT/NAME.html" in expected
assert "`slopfab projects update NAME`" in expected
assert "edit only OUTPUT/NAME/project.json" in expected

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
    run(fresh, 'system', 'version')
    assert not (fresh / "AGENTS.md").exists(), "human-mode invocation wrote instructions"
    events = run(fresh, "--agent", 'system', 'version')
    assert len(events) == 1 and events[0]["version"]
    context = events[0]["agent_context"]
    assert Path(context["path"]) == fresh / "AGENTS.md"
    assert context["instructions"] == expected
    assert (fresh / "AGENTS.md").read_bytes() == expected.encode("utf-8")
    original_time = (fresh / "AGENTS.md").stat().st_mtime_ns
    for command in (("system", "version"), ("help",), ("system", "capabilities")):
        assert all("agent_context" not in event for event in run(fresh, "--agent", *command))
    assert (fresh / "AGENTS.md").stat().st_mtime_ns == original_time

    # Older and unversioned files are replaced, including legacy generated files.
    for n, content in enumerate((b"User instructions\r\nKeep these.\r\n", b"",
                                 b"# Slopfab media generation\nOld bundled instructions.\n",
                                 f"{version - 1}\r\nOld version\r\n".encode())):
        existing = root / f"existing-{n}"
        existing.mkdir()
        (existing / "AGENTS.md").write_bytes(content)
        run(existing, 'system', 'version')
        assert (existing / "AGENTS.md").read_bytes() == content, "non-agent launch upgraded instructions"
        events = run(existing, "--agent", "help")
        assert events[0]["agent_context"]["instructions"] == expected
        assert (existing / "AGENTS.md").read_bytes() == expected.encode("utf-8")
        assert all("agent_context" not in event for event in run(existing, "--agent", "help"))
        assert not list(existing.glob("AGENTS.md.*.tmp"))

    # Versions compare numerically; equal/newer content and timestamps stay intact.
    for n, existing_version in enumerate((str(version), f"000{version}", str(version + 1),
                                           str(version + 10), "9" * 80)):
        preserved = root / f"preserved-{n}"
        preserved.mkdir()
        content = f"{existing_version}\r\nCustom instructions at this version.\r\n".encode()
        file = preserved / "AGENTS.md"
        file.write_bytes(content)
        timestamp = file.stat().st_mtime_ns
        events = run(preserved, "--agent", "help")
        assert all("agent_context" not in event for event in events)
        assert file.read_bytes() == content and file.stat().st_mtime_ns == timestamp

    # The first invocation may be a failing command or a generation dry run.
    failed = root / "failed"
    failed.mkdir()
    events = run(failed, "--agent", "unknown-command", code=2)
    assert events[0]["type"] == "error" and events[0]["agent_context"]["instructions"] == expected
    planned = root / "planned"
    planned.mkdir()
    events = run(planned, "--agent", 'requests', 'validate', "image", "--prompt", "test")
    assert sum("agent_context" in event for event in events) == 1
    assert events[-1]["ok"] and (planned / "AGENTS.md").exists()

    # The file belongs to the working directory, not the shared config folder.
    assert not (root / "AGENTS.md").exists()
    simultaneous = root / "concurrent"
    simultaneous.mkdir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda _: run(simultaneous, "--agent", 'system', 'version'), range(6)))
    assert sum("agent_context" in event for events in responses for event in events) == 1
    assert (simultaneous / "AGENTS.md").read_bytes() == expected.encode("utf-8")
    (simultaneous / "AGENTS.md").write_text(f"{version - 1}\nOld instructions\n", encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda _: run(simultaneous, "--agent", 'system', 'version'), range(6)))
    assert sum("agent_context" in event for events in responses for event in events) == 1
    assert (simultaneous / "AGENTS.md").read_bytes() == expected.encode("utf-8")
    assert not list(simultaneous.glob("AGENTS.md.*.tmp"))

    # Do not follow even a dangling AGENTS.md symlink to create another file.
    linked = root / "linked"
    linked.mkdir()
    target = root / "not-created.md"
    try:
        (linked / "AGENTS.md").symlink_to(target)
    except OSError:
        pass
    else:
        assert all("agent_context" not in event for event in run(linked, "--agent", 'system', 'version'))
        assert not target.exists()
        target.write_text("0\nKeep the linked file.\n", encoding="utf-8")
        assert all("agent_context" not in event for event in run(linked, "--agent", 'system', 'version'))
        assert target.read_text(encoding="utf-8") == "0\nKeep the linked file.\n"
        assert (linked / "AGENTS.md").is_symlink()

print("PASS: versioned AGENTS.md upgrades, current/future preservation, per-folder context and concurrent launches")
