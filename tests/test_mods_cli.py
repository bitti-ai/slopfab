"""Mod discovery, prompt loading, malformed metadata, conflicts and live updates."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())

with tempfile.TemporaryDirectory(prefix="slopfab-mods-") as directory:
    root = Path(directory)
    config = root / "global" / "config.json"
    work = root / "work"
    work.mkdir()
    global_mods = config.parent / "mods"
    local_mods = work / "mods"

    def run(*args, code=0, machine=True, cwd=work):
        command = [exe, "--config", str(config), *args]
        if machine:
            command.append("--agent")
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", cwd=cwd, timeout=20)
        assert result.returncode == code, (command, result.stdout, result.stderr)
        if not machine:
            return result.stdout
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(events) == 1, events
        assert events[0]["schema_version"] == 1
        return events[0]

    def write(name, header=None, prompt="Follow these instructions.\n", base=local_mods):
        path = base / name / "MOD.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        if header is None:
            header = f"name: {name}\ndescription: Make a video"
        path.write_bytes(f"---\n{header}\n---\n{prompt}".encode("utf-8"))
        return path

    empty = run("mods", "list")
    assert empty["mods"] == [] and empty["errors"] == []
    assert Path(empty["roots"][0]) == global_mods
    assert Path(exe).parent / "mods" in map(Path, empty["roots"])
    assert local_mods in map(Path, empty["roots"])
    assert "No active mods" in run("mods", machine=False)
    assert run("help", "mods")["command"] == "mods"
    assert run("mods", "--help")["command"] == "mods"
    assert run("system", "capabilities")["data"]["mods"]["index_command"] == "slopfab mods list"

    prompt = "# Product shot\n\nRead ./camera.md.\nGenerate an image of cafÃ© lighting.\n---\nDo not lose this divider.\n"
    product = write("product", "name: product\ndescription: >-\n  Create a product shot\n  with soft lighting.\nauthor: 'Sam''s studio'\nversion: \"1.0\"", prompt)
    write("music", "name: music\ndescription: 'Music: with # lyrics' # comment", base=global_mods)
    disabled = write("disabled", "name: disabled\ndescription: Hidden\nenabled: false # disabled")
    (local_mods / "ignored.txt").write_text("not a mod")
    ignored = local_mods / "ignored"
    ignored.mkdir()
    (ignored / "MOD.txt").write_text("not markdown")
    index = run("mods", "index")
    assert [m["name"] for m in index["mods"]] == ["music", "product"]
    assert index["errors"] == []
    assert "prompt" not in index["mods"][0]
    assert index["mods"][1]["description"] == "Create a product shot with soft lighting."
    assert len(run("mods", "list", "--all")["mods"]) == 3
    loaded = run("mods", "load", "product")["mod"]
    assert loaded["prompt"] == prompt
    assert loaded["metadata"]["author"] == "Sam's studio"
    assert Path(loaded["directory"]) == product.parent
    assert Path(loaded["path"]) == product
    assert run("mods", "load", "product", machine=False) == product.read_text(encoding="utf-8")
    assert run("mods", "load", "disabled", code=4)["error"]["code"] == "MOD_DISABLED"
    assert run("mods", "load", "missing", code=3)["error"]["code"] == "MOD_NOT_FOUND"

    # Reindex on every invocation: prompt edits, enabling and deletion are live.
    disabled.write_text(disabled.read_text().replace("false", "true"))
    assert len(run("mods", "list")["mods"]) == 3
    disabled.unlink()
    product.write_bytes(("\ufeff---\r\nname: product\r\ndescription: |\r\n  First\r\n  Second\r\n---\r\nNew prompt\r\n").encode())
    updated = run("mods", "load", "product")["mod"]
    assert updated["prompt"] == "New prompt\r\n"
    assert updated["description"] == "First\nSecond\n"

    # No shadowing: duplicate names are omitted and diagnosed, not silently chosen.
    duplicate = write("product", base=global_mods)
    conflict = run("mods", "list")
    assert [m["name"] for m in conflict["mods"]] == ["music"]
    assert conflict["errors"][0]["code"] == "MOD_AMBIGUOUS"
    assert len(conflict["errors"][0]["paths"]) == 2
    assert run("mods", "load", "product", code=3)["error"]["code"] == "MOD_AMBIGUOUS"
    duplicate.unlink()

    bad = write("bad")
    for text in ["No frontmatter", "---\nname: bad\n", "---\nname: wrong\ndescription: x\n---\nx",
                 "---\nname: bad\ndescription: x\nname: bad\n---\nx",
                 "---\nname: bad\ndescription: x\nenabled: 'false'\n---\nx",
                 "---\nname: bad\ndescription: [a,b]\n---\nx",
                 "---\nname: bad\nunknown: x\ndescription: x\n---\nx",
                 "---\nname: bad\ndescription: x\n---\n  \n",
                 "---\nname: bad\ndescription: x\n---\nx\0y",
                 "x" * (1024 * 1024 + 1)]:
        bad.write_text(text)
        result = run("mods", "list")
        assert [e["code"] for e in result["errors"]] == ["INVALID_MOD"], result
        assert len(result["mods"]) == 2, result
        assert run("mods", "load", "bad", code=2)["error"]["code"] == "INVALID_MOD"
        assert run("mods", "load", "music")["mod"]["name"] == "music"
    for invalid_bytes in (b"\xff", b"\xc0\xaf", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe2\x82"):
        bad.write_bytes(b"---\nname: bad\ndescription: x\n---\n" + invalid_bytes)
        assert run("mods", "load", "bad", code=2)["error"]["code"] == "INVALID_MOD"
    write("bad", 'name: bad\ndescription: "Quoted \\"text\\" and \\u00e9"', "Emoji: \U0001f3ac\n")
    assert run("mods", "load", "bad")["mod"]["description"] == 'Quoted "text" and \u00e9'
    write("bad", 'name: bad\ndescription: >-\n  First paragraph.\n\n  Second paragraph.')
    assert run("mods", "load", "bad")["mod"]["description"] == "First paragraph.\nSecond paragraph."
    bad.unlink()

    for args in [("load", "../product"), ("load", str(product)), ("load",),
                 ("load", "music", "extra"), ("list", "extra"), ("unknown",)]:
        assert run("mods", *args, code=2)["error"]["code"] == "INVALID_REQUEST"

    # Working in the global config directory must not index its mods twice.
    dedup = run("mods", "list", cwd=config.parent)
    assert len(dedup["roots"]) == 2
    assert [m["name"] for m in dedup["mods"]] == ["music"]
    assert dedup["errors"] == []

print("PASS: mod indexing, Markdown context, metadata, live edits and conflict handling")
