"""Project storage, generation planning and offline page contracts; no models/GPU."""
import base64
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import wave

exe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="slopfab-projects-") as temporary:
    root = Path(temporary)
    config = root / "config.json"
    output = root / "output with spaces å"

    def run(*args, code=0, payload=None):
        process = subprocess.run([exe, "--config", str(config), "--agent", *args], cwd=root,
                                 input=json.dumps(payload) if payload is not None else None,
                                 capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert process.returncode == code, (args, process.returncode, process.stdout, process.stderr)
        events = [json.loads(line) for line in process.stdout.splitlines()]
        assert events, (args, process.stderr)
        return events[-1]

    run("settings", "set", "output_directory", str(output))
    assert Path(run("settings", "get", "output_directory")["value"]) == output
    title = '</script><img src=x onerror="window.injected=1"> & a project'
    created = run("projects", "create", "demo", "--title", title)["project"]
    directory = output / "demo"
    assert Path(created["directory"]) == directory
    assert (directory / "PROJECT.md").is_file()
    assert "exact" in created["naming_rules"]["manifest"]
    html = (directory / "index.html").read_text(encoding="utf-8")
    assert title not in html
    data = json.loads(re.search(r'<script id="project-data" type="application/json">(.*?)</script>', html, re.S)[1])
    assert data["title"] == title
    assert "fetch(" not in html and "XMLHttpRequest" not in html
    assert run("projects", "list")["projects"][0]["name"] == "demo"
    run("projects", "create", "demo", code=6)
    run("projects", "create", "../escape", code=2)
    run("projects", "create", "con", code=2)
    run("projects", "show", "missing", code=3)
    run("projects", "add", "missing", "--file", "missing.png", code=3)
    assert not (output / "missing").exists()

    png = root / "source # one.png"
    png.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aGd8AAAAASUVORK5CYII="))
    movie = root / "source.y4m"
    movie.write_bytes(b"YUV4MPEG2 W2 H2 F24:1 Ip A1:1 C420jpeg\nFRAME\n" + bytes(6))
    preview = root / "preview.mp4"
    preview.write_bytes(b"synthetic registration fixture")
    wav = root / "track.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(bytes(1600))
    request = root / "request.json"
    request.write_text(json.dumps({"schema_version": 1, "task": "image", "prompt": "Original direction"}))
    image = run("projects", "add", "demo", "--file", str(png), "--id", "cover",
                "--title", "--agent", "--request", str(request), "--prompt", title)["item"]
    assert image["file"] == "assets/cover/v1/source.png" and image["title"] == "--agent"
    assert image["request"]["prompt"] == "Original direction"
    assert (directory / image["file"]).read_bytes() == png.read_bytes()
    run("projects", "add", "demo", "--file", str(wav), "--id", "score")
    run("projects", "add", "demo", "--file", str(movie), "--id", "film", "--preview", str(preview))
    run("projects", "add", "demo", "--file", str(movie), "--id", "clip-a", "--kind", "clip", "--parent", "film", "--position", "1", "--start", "0", "--end", "2.5")
    run("projects", "add", "demo", "--file", str(movie), "--kind", "clip", code=2)
    run("projects", "add", "demo", "--file", str(movie), "--kind", "clip", "--parent", "cover", code=2)
    run("projects", "add", "demo", "--file", str(movie), "--kind", "image", code=2)
    run("projects", "update", "demo", "film", "--preview", str(png), code=2)
    run("projects", "update", "demo", "clip-a", "--start", "3", "--end", "2", code=2)
    run("projects", "add", "demo", "--file", str(png), "--id", "cover", code=2)

    def project():
        return run("projects", "show", "demo")["project"]

    def item(id):
        return next(i for i in project()["items"] if i["id"] == id)

    assert item("film")["needs_rebuild"]
    current = project()
    before = (directory / "project.json").read_bytes()
    run("projects", "remove", "demo", "clip-a", "--if-revision", str(current["revision"] - 1), code=7)
    assert (directory / "project.json").read_bytes() == before
    run("projects", "remove", "demo", "clip-a", "--if-revision", str(current["revision"]))
    assert item("clip-a")["removed"] and (directory / item("clip-a")["file"]).exists()
    run("projects", "restore", "demo", "clip-a")
    assert not item("clip-a")["removed"]
    updated = run("projects", "update", "demo", "cover", "--file", str(png))["item"]
    assert updated["file"] == "assets/cover/v2/source.png"
    assert updated["history"][0]["file"] == image["file"]
    assert (directory / image["file"]).exists()
    run("projects", "update", "demo", "film", "--file", str(preview))
    assert not item("film")["needs_rebuild"] and "preview" not in item("film")
    run("projects", "update", "demo", "clip-a", "--position", "0")
    assert item("film")["needs_rebuild"]

    # Dry runs resolve project paths but cannot create files or modify metadata.
    before = (directory / "project.json").read_bytes()
    plan = run("generate", "image", "--project", "demo", "--prompt", "test", "--dry-run")["plans"][0]
    assert Path(plan["output"]).is_relative_to(directory / "assets")
    assert not Path(plan["output"]).exists()
    plan = run("generate", "image", "--project", "demo", "--project-item", "cover", "--prompt", "new", "--dry-run")["plans"][0]
    assert Path(plan["output"]) == directory / "assets/cover/v3/source.png"
    run("generate", "video", "--project", "demo", "--project-item", "cover", "--prompt", "new", "--dry-run", code=2)
    run("generate", "image", "--project", "demo", "--project-item", "cover", "--count", "2", "--prompt", "new", "--dry-run", code=2)
    run("generate", "image", "--project", "demo", "--out", "../escape.png", "--prompt", "new", "--dry-run", code=2)
    run("generate", "image", "--project", "demo", "--overwrite", "--prompt", "new", "--dry-run", code=2)
    run("generate", "image", "--project-item", "cover", "--prompt", "new", "--dry-run", code=2)
    payload = {"schema_version": 1, "task": "image", "project": "demo", "project_item": "cover", "prompt": "new", "output_directory": str(output)}
    plan = run("validate", "--request", "-", payload=payload)["plans"][0]
    assert Path(plan["output"]) == directory / "assets/cover/v3/source.png"
    run("generate", "image", "--project", "demo", "--count", "2", "--prompt", "new", "--dry-run")
    assert (directory / "project.json").read_bytes() == before
    (directory / "index.html").unlink()
    run("projects", "render", "demo")
    assert (directory / "index.html").is_file()
    assert (directory / "project.json").read_bytes() == before

    # Reject symlink escapes on systems where symlink creation is available.
    outside = root / "outside"
    outside.mkdir()
    try:
        (output / "escaped").symlink_to(outside, target_is_directory=True)
    except OSError:
        pass
    else:
        run("projects", "create", "escaped", code=2)
        (directory / "escape").symlink_to(outside, target_is_directory=True)
        run("generate", "image", "--project", "demo", "--out", "escape/out.png", "--prompt", "new", "--dry-run", code=2)
        assert not list(outside.iterdir())

    help_text = run("help", "projects")["options"]
    assert "naming" in help_text and "--preview" in help_text
    assert "projects" in run("capabilities")["data"]
    assert "project" in run("schema")["data"]["properties"]

print("PASS: projects, safe paths, versions, clip edits, naming contract, HTML embedding and generation planning")
