"""Project storage, generation planning and offline page contracts; no models/GPU."""
import base64
import json
import os
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

    def run(*args, code=0, payload=None, env=None):
        process = subprocess.run([exe, "--config", str(config), "--agent", *args], cwd=root,
                                 input=json.dumps(payload) if payload is not None else None,
                                 capture_output=True, text=True, encoding="utf-8", timeout=30, env=env)
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
    assert Path(created["index_path"]) == output / "demo.html"
    assert not (directory / "index.html").exists()
    assert (directory / "PROJECT.md").is_file()
    assert "exact" in created["naming_rules"]["manifest"]
    html = (output / "demo.html").read_text(encoding="utf-8")
    assert title not in html
    assert 'rel="icon" type="image/svg+xml" href="data:image/svg+xml,' in html
    data = json.loads(re.search(r'<script id="project-data" type="application/json">(.*?)</script>', html, re.S)[1])
    assert data["title"] == title
    assert data["relative_directory"] == "demo"
    assert Path(data["index_path"]) == output / "demo.html"
    assert "fetch(" not in html and "XMLHttpRequest" not in html
    template = (Path(__file__).resolve().parents[1] / "templates/project.html").read_text(encoding="utf-8")

    def page_data():
        page = (output / "demo.html").read_text(encoding="utf-8")
        match = re.search(r'(<script id="project-data" type="application/json">)(.*?)(</script>)', page, re.S)
        assert match
        assert page == template.replace("__SLOPFAB_PROJECT_DATA__", match[2].strip())
        return json.loads(match[2])

    page_data()
    assert "edit only project.json" in created["naming_rules"]["page_updates"]
    listed = run("projects", "list")["projects"][0]
    assert listed["name"] == "demo"
    assert Path(listed["index_path"]) == output / "demo.html"
    run("projects", "create", "demo", code=6)
    run("projects", "create", "../escape", code=2)
    run("projects", "create", "con", code=2)
    run("projects", "show", "missing", code=3)
    run("projects", "open", "missing", code=3)
    run("projects", "open", code=2)
    run("projects", "open", "demo", "extra", code=2)
    run("projects", "open", "../escape", code=2)
    run("projects", "open", "demo", "--title", "ignored", code=2)
    run("projects", "add", "missing", "--file", "missing.png", code=3)
    run("projects", "update", "missing", code=3)
    run("projects", "update", code=2)
    run("projects", "update", "demo", "--title", "ignored", code=2)
    run("projects", "update", "demo", "--file", "ignored.png", code=2)
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
    assert page_data()["items"] == json.loads((directory / "project.json").read_text())["items"]
    canonical = directory / "assets/manual/v1/source.png"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(png.read_bytes())
    manual = run("projects", "add", "demo", "--file", str(canonical), "--id", "manual")["item"]
    assert manual["file"] == "assets/manual/v1/source.png"
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
    previous_media_revision = item("film")["media_revision"]
    run("projects", "update", "demo", "film", "--file", str(preview))
    assert not item("film")["needs_rebuild"] and "preview" not in item("film")
    assert item("film")["media_revision"] == project()["revision"] > previous_media_revision
    film_media_revision = item("film")["media_revision"]
    run("projects", "update", "demo", "clip-a", "--position", "0")
    assert item("film")["needs_rebuild"]
    assert item("film")["media_revision"] == film_media_revision

    # The JSON is the sole page source. Direct edits and new external media are
    # published through update NAME, restoring any hand-edited HTML to the template.
    manifest = directory / "project.json"
    edited = json.loads(manifest.read_text(encoding="utf-8"))
    edited["title"] = "Updated project title"
    edited["revision"] += 1
    external = directory / "assets/external/v1/source.png"
    external.parent.mkdir(parents=True)
    external.write_bytes(png.read_bytes())
    edited["items"].append({"id": "external", "kind": "image", "title": "New result",
                            "file": "assets/external/v1/source.png", "version": 1,
                            "prompt": title, "removed": False, "history": []})
    manifest.write_text(json.dumps(edited, indent=2), encoding="utf-8")
    edited_bytes = manifest.read_bytes()
    (output / "demo.html").write_text("<html>agent changed the layout</html>", encoding="utf-8")
    rebuilt = run("projects", "update", "demo", "--output-dir", str(output),
                  "--if-revision", str(edited["revision"]))
    assert rebuilt["updated"] is True
    assert Path(rebuilt["project"]["index_path"]) == output / "demo.html"
    assert manifest.read_bytes() == edited_bytes
    assert page_data()["title"] == edited["title"] and page_data()["items"] == edited["items"]
    assert title not in (output / "demo.html").read_text(encoding="utf-8")
    assert external.read_bytes() == png.read_bytes()
    assert not (directory / "index.html").exists()
    assert "only review-page source file" in (directory / "PROJECT.md").read_text(encoding="utf-8")
    valid_page = (output / "demo.html").read_bytes()
    run("projects", "update", "demo", "--if-revision", str(edited["revision"] - 1), code=7)
    edited["items"][-1]["file"] = "../outside.png"
    manifest.write_text(json.dumps(edited), encoding="utf-8")
    run("projects", "update", "demo", code=2)
    assert (output / "demo.html").read_bytes() == valid_page
    manifest.write_bytes(edited_bytes)
    # A later registration automatically refreshes the page after JSON editing.
    run("projects", "update", "demo", "external", "--title", "Registered update")
    assert page_data()["items"][-1]["title"] == "Registered update"

    # Dry runs resolve project paths but cannot create files or modify metadata.
    before = (directory / "project.json").read_bytes()
    plan = run("requests", "run", "image", "--project", "demo", "--prompt", "test", "--dry-run")["plans"][0]
    assert Path(plan["output"]).is_relative_to(directory / "assets")
    assert not Path(plan["output"]).exists()
    plan = run("requests", "run", "image", "--project", "demo", "--project-item", "cover", "--prompt", "new", "--dry-run")["plans"][0]
    assert Path(plan["output"]) == directory / "assets/cover/v3/source.png"
    run("requests", "run", "video", "--project", "demo", "--project-item", "cover", "--prompt", "new", "--dry-run", code=2)
    run("requests", "run", "image", "--project", "demo", "--project-item", "cover", "--count", "2", "--prompt", "new", "--dry-run", code=2)
    run("requests", "run", "image", "--project", "demo", "--out", "../escape.png", "--prompt", "new", "--dry-run", code=2)
    run("requests", "run", "image", "--project", "demo", "--overwrite", "--prompt", "new", "--dry-run", code=2)
    run("requests", "run", "image", "--project-item", "cover", "--prompt", "new", "--dry-run", code=2)
    payload = {"schema_version": 1, "task": "image", "project": "demo", "project_item": "cover", "prompt": "new", "output_directory": str(output)}
    plan = run("requests", "validate", "--request", "-", payload=payload)["plans"][0]
    assert Path(plan["output"]) == directory / "assets/cover/v3/source.png"
    run("requests", "run", "image", "--project", "demo", "--count", "2", "--prompt", "new", "--dry-run")
    assert (directory / "project.json").read_bytes() == before
    (output / "demo.html").unlink()
    missing_index = run("projects", "open", "demo", code=3)
    assert "projects update demo" in json.dumps(missing_index)
    assert not (output / "demo.html").exists()
    run("projects", "update", "demo")
    assert (output / "demo.html").is_file()
    assert (directory / "project.json").read_bytes() == before
    rebuilt_page = (output / "demo.html").read_bytes()
    run("projects", "render", "demo")
    assert (output / "demo.html").read_bytes() == rebuilt_page

    # Exercise the desktop launcher without opening a real browser.
    if os.name != "nt":
        launchers = root / "launchers"
        launchers.mkdir()
        opener = launchers / ("open" if sys.platform == "darwin" else "xdg-open")
        captured = root / "opened-path.txt"
        env = dict(os.environ, PATH=str(launchers), SLOPFAB_TEST_OPEN_PATH=str(captured))
        opener.write_text('#!/bin/sh\nprintf "%s\\n" "$#" "$1" > "$SLOPFAB_TEST_OPEN_PATH"\necho launcher-output\n')
        opener.chmod(0o755)
        for options in ((), ("--output-dir", str(output))):
            opened = run("projects", "open", "demo", *options, env=env)
            assert opened["opened"] is True
            assert Path(opened["project"]["index_path"]) == output / "demo.html"
            assert captured.read_text().splitlines() == ["1", str(output / "demo.html")]
        assert (directory / "project.json").read_bytes() == before
        opener.write_text("#!/bin/sh\nexit 1\n")
        assert "OPEN_FAILED" in json.dumps(run("projects", "open", "demo", code=6, env=env))
        opener.unlink()
        assert "OPEN_FAILED" in json.dumps(run("projects", "open", "demo", code=6, env=env))

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
        run("requests", "run", "image", "--project", "demo", "--out", "escape/out.png", "--prompt", "new", "--dry-run", code=2)
        assert not list(outside.iterdir())
        (outside / "index.html").write_text("outside project")
        (output / "demo.html").unlink()
        (output / "demo.html").symlink_to(outside / "index.html")
        run("projects", "open", "demo", code=2)
        run("projects", "render", "demo", code=2)
        run("projects", "update", "demo", code=2)
        assert (outside / "index.html").read_text() == "outside project"

    help_text = run("help", "projects")["options"]
    assert "naming" in help_text and "--preview" in help_text
    assert "projects open NAME" in help_text
    assert "projects update NAME" in help_text
    assert "projects" in run("system", "capabilities")["data"]
    assert run("system", "capabilities")["data"]["projects"]["open"] == "slopfab projects open NAME"
    assert run("system", "capabilities")["data"]["projects"]["update"] == "slopfab projects update NAME"
    assert "project" in run("requests", "schema")["data"]["properties"]

print("PASS: projects, safe paths, versions, clip edits, naming contract, HTML embedding and generation planning")
