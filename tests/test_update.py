"""Updater fixtures: no network, real ZIPs and isolated installation directories."""
import copy
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile

DRIVER = Path(sys.argv[1]).resolve()
CLI = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from package_bundle import package as make_bundle
FILES = {
    "slopfab.exe": b"MZ" + b"new-windows" * 100,
    "slopfab": b"\x7fELF" + b"new-linux" * 100,
    "mods/download/MOD.md": b"new download docs",
    "mods/download/main.js": b"new download script",
    "mods/download/catalog.json": b'{"models":[]}',
    "mods/licenses/MOD.md": b"new licenses docs",
    "mods/licenses/main.js": b"new licenses script",
    "mods/licenses/licenses.json": b'{"licenses":[]}',
}


def run(*args, ok=True):
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=45)
    assert (result.returncode == 0) == ok, (args, result.returncode, result.stdout, result.stderr)
    return result.stdout + result.stderr


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(path, files=None, mode=zipfile.ZIP_DEFLATED, symlink=None):
    with zipfile.ZipFile(path, "w", compression=mode) as out:
        for name, data in (FILES if files is None else files).items():
            info = zipfile.ZipInfo(name)
            info.compress_type = mode
            info.create_system = 3
            info.external_attr = ((stat.S_IFLNK | 0o777) if name == symlink else
                                  (stat.S_IFREG | 0o644)) << 16
            out.writestr(info, data)


def extract(archive_path, destination, ok=True, sha=None):
    return run(DRIVER, "extract", archive_path, destination, sha or digest(archive_path),
               archive_path.stat().st_size, ok=ok)


def installed(root):
    root.mkdir()
    for name in FILES:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"previous:" + name.encode())
    for name in ["weights/model.bin", "output/project.html", "settings.json", "mods/custom/main.js"]:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"user data")


def preserved(root):
    for name in ["weights/model.bin", "output/project.html", "settings.json", "mods/custom/main.js"]:
        assert (root / name).read_bytes() == b"user data"


@contextmanager
def test_directory():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="slopfab-update-") as directory:
        try:
            yield directory
        finally:
            os.chdir(previous)


with test_directory() as directory:
    work = Path(directory)
    os.chdir(work)
    package = work / "package.zip"
    archive(package)
    tag = "v0.2.0"
    name = "slopfab-0.2.0-windows-linux-x64.zip"
    release = {"draft": False, "prerelease": False, "tag_name": tag, "assets": [{
        "name": name, "browser_download_url": f"https://github.com/bitti-ai/slopfab/releases/download/{tag}/{name}",
        "digest": "sha256:" + digest(package), "size": package.stat().st_size}]}
    metadata = work / "release.json"

    def check_release(value, installed_version="0.1.0", ok=True):
        metadata.write_text(json.dumps(value))
        return run(DRIVER, "release", metadata, installed_version, ok=ok)

    assert check_release(release) == "available"
    assert check_release(release, "0.2.0") == "up_to_date"
    assert check_release(release, "0.10.0") == "up_to_date"
    for field, value in [("draft", True), ("prerelease", True), ("tag_name", "v0.2.0-rc1"),
                         ("tag_name", "v../../evil"), ("assets", []),
                         ("assets", release["assets"] * 2)]:
        bad = copy.deepcopy(release)
        bad[field] = value
        check_release(bad, ok=False)
    for field, value in [("digest", None), ("digest", "sha256:" + "z" * 64),
                         ("browser_download_url", "https://evil.invalid/package.zip"),
                         ("size", 0), ("size", 513 * 1024 * 1024), ("size", 12.5)]:
        bad = copy.deepcopy(release)
        bad["assets"][0][field] = value
        check_release(bad, ok=False)

    payload = work / "payload"
    extract(package, payload)
    for name, data in FILES.items():
        assert (payload / name).read_bytes() == data
    if os.name != "nt":
        assert (payload / "slopfab").stat().st_mode & stat.S_IXUSR
    extract(package, work / "bad-hash", ok=False, sha="0" * 64)
    assert not (work / "bad-hash").exists()
    extract(package, payload, ok=False)  # Never reuse an existing staging directory.
    for i, path in enumerate(["../outside", "/absolute", "mods/../../outside", "C:/outside", "mods\\outside"]):
        bad = dict(FILES)
        del bad["mods/licenses/MOD.md"]
        bad[path] = b"bad"
        archive(work / "bad.zip", bad)
        extract(work / "bad.zip", work / f"bad-path-{i}", ok=False)
        assert not (work / f"bad-path-{i}").exists()
    archive(work / "bad.zip", symlink="mods/download/main.js")
    extract(work / "bad.zip", work / "bad-link", ok=False)
    missing = dict(FILES)
    del missing["slopfab"]
    archive(work / "bad.zip", missing)
    extract(work / "bad.zip", work / "missing", ok=False)
    archive(work / "bad.zip", mode=zipfile.ZIP_STORED)
    broken = bytearray((work / "bad.zip").read_bytes())
    # Damage the data, keeping the archive parseable and its external SHA valid.
    offset = 30 + struct.unpack_from("<H", broken, 26)[0] + struct.unpack_from("<H", broken, 28)[0]
    broken[offset] ^= 1
    (work / "bad.zip").write_bytes(broken)
    assert "checksum" in extract(work / "bad.zip", work / "bad-crc", ok=False)
    archive(work / "stored.zip", mode=zipfile.ZIP_STORED)
    extract(work / "stored.zip", work / "stored")
    # Exercise the actual release packager, including generated third-party licenses.
    bundles = make_bundle(payload, payload, work / "bundles", "0.2.0")
    extract(next(path for path in bundles if path.suffix == ".zip"), work / "real-bundle")
    licenses = json.loads((work / "real-bundle/mods/licenses/licenses.json").read_text(encoding="utf-8"))
    assert any(entry["source"] == "third_party/miniz/LICENSE" for entry in licenses)
    (work / "truncated.zip").write_bytes(package.read_bytes()[:-30])
    extract(work / "truncated.zip", work / "truncated", ok=False)

    root = work / "installation"
    installed(root)
    run(DRIVER, "install", root, payload, work / "backup")
    for name, data in FILES.items():
        assert (root / name).read_bytes() == data
        assert (work / "backup" / name).read_bytes() == b"previous:" + name.encode()
    preserved(root)

    blocked = work / "blocked"
    installed(blocked)
    (blocked / "slopfab.exe").unlink()
    (blocked / "slopfab.exe").mkdir()
    extract(package, work / "blocked-payload")
    run(DRIVER, "install", blocked, work / "blocked-payload", work / "blocked-backup", ok=False)
    assert not (work / "blocked-backup").exists()
    assert (blocked / "mods/download/main.js").read_bytes().startswith(b"previous:")

    linked = work / "linked"
    installed(linked)
    outside = work / "outside"
    outside.mkdir()
    shutil.rmtree(linked / "mods/download")
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(linked / "mods/download"), str(outside)],
                                capture_output=True)
        assert result.returncode == 0, result.stderr
    else:
        (linked / "mods/download").symlink_to(outside, target_is_directory=True)
    extract(package, work / "linked-payload")
    run(DRIVER, "install", linked, work / "linked-payload", work / "linked-backup", ok=False)
    assert not list(outside.iterdir())
    # Remove the junction itself before TemporaryDirectory cleanup.
    if os.name == "nt":
        os.rmdir(linked / "mods/download")

    assert "update" in run(CLI, "help")
    assert "--check" in run(CLI, "update", "--check", "--help")
    run(CLI, "update", "--unexpected", ok=False)

    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        # Fail on the last executable replacement after mods have been replaced.
        rollback = work / "rollback"
        installed(rollback)
        extract(package, work / "rollback-payload")
        handle = kernel.CreateFileW(str(rollback / "slopfab.exe"), 0x80000000, 1, None, 3, 0, None)
        assert handle != ctypes.c_void_p(-1).value
        try:
            text = run(DRIVER, "install", rollback, work / "rollback-payload", work / "rollback-backup", ok=False)
            assert "previous files restored" in text
        finally:
            kernel.CloseHandle(handle)
        for name in FILES:
            assert (rollback / name).read_bytes() == b"previous:" + name.encode()
        preserved(rollback)

        for corrupt in [False, True]:
            target = work / ("handoff-corrupt" if corrupt else "handoff")
            installed(target)
            shutil.copyfile(CLI, target / "slopfab.exe")
            stage = target / ".slopfab-update-fixture"
            stage.mkdir()
            shutil.copyfile(CLI, stage / "updater.exe")
            shutil.copyfile(package, stage / "package.zip")
            (stage / "plan.json").write_text(json.dumps({
                "version": "0.2.0", "sha256": "0" * 64 if corrupt else digest(package),
                "bytes": package.stat().st_size, "previous_sha256": digest(target / "slopfab.exe")}))
            # Wait on a real process, rather than assuming an arbitrary PID does not exist.
            parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
            try:
                run(stage / "updater.exe", "_update-install", parent.pid, ok=not corrupt)
            finally:
                parent.wait(timeout=5)
            status = json.loads((target / ".slopfab-update-result.json").read_text())
            assert status["status"] == ("failed" if corrupt else "installed"), status
            if not corrupt:
                for name, data in FILES.items():
                    assert (target / name).read_bytes() == data
                assert digest(stage / "backup/slopfab.exe") == digest(CLI)
            else:
                assert digest(target / "slopfab.exe") == digest(CLI)
            preserved(target)

print("Updater release, checksum, archive, preservation, rollback and handoff checks passed")
