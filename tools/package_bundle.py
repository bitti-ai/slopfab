"""Create a release containing both native CLIs. No single-platform fallback."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(windows: Path, linux: Path, destination: Path, version: str):
    binaries = {"windows/slopfab.exe": windows / "slopfab.exe", "linux/slopfab": linux / "slopfab"}
    for relative, source in binaries.items():
        if not source.is_file():
            raise ValueError(f"Both platform binaries are required; missing {source}")
        with source.open("rb") as stream:
            magic = stream.read(4)
        if not (magic.startswith(b"MZ") if relative.startswith("windows/") else magic == b"\x7fELF"):
            raise ValueError(f"Wrong executable format: {source}")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Version must be MAJOR.MINOR.PATCH")
    destination.mkdir(parents=True, exist_ok=True)
    name = f"slopfab-{version}-windows-linux-x64"
    with tempfile.TemporaryDirectory(prefix="slopfab-package-") as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        for platform, source, patterns in (
            ("windows", windows, ("slopfab.exe", "slopfab.dll", "slopfab_c.lib", "capi.h", "*-LICENSE.txt", "build-info.txt")),
            ("linux", linux, ("slopfab", "libslopfab.so*", "capi.h", "*-LICENSE.txt", "build-info.txt")),
        ):
            target = stage / platform
            target.mkdir()
            for pattern in patterns:
                for path in source.glob(pattern):
                    if path.is_file():
                        shutil.copyfile(path, target / path.name)
            if platform == "linux":
                (target / "slopfab").chmod(0o755)
        for filename in ("README.md", "LICENSE"):
            shutil.copyfile(ROOT / filename, stage / filename)
        shutil.copytree(ROOT / "docs", stage / "docs", ignore=shutil.ignore_patterns("AGENTS.md*"))
        for folder in ("workflows", "mods", "motions"):
            shutil.copytree(ROOT / "examples" / folder, stage / "examples" / folder,
                            ignore=shutil.ignore_patterns("AGENTS.md*"))
        licenses = stage / "licenses"
        licenses.mkdir()
        for dependency in ("sageattention", "motioncache", "vulkan", "seedvr2", "quickjs", "thorvg", "whisper"):
            source = ROOT / "third_party" / dependency / ("LICENSE.md" if dependency == "vulkan" else "LICENSE")
            shutil.copyfile(source, licenses / f"{dependency}-LICENSE.txt")
        shutil.copyfile(ROOT / "third_party/thorvg/LODEPNG-LICENSE.txt", licenses / "LODEPNG-LICENSE.txt")
        shutil.copyfile(ROOT / "third_party/seedvr2/NOTICE", licenses / "SEEDVR2-NOTICE.txt")
        manifest = {"schema_version": 1, "version": version, "binaries": {}}
        for relative in binaries:
            path = stage / relative
            manifest["binaries"][relative] = {"sha256": checksum(path), "bytes": path.stat().st_size}
        (stage / "package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        sums = [f"{checksum(path)}  {path.relative_to(stage).as_posix()}" for path in files]
        (stage / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="ascii")
        files.append(stage / "SHA256SUMS")
        archives = []
        for suffix in (".zip", ".tar.gz"):
            final = destination / (name + suffix)
            with tempfile.NamedTemporaryFile(prefix=".package-", dir=destination, delete=False) as pending_file:
                pending = Path(pending_file.name)
            try:
                if suffix == ".zip":
                    with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as archive:
                        for path in files:
                            info = zipfile.ZipInfo(f"{name}/{path.relative_to(stage).as_posix()}")
                            info.create_system = 3
                            mode = 0o755 if path == stage / "linux/slopfab" else 0o644
                            info.external_attr = (0o100000 | mode) << 16
                            info.compress_type = zipfile.ZIP_DEFLATED
                            with path.open("rb") as source, archive.open(info, "w") as target:
                                shutil.copyfileobj(source, target)
                else:
                    with tarfile.open(pending, "w:gz") as archive:
                        def permissions(info):
                            info.mode = 0o755 if info.isdir() or info.name == name + "/linux/slopfab" else 0o644
                            return info
                        archive.add(stage, arcname=name, filter=permissions)
                pending.replace(final)
                archives.append(final)
            finally:
                pending.unlink(missing_ok=True)
        return archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=ROOT / "dist/windows")
    parser.add_argument("--linux-dir", type=Path, default=ROOT / "dist/linux")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--version", default=re.search(r"project\(slopfab .*?VERSION ([0-9.]+)", (ROOT / "CMakeLists.txt").read_text())[1])
    args = parser.parse_args()
    try:
        for archive in package(args.windows_dir, args.linux_dir, args.output_dir, args.version):
            print(f"package: wrote {archive}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"package: {error}\n")


if __name__ == "__main__":
    main()
