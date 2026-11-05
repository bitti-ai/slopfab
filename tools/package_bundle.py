"""Package existing Windows and Linux CLIs with download and licenses mods."""
import argparse
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def license_entries():
    paths = [ROOT / "LICENSE"]
    paths.extend(sorted(path for path in (ROOT / "third_party").rglob("*")
                        if path.is_file() and ("LICENSE" in path.name.upper()
                                               or path.name.upper() in ("NOTICE", "COPYING"))))
    return [{"source": path.relative_to(ROOT).as_posix(),
             "text": path.read_text(encoding="utf-8")} for path in paths]


def package(windows: Path, linux: Path, destination: Path, version: str):
    binaries = {"slopfab.exe": windows / "slopfab.exe", "slopfab": linux / "slopfab"}
    for relative, source in binaries.items():
        if not source.is_file():
            raise ValueError(f"Both platform binaries are required; missing {source}")
        with source.open("rb") as stream:
            magic = stream.read(4)
        if not (magic.startswith(b"MZ") if relative == "slopfab.exe" else magic == b"\x7fELF"):
            raise ValueError(f"Wrong executable format: {source}")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Version must be MAJOR.MINOR.PATCH")
    destination.mkdir(parents=True, exist_ok=True)
    name = f"slopfab-{version}-windows-linux-x64"
    with tempfile.TemporaryDirectory(prefix="slopfab-package-") as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        for relative, source in binaries.items():
            shutil.copyfile(source, stage / relative)
        (stage / "slopfab").chmod(0o755)
        for mod in ("download", "licenses"):
            shutil.copytree(ROOT / "examples/mods" / mod, stage / "mods" / mod)
        (stage / "mods/licenses/licenses.json").write_text(
            json.dumps(license_entries(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        archives = []
        for suffix in (".zip", ".tar.gz"):
            final = destination / (name + suffix)
            with tempfile.NamedTemporaryFile(prefix=".package-", dir=destination, delete=False) as pending_file:
                pending = Path(pending_file.name)
            try:
                if suffix == ".zip":
                    with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as archive:
                        for path in files:
                            info = zipfile.ZipInfo(path.relative_to(stage).as_posix())
                            info.create_system = 3
                            mode = 0o755 if path == stage / "slopfab" else 0o644
                            info.external_attr = (0o100000 | mode) << 16
                            info.compress_type = zipfile.ZIP_DEFLATED
                            with path.open("rb") as source, archive.open(info, "w") as target:
                                shutil.copyfileobj(source, target)
                else:
                    with tarfile.open(pending, "w:gz") as archive:
                        def permissions(info):
                            info.mode = 0o755 if info.isdir() or info.name == "slopfab" else 0o644
                            return info
                        for path in sorted(stage.iterdir()):
                            archive.add(path, arcname=path.name, filter=permissions)
                pending.replace(final)
                archives.append(final)
            finally:
                pending.unlink(missing_ok=True)
        return archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=ROOT / "build/Release",
                        help="directory containing slopfab.exe (default: checkout/build/Release)")
    parser.add_argument("--linux-dir", type=Path, default=ROOT / "build/linux-cli",
                        help="directory containing slopfab (default: checkout/build/linux-cli)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist",
                        help="archive destination (default: checkout/dist)")
    parser.add_argument("--version", default=(ROOT / "VERSION").read_text(encoding="ascii").strip(),
                        help="package version override (default: checkout/VERSION)")
    args = parser.parse_args()
    try:
        for archive in package(args.windows_dir, args.linux_dir, args.output_dir, args.version):
            print(f"package: wrote {archive}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"package: {error}\n")


if __name__ == "__main__":
    main()
