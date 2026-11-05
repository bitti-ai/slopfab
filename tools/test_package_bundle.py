import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from package_bundle import ROOT, package


class PackageTests(unittest.TestCase):
    def test_requires_both_platforms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            windows, linux = root / "windows", root / "linux"
            windows.mkdir()
            linux.mkdir()
            (windows / "slopfab.exe").write_bytes(b"MZwindows fixture")
            with self.assertRaisesRegex(ValueError, "Both platform"):
                package(windows, linux, root / "out", "1.2.3")
            self.assertFalse((root / "out").exists())
            (linux / "slopfab").write_bytes(b"MZwrong platform")
            with self.assertRaisesRegex(ValueError, "Wrong executable"):
                package(windows, linux, root / "out", "1.2.3")

    def test_archives_have_only_flat_binaries_and_mods(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            windows, linux = root / "windows", root / "linux"
            windows.mkdir()
            linux.mkdir()
            win_bytes, linux_bytes = b"MZwindows fixture", b"\x7fELFlinux fixture"
            (windows / "slopfab.exe").write_bytes(win_bytes)
            (linux / "slopfab").write_bytes(linux_bytes)
            for name in ("private-settings.json", "slopfab.dll", "capi.h", "build-info.txt"):
                (windows / name).write_text("must not ship")
            (linux / "libslopfab.so").write_text("must not ship")
            archives = package(windows, linux, root / "out", "1.2.3")
            self.assertEqual([path.name for path in archives], [
                "slopfab-1.2.3-windows-linux-x64.zip", "slopfab-1.2.3-windows-linux-x64.tar.gz"])
            expected = {"slopfab.exe", "slopfab", "mods/download/MOD.md",
                        "mods/download/main.js", "mods/download/catalog.json",
                        "mods/licenses/MOD.md", "mods/licenses/main.js", "mods/licenses/licenses.json"}
            with zipfile.ZipFile(archives[0]) as archive:
                self.assertEqual(set(archive.namelist()), expected)
                contents = {name: archive.read(name) for name in expected}
                self.assertEqual(contents["slopfab.exe"], win_bytes)
                self.assertEqual(contents["slopfab"], linux_bytes)
                self.assertEqual((archive.getinfo("slopfab").external_attr >> 16) & 0o777, 0o755)
                for mod in ("download", "licenses"):
                    for filename in ("MOD.md", "main.js"):
                        self.assertEqual(contents[f"mods/{mod}/{filename}"],
                                         (ROOT / "examples/mods" / mod / filename).read_bytes())
                licenses = json.loads(contents["mods/licenses/licenses.json"])
                sources = {entry["source"] for entry in licenses}
                self.assertEqual(len(sources), len(licenses))
                self.assertTrue({"LICENSE", "third_party/seedvr2/NOTICE", "third_party/demucs/LICENSE",
                                 "third_party/thorvg/LODEPNG-LICENSE.txt",
                                 "third_party/thorvg/src/loaders/webp/LICENSE"} <= sources)
                expected_sources = {"LICENSE"} | {
                    path.relative_to(ROOT).as_posix() for path in (ROOT / "third_party").rglob("*")
                    if path.is_file() and ("LICENSE" in path.name.upper()
                                          or path.name.upper() in ("NOTICE", "COPYING"))}
                self.assertEqual(sources, expected_sources)
                for entry in licenses:
                    self.assertEqual(entry["text"], (ROOT / entry["source"]).read_text(encoding="utf-8"))
                catalog = json.loads(contents["mods/download/catalog.json"])
                self.assertIn("whisper", catalog)
            with tarfile.open(archives[1]) as archive:
                self.assertEqual({member.name for member in archive if member.isfile()}, expected)
                self.assertEqual({member.name.split("/")[0] for member in archive},
                                 {"slopfab", "slopfab.exe", "mods"})
                for name in expected:
                    self.assertEqual(archive.extractfile(name).read(), contents[name])
                self.assertEqual(archive.getmember("slopfab").mode, 0o755)
            self.assertEqual(len(list((root / "out").iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
