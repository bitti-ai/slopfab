import hashlib
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

    def test_archives_include_both_binaries_checksums_and_execute_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            windows, linux = root / "windows", root / "linux"
            windows.mkdir()
            linux.mkdir()
            win_bytes, linux_bytes = b"MZwindows fixture", b"\x7fELFlinux fixture"
            (windows / "slopfab.exe").write_bytes(win_bytes)
            (linux / "slopfab").write_bytes(linux_bytes)
            (windows / "private-settings.json").write_text("must not ship")
            archives = package(windows, linux, root / "out", "1.2.3")
            prefix = "slopfab-1.2.3-windows-linux-x64/"
            with zipfile.ZipFile(archives[0]) as archive:
                self.assertEqual(archive.read(prefix + "windows/slopfab.exe"), win_bytes)
                self.assertEqual(archive.read(prefix + "linux/slopfab"), linux_bytes)
                self.assertIn(b"Meta Platforms", archive.read(prefix + "licenses/demucs-LICENSE.txt"))
                manifest = json.loads(archive.read(prefix + "package.json"))
                self.assertEqual(manifest["version"], "1.2.3")
                self.assertEqual(archive.read(prefix + "VERSION"), b"1.2.3\n")
                self.assertEqual(manifest["binaries"]["linux/slopfab"]["sha256"], hashlib.sha256(linux_bytes).hexdigest())
                for platform in ("windows", "linux"):
                    for filename in ("MOD.md", "main.js", "catalog.json"):
                        self.assertEqual(archive.read(prefix + f"{platform}/mods/download/{filename}"),
                                         (ROOT / "examples/mods/download" / filename).read_bytes())
                for line in archive.read(prefix + "SHA256SUMS").decode().splitlines():
                    digest, relative = line.split("  ", 1)
                    self.assertEqual(hashlib.sha256(archive.read(prefix + relative)).hexdigest(), digest)
                self.assertEqual((archive.getinfo(prefix + "linux/slopfab").external_attr >> 16) & 0o777, 0o755)
                self.assertFalse(any("private-settings" in path for path in archive.namelist()))
            with tarfile.open(archives[1]) as archive:
                self.assertEqual(archive.extractfile(prefix + "windows/slopfab.exe").read(), win_bytes)
                self.assertEqual(archive.extractfile(prefix + "linux/slopfab").read(), linux_bytes)
                self.assertTrue(archive.getmember(prefix + "linux/slopfab").mode & 0o111)
                for platform in ("windows", "linux"):
                    catalog = json.load(archive.extractfile(prefix + f"{platform}/mods/download/catalog.json"))
                    self.assertIn("whisper", catalog)
                    self.assertIn("minimaxh3", catalog)
            self.assertEqual(len(list((root / "out").iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
