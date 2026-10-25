"""Device-free music CLI defaults and unsupported attention checks."""
from pathlib import Path
import subprocess
import sys

exe = str(Path(sys.argv[1]).resolve())
for family, rate in (("yue2", "48000"), ("minimax_music3", "44100")):
    base = [exe, "generate", "--dry-run", "--model", family, "--music-duration", "12"]
    default = subprocess.run(base, capture_output=True, text=True)
    assert default.returncode == 0, default.stderr
    assert rate in default.stdout and "12.000000" in default.stdout
    explicit = subprocess.run(base + ["--attention", "sage2"], capture_output=True, text=True)
    assert explicit.returncode != 0 and "attention overrides" in explicit.stderr
print("Music CLI defaults and attention validation passed")
