"""Model-free checks for audio count validation and explicit CLI precedence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


exe = str(Path(sys.argv[1]).resolve())


def run(*args, success=True):
    result = subprocess.run(
        [exe, "generate", "--prompt", "test", "--transformer",
         "missing-audio-steps-fixture.safetensors", "--steps", "3", "--dry-run", *map(str, args)],
        capture_output=True, text=True)
    if (result.returncode == 0) != success:
        raise AssertionError((args, result.returncode, result.stdout, result.stderr))
    return result.stdout + result.stderr


assert "video 2, audio 2" in run()
assert "video 2, audio 3" in run("--audio-steps", "4")
assert "4 model evaluations" in run("--audio-steps", "4")
for value in ("0", "1", "-1", "2.5", "4junk", "bad", "1000001", "99999999999999999999"):
    run("--audio-steps", value, success=False)
run("--audio-steps", success=False)
with tempfile.TemporaryDirectory(prefix="slopfab-audio-steps-") as directory:
    path = Path(directory) / "sampling.json"
    path.write_text(json.dumps({"version": 1, "audio_steps": 9,
                                "base_sigmas": [1, 0.5, 0]}), encoding="utf-8")
    for args in (("--sampling-settings", path, "--audio-steps", "4"),
                 ("--audio-steps", "4", "--sampling-settings", path)):
        output = run(*args)
        assert "video 2, audio 3" in output
        assert "4 model evaluations" in output
print("audio steps CLI checks passed")
