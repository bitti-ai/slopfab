"""Device-free music CLI defaults and unsupported attention checks."""
from pathlib import Path
import subprocess
import sys

exe = str(Path(sys.argv[1]).resolve())
for family, rate in (("yue2", "48000"), ("minimax_music3", "44100")):
    base = [exe, 'requests', 'run', "--dry-run", "--model", family, "--music-duration", "12", "--vulkan-arithmetic", "portable"]
    default = subprocess.run(base, capture_output=True, text=True)
    assert default.returncode == 0, default.stderr
    assert rate in default.stdout and "12.000000" in default.stdout
    vulkan = subprocess.run(base + ["--inference-backend", "vulkan", "--vulkan-arithmetic", "portable"], capture_output=True, text=True)
    assert vulkan.returncode == 0, vulkan.stderr
    exact = subprocess.run(base + ["--inference-backend", "vulkan", "--vulkan-arithmetic", "exact"], capture_output=True, text=True)
    assert exact.returncode != 0 and "requires portable arithmetic" in exact.stderr
    explicit = subprocess.run(base + ["--attention", "sage2"], capture_output=True, text=True)
    assert explicit.returncode != 0 and "attention overrides" in explicit.stderr
base = [exe, 'requests', 'run', "--dry-run", "--model", "yue2", "--vulkan-arithmetic", "portable"]
for sampler, predictions in (("dpmpp_2m", 6), ("midpoint", 12)):
    fast = subprocess.run(base + ["--steps", "6", "--music-settings",
                                  '{"flow_sampler":"' + sampler + '"}'], capture_output=True, text=True)
    assert fast.returncode == 0, fast.stderr
    assert f"6 ({sampler}), {predictions} predictions per chunk" in fast.stdout
bad = subprocess.run(base + ["--model", "minimax_music3", "--music-settings",
                             '{"flow_sampler":"dpmpp_2m"}'], capture_output=True, text=True)
assert bad.returncode != 0 and "only supported by YuE2" in bad.stderr
print("Music CLI defaults, flow samplers and attention validation passed")

