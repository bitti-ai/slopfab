"""Compare raw float32 RGB outputs from seedvr2bench (requires NumPy)."""
import argparse
import json
import math
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference")
parser.add_argument("candidate")
args = parser.parse_args()
reference = np.fromfile(args.reference, dtype=np.float32)
candidate = np.fromfile(args.candidate, dtype=np.float32)
if not reference.size or reference.shape != candidate.shape:
    raise SystemExit("outputs must be nonempty and have equal float counts")
if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
    raise SystemExit("outputs contain non-finite pixels")
delta = candidate.astype(np.float64) - reference
mse = float(np.mean(delta * delta))
norm = float(np.linalg.norm(reference.astype(np.float64)))
print(json.dumps({
    "elements": int(reference.size),
    "bit_identical": bool(np.array_equal(reference.view(np.uint32), candidate.view(np.uint32))),
    "max_absolute_error": float(np.max(np.abs(delta))),
    "rmse": math.sqrt(mse),
    "relative_l2": float(np.linalg.norm(delta)) / norm if norm else None,
    "psnr_db": -10 * math.log10(mse) if mse else None,
}, indent=2))
