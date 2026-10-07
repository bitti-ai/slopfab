"""Measure known two-input-pixel translations in Real-ESRGAN probe dumps.

All paths contain interleaved RGB float32 at four times input resolution.
The crop excludes frame edges; this controlled metric is not optical flow or
a perceptual quality score. The full-frame reference uses the same model.
"""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("width", type=int, help="input width")
parser.add_argument("height", type=int, help="input height")
parser.add_argument("reference", help="untiled probe output")
parser.add_argument("candidates", nargs="+")
parser.add_argument("--border", type=int, default=32, help="input pixels cropped from each edge")
args = parser.parse_args()
w, h, border, shift = args.width*4, args.height*4, args.border*4, 8
if args.border < 0 or h <= 2*border or w <= 2*border+shift:
    parser.error("invalid image/crop geometry")

def read(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % (h*w*3) or raw.size < 2*h*w*3 or not np.isfinite(raw).all():
        parser.error("expected complete finite float RGB frames: " + path)
    return raw.reshape(-1, h, w, 3).astype(np.float64)

reference = read(args.reference)
rows = []
for path in [args.reference] + args.candidates:
    frames = read(path)
    if frames.shape != reference.shape:
        parser.error("candidate and reference dimensions must match")
    previous = frames[:-1, border:h-border, border+shift:w-border]
    current = frames[1:, border:h-border, border:w-border-shift]
    motion_rmse = np.sqrt(np.mean((current-previous)**2, axis=(1, 2, 3)))
    error = frames[:, border:h-border, border:w-border] - reference[:, border:h-border, border:w-border]
    rows.append({"file": str(Path(path)), "frames": len(frames),
                 "motion_aligned_rmse_mean": float(motion_rmse.mean()),
                 "motion_aligned_rmse_max": float(motion_rmse.max()),
                 "untiled_reference_rmse": float(np.sqrt(np.mean(error**2))),
                 "untiled_reference_max_abs": float(np.abs(error).max())})
print(json.dumps(rows, indent=2))
