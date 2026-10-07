"""Measure controlled static/panning RGB float dumps from seedvr2_temporalbench.

The pan probe translates content left by two pixels per frame. Align frames
with --shift 2; ordinary real videos need optical flow, not this metric.
These consistency measurements do not establish perceptual quality.
"""
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("file")
parser.add_argument("width", type=int)
parser.add_argument("height", type=int)
parser.add_argument("--shift", type=int, default=0)
args = parser.parse_args()
if args.width < 1 or args.height < 1 or not 0 <= args.shift < args.width:
    parser.error("invalid geometry or horizontal shift")
raw = np.fromfile(args.file, dtype=np.float32)
frame_size = args.width * args.height * 3
if raw.size % frame_size or raw.size < frame_size * 2 or not np.isfinite(raw).all():
    parser.error("expected at least two complete finite RGB frames")
frames = raw.reshape(-1, args.height, args.width, 3).astype(np.float64)
previous, current = frames[:-1], frames[1:]
if args.shift:
    previous, current = previous[:, :, args.shift:], current[:, :, :-args.shift]
rmse = np.sqrt(np.mean((current - previous) ** 2, axis=(1, 2, 3)))
print(json.dumps({
    "frames": len(frames),
    "shift": args.shift,
    "mean_adjacent_rmse": float(rmse.mean()),
    "max_adjacent_rmse": float(rmse.max()),
    "adjacent_rmse": rmse.tolist(),
}, indent=2))
