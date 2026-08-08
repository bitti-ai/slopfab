#!/usr/bin/env python3
"""Compare decoded RGB video and float32 audio from two media files."""

import argparse
import array
import math
import subprocess


def decoder(path, selector, fmt, codec):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-map", selector,
         "-f", fmt, "-c:" + selector[-1], codec, "-"],
        stdout=subprocess.PIPE,
    )


def compare_streams(lhs, rhs, typecode, scale):
    count = 0
    sum_l = sum_r = sum_ll = sum_rr = sum_lr = 0.0
    sum_abs = sum_diff_sq = 0.0
    item = array.array(typecode).itemsize
    while True:
        a = lhs.stdout.read(65536 - 65536 % item)
        b = rhs.stdout.read(len(a)) if a else rhs.stdout.read(1)
        if not a and not b:
            break
        if len(a) != len(b):
            raise RuntimeError("decoded stream lengths differ")
        av = array.array(typecode); av.frombytes(a)
        bv = array.array(typecode); bv.frombytes(b)
        for x, y in zip(av, bv):
            x = float(x); y = float(y)
            d = x - y
            count += 1
            sum_l += x; sum_r += y
            sum_ll += x * x; sum_rr += y * y; sum_lr += x * y
            sum_abs += abs(d); sum_diff_sq += d * d
    if lhs.wait() or rhs.wait() or count == 0:
        raise RuntimeError("decoder failed or produced no samples")
    covariance = sum_lr - sum_l * sum_r / count
    var_l = sum_ll - sum_l * sum_l / count
    var_r = sum_rr - sum_r * sum_r / count
    cosine = sum_lr / math.sqrt(sum_ll * sum_rr)
    correlation = covariance / math.sqrt(var_l * var_r)
    rmse = math.sqrt(sum_diff_sq / count)
    psnr = math.inf if rmse == 0 else 20.0 * math.log10(scale / rmse)
    return count, cosine, correlation, sum_abs / count, rmse, psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("actual")
    args = parser.parse_args()

    video = compare_streams(
        decoder(args.reference, "0:v:0", "rawvideo", "rawvideo"),
        decoder(args.actual, "0:v:0", "rawvideo", "rawvideo"), "B", 255.0)
    audio = compare_streams(
        decoder(args.reference, "0:a:0", "f32le", "pcm_f32le"),
        decoder(args.actual, "0:a:0", "f32le", "pcm_f32le"), "f", 1.0)
    print("video values=%d cosine=%.9f correlation=%.9f MAE=%.6f RMSE=%.6f PSNR=%.6f" % video)
    print("audio values=%d cosine=%.9f correlation=%.9f MAE=%.9f RMSE=%.9f PSNR=%.6f" % audio)


if __name__ == "__main__":
    main()
