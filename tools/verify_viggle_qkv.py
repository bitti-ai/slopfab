"""Compare sampled checkpoint Q/K/V rows with original Viggle weights.

Requires NumPy, PyTorch and safetensors. Downloads only safetensors headers and
eight rows per sampled projection using HTTP ranges. Does not modify weights.
Exit 2 means the declared layout disagrees with the sampled original weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np
from safetensors import safe_open
import torch


REVISION = "16e05b96cf715035544f0330d1d29b123340a2ba"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("out/viggle-qkv-audit"))
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/Viggle/Viggle-Animate/resolve/{args.revision}/transformer/"

    def fetch(name, start=None, stop=None):
        url = base + name
        headers = {}
        if start is not None:
            if stop < start or stop - start >= 16 * 1024 * 1024:
                raise ValueError("Unexpectedly large tensor range")
            headers["Range"] = f"bytes={start}-{stop}"
            url += f"?slopfab_range={start}-{stop}"
        cache = args.cache / (hashlib.sha256(url.encode()).hexdigest() + ".bin")
        if cache.exists():
            data = cache.read_bytes()
        else:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
                if start is not None and response.status != 206:
                    raise RuntimeError("Server ignored range; refusing full shard download")
                limit = stop - start + 1 if start is not None else 16 * 1024 * 1024
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise RuntimeError("Server response exceeds requested size")
            with cache.open("xb") as output:
                output.write(data)
        if start is not None and len(data) != stop - start + 1:
            raise RuntimeError("Truncated range response")
        return data

    index = json.loads(fetch("diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    shard_headers = {}

    def original_rows(key, row, count=8):
        shard = index[key]
        if shard not in shard_headers:
            size = struct.unpack("<Q", fetch(shard, 0, 7))[0]
            shard_headers[shard] = (8 + size, json.loads(fetch(shard, 8, 7 + size)))
        offset, header = shard_headers[shard]
        tensor = header[key]
        if tensor["dtype"] != "BF16" or len(tensor["shape"]) != 2:
            raise ValueError("Expected upstream BF16 projection")
        columns = tensor["shape"][1]
        start = offset + tensor["data_offsets"][0] + row * columns * 2
        data = fetch(shard, start, start + count * columns * 2 - 1)
        return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).float().numpy().reshape(count, columns)

    # Regular Hadamard used by comfy_quant_v1, not the Walsh matrix.
    h4 = np.ones((4, 4), dtype=np.float32)
    h4[np.arange(4), 3 - np.arange(4)] = -1
    rotation = np.kron(np.kron(np.kron(h4, h4), h4), h4) / 16
    results = []
    with safe_open(args.checkpoint, framework="pt") as local:
        metadata = local.metadata() or {}
        keys = set(local.keys())

        def local_rows(prefix, row):
            values = local.get_slice(prefix + ".weight")[row:row + 8].float().numpy()
            if prefix + ".weight_scale" in keys:
                values *= local.get_slice(prefix + ".weight_scale")[row:row + 8].float().numpy()
                tag = json.loads(bytes(local.get_tensor(prefix + ".comfy_quant").tolist()))
                if tag.get("convrot"):
                    if tag.get("convrot_groupsize", 256) != 256:
                        raise ValueError("Audit supports ConvRot group 256 only")
                    values = (values.reshape(-1, 256) @ rotation).reshape(values.shape)
            return values

        pairs = [(f"blocks.{i}", f"transformer_blocks.{i}") for i in (0, 25, 49)]
        pairs += [(f"token_refiner.blocks.{i}", f"token_refiner.refiner_blocks.{i}") for i in (0, 1)]
        for prefix, upstream in pairs:
            for part, name in enumerate(("q", "k", "v")):
                # Head 1 distinguishes both layouts even for Q (head 0 starts at 0 in both).
                original = original_rows(upstream + f".attn.to_{name}.weight", 128)
                result = {"projection": prefix + "." + name}
                for layout, row in (("contiguous", part * 7168 + 128), ("interleaved", (3 + part) * 128)):
                    converted = local_rows(prefix + ".attn.qkv_proj", row)
                    result[layout] = {
                        "relative_error": float(np.linalg.norm(converted - original) / np.linalg.norm(original)),
                        "cosine": float(np.dot(converted.ravel(), original.ravel()) /
                                        (np.linalg.norm(converted) * np.linalg.norm(original))),
                    }
                results.append(result)
                print(json.dumps(result), flush=True)

    supported = [layout for layout in ("contiguous", "interleaved")
                 if all(row[layout]["relative_error"] < 0.05 for row in results)]
    observed = supported[0] if len(supported) == 1 else "inconclusive"
    declared = metadata.get("qkv_layout", "contiguous")
    report = {"checkpoint": str(args.checkpoint), "upstream_revision": args.revision,
              "declared_layout": declared, "observed_sample_layout": observed, "samples": results}
    if args.report:
        with args.report.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
    print(f"Declared: {declared}; observed in sampled rows: {observed}", flush=True)
    if observed == "inconclusive":
        raise SystemExit(1)
    if declared != observed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
