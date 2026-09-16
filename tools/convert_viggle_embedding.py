"""Convert Viggle's frozen .pt conditioning to SlopFab safetensors, without Qwen.

Requires torch and numpy. The BF16 -> F32 cast preserves every stored value.
Usage: python tools/convert_viggle_embedding.py fixed_embed_fwd_anyframe.pt output.safetensors
"""
import argparse
import json
from pathlib import Path
import struct

import torch


def convert(source: Path, output: Path):
    fixed = torch.load(source, map_location="cpu", weights_only=True)
    embedding = fixed["prompt_embeds"]
    tags = fixed["text_token_tags"]
    if tuple(embedding.shape) != (1, 362, 5120):
        raise ValueError("Expected Viggle prompt_embeds [1,362,5120]")
    if tuple(tags.shape) != (362,) or tags.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected integer text_token_tags [362]")
    if not torch.isfinite(embedding).all() or not ((tags >= 0) & (tags <= 2)).all():
        raise ValueError("Invalid conditioning values or modality tags")
    values = embedding[0].float().contiguous().numpy().astype("<f4").tobytes()
    tag_bytes = tags.to(torch.int32).contiguous().numpy().astype("<i4").tobytes()
    header = {
        "__metadata__": {"source": "Viggle/Viggle-Animate",
                         "presentation": fixed["presentation"]},
        "prompt_embedding": {"dtype": "F32", "shape": [362, 5120],
                             "data_offsets": [0, len(values)]},
        "text_token_tags": {"dtype": "I32", "shape": [362],
                            "data_offsets": [len(values), len(values) + len(tag_bytes)]},
    }
    encoded = json.dumps(header, ensure_ascii=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(values)
        stream.write(tag_bytes)
    print(f"Wrote {output}: 362 x 5120 F32 values and 362 modality tags")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
