"""Reproduce embedded Whisper assets from a pinned MIT-licensed upstream revision."""
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np

REVISION = "86098128c0b4f24f0e2aa2994de830614b474227"
BASE = f"https://raw.githubusercontent.com/openai/whisper/{REVISION}/"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "resources/whisper"

def read(name):
    with urllib.request.urlopen(BASE + name, timeout=60) as response:
        return response.read()

def assignment(source, key):
    tree = ast.parse(source)
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == key for t in node.targets))

DEST.mkdir(parents=True, exist_ok=True)
for name in ("gpt2.tiktoken", "multilingual.tiktoken", "mel_filters.npz"):
    (DEST / name).write_bytes(read("whisper/assets/" + name))
(ROOT / "third_party/whisper/LICENSE").write_bytes(read("LICENSE"))
heads = assignment(read("whisper/__init__.py"), "_ALIGNMENT_HEADS")
languages = assignment(read("whisper/tokenizer.py"), "LANGUAGES")
profiles = {}
for name, width, layers in [("tiny", 384, 4), ("base", 512, 6), ("small", 768, 12),
                            ("medium", 1024, 24), ("large-v1", 1280, 32),
                            ("large-v2", 1280, 32), ("large-v3", 1280, 32),
                            ("large-v3-turbo", 1280, 32)]:
    for english in ([False, True] if width < 1280 else [False]):
        key = name + (".en" if english else "")
        decoder = 4 if name.endswith("turbo") else layers
        mask = np.frombuffer(gzip.decompress(base64.b85decode(heads[key])), np.bool_).reshape(decoder, width // 64)
        profiles[key] = dict(width=width, heads=width // 64, encoder_layers=layers,
                             decoder_layers=decoder, mels=128 if name.startswith("large-v3") else 80,
                             vocab=51864 if english else 51866 if name.startswith("large-v3") else 51865,
                             alignment_heads=np.argwhere(mask).tolist())
filters = np.load(DEST / "mel_filters.npz")
data = dict(source=f"https://github.com/openai/whisper/tree/{REVISION}",
            languages=list(languages.items()), profiles=profiles,
            mel_80=filters["mel_80"].ravel().tolist(), mel_128=filters["mel_128"].ravel().tolist())
(DEST / "profiles.json").write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
names = ["gpt2.tiktoken", "multilingual.tiktoken", "mel_filters.npz", "profiles.json"]
(DEST / "SHA256SUMS").write_text("".join(hashlib.sha256((DEST / n).read_bytes()).hexdigest() + "  " + n + "\n" for n in names), newline="\n")
