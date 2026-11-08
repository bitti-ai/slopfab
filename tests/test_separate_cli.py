"""Offline separation checks with a tiny, synthetic HTDemucs checkpoint."""
import json
import math
import shutil
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())


def checkpoint(path, dtype="F32", mutate=None, metadata=None):
    tensors = {}

    def tensor(name, shape, values=None):
        tensors[name] = (shape, values)

    def affine(prefix, out, shape):
        tensor(prefix + ".weight", shape)
        tensor(prefix + ".bias", [out])

    for freq in (True, False):
        for i in range(4):
            c = 8 * 2**i
            cin = c // 2 if i else (4 if freq else 2)
            cout = c // 2 if i else (16 if freq else 8)
            enc = ("encoder." if freq else "tencoder.") + str(i)
            dec = ("decoder." if freq else "tdecoder.") + str(3 - i)
            affine(enc + ".conv", c, [c, cin, 8, 1] if freq else [c, cin, 8])
            affine(enc + ".rewrite", 2*c, [2*c, c, 1, 1] if freq else [2*c, c, 1])
            affine(dec + ".conv_tr", cout, [c, cout, 8, 1] if freq else [c, cout, 8])
            affine(dec + ".rewrite", 2*c, [2*c, c, 3, 3] if freq else [2*c, c, 3])
            for prefix in (enc, dec):
                for d in range(2):
                    p = prefix + f".dconv.layers.{d}"
                    affine(p + ".0", c//8, [c//8, c, 3])
                    affine(p + ".1", c//8, [c//8])
                    affine(p + ".3", 2*c, [2*c, c//8, 1])
                    affine(p + ".4", 2*c, [2*c])
                    tensor(p + ".6.scale", [c])
    tensor("freq_emb.embedding.weight", [512, 8])
    for suffix in ("", "_t"):
        affine("channel_upsampler" + suffix, 8, [8, 64, 1])
        affine("channel_downsampler" + suffix, 64, [64, 8, 1])
        affine("crosstransformer.norm_in" + suffix, 8, [8])
        for i in range(2):
            p = f"crosstransformer.layers{suffix}.{i}"
            a = p + (".cross_attn" if i else ".self_attn")
            tensor(a + ".in_proj_weight", [24, 8])
            tensor(a + ".in_proj_bias", [24])
            affine(a + ".out_proj", 8, [8, 8])
            affine(p + ".linear1", 32, [32, 8])
            affine(p + ".linear2", 8, [8, 32])
            for n in (".norm1", ".norm2", ".norm_out") + ((".norm3",) if i else ()):
                affine(p + n, 8, [8])
            tensor(p + ".gamma_1.scale", [8])
            tensor(p + ".gamma_2.scale", [8])
    # Unequal stems, exactly representable in each supported storage dtype.
    tensor("tdecoder.3.conv_tr.bias", [8], [0.125 * (i + 1) for i in range(8)])
    if mutate:
        mutate(tensors)
    header, payload = {}, bytearray()
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, (shape, values) in tensors.items():
        values = values or [0.] * math.prod(shape)
        if dtype == "BF16":
            blob = b"".join(struct.pack("<f", v)[2:] for v in values)
        else:
            blob = struct.pack("<" + ("f" if dtype == "F32" else "e") * len(values), *values)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(blob)]}
        payload.extend(blob)
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def wav(path, samples, channels=2, rate=44100):
    payload = struct.pack("<" + "f"*len(samples), *samples)
    fmt = struct.pack("<HHIIHH", 3, channels, rate, rate*channels*4, channels*4, 32)
    chunks = b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(chunks)+4) + b"WAVE" + chunks)


def read_wav(path):
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    offset, samples = 12, None
    while offset < len(data):
        name, count = data[offset:offset+4], struct.unpack_from("<I", data, offset+4)[0]
        body = data[offset+8:offset+8+count]
        if name == b"fmt ":
            assert struct.unpack_from("<HHI", body) == (3, 2, 44100)
        if name == b"data":
            samples = struct.unpack("<" + "f"*(len(body)//4), body)
        offset += 8 + count + count % 2
    assert samples is not None
    return samples


with tempfile.TemporaryDirectory(prefix="slopfab-separate-") as directory:
    root = Path(directory)

    def run(*args, ok=True):
        result = subprocess.run([exe, *map(str, args)], cwd=root, capture_output=True,
                                text=True, encoding="utf-8", timeout=40)
        assert (result.returncode == 0) == ok, (args, result.stdout, result.stderr)
        return result

    assert "--two-stems" in run("help", 'audio', 'separate').stdout
    assert '"separation"' in run("--agent", 'system', 'capabilities').stdout
    audio = [1.3 * math.sin(t * 0.07 + c) for t in range(3501) for c in range(2)]
    audio = list(struct.unpack("<" + "f"*len(audio), struct.pack("<" + "f"*len(audio), *audio)))
    wav(root / "song.wav", audio)
    checkpoint(root / "model.safetensors")
    base = ['audio', 'separate', "song.wav", "--model", "model.safetensors", "--segment", "0.05"]
    result = run("--agent", *base, "--out", "four")
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["ok"] and len(events[-1]["artifacts"]) == 4
    assert events[-1]["data"]["frames"] == len(audio)//2
    four = [read_wav(root / "four" / (name + ".wav")) for name in ("drums", "bass", "other", "vocals")]
    assert all(len(stem) == len(audio) and all(map(math.isfinite, stem)) for stem in four)
    assert four[0] != four[3]
    for dtype in ("F32", "F16", "BF16"):
        checkpoint(root / "model.safetensors", dtype)
        run(*base, "--two-stems", "vocals", "--out", dtype)
        vocals = read_wav(root / dtype / "vocals.wav")
        instrumental = read_wav(root / dtype / "instrumental.wav")
        assert max(abs(a-b) for a, b in zip(vocals, four[3])) < 1e-6
        assert max(abs(v+i-a) for v, i, a in zip(vocals, instrumental, audio)) < 3e-7
    run(*base, "--two-stems", "bass", "--backend", "auto", "--out", "bass")
    bass = read_wav(root / "bass/bass.wav")
    no_bass = read_wav(root / "bass/no_bass.wav")
    assert max(abs(v+i-a) for v, i, a in zip(bass, no_bass, audio)) < 3e-7
    before = (root / "four/vocals.wav").read_bytes()
    run(*base, "--out", "four", ok=False)
    assert (root / "four/vocals.wav").read_bytes() == before
    run(*base, "--out", "four", "--overwrite")
    assert not list(root.rglob("*.slopfab-reserved"))
    for extra in (["--unknown"], ["--two-stems", "piano"], ["--out"],
                  ["--overlap", "nan"], ["--overlap", "1"], ["--overlap", "0.2junk"],
                  ["--overlap", "-1"], ["--backend", "cuda"], ["--model", "again"]):
        run(*base, *extra, ok=False)
    for value in ("0", "8", "-2", "nan", "inf", "0.2junk"):
        run(*base[:-2], "--segment", value, ok=False)
    run("audio", "separate", ok=False)
    run("--agent", 'audio', 'separate', "missing.wav", "--model", "missing.safetensors", ok=False)
    wav(root / "vocals.wav", audio)
    result = run("audio", "separate", "vocals.wav", "--model", "model.safetensors", "--out", ".", "--overwrite", ok=False)
    assert "must differ" in result.stderr
    checkpoint(root / "bad.safetensors", mutate=lambda tensors: tensors.pop("encoder.0.conv.weight"))
    run("audio", "separate", "song.wav", "--model", "bad.safetensors", "--out", "bad", ok=False)
    assert not list((root / "bad").glob("*.wav"))
    assert not list(root.rglob("*.tmp")) and not list(root.rglob("*.slopfab-reserved"))
    for mutation, message in [
        (lambda t: t.update({"encoder.0.conv.weight": ([8, 4, 7, 1], None)}), "shape"),
        (lambda t: t.update({"unexpected.weight": ([1], None)}), "unsupported tensor"),
        (lambda t: t.update({"tdecoder.3.conv_tr.bias": ([8], [float("nan")]*8)}), "non-finite tensor"),
    ]:
        checkpoint(root / "bad.safetensors", mutate=mutation)
        result = run("audio", "separate", "song.wav", "--model", "bad.safetensors", "--out", "bad", ok=False)
        assert message in result.stderr, result.stderr
    checkpoint(root / "bad.safetensors", metadata={"sources": "vocals,other,bass,drums"})
    assert "incompatible metadata" in run("audio", "separate", "song.wav", "--model", "bad.safetensors", "--out", "bad", ok=False).stderr
    # Silence, a single mono sample, and non-finite input must all terminate safely.
    wav(root / "silence.wav", [0.] * 16, channels=1)
    run("audio", "separate", "silence.wav", "--model", "model.safetensors", "--out", "silence")
    assert all(v == 0 for v in read_wav(root / "silence/vocals.wav"))
    wav(root / "one.wav", [0.25], channels=1)
    run("audio", "separate", "one.wav", "--model", "model.safetensors", "--segment", "0.05", "--out", "one")
    assert len(read_wav(root / "one/vocals.wav")) == 2
    wav(root / "nan.wav", [float("nan"), 0.])
    run("audio", "separate", "nan.wav", "--model", "model.safetensors", ok=False)
    wav(root / "rate.wav", [0.1 * math.sin(t * 0.1) for t in range(960)], channels=1, rate=48000)
    run("audio", "separate", "rate.wav", "--model", "model.safetensors", "--ffmpeg", "missing-ffmpeg", ok=False)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(root / "rate.wav"),
                        str(root / "compressed.flac")], check=True, timeout=20)
        for source in ("rate.wav", "compressed.flac"):
            run("audio", "separate", source, "--model", "model.safetensors", "--segment", "0.05",
                "--out", source + "-stems", "--ffmpeg", ffmpeg)
            assert len(read_wav(root / (source + "-stems") / "vocals.wav")) == 882 * 2
    assert not list(root.rglob("*.tmp")) and not list(root.rglob("*.slopfab-reserved"))

print("Demucs CLI separation, dtypes, reconstruction, validation and output protection passed")
