"""Offline analysis. Not part of the build, not part of the runtime, not
imported by anything that ships — the no-Python constraint is about what the
binary needs at run time, and this is the harness that pinned a fact no shape
check could have caught.

WHAT IT ESTABLISHED
-------------------
The nvfp4 text-encoder checkpoint's `weight_scale` is *not* the row-major
[out, in/16] its header shape suggests. Its bytes are in the 128x4 tile
swizzle, and the E2M1 nibble order is the opposite of the obvious reading:
the HIGH nibble holds the even-indexed element.

Both errors are the silent kind. Reading the scales row-major gives relative
L2 0.77 against a correct reference at correlation +0.79 — finite, correctly
shaped, plausibly scaled garbage. Getting the nibble order wrong gives
correlation +0.0008, which is to say none at all, and still produces output
that passes every finiteness and magnitude check in the test suite.

THE SIGNATURE OF THIS CLASS OF BUG
----------------------------------
Right marginal distribution, wrong pairing. This is the part that generalises,
and it is why every aggregate statistic looked healthy while the weights were
scrambled:

  - the stored block scales matched `amax/6` of their block to 0.4% in the
    median, and so did the scales implied by an independent reference. Both
    marginals were correct.
  - but the *per-block ratio* between them had a log2 spread of 0.64, i.e. the
    two were pairing block j of row o with entirely different blocks.

Any permutation of a tensor preserves its histogram, its norm, its min and its
max. So a checkpoint whose summary statistics are all exactly right and whose
per-element agreement is poor has an ordering bug, not a scaling bug — and you
can see that without a second checkpoint to compare against, by fitting the
scale each block *would* need and comparing its distribution with the stored
one. Matching distributions plus non-matching pairs is conclusive.

METHOD
------
The two shipped builds of Qwen3-VL-32B are the same model quantised twice, so
one is a reference for the other. `model.embed_tokens` agrees between them at
relative L2 0.0094 / correlation +0.99996, which is the control that says so
before any conclusion is drawn from the comparison. ConvRot is orthogonal
within each 256-wide group, so de-rotating the int8 build with the same
Hadamard recovers the unrotated weight and the two become directly comparable
elementwise.

Elementwise, not by norms: norms cannot see a permutation, which is the whole
point above.

USAGE
-----
  python tools/nvfp4_layout_probe.py [nvfp4.safetensors] [int8_convrot.safetensors]

Needs numpy and the two licence-restricted checkpoints under weights/, which
are not committed. Prints the grid; asserts nothing.
"""
import json
import mmap
import sys

import numpy as np

DEFAULT_NVFP4 = "weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_INT8 = "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors"

NUMPY_DTYPE = {"F32": np.float32, "F64": np.float64, "I64": np.int64, "I32": np.int32,
               "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}

# E2M1: 1 sign, 2 exponent, 1 mantissa. Sixteen values, so a table is simplest.
E2M1 = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                 -0., -.5, -1., -1.5, -2., -3., -4., -6.], np.float32)


def e4m3_table():
    """OCP E4M3: bias 7, no infinities, 0x7F/0xFF reserved for NaN."""
    t = np.zeros(256, np.float32)
    for v in range(256):
        sign = -1.0 if (v & 0x80) else 1.0
        exp, man = (v >> 3) & 0xF, v & 0x7
        if exp == 0:
            t[v] = sign * man * 2.0 ** -9
        elif exp == 0xF and man == 7:
            t[v] = np.nan
        else:
            t[v] = sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)
    return t


E4M3 = e4m3_table()


class SafeTensors:
    def __init__(self, path):
        self.file = open(path, "rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        n = int.from_bytes(self.map[:8], "little")
        self.header = json.loads(self.map[8:8 + n].decode("utf-8"))
        self.base = 8 + n

    def shape(self, name):
        return self.header[name]["shape"]

    def raw(self, name):
        start, end = self.header[name]["data_offsets"]
        return np.frombuffer(self.map, np.uint8, end - start, self.base + start)

    def get(self, name):
        entry = self.header[name]
        dtype, buf = entry["dtype"], self.raw(name)
        if dtype in NUMPY_DTYPE:
            out = buf.view(NUMPY_DTYPE[dtype])
        elif dtype == "BF16":
            out = (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            out = buf.view(np.float16).astype(np.float32)
        elif dtype == "F8_E4M3":
            out = E4M3[buf]
        else:
            raise ValueError("unhandled dtype " + dtype)
        return out.reshape(entry["shape"])


def scale_offsets(rows, blocks):
    """Byte offset of the e4m3 scale for (output row m, block k), where
    `blocks` = in_features / 16.

    The tile is 128 rows x 4 blocks = 512 bytes, laid out row-of-tiles major.
    Inside a tile the slowest index is (m % 32) at stride 16, then which
    quarter of the 128 rows at stride 4, then (k % 4) at stride 1 — so four
    consecutive blocks of one row are four consecutive bytes, which is the one
    part of this visible by eye.

    Requires rows % 128 == 0 and blocks % 4 == 0. Both hold for all seven
    linears of the shipped checkpoint with nothing left over. A padded layout
    is plausible but no file exercises it, so the caller should refuse it
    rather than guess.
    """
    m = np.arange(rows)[:, None]
    k = np.arange(blocks)[None, :]
    tile = (m // 128) * (blocks // 4) + (k // 4)
    return tile * 512 + (m % 32) * 16 + ((m % 128) // 32) * 4 + (k % 4)


def dequant_nvfp4(st, prefix, high_nibble_even=True, swizzled=True):
    packed = st.raw(prefix + ".weight").reshape(st.shape(prefix + ".weight"))
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    out = np.empty((rows, cols), np.float32)
    low, high = E2M1[packed & 0x0F], E2M1[packed >> 4]
    out[:, 0::2], out[:, 1::2] = (high, low) if high_nibble_even else (low, high)

    raw = np.asarray(st.raw(prefix + ".weight_scale"))
    blocks = cols // 16
    idx = scale_offsets(rows, blocks).ravel() if swizzled else np.arange(raw.size)
    out *= np.repeat(E4M3[raw[idx]].reshape(rows, blocks), 16, axis=1)
    return out * float(np.asarray(st.get(prefix + ".weight_scale_2")).reshape(-1)[0])


def hadamard_256():
    """H = kron^4(h4) / 16 with the *regular* h4, not the Sylvester one.
    Symmetric, orthogonal and involutory, so it de-rotates as well as rotates.
    The wrong h4 gives relative error 1.4 rather than an exception — see
    docs/convrot_notes.md."""
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], np.float64)
    m = np.array([[1.0]])
    while m.shape[0] < 256:
        m = np.kron(h4, m)
    return (m / 16.0).astype(np.float32)


def derotate(w, hadamard):
    rows, cols = w.shape
    return (w.reshape(rows * (cols // 256), 256) @ hadamard).reshape(rows, cols)


def score(candidate, reference):
    err = np.sqrt(((candidate - reference).astype(np.float64) ** 2).sum())
    mag = np.sqrt((reference.astype(np.float64) ** 2).sum())
    corr = np.corrcoef(candidate.ravel()[::11].astype(np.float64),
                       reference.ravel()[::11].astype(np.float64))[0, 1]
    return err / mag, corr


def main(argv):
    nvfp4 = SafeTensors(argv[1] if len(argv) > 1 else DEFAULT_NVFP4)
    int8 = SafeTensors(argv[2] if len(argv) > 2 else DEFAULT_INT8)
    hadamard = hadamard_256()

    # Control first: are these two files the same model at all? Everything
    # below is meaningless if not.
    rows = np.arange(0, 151936, 997)
    quantised = nvfp4.get("model.embed_tokens.weight")[rows].astype(np.float32)
    per_row = nvfp4.get("model.embed_tokens.weight_scale").reshape(-1)[rows][:, None]
    reference = int8.get("model.embed_tokens.weight")[rows].astype(np.float32)
    print("control -- embed_tokens, int8*scale vs the other build's bf16 table:")
    print("   multiply  relL2 %.4f  corr %+.5f" % score(quantised * per_row, reference))
    print("   divide    relL2 %.4f  corr %+.5f" % score(quantised / per_row, reference))

    def grid(prefix, fold, fold_name):
        rotated = (int8.get(prefix + ".weight").astype(np.float32) *
                   int8.get(prefix + ".weight_scale").reshape(-1, 1))
        plain = derotate(rotated, hadamard)
        print("\n%s" % prefix)
        for high in (True, False):
            for mult, name in ((fold, "* " + fold_name), (1.0 / fold, "/ " + fold_name),
                               (np.ones_like(fold), "no fold")):
                w = dequant_nvfp4(nvfp4, prefix, high_nibble_even=high)
                print("   %-10s %-18s relL2 %7.4f  corr %+.6f"
                      % ("high=even" if high else "low=even", name,
                         *score(w * mult[None, :], plain)))
        w = dequant_nvfp4(nvfp4, prefix, high_nibble_even=True, swizzled=False)
        print("   %-10s %-18s relL2 %7.4f  corr %+.6f   <- scales read row-major"
              % ("high=even", "* " + fold_name, *score(w * fold[None, :], plain)))

    # Two of the seven linears store the AWQ scale; the other five had it folded
    # into the preceding norm, so for those the fold is recovered as the ratio
    # between the two builds' norm weights.
    for prefix in ("model.layers.0.self_attn.o_proj", "model.layers.0.mlp.down_proj"):
        grid(prefix, np.asarray(nvfp4.get(prefix + ".pre_quant_scale")).reshape(-1),
             "pre_quant_scale")

    for prefix, norm in (("model.layers.0.self_attn.q_proj", "input_layernorm"),
                         ("model.layers.0.mlp.gate_proj", "post_attention_layernorm")):
        ratio = (np.asarray(nvfp4.get("model.layers.0.%s.weight" % norm)).reshape(-1) /
                 np.asarray(int8.get("model.layers.0.%s.weight" % norm)).reshape(-1))
        grid(prefix, ratio, "norm ratio")

    # The folding claim itself: the norms that absorbed a scale must differ
    # between the builds, and the ones that could not absorb anything must not.
    print("\nfolding control -- layer 0 norms, nvfp4 vs int8:")
    for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
        a = np.asarray(nvfp4.get("model.layers.0." + name)).reshape(-1)
        b = np.asarray(int8.get("model.layers.0." + name)).reshape(-1)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-30)
        print("   %-34s identical %-5s  median rel diff %.4f"
              % (name, bool(np.array_equal(a, b)), np.median(rel)))


if __name__ == "__main__":
    main(sys.argv)
