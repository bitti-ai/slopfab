# ConvRot (int8) — what is settled and what is not

Both quantised checkpoints we intend to load are `*_int8_convrot`, and the
`comfy_quant` blob attached to every quantised tensor reads:

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

**Read this before writing the int8 GEMM path.** Getting ConvRot wrong does not
crash — it produces a well-scaled, plausible-looking tensor of noise.

## Settled

- **Scales are per-output-channel**, not per-tensor, despite the format being
  named `int8_tensorwise`. `weight_scale` has shape `[out_features, 1]`, and
  the observed shapes agree (`[5120,1]`, `[25600,1]`, `[8192,1]`, `[1024,1]`).
- **The rotation is applied offline to the input (contraction) dimension** of
  each weight, in groups of 256:

  ```
  H = regular_hadamard(gs, dtype=W.dtype)      # built at the weight dtype, bf16
  w = (W.view(out, in // gs, gs) @ H.T).reshape(out, in)
  ```

- **Group size must be a power of four** (4, 16, 64, 256, 1024). 256 = 4^4.
  The likely reason is exactness: for a power of four, `sqrt(gs)` is a power of
  two, so a `1/sqrt(gs)` normalisation is exactly representable in binary
  floating point and the rotation is lossless at the storage dtype.
- **Biases are copied through unquantised** and are not rotated.
- **Layers whose `in_features` is not divisible by the group size skip rotation
  entirely.** For Qwen3-VL every `in_features` is 5120 or 25600, both divisible
  by 256, so every quantised layer here is rotated. Do not assume this holds
  for the transformer — check per tensor.

## Not settled — resolve before trusting any output

### 1. Are activations rotated at inference?

Two sources disagree. The comfy-quants format doc says rotation happens offline
and describes runtime as only "dynamically row-quantizing" activations; a
separate description of the same runtime says activations are "online-rotated"
inside the kernel.

**The mathematics is not ambiguous, and it says they must be.** With `R`
orthogonal and stored `W_rot = W R^T`:

```
x W^T  =  (x R^T) (W R^T)^T  =  x_rot W_rot^T
```

so recovering the true output requires the *same* rotation on the activation's
last dimension. Skipping it computes `x H W^T`, which is not the intended
result. The format doc is most likely describing where the *weight* rotation is
performed, not claiming activations are exempt.

Since the transform is block-diagonal with 256-wide blocks, applying it to an
activation is a fast Walsh–Hadamard transform per block — 8 butterfly stages,
negligible against the GEMM.

### 2. Which Hadamard, and is it normalised? — PARTLY SETTLED BY MEASUREMENT

The experiment in §3 below was run. Ground truth came from range-fetching 512
bytes out of the 48 GB bf16 checkpoint — the first 256 values of
`model.layers.0.self_attn.k_proj.weight` — and comparing against the same row
dequantised from the int8 file (`int8 * weight_scale[0]`,
`weight_scale[0] = 0.0007675675`).

**Result 1 — the Hadamard is normalised, i.e. orthogonal.**

```
||W||     = 0.415668
||W_rot|| = 0.415876
ratio     = 1.000501
```

A raw `±1` Hadamard would have given a ratio of 16 (Parseval). The 0.05 %
excess is int8 quantisation error. So `regular_hadamard(gs)` returns entries
`±1/sqrt(gs) = ±1/16` and the transform preserves norm. **No separate
normalisation constant is needed.**

**Result 2 — it is NOT the Sylvester/Walsh construction.**

Applying an unnormalised Sylvester FWHT to `W` and dividing by 16 gives a
relative error of **1.497** against `W_rot` — i.e. completely different, not a
near miss:

```
W_rot[0..3]      = -0.020724, 0.006908, -0.052195, -0.042984
FWHT(W)/16[0..3] = -0.054185,  0.015715,  0.04392,   0.01925
```

So `regular_hadamard` really does mean a *regular* Hadamard matrix (constant
row sums, order `4m²`, here `m = 8`) and **not** the ordinary Walsh–Hadamard
matrix. A fast Walsh–Hadamard transform is therefore the wrong kernel, however
tempting its speed.

**Still open:** the exact regular-Hadamard construction and its row/column
ordering. Two ways to close it, in order of preference:

1. Read `regular_hadamard` in the comfy-quants source.
2. Recover `H` numerically. `W_rot = W H^T` holds for every one of the 1024
   rows, so with 256+ rows of both matrices `H` is over-determined:
   `H^T = (WᵀW)⁻¹ Wᵀ W_rot` over one 256-column block. That needs ~10 MB
   range-fetched from the bf16 file and a 256×256 solve. It yields the matrix
   itself, which can then be checked for the defining property (all row sums
   equal) and stored as a constant.

Note the transform is fixed and shared by every layer, so once recovered it can
be baked in as a 256×256 constant — 64 KiB in fp32 — and applied as a small
GEMM rather than a bespoke butterfly kernel.

### 2b. Original question: which Hadamard, and is it normalised?

The doc says *regular* Hadamard, which is a specific thing: a Hadamard matrix
whose row and column sums are all equal (`±sqrt(n)`), existing for orders
`n = 4m²`. 256 = 4·8² qualifies. This is **not** necessarily the ordinary
Sylvester/Walsh construction, and the two differ by a permutation and sign
pattern — which changes the result.

It is also unclear whether `regular_hadamard(gs)` returns entries `±1`
(requiring an explicit `1/16`) or `±1/16` already normalised. The doc's remark
that "rotation and division happen at the source weight dtype" implies a
division exists somewhere.

### 3. How to settle both, cheaply

Comfy-Org ships the **unquantised** text encoder alongside the quantised one:

```
Comfy-Org/MiniMax-H3 : text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors   (47.97 GB)
```

Download one shard's worth, take a single layer — `model.layers.0.self_attn.k_proj.weight`
is only 5 MB quantised — and solve directly:

1. Dequantise the int8 weight: `W_rot = int8 * weight_scale`.
2. Read the true `W` from the bf16 checkpoint.
3. For each candidate `H` (Sylvester vs regular; normalised vs not), check
   whether `W_rot ≈ W @ H.T` blockwise.

Whichever candidate matches to within int8 quantisation error is the answer,
and it also fixes the normalisation constant. This is a decisive experiment
against ground truth, not an inference — do it before writing the kernel.

A cheaper *partial* check, if the bandwidth is not available: a correct
de-rotation should leave `W` with visible outlier structure (that is what the
rotation exists to suppress), while a wrong one leaves it Gaussian. Suggestive,
not conclusive.

## Consequence for the runtime design

Weights cannot be dequantised to fp16 up front: 23.3 GB of int8 becomes 46 GB,
which does not fit in 32 GB. The plan is to hold int8 on the device and
dequantise each weight into a reusable scratch buffer immediately before its
GEMM (largest is 25600×5120 → 262 MB in fp16). Text encoding runs once per
generation, so the extra bandwidth is not on the critical path.
