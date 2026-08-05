# vidfab

A from-scratch C++/CUDA implementation of [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3),
targeting a single RTX 5090 with no Python at runtime.

**Status: `vidfab generate` works end to end.** A text prompt goes in and a
real MP4 comes out — Qwen3-VL conditioner, 50-block transformer, flow-matching
denoise loop, both VAEs, H.264/AAC muxing, no Python anywhere. Output is
coherent, prompt-faithful video. See [Roadmap](#roadmap) and
[Known numerical gap](#known-numerical-gap--open).

## Why

H3 is a 33B omni-modal video model with open weights. The reference stack is
diffusers + PyTorch. This is a dependency-light native implementation: the goal
is to generate video from a single binary, with the CUDA runtime as the only
hard dependency, and to match the reference within a stated tolerance rather
than approximately.

## Dependencies

| Dependency | Why | Linkage |
|---|---|---|
| CUDA runtime + cuBLAS | kernels, GEMM | static |
| C++17 standard library | — | — |
| ffmpeg | MP4/AAC muxing only | **dynamic, resolved at runtime** |

There is deliberately no JSON library and no test framework — both are
hand-written and small. Nothing in the decode path allocates through a
third-party abstraction.

ffmpeg is loaded with `LoadLibrary`/`dlopen` at first use and is never linked
at build time; the build system does not reference an ffmpeg header or library
at all, and the handful of ABI declarations needed to drive it live in
`src/video/ffmpeg_abi.h`. That is a licensing requirement — LGPL compliance
depends on being able to substitute your own build, which `VIDFAB_FFMPEG_DIR`
exists for — and it means a missing or unusable ffmpeg degrades to the `.y4m` +
`.wav` writers rather than failing the run. Every struct offset the muxer
relies on is validated at load time by driving ffmpeg's own allocators and
reading the value back, so a layout that has moved is reported by name instead
of corrupting memory.

## Build

Requires CMake 3.24+, a C++17 compiler, and the CUDA toolkit.

```sh
cmake -S . -B build
cmake --build build --config Release
```

The default CUDA architecture is `120a` (Blackwell / RTX 50 series). The `a`
is load-bearing rather than decorative: `ptxas` rejects the `.block_scale`
operand plain `sm_120` does not have, and that operand is the whole of native
nvfp4. Override with `-DCMAKE_CUDA_ARCHITECTURES=90` for Hopper, which has no
nvfp4 at all. The core library and CLI build without CUDA; the decoder does
not.

## Usage

```sh
# Resolve a request: canvas, frame alignment, packed sequence length, both
# sigma schedules. Reads no weights, so it is instant.
vidfab generate --prompt "..." --aspect 16:9 --frames 124 --steps 50 --dry-run

# The real thing: prompt -> conditioner -> transformer -> denoise -> VAEs -> MP4.
vidfab generate --prompt "integrated_multimodal_description: ..." \
                --frames 22 --aspect 1:1 --steps 30 --seed 11 \
                --tokenizer      <tokenizer.json> \
                --text-encoder   weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
                --transformer    weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors \
                --vae            weights/vae/minimax_h3_video_vae_fp16.safetensors \
                --audio-vae      weights/vae/minimax_h3_audio_vae_fp32.safetensors \
                --out cat.mp4

# The quantisation of each checkpoint is read out of the file, so there is no
# flag for it and the pair need not match. `generate.cmd` wraps all of this.

# Every command documents itself.
vidfab generate --help

# Everything downstream of the denoiser only, against the real checkpoints:
# seeded noise -> unpatchify -> video VAE -> audio VAE -> H.264/AAC in an MP4.
vidfab generate --synthetic-latents --frames 22 --aspect 1:1 \
                --vae weights/vae/minimax_h3_video_vae_fp16.safetensors \
                --audio-vae weights/vae/minimax_h3_audio_vae_fp32.safetensors \
                --out out.mp4

# Inspect a checkpoint: tensor names, shapes, dtype breakdown, metadata
vidfab inspect weights/vae/minimax_h3_video_vae_fp16.safetensors --list --prefix decoder

# Decode a latent to video
vidfab decode --vae weights/vae/minimax_h3_video_vae_fp16.safetensors \
              --latent latent.safetensors --out out.y4m

# Without --latent, decodes a deterministic synthetic latent (smoke test)
vidfab decode --vae <vae.safetensors> --shape 7 48 48 --out out.y4m --ppm frame0.ppm

# Diff two checkpoints or two activation dumps, tensor by tensor
vidfab compare reference.safetensors actual.safetensors --abs-tol 1e-3

# Report CUDA devices and supported numeric formats
vidfab devices
```

`decode --dump <f>` writes raw fp32 pixels as safetensors, so two runs can be
compared at float precision instead of after 8-bit quantisation.
`generate --dump-latents <f>` writes the denoiser's own output — the packed
video and audio rows — before either VAE sees it. That is the diff point for a
change to the transformer: 7.5 MB a side at the default geometry rather than
400 MB, and no 9 GB decoder between the change and the comparison. Two runs of
the same seed and geometry must agree exactly, so
`vidfab compare a b --abs-tol 0` is the whole test.

`--sampler euler|ab2` selects the integrator; `euler` is the default and is the
reference's own update, unchanged. `ab2` is Adams-Bashforth 2, second order at
the same one forward pass per step. It was measured against Euler at matched
step counts and **does not buy fewer evaluations on this checkpoint** — see
[the sweep](#a-second-order-sampler--measured-and-ruled-out-as-a-step-count-reduction)
before reaching for it.

## Getting weights

The video VAE is a single standalone file (4.85 GB):

```
Comfy-Org/MiniMax-H3 : vae/minimax_h3_video_vae_fp16.safetensors
```

The repository is public and ungated. Full-precision transformer and text
encoder come from `MiniMaxAI/MiniMax-H3`; quantised variants (fp8, int8, nvfp4,
int4) from `Comfy-Org/MiniMax-H3` and `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`.

Weights are **not** redistributed here. They are covered by the MiniMax H3
Community License.

## Verification

Correctness is defined as agreement within tolerance, not bit-exactness —
that is not achievable across different GEMM implementations, and error
compounds over denoising steps.

Tests self-register with a shared harness (`tests/harness.h`), so each area
contributes its own translation unit and adding tests never touches a file
someone else is editing.

- **Host tests**: JSON parser, dtype conversions including fp8 E4M3 and fp4
  E2M1, safetensors loading and its rejection cases, comparison statistics, the
  flow scheduler and its second-order sampler, token packing, request
  resolution, the AdaLN table, the tokenizer, latent noise, the WAV writer and
  the colour transform.
  **2012 checks** — but only with `ref/` and the checkpoints present. Three
  tests skip themselves without them (two tokenizer goldens, which need
  `ref/FL2VA/text_encoder/tokenizer.json`, and one transformer case that needs
  a real checkpoint), so a clean clone reports **1952** and is not failing.
  `ref/` is licence-restricted and not redistributable, so 1952 is the number
  most people will see; run from the repository root to get 2012.
  Both figures are measured on the merged tree rather than added up from
  branches: contributors state a delta and the absolute is set here, because
  two branches each correctly adding to the same baseline is how this number
  went wrong before.
- **GPU kernel tests**: every kernel against independent CPU references written
  from the spec rather than from the kernel. **1053 checks**, plus 11 DEFERRED,
  measured with the checkpoints present — a tree without `weights/` skips the
  cases that need them and reports fewer, which is not a failure.
  These exist because the failure modes here are silent — a wrong QKV de-interleave, a wrong
  depth-to-space ordering, or a transposed GEMM all produce plausible output.
  Two of the newest close holes that had been open since the fused kernel
  landed: nothing had ever run it at `head_dim = 64`, and nothing pinned the
  precision of its probability tile. The second bars at 1.10× of the bf16
  output floor and currently measures **1.01×**, so a change dropping P to
  bf16 — which models at 1.21–1.31× depending on amplitude — fails it instead
  of passing unnoticed.

Two habits do most of the work. Where a wrong implementation is *plausible*
rather than merely broken, the test computes the wrong form too and asserts the
kernel does not match it — a missing `1 +` in AdaLN, a gate applied to the sum
instead of the branch, swapped SwiGLU halves, RoPE pairing `j` with `j + 64`
instead of `j + 48`, a Sylvester Hadamard instead of the regular one, a skipped
ConvRot activation rotation. And where an external reference exists, it is used:
the tokenizer is pinned to HuggingFace's output over 98 cases, the rotary grids
to numpy at zero tolerance, and the MP4 colour transform to the bytes in a
`.y4m` written by the other code path.

The packing tests are worth singling out. They compare the float64 rotary grids
against golden values taken from numpy **with zero tolerance**, because numpy's
`linspace` computes `arange(n) * ((stop - start) / n) + start` and
`(left + ratio) - left` is not `ratio` in float64 — reusing `ratio` puts the
width grid one ulp off, which a loose tolerance would happily accept.

```sh
build/Release/vidfab_tests.exe
build/Release/vidfab_kernel_tests.exe
```

**What is not yet verified:** agreement with the reference implementation
itself. Everything above checks internal consistency against
`docs/vae_decoder_spec.md`. Closing that gap requires activation tensors dumped
from the reference PyTorch pipeline; `vidfab compare` is built to consume them.

## Architecture notes

`docs/vae_decoder_spec.md`, `docs/transformer_spec.md`,
`docs/text_encoder_spec.md` and `docs/audio_vae_spec.md` document each stage
layer by layer with citations into the reference source by file and line;
`docs/convrot_notes.md` covers the int8 rotation.

### The nvfp4 conditioner

Both builds of the conditioner load, and which one a file is comes from its own
`comfy_quant` descriptors rather than from a flag — they differ in tensor count
(34 per layer against 25), dtype and shape, so a mismatch is a validation error
rather than wrong numbers. Four things about the nvfp4 build were established by
comparing it elementwise against the int8 build of the same model rather than by
reading a convention off the file, because every one of them fails silently:

- **The HIGH nibble holds the even-indexed element.** The other way round
  correlates with the truth at +0.0008 — that is, not at all — while still
  producing finite, correctly shaped, plausibly scaled output.
- **`weight_scale` is not row-major.** Its declared shape is `[out, in/16]` but
  its bytes are in a 128x4 tile swizzle. Read row-major it scores relative L2
  0.77 at correlation +0.79: well-scaled noise, invisible to every shape and
  finiteness check in the suite.
- **The activation is multiplied by `pre_quant_scale`**, the stored weight
  having already been divided: `y = (x*s) @ (W/s)^T`. Multiplying scores
  relative L2 0.10 against the int8 build, dividing 0.78, skipping it 0.50.
- **Only `o_proj` and `down_proj` carry that scale.** The other five had it
  folded into the preceding norm, which is why this build's `input_layernorm`
  and `post_attention_layernorm` differ from the int8 build's while `q_norm`
  and `k_norm` — which sit after the projections and have nothing to absorb —
  are bitwise identical to it.

All 350 quantised linears declare `full_precision_matrix_mult`, so every one of
them dequantises and runs bf16 and none may ever take a native fp4 GEMM. The
file says so and validation insists on it; nothing infers it from which scales
happen to be present. The build is also *not* ConvRot-rotated, unlike the int8
one, and the embedding table does not follow the linears — it is I8 with a
per-row F32 scale here and BF16 there.

End to end the two builds' `hidden_states[50]` agree to 1.9% of the int8
output's norm, which is the gap between two quantisations of one model and not
an error bar. `tools/nvfp4_layout_probe.py` is the offline harness that pinned
all of this and prints the grid.

Three things about the video decoder are worth knowing before reading the code,
because each produces plausible-but-wrong output if taken the obvious way:

- **The decoder has no convolutional upsampler.** The `block_out_channels` and
  `spatial_downsample_factors` in `vae/config.json` describe the *encoder*. The
  shipped checkpoint sets `use_vit_decoder: true`; all 16× spatial and 4×
  temporal upsampling is one `Linear(2048 → 3072)` plus depth-to-space.
- **Fused QKV is interleaved per head** (`r = h*192 + s`), not `[Q|K|V]`
  blocked.
- **Depth-to-space is channel-major**: `c*1024 + pt*256 + ph*16 + pw`. This is
  neither `nn.PixelShuffle` ordering nor the common `(pt, ph, pw, c)` ordering.

Decode is chunked temporally (7-token windows at stride 5, 5-frame cross-fade)
and tiled spatially (256 px tiles, ≥64 px overlap). RoPE coordinates are
length-normalised per tile, so tiling changes activations everywhere rather
than only at seams.

## Roadmap

| Stage | State |
|---|---|
| safetensors loader, dtype layer, compare harness | done |
| CUDA device layer | done |
| Video VAE decoder | done |
| Flow-matching sampler | done |
| Token packing, rotary grids, request resolution | done |
| Rank-8 AdaLN table lookup | done |
| Shared kernel layer, quantised linear, blocked attention | done |
| Audio VAE (DAC + BigVGAN) | done |
| WAV writer, MP4/AAC muxing | done |
| `generate` back half (unpatchify → VAEs → mux) | done |
| `generate` end to end, real prompt to MP4 | done |
| H3-Omni-Transformer, 50 layers | done |
| Qwen3-VL-32B text encoder (int8 ConvRot, 50 layers) | done |
| Qwen3-VL-32B text encoder (nvfp4 AWQ, 50 layers) | done |
| Fused attention (FlashAttention-2, `mma.sync`) | done |
| `cp.async` double-buffered K/V staging | not started |
| Native nvfp4 GEMM (`mma.sync` block-scaled) | landed, **off by default** |
| Native fp8/int4 GEMM | not started |

## Native nvfp4 GEMM — correct, and off by default

`LinearRunner::set_native` selects a hand-written block-scaled tensor-core GEMM
that feeds the checkpoint's bytes straight into
`mma.sync.aligned.m16n8k64.…block_scale` with nothing dequantised — 443–738
TFLOP/s, 2.6–4.1× the dequantise-then-cuBLAS path.

It is off by default anyway, and the reason is worth stating precisely, because
the obvious summary of it is wrong.

**The kernel is correct.** Fed activations that already sit exactly on the fp4
grid, it agrees with the reference at rms_rel **1.6e-3** — bf16 output rounding
and nothing else — and at **1.7e-3** on real `blocks.0` weights from the
checkpoint.

**The ~9% gap is the format, not the kernel.** Neither checkpoint ships an
`input_scale`, so activations are quantised dynamically to E2M1 at `amax/6` per
16-element block. E2M1 carries one mantissa bit, and a dot product cannot
average that away because signal and error both grow as √K. Three properties
say format rather than defect: correlation **0.9955**, where a misread layout
gives 0.00003; flatness in K from 128 to 5376, where a real bug would shrink;
and flatness in output magnitude, which is why every large *relative* error sits
on an output that cancelled towards zero.

So a native fp4-activation GEMM cannot meet a 1e-2 relative bound against a
bf16-activation reference at any K, and no kernel work will close it. The test
therefore compares the kernel against an **fp4-activation** reference at the
same 1e-3 / 1e-2 — pointing the bound at what it can describe — and prints the
bf16 gap as a measurement beside it. **The tolerance was not loosened.**

### What ~9% per layer does over a whole generation

Measured, rather than argued about. `VIDFAB_NATIVE_NVFP4=1` exists so the same
seed can be run both ways; 22 frames at 1:1, 30 steps, seed 11, everything else
identical:

| | per step | denoise | total | video latents |
|---|---|---|---|---|
| dequantise-then-cuBLAS | 1.19 s | 34.5 s | 48.7 s | mean +0.0659, std 1.063 |
| **native nvfp4** | **0.68 s** | **19.7 s** | **25.8 s** | mean +0.0135, std 1.089 |

**1.75× per step, 1.89× end to end.** And the output is *different video*:

```
mean |diff|  13.5 levels of 255      p99 100, max 183
correlation  0.874 overall,  0.629 on frame 0 luma
means        73.3 vs 74.1     stds  48.3 vs 48.0
```

The global statistics match almost exactly while the correlation does not,
which is the signature of a **different sample rather than a degraded one** — a
denoiser whose trajectory was perturbed early converges somewhere else in the
same distribution. That is what 50 blocks and 29 steps do with a 9% per-layer
perturbation, and no per-tensor tolerance would have predicted it.

So the switch stays **off by default**. It is not broken and it is not noise on
top of the reference image; it is a lossier model that generates its own
equally plausible video, almost twice as fast. Whether that trade is worth
taking is a judgement about output quality that wants eyes on a set of samples,
not another number — which is exactly why it is a flag and not a default.

## A second-order sampler — measured, and ruled out as a step-count reduction

`--sampler ab2` is Adams-Bashforth 2: `x_{n+1} = x_n + h(1.5 v_n - 0.5 v_{n-1})`,
second order at the same one forward pass per step, with the first step of a
trajectory falling back to Euler. It is correct — the linear-ODE test shows it
converging at **3.99× per halving** against Euler's 2.0×, and `--sampler euler`
is bit-identical to the pre-change update by cross-build comparison.

**It does not buy fewer evaluations on this checkpoint.** The estimate going in
was 1.5–1.8×. Measured, it is ~1.26×, video only, at the bottom of the step
range, and nothing at all at the default 50.

The reason is not the sampler. **From 25 steps up this checkpoint is
insensitive to step count**, and the residual differences between runs are
chaotic rather than convergent. One control establishes it: `ab2` and `euler`
at the *same* 49 evaluations differ by rel_L2 **0.2210** on the video latents.
That is a noise floor, and it swallows nearly every measurement in the sweep —
including, absurdly, `ab2` at 19 evaluations landing **closer** to the 50-step
Euler reference (0.115) than `ab2` at 49 evaluations does (0.221). A metric
where 19 evaluations beat 49 of the same sampler is not measuring convergence.

The Euler controls at matched step counts are what make that readable, and
they are non-monotone:

| video rel_L2 vs euler@50 | 20 | 25 | 30 | 40 |
|---|---|---|---|---|
| euler | **0.4232** | 0.2208 | 0.2056 | 0.2095 |
| ab2 | **0.1151** | 0.2339 | 0.1598 | 0.2088 |

Euler at 40 steps is no closer to Euler at 50 than Euler at 25 is. Audio is
worse still — 0.0821 at 25 steps, 0.3804 at 30, 0.0903 at 40. Meanwhile the
global statistics match almost exactly across every run (mean 0.050–0.055
against the reference's 0.0513, std 1.048–1.057 against 1.0530, correlation
0.97–0.99). That is the same signature as the nvfp4 section above: **a
different sample, not a degraded one.**

One effect is real and replicated. At 20 grid points Euler genuinely degrades —
it is the worst point in the table — and AB2 does not, on **4 of 4 seeds**,
each against its own 50-step reference. Four seeds, but **one geometry and one
prompt**: enough to say Euler at 20 degrades and AB2 at 20 does not, not enough
to say how that scales to 124 frames, where the packed sequence is 9× longer
and the sigma grid is identical while the trajectory is not. Read the 1.26× as
specific to what was measured.

| seed | euler@20 | ab2@20 | correlation |
|---|---|---|---|
| 11 | 0.4232 | 0.1151 | 0.910 → 0.993 |
| 12 | 0.3771 | 0.2224 | 0.928 → 0.975 |
| 13 | 0.3514 | 0.2187 | 0.938 → 0.976 |
| 14 | 0.4366 | 0.2947 | 0.904 → 0.956 |

So AB2 at 20 grid points sits in the band Euler needs 25 for: 19 evaluations
against 24, **~1.26×**, video only. Audio shows no effect (2 of 4 seeds, ratios
0.77–1.66). The variable-step coefficient variant was tried and dropped: it won
on the linear ODE and the advantage did not transfer to the checkpoint.

`ab2` stays as an off-by-default flag, because the code is correct and someone
who wants 20 grid points should have it. **There is deliberately no `--steps`
recommendation here** — the evidence does not support one. The default stays 50.

These are quality comparisons run at one base commit with matched geometry
(22 frames, 1:1, 4170 packed rows) and matched seeds, so the *ratios* transfer;
the wall times were taken before the attention staging rewrite and do not.

## Performance

Measured on the RTX 5090, and the first number is the one that recalibrates
everything else:

**The sustained dense bf16 GEMM ceiling on this card is 216–222 TFLOP/s, not
the 419 on the spec sheet.** At 575 W it holds about 2.45 GHz, and cuBLAS on a
16384³ GEMM gets 220. Every efficiency claim below is against that.

| op | shape | time | achieved |
|---|---|---|---|
| attention, blocked | seq 37710, 56 heads, dim 128 | 561 ms | 72.6 TFLOP/s |
| **attention, fused** | seq 37710, 56 heads, dim 128 | **228.4 ms** | **178.5 TFLOP/s** |
| `qkv_proj` | `[21504, 5376]` × 37710 rows | 36.2 ms | 234 TFLOP/s |
| `attn.out_proj` | `[5376, 7168]` | 12.1 ms | 237 TFLOP/s |
| `mlp.fc1` | `[28672, 5376]` | 48.3 ms | 232 TFLOP/s |
| `mlp.fc2` | `[5376, 14336]` | 24.0 ms | 231 TFLOP/s |
| video VAE decode | 124 frames at 1344×768 | 52.7 s | — |
| audio VAE decode | 5.2 s of 32 kHz stereo | 0.25 s | — |

### How long a generation actually takes

Measured end to end on the RTX 5090, and worth reading before you run the
default:

| request | rows | per step | total |
|---|---|---|---|
| 22 frames, 1:1, 30 steps | 4 167 | 1.07 s | **~50 s** |
| **124 frames, 16:9, 50 steps (the default)** | 37 710 | **19.1 s** | **~16 min** |

Both rows are the nvfp4 pair, measured with `VIDFAB_PROFILE=1` on an idle card,
whose device timeline accounts for 100.00% of a step at 0.24% overhead. **An
earlier revision of this table said 38.8 s and ~33 min**; that was an fp8
carry-over never re-measured against the nvfp4 pair, and it was 36% too high.
Its own per-layer table already contradicted it, summing to 23.5 s — a
discrepancy visible in this file for some time, which is the argument for
quoting a geometry and an idle card beside every number rather than a
percentage on its own.

The default row is post-staging-rewrite: the same measurement read 24.85 s
before it and 19.09 s after, and 98.5% of that 5.76 s came out of `attn.fused`
alone, which is the check that nothing else moved. **The quick row has not been
re-measured since**, so treat 1.07 s as an upper bound.

The default is the reference model's own default and it is genuinely that
slow — the fused attention kernel alone is 61.5% of a step (down from 70.4%
before the staging rewrite) and scales with the square of the packed sequence,
so the 9× row increase costs 18× the time. It is not hung: a
progress line reports seconds per step and a running ETA from the first step
onward. **If you just want to see it work, use `--frames 22 --aspect 1:1
--steps 30` and wait under a minute.**

Roughly 5.5 s of the fixed cost is loading 19.6 GiB of transformer weights from
a warm page cache — but 22 s from a cold one, and up to 94 s before this was
fixed. See [Cold weight loading](#cold-weight-loading). ~5 s is the
conditioner, which streams its 24.4 GB rather than resident-loading it.

**Measure with the GPU idle.** A contended card does not halve throughput here,
it costs up to 14x: the conditioner's attention is a dependent chain of short
launches and WDDM time-slicing destroys dependent chains rather than slowing
them proportionally. The same encoder attention measured 2.55 ms with another
process on the card and 0.159 ms with it idle. Check `nvidia-smi` before
believing any number in this file.

### Conditioner residency

Both builds of the conditioner, both modes, measured on one idle 5090 over a
190-token prompt in a single run:

| build | mode | load | encode (warm) | peak |
|---|---|---|---|---|
| int8 + ConvRot | streaming *(default)* | 0.00 s | **0.66 s** | 1.20 GB |
| int8 + ConvRot | resident | 4.95 s | 0.11 s | 23.08 GB |
| nvfp4 + AWQ | streaming *(default)* | 0.00 s | **0.48 s** | 0.80 GB |
| nvfp4 + AWQ | resident | 9.30 s | 0.12 s | 13.09 GB |

nvfp4 is smaller in both modes and faster in streaming, where the encode is
bound by how many bytes cross PCIe rather than by arithmetic — half the weight
bytes, roughly two-thirds the time. Resident is a dead heat, because there both
builds dequantise to bf16 and run the same cuBLAS GEMM; nvfp4's resident load is
*slower* (9.30 s against 4.95 s) despite reading a smaller file, which is the
50 x 7 extra `weight_scale_2` reads and the larger tensor count, not bandwidth.

Streaming is the default because the transformer needs 19.6 GiB (fp8) or
12.5 GB (nvfp4) later in the same process. Its encode used to be 2.9 s, of which **96% was a single-threaded
host `memcpy`** staging weights into pinned memory — and most of *that* was soft
page faults on the 27 GB mapping, not memcpy bandwidth. The mapping is now
page-locked once with `cudaHostRegister` and each weight DMAs straight out of
it, which lifted H2D from 8.7 GB/s to ~42 GB/s and removed 930 MB of pinned
staging. Registration is best-effort; if it fails the old staging path still
runs, just slower.

### Cold weight loading

The first run of the day pays for reading the checkpoint off the drive, and it
was paying about three times what the drive charges. `SafeTensors` maps the
file and every loader then demand-faults it in, which is a synchronous walk
that never gives the SSD any queue depth. One `PrefetchVirtualMemory` over the
mapping — `MADV_WILLNEED` on POSIX — replaces that with one asynchronous read.

Measured end to end through `generate --bench-load`, cache evicted before each
cold sample, card confirmed idle, n=2 cold and n=3 warm per configuration:

| checkpoint | | cold | warm |
|---|---|---|---|
| nvfp4, 12.5 GB | demand faulted | 26.4, 28.0 s | 3.34–3.48 s |
| | **prefetched** | **12.6, 13.1 s** | 3.67–3.90 s |
| fp8, 21.0 GB | demand faulted | 45.0, **94.3** s | 5.47–5.69 s |
| | **prefetched** | **22.1, 24.6 s** | 6.21–6.56 s |

**14 s off a cold nvfp4 run and 21–70 s off a cold fp8 one.** That is seconds
off a fresh run and never a percentage of a step — this is outside the denoise
loop entirely. Against the default 50-step run at ~16 min it is 1–7%; against
the ~50 s quick geometry the nvfp4 saving alone is over a quarter of the whole
run, and the fp8 tail case was longer than the generation it preceded. That
contrast is the honest way to read it.

**It also removes a tail, and that tail was the original mystery.** Two samples
of the same cold fp8 file demand-faulted took 45.0 s and 94.3 s — a 2.1×
spread on identical work, which is the same instability an earlier profiler saw
as 22.6 s and 60.7 s and could not explain. Prefetched, the same file is 22.1
and 24.6 s. The win is as much that the number becomes predictable as that it
becomes smaller.

**Warm loads are ~0.4 s (nvfp4) and ~0.8 s (fp8) slower**, consistently and
outside the noise: prefetching an already-resident 21 GB mapping still has to
walk five million pages into the working set. Paying that to save 14–70 s on
the first load is the right trade, but it is a real cost and not a free win.

**Access order is not the problem, which took measuring to establish.** Every
loader walks tensors in `std::map` name order, and that is emphatically not
file order — 453 backward seeks over 1632 GB cumulative for nvfp4, 402 over
615 GB for fp8. Sorting the upload by file offset looks like the obvious fix
and buys **nothing**: replaying the exact extents unbuffered gives 2.19 GB/s in
name order against 2.26 GB/s in file order, and mapped-and-cold the two are
21.7 s against 21.6 s. Even *with* prefetch the two orders are identical at
7.0 s. The sort is deliberately not implemented.

The measurement method mattered as much as the result, because "cold" on
Windows is not a controlled condition. `tools/loadprobe.cpp` replays a
checkpoint's real extent list under `FILE_FLAG_NO_BUFFERING`, which bypasses
the cache and so always measures the drive: spread 1.01× over five samples,
against the 2.7× the old figures had. For the mapped path it drops the file's
own cached pages by opening it unbuffered, and then *verifies* the eviction
with a probe read rather than assuming it — the first attempt read 62 GB past
the target with `FILE_FLAG_SEQUENTIAL_SCAN` set, which retires pages as they
are consumed, so the flood evicted itself and left the target fully cached.
That same probe later caught two contaminated samples in flight.

Weights are bit-identical, which is the point of a readahead change and is
checked rather than argued: `VIDFAB_ARENA_HASH=1` hashes the finished arena,
and ten loads of each checkpoint across both settings of `VIDFAB_NO_PREFETCH`
agree exactly — nvfp4 `7da30a6df7e45676` over 12,615,830,272 bytes and 930
records, fp8 `190cdce19da29c2d` over 21,045,398,272 bytes and 730 records.

**Resident mode has a bad tail on a shared card.** If another process wants the
memory, WDDM evicts the 22.7 GB weight arena to system RAM and faults it back
over PCIe — an encode measured at 485 s once under contention, recovering to
0.9 s on the next call. The residency check samples free VRAM once at load and
cannot see a later competing allocation. Prefer streaming unless the card is
yours alone.

So the four linear layers are already at the machine ceiling and are not worth
touching — `cublasLt` heuristic search, a larger cuBLAS workspace and row
alignment were all measured and buy nothing.

Attention **was** 86% of a denoising step and bandwidth-bound: the blocked path
moves ~1131 GB per layer at 1.41 TB/s while running at only 72.6 of 216
TFLOP/s, because the score tile round-trips through HBM at ~12 bytes per
element. Its own floor was ~682 ms, so tuning it further was pointless — the
round trip had to go, not shrink.

The tile-budget sweep (192 MiB → 1010 ms, 640 MiB → 749 ms, 1536 MiB → 711 ms)
is explained by `t ≈ (12·H·S² + 16·H·D·S²/bk) / 1.4 TB/s`, where the second
term is the fp32 accumulator being read-modify-written twice per key block.
That model reproduces all three points to within 3%, and both of its terms are
exactly what the fused kernel deletes.

### The fused kernel

`AttentionBackend::kFused` keeps S and P in registers and never writes them
anywhere. **228.4 ms against the blocked path's 561, at 178.5 TFLOP/s — 82% of
the machine ceiling, and no workspace at all against 1.63 GiB.**

It first landed at 349 ms and 117 TFLOP/s. The remaining 121 ms was not in the
mma pipeline at all but in the K/V staging loop, which re-derived `r = i / D`,
`c = i % D` and a fresh 64-bit global address for **every 2-byte element** —
172 instructions per trip of which only 128 moved data, between two
`__syncthreads()` where no `mma` could issue. Hoisting the loop invariants and
giving each thread eight contiguous columns turns K's staging into four
`LDG.E.128` and four `STS.128` instead of 32 scalar pairs, and takes the
staging from 1376 instructions per warp per key block to ~195. The obvious
lane-to-row mapping for this is wrong in a way that costs most of the win: it
puts all 32 of V's transposing `STS.U16` in one bank. The 8×4 remap that fixes
it is in the code beside the reason.

The load-bearing decision is `mma.sync` instead of `nvcuda::wmma`. `wmma` does
not specify which row an accumulator element belongs to, and O needs a *per-row*
rescale on every key block, which forces O into shared memory — the first
attempt did exactly that and came out at 1312 ms, **2.3× slower than the path it
was replacing**. `mma.sync.aligned.m16n8k16` does specify the layout: a thread's
four accumulator registers hold rows `groupID` and `groupID+8`. So the rescale
is four multiplies, the row reduction is two shuffles across the four
consecutive lanes of a group, and — the part that pays for the rest — the QK
accumulator tiles at n-offset 0 and 8 *are* the four A-fragment registers the PV
`mma` wants, so S becomes P by a register permute with no transpose and no
store.

Two things that are easy to get wrong and are load-bearing here:

- **The fp32 shared tiles could not be de-conflicted by padding.** `wmma` wants
  an fp32 `ldm` that is a multiple of 4 floats, and for any such stride, rows 8
  apart — the pair every 16×16 fragment touches — land in the same bank.
  Deleting the tiles was the only fix.
- **The grid is `(query_tile, head)` and the order matters.** CUDA dispatches x
  fastest, so resident blocks share a head and walk one K/V stream together.
  That stream is 19.3 MB, which lives in L2. The 638 GB of K/V re-reads is L2
  traffic, not DRAM; swapping the dimensions would make it DRAM traffic.

Remaining headroom is `cp.async` double-buffering of the K/V stage, which is
still two barriers and 2-byte scalar loads. That needs V pre-converted to fp16
once per call — 541 MB — so it trades the zero-workspace property for an
estimated 290–340 ms.

## Audio level — settled

Early runs produced very quiet audio (mean −59.7 dB, peak −38.5 dB), which is
consistent with "quiet room tone" *and* with a gain error. That ambiguity is
the same silent-wrong shape as every other trap here, so it was resolved by
experiment rather than by listening.

Same seed, same geometry, same everything — only the soundscape clause of the
prompt changed:

| soundscape in the prompt | mean | peak |
|---|---|---|
| "quiet room tone, soft paw-steps on wood" | −59.7 dB | −38.5 dB |
| "loud clattering pots, a barking dog, a slamming door" | **−20.6 dB** | **−2.0 dB** |

A 39 dB spread driven by conditioning alone. **A constant gain error cannot do
that**, so the level is the model obeying the prompt.

The decoder's own gain is pinned separately, against a float64 NumPy
transcription of the reference decode path, which it matches to **2.15e-7**.
That test now carries a mutation check: it applies a ±1%, +5% gain to the real
output and requires the assertions to fail. Adding it immediately showed the
original tolerances were too loose to catch 1% — the golden comparison used a
1e-2 relative bound, so a 1% gain error passed *by construction*. Tolerances
are now 1e-5/1e-4 and the RMS bound is 1e-4, which the implementation clears by
three orders of magnitude.

## Known numerical gap — resolved as distributed rounding

`transformer_forward_vs_cpu_reference` reports ten **deferred** checks: the
model disagrees with a CPU reference by ~0.8% in the mean and 2-3.5% at the max.
This was investigated to a conclusion. **It is accumulated bf16 rounding, not a
defect**, and the evidence is a stage-by-stage bisect rather than an argument.

### The bisect

`transformer_refiner_bisect` walks both implementations through the same six
boundaries and reports the fraction of elements that differ at each:

| stage | L=1 | L=2 | L=5 |
|---|---|---|---|
| `condition_proj` | 0.00% | **0.00%** | **0.00%** |
| refiner block 0, after attention | 0.00% | 31.6% | 26.4% |
| refiner block 0, after FFN | 0.00% | 56.3% | 57.5% |
| refiner block 1, after attention | 0.00% | 70.7% | 66.4% |
| refiner block 1, after FFN | 0.00% | 77.7% | 73.3% |
| `final_norm` | 0.00% | 78.5% | 72.5% |

Four things in that table settle it.

1. **`condition_proj` is bit-exact.** It is a per-row GEMM plus a bias, so there
   is no reduction order for the two implementations to disagree about. The
   linear path, the bias add and the fp32→bf16 narrowing are all exactly right.
2. **Divergence begins at the first operation that mixes rows.** Attention is
   the first place where cuBLAS's reduction over the key dimension can differ
   from the reference's. At `L = 1` it cannot: softmax over one key is
   identically 1 and attention is the identity on `v` — and every stage is
   bit-exact, which is the control.
3. **It grows monotonically at every subsequent stage** — 26% → 58% → 66% →
   73%. A single wrong operation produces a jump followed by a plateau. This is
   accumulation.
4. **The magnitude never exceeds one bf16 ULP** — 1.562e-02 (2⁻⁶) early,
   3.125e-02 (2⁻⁵) once magnitudes cross into [4, 8) — and the signed mean stays
   near zero and changes sign between stages. Both are the signature of
   symmetric rounding, not bias.

Two fp32 values differing by ~0.2% land on different bf16 values roughly 70% of
the time, which is the rate the table converges to.

### What would change the conclusion

The bisect now asserts the shape of this result, so a regression from rounding
to arithmetic fails loudly rather than being absorbed: `condition_proj` must
stay bit-exact, no stage may exceed one bf16 ULP, disagreement must not fall
between stages, and every stage at `L = 1` must be exact. If any of those
breaks, the explanation above is wrong and there is a real defect.

### Four hypotheses ruled out on the way

Each was an experiment against the real checkpoint, not reasoning.

| hypothesis | result |
|---|---|
| Truncating `fp32 -> bf16` conversion | **Ruled out.** Signed mean `+1.66e-04` against a half-ULP bound of `1.95e-03`. Still asserted. |
| Reference softmax rounds its numerator but not its denominator | **Ruled out, and backwards.** The GPU has the same asymmetry, so making the reference self-consistent moved it *away*: 72.5% → 73.75%, audio mean 0.82% → 1.05%. |
| The fp16 score tile in attention | **Ruled out.** Modelling it moved 72.5% → 72.03%. |
| Reference accumulates in fp64, GPU in fp32 | **Ruled out.** Switching the reference to fp32 changed *nothing*, to three decimals. |

The DEFER lines stay, still printing real numbers every run, because the
assertions they replaced are what found this and are worth keeping pointed at
it. The mean error is inside the project's 1% per-tensor tolerance and that
bound still asserts; only the max exceeds its 3% bar, in some geometries.

## Memory budget

The card is a 32 GB RTX 5090. Resident weights, measured from the checkpoints:

| Stage | Device weights |
|---|---|
| Qwen3-VL conditioner, int8 + ConvRot | 22.7 GB (23.1 GB peak; +1.6 GB embedding, kept on the host) |
| Qwen3-VL conditioner, nvfp4 + AWQ | 12.8 GB (13.1 GB peak; +0.78 GB embedding, kept on the host) |
| H3 transformer, fp8 | 19.3 GB |
| H3 transformer, nvfp4 | 12.5 GB |
| Video VAE decoder | 9.0 GB (fp16 on disk, widened to fp32) |
| Audio VAE | 0.6 GB |

The pipeline runs strictly in sequence — encode the prompt, free the
conditioner, load the transformer, denoise, free it, decode — and that ordering
used to be forced: 23.1 GB and 19.3 GB do not co-exist on a 32 GB card.

**nvfp4 changes that, and the sequencing stays anyway.** The nvfp4 pair is
13.1 GB and 12.5 GB, which does fit together with room to spare. But three of
the four checkpoint combinations still do not, the saving from keeping both
loaded is one 0.12 s encode against a denoising run measured in tens of
minutes, and dropping the sequencing would turn a mixed pair into an
out-of-memory failure at the worst possible moment. So the swap is kept as a
matter of it costing nothing, not of it being unavoidable. Ruled out rather
than not tried.

The vision tower in the conditioner checkpoint (1.19 GB in the nvfp4 build) is
never loaded — it is reached only by the keyframe path, which this port does
not implement.

## Licence

The code in this repository is the author's. Model weights, configuration files
and the reference material under `ref/` are covered by the MiniMax H3 Community
License and are not redistributed.
