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

`--dump <f>` writes raw fp32 pixels as safetensors, so two runs can be compared
at float precision instead of after 8-bit quantisation.

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
  flow scheduler, token packing, request resolution, the AdaLN table, the
  tokenizer, latent noise, the WAV writer and the colour transform.
  **1669 checks.**
- **GPU kernel tests**: every kernel against independent CPU references written
  from the spec rather than from the kernel. **821 checks**, plus 11 DEFERRED.
  These exist because the failure modes here are silent — a wrong QKV de-interleave, a wrong
  depth-to-space ordering, or a transposed GEMM all produce plausible output.

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
| Native fp8/nvfp4/int4 GEMM | not started |

## Performance

Measured on the RTX 5090, and the first number is the one that recalibrates
everything else:

**The sustained dense bf16 GEMM ceiling on this card is 216–222 TFLOP/s, not
the 419 on the spec sheet.** At 575 W it holds about 2.45 GHz, and cuBLAS on a
16384³ GEMM gets 220. Every efficiency claim below is against that.

| op | shape | time | achieved |
|---|---|---|---|
| attention, blocked | seq 37710, 56 heads, dim 128 | 561 ms | 72.6 TFLOP/s |
| **attention, fused** | seq 37710, 56 heads, dim 128 | **349 ms** | **117 TFLOP/s** |
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
| 22 frames, 1:1, 30 steps | 4 167 | 1.5 s | **~50 s** |
| **124 frames, 16:9, 50 steps (the default)** | 37 710 | **38.8 s** | **~33 min** |

The default is the reference model's own default and it is genuinely that
slow — attention is 86% of a step and scales with the square of the packed
sequence, so the 9× row increase costs 26× the time. It is not hung: a
progress line reports seconds per step and a running ETA from the first step
onward. **If you just want to see it work, use `--frames 22 --aspect 1:1
--steps 30` and wait under a minute.**

Roughly 25 s of the fixed cost is loading 19.6 GiB of transformer weights, and
~5 s is the conditioner, which streams its 24.4 GB rather than resident-loading
it.

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
anywhere. **349 ms against the blocked path's 561, at 117 TFLOP/s — 54% of the
machine ceiling, and no workspace at all against 1.63 GiB.**

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
