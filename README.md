# vidfab

A from-scratch C++/CUDA implementation of [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3),
targeting Ampere RTX 30-series and Blackwell RTX 50-series GPUs with no Python
at runtime.

**Status: `vidfab generate` works end to end.** A text prompt goes in and a
real MP4 comes out — Qwen3-VL conditioner, 50-block transformer, flow-matching
denoise loop, both VAEs, H.264/AAC muxing, no Python anywhere. Output is
coherent, prompt-faithful video. See [Roadmap](#roadmap) and
[Known numerical gap](#known-numerical-gap--open).

## Why

H3 is a 33B omni-modal video model with open weights. The reference stack is
diffusers + PyTorch. This is a dependency-light native implementation: the goal
is to generate video from native executables, with an installed CUDA toolkit as
the only accelerator dependency, and to match the reference within a stated
tolerance rather than approximately.

## Dependencies

| Dependency | Why | Linkage |
|---|---|---|
| CUDA 12 or 13 toolkit | kernels, GEMM | runtime embedded; installed cuBLAS dynamic |
| C++17 standard library | — | — |
| ffmpeg | MP4/AAC muxing only | **dynamic, resolved at runtime** |
| Vulkan 1.2 loader | optional exact neural inference and RGB-to-YUV conversion | dynamic, SDK-free |

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

Requires CMake 3.24+, a C++17 compiler, and CUDA toolkit 12 or 13. CUDA and
cuBLAS DLLs are not shipped with vidfab.

```sh
cmake -S . -B build
cmake --build build --config Release
```

The default is a `86;120a` fat binary. SM86 serves Ampere / RTX 30-series GPUs;
SM120a serves Blackwell / RTX 50-series GPUs. The `a` is load-bearing rather
than decorative: `ptxas` rejects the `.block_scale` operand plain `sm_120`
does not have, and that operand is the whole of native NVFP4. Ampere runs the
BF16 materialization path instead. SM89 / RTX 40-series specialization is
planned but not part of this build yet.

On Windows, a CUDA build produces a CUDA-free `vidfab.exe` launcher and either
`vidfab-cuda12.exe` or `vidfab-cuda13.exe`. A release contains both backends.
The launcher prefers an installed CUDA 13 toolkit and falls back to CUDA 12,
requiring the matching `cublas64_<major>.dll` and `cublasLt64_<major>.dll` in
that toolkit's `bin` directory. Override selection with
`--cuda-version=auto|13|12` or `VIDFAB_CUDA_VERSION`. The launcher removes its
own option before forwarding the normal vidfab CLI, and each backend can also
be invoked directly. The GPU architecture is selected independently by the
backend fat binary after launch.

### The C API

`vidfab.dll` exports the flat C ABI in `include/vidfab/capi.h`, so Rust, C#,
Python, Go and plain C can drive the pipeline without depending on a C++ ABI.
It is built by default; `-DVIDFAB_BUILD_C_API=OFF` turns it off.

```sh
cmake -S . -B build-dll -DVIDFAB_WITH_FFMPEG=OFF
cmake --build build-dll --config Release --target vidfab_c
```

`vidfab_core` and `vidfab_cuda` are internal static libraries that link into
`vidfab.dll`. It is therefore the only vidfab DLL, and its export table contains
exactly the stable C entry points rather than the internal C++ symbols.

It is not dependency-free, though, and the mistake is invisible on a machine
with the CUDA toolkit installed. The CUDA runtime is embedded, but cuBLAS is
not: `vidfab.dll` imports `cublas64_<major>.dll` at load time, which pulls
`cublasLt64_<major>.dll` with it. On a development box those resolve off
`PATH`; on a consumer's machine the process fails at `LoadLibrary` with no
useful message. They are intentionally not installed or packaged by vidfab.
Hosts using the C API must prepend the matching toolkit `bin` directory to
`PATH`, just as the CLI launcher does.

**It produces pixels, not files.** A generation hands back decoded frames as
planar float RGB and audio as interleaved float PCM, and writes nothing to
disk. Encoding, muxing and playback belong to the host. That is why the DLL
needs no FFmpeg at all: the muxer is the only thing in this project that loads
it, and the C API never reaches the muxer.

Generation is asynchronous — `vidfab_generation_start` returns as soon as the
request is known to be satisfiable, and the work proceeds on a worker thread,
so a UI stays responsive across a run that takes minutes. Progress callbacks
arrive on that worker thread, not the caller's.

```c
#include <vidfab/capi.h>

vidfab_request* req = vidfab_request_create();
vidfab_request_set_prompt(req, "a cat playing a piano, warm lamplight");
vidfab_request_set_model_path(req, VIDFAB_MODEL_TRANSFORMER, "transformer.safetensors");
vidfab_request_set_model_path(req, VIDFAB_MODEL_TEXT_ENCODER, "text_encoder.safetensors");
vidfab_request_set_model_path(req, VIDFAB_MODEL_VIDEO_VAE, "video_vae.safetensors");
vidfab_request_set_frames(req, 124);

/* Reads no weights, so it is instant: validate and cost the request first. */
vidfab_plan plan;
if (vidfab_resolve_plan(req, &plan) != VIDFAB_OK) {
  fprintf(stderr, "%s\n", vidfab_last_error());
  return 1;
}

vidfab_generation* gen = NULL;
if (vidfab_generation_start(req, on_progress, NULL, &gen) == VIDFAB_OK) {
  if (vidfab_generation_wait(gen, -1) == VIDFAB_OK) {
    vidfab_output out;
    vidfab_generation_output(gen, &out);
    /* out.video is [channels][frames][height][width], fp32 in [0,1], owned by
       `gen`. One frame as packed RGBA8, for a texture upload: */
    uint8_t* rgba = malloc((size_t)out.width * out.height * 4);
    vidfab_generation_frame_rgba8(gen, 0, rgba, (size_t)out.width * out.height * 4);
  } else {
    fprintf(stderr, "%s\n", vidfab_generation_error(gen));
  }
  vidfab_generation_destroy(gen);  /* the pixels die with the handle */
}
vidfab_request_destroy(req);
```

From Rust the same flow is a `bindgen` run over `capi.h` and a `Drop` impl per
handle; the destructors all accept null, so the `Drop` needs no guard. Three
rules carry across every binding:

- **Status codes are plain `int`, not an enum**, so a caller linked against an
  older header can hold a code this header does not name. Mapping them into a
  Rust enum needs a catch-all arm.
- **Every pointer in `vidfab_output` is owned by the generation** and dangles
  after `vidfab_generation_destroy`. At the default geometry the video plane
  alone is over 2 GB, so it is handed over by pointer rather than copied — copy
  it out if it must outlive the handle.
- **The only pointer you free is the `char*` from `vidfab_describe_plan`**, and
  it is freed by `vidfab_free_string`. Calling the host's own `free` on it
  crosses CRTs, which on Windows is a crash often enough to matter.

`vidfab_last_error()` is thread-local and holds the reason the *calling* thread
last failed. A run's failure message is not there — the run fails on a worker
thread the caller never enters — so use `vidfab_generation_error()` for that.

### Building without FFmpeg

FFmpeg is loaded at runtime and never linked, so `-DVIDFAB_WITH_FFMPEG=OFF`
does not change what the binary needs in order to *start*. It changes what is
compiled in at all: the muxer is replaced by a stub, and reference images are
decoded by the platform instead of by libavcodec.

|                      | `ON` (default)                     | `OFF`                                  |
| -------------------- | ---------------------------------- | -------------------------------------- |
| `vidfab` output      | MP4, falling back to `.y4m`+`.wav` | `.y4m` + `.wav` only                   |
| Reference images     | anything FFmpeg demuxes, incl. video | PPM, plus WIC formats on Windows     |
| Shipped beside `.exe`| five FFmpeg DLLs                   | nothing                                |

On Windows the platform decoder is WIC, which is part of the OS and reads PNG,
JPEG, BMP, GIF and TIFF with nothing shipped or linked beyond
`windowscodecs.lib`. Elsewhere there is no equivalent single API, so a build
without FFmpeg reads binary PPM and says so — which is a 15-byte header and the
bytes, if a host needs to hand over pixels it already holds.

It is one switch for the whole build tree rather than per target. `vidfab_cuda`
links `vidfab_core`, so an FFmpeg-free DLL beside an MP4-capable exe would need
a second copy of both libraries, and `vidfab_cuda` costs minutes to compile.
The C API does not need the distinction anyway — it never reaches the muxer in
either configuration — so the switch really only decides what `vidfab.exe` can
write.

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

# Without --out, videos go to output/video-YYYYMMDD-HHMMSS.mp4. Generate
# several variations from one prompt with --count. A supplied seed increments
# for each video (11, 12, 13 here); without --seed, or with a negative one,
# every video gets a fresh random seed. Multi-video filenames also receive
# -001, -002, ... suffixes.
vidfab generate --prompt "three variations of a moonlit forest" \
                --count 3 --seed 11

# Long Context-IR prompts do not belong on a command line. --prompt-file reads
# one out of a UTF-8 text file instead; a BOM, CRLF endings and blank space
# around the text are stripped. It replaces --prompt and cannot join it.
vidfab generate --prompt-file prompts/moonlit-forest.txt --steps 30

# The quantisation of each checkpoint is read out of the file, so there is no
# flag for it and the pair need not match. `generate.cmd` wraps all of this.

# Name the canvas outright instead of asking for a shape. --aspect picks the
# largest canvas of that ratio inside the trained 1344x768 area; --resolution
# takes what you give it. Both axes must be a multiple of 32 and the ratio must
# stay inside 1:4 .. 4:1, but the area is *not* capped -- a larger canvas is
# allowed, warned about, and costs attention time with the square of its area.
vidfab generate --prompt "..." --resolution 1024x512 --dry-run

# Condition a run with MiniMax H3 Ref2VA images. Repeat --reference-image in
# subject/style/scene order; that order is preserved in the multimodal prompt
# and packed sequence. At most nine images are accepted. This mode requires a
# Ref2VA transformer checkpoint (FL2VA/text-to-video weights are not compatible).
vidfab generate --prompt "integrated_multimodal_description: ..." \
                --reference-image subject.png \
                --reference-image style.jpg \
                --transformer weights/transformer/minimax_h3_ref2va_pruned_fp8_scaled.safetensors \
                --tokenizer <tokenizer.json> --text-encoder <text-encoder.safetensors> \
                --vae <video-vae.safetensors> --audio-vae <audio-vae.safetensors> \
                --out referenced.mp4

# The unpruned bitsandbytes NF4 Ref2VA build is not currently supported. Its
# execution path is known to produce invalid tiled video; use the validated
# pruned FP8 Ref2VA checkpoint above.

# The pruned FP8 Ref2VA and FL2VA archives have the same tensor schema and no
# identifying metadata. Keep `ref2va` in the Ref2VA filename: vidfab uses that
# distribution name to reject accidental image conditioning with FL2VA weights.

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

`generate --output-accelerator vulkan` moves the final planar RGB to BT.709
limited-range YUV420 conversion onto a Vulkan compute device. The default is
`cpu`. This option accelerates output conversion only and is independent of
`--inference-backend vulkan`, which runs the exact captured-prompt T2VA
denoiser and both VAEs. Until the native conditioner lands that path requires
`--prompt-embedding <capture.safetensors>` with F32
`prompt_embedding [L,5120]`; it never invokes the CUDA conditioner. Vulkan is
loaded dynamically and an explicit request fails clearly when no compatible
device or loader is available; it never silently falls back.

Both VAE arguments also accept the compact checkpoints in `weights/vae`:
`video_vae_nf4.safetensors` and `audio_vae_nf4.safetensors`. The video VAE keeps
bitsandbytes NF4 matrices packed on device and expands only the matrix currently
being used into reusable FP16 workspace. The audio file's decode graph is BF16
weight-normalized; its NF4 pre-block is not part of synthesis.

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
  resolution, the AdaLN table, the step cache's compute-or-skip decision, the
  tokenizer, latent noise, the WAV writer and the colour transform.
  **2198 checks** — but only with `ref/` and the checkpoints present. Three
  tests skip themselves without them (two tokenizer goldens, which need
  `ref/FL2VA/text_encoder/tokenizer.json`, and one transformer case that needs
  a real checkpoint), so a clean clone reports **2138** and is not failing.
  `ref/` is licence-restricted and not redistributable, so 2138 is the number
  most people will see; run from the repository root to get 2198.
  Both figures are measured on the merged tree rather than added up from
  branches: contributors state a delta and the absolute is set here, because
  two branches each correctly adding to the same baseline is how this number
  went wrong before.
- **GPU kernel tests**: every kernel against independent CPU references written
  from the spec rather than from the kernel. **1122 checks**, plus 11 DEFERRED,
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
| SageAttention2.2 (SM86 FP16 P/V, SM120 FP8 P/V) | done; default |
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

**Euler is the reference throughout, and deliberately so.** It is the update the
reference implementation ships (spec §7.3), so "distance from `euler` at 50
steps" is the right axis for a port whose job is to match that reference. A
large `ab2` distance is therefore a statement about how far a different
integrator travels, not a verdict on it.

**Which of the two makes preferable video is left to the user.** They are one
flag apart, cost the same per step, and the answer is a matter of taste across a
distribution of samples rather than something this project should assert. One
viewer comparing the two at 40 steps, same seed and prompt, preferred `ab2` on
both video and audio — one person, one clip, recorded because it is worth
knowing that the question is live, and not pursued further because it is a
choice rather than a defect.

One consequence worth stating for the sections that reuse the **0.2210** figure
as a noise floor: it is measured between *two different integrators*, not
between two draws of one. It bounds how far two defensible results can sit
apart, which is what a floor is for — but it is not a pure-chance quantity, and
a ratio against it should be read as "compared to the spread between samplers"
rather than "compared to nothing at all".

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

## Step caching — shipped, and deliberately uncharacterised

`--cache-threshold <x>` reuses the previous step's velocity instead of calling
the transformer, when the timestep conditioning has barely moved. `0` is off and
is the default. `--cache-warmup <n>` forces the first `n` steps to evaluate, and
`--skip-every <n>` is a calibration-free fixed-interval alternative; the two
skipping modes are rejected together, as is `--sampler ab2` with either, because
AB2 extrapolates from `v_{n-1}` and a reused velocity makes that extrapolation
run through a point the model never visited.

**This architecture suits the technique unusually well.** TeaCache and its
relatives need a per-model polynomial fitted offline, because the quantity they
want — how much the conditioning moved — is buried in a timestep MLP. Here the
*entire* timestep conditioning is an 8-vector, `adaln_t_table[row(t)]`, shared
by all 51 AdaLN consumers (spec §3.4). So `‖c(t_i) − c(t_last)‖` is exact,
calibration-free and costs microseconds, and the previous velocities are already
sitting in host vectors the denoise loop allocates anyway.

**What is measured.** The conditioning trajectory, computed on the host from the
real checkpoint: per-step movement is 0.077–0.083 across the first half of a
30-step schedule, then ramps to 0.83 at the last step. The resulting skip counts,
of 29 evaluations:

| threshold | 0.08 | 0.10 | 0.15 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|
| evaluations skipped | 0 | 6 | 9 | 12 | 15 | 19 |

Each skipped evaluation is **19.1 s** at the default geometry, so threshold 0.20
would take a 50-step run from ~16 min to roughly ~11.

**What is not measured: any of it.** No quality comparison has been run at any
threshold — not one sample, not one seed. The flag is shipped because the code
is verified, not because the trade is understood, and **a user turning it on is
the first person to find out what it costs.** Two things are mechanically
guaranteed and neither is a quality claim: with the flag off the output is
bit-identical to a build without the feature, and the loop provably skips exactly
the steps the planner chose — the planner's decisions and the loop's recorded
decisions are asserted equal, along with the steps the substituted velocity was
actually used on. Warmup has a floor of 2 because steps 0 and 1 have no previous
velocity to reuse, and the last step always evaluates because `sigma_next = 0`
makes `x_next = denoised`, so a stale velocity there writes into the output
undamped.

**And there is a known, unresolved tension a user should walk into knowingly.**
The conventional guidance for this technique is that early steps matter most and
should never be skipped. On this checkpoint the indicator says the opposite: with
shift 12 the video timestep moves 0.003 per step early and 0.17 late, so the
skips are **strongly front-loaded** — at threshold 0.20 with the default warmup,
8 of 12 skips fall in the first half, and the last several steps always evaluate.
Whether "the conditioning barely moved" and "an error here will not compound over
the remaining steps" coincide is exactly the question no measurement here
answers. The flag defaults to off for that reason.

## Temporal locality — probed before the banding kernel was written

Frame-banded attention (each query attending only within ±N latent frames) is
the largest remaining win available here. It is lossy, so before anyone spent a
day on the kernel the question *"does this model tolerate losing distant
temporal attention at all?"* was asked with a deliberately cruder experiment
that needs no kernel: **run one request as three overlapping shorter ones and
cross-fade the latents.**

`vidfab_chunkprobe` drives it and links `vidfab_core` only, so it starts no
CUDA context and can run while the card is busy. `--frames 15` snaps to 22
pixel frames (7 latent), `--frames 45` to 56 (17 latent), and three chunks at a
stride of 5 latent frames tile 17 exactly with a 2-frame overlap. The stride
must be a multiple of 5 — `span[j] = (5/3)·FRAMES_PER_LATENT[j % 5]` repeats
with period 5, so any other stride would give each chunk a different internal
rotary spacing from the frames it stands in for, and the probe would be
measuring its own misconfiguration. `resolve_chunk_plan` refuses it.

Each chunk starts from **its slice of the full request's noise field**, not an
independent draw — a band restricts attention over one noise field, so a slice
is the closer analogue, and independent draws would have guaranteed no two
chunks could agree. Slicing and blending the field back reproduces it bit
exactly for video, and to one fp32 ulp for audio.

**Chunking is strictly more damaging than a band**, which is the point: no
cross-chunk attention at all rather than a soft window, a separate audio
schedule per chunk, and a rotary grid that restarts at every boundary.

### The floor does not transfer between geometries

Measured at the probe's own geometry — 56 frames, 1:1, 10 040 packed rows, 29
evaluations, `euler` against `ab2`, two legitimate integrations of one ODE:

| | video rel_L2 | correlation | audio rel_L2 |
|---|---|---|---|
| **the floor** (euler vs ab2) | **0.4346** | 0.9053 | 0.3668 |
| chunked and cross-faded | **0.7771** | 0.6960 | 0.7199 |

The floor here is **roughly twice the 0.2210 measured at 4170 rows** for the
sampler sweep above. Floors are geometry-specific and inheriting one is a way
to be wrong by a factor of two; measure it beside every comparison.

### The result: a different scene, not a seam

Chunking lands at **1.79× the floor** with correlation falling 0.905 → 0.696,
while the global statistics still match (video mean −0.0269 vs −0.0358, std
1.0901 vs 1.0834). That is the same "different sample, not degraded" signature
as the nvfp4 and AB2 sections — but here it is *not* benign, and the numbers
that say so are not the aggregate ones.

**There is no seam.** The frame-to-frame step profile deviates from the
reference only 1.29× more at the chunk overlaps than away from them, and the
single worst step in the clip is not at a boundary. Mean luma across chunk
interiors spreads 13.95 levels of 255 against the reference's own 17.24 over
the same windows, so there is no exposure drift either. Every artefact the
probe was designed to catch at the boundaries is absent.

**What fails is scene identity.** Chunk 0's last five frames and chunk 1's
first five cover *the same five pixel frames* of the same request, from the
same sliced noise, with the same prompt — and they render different
locomotives (gold boiler domes against none), different backgrounds (leafy
greenery against a lawn), and a different camera height. Each chunk is
individually coherent and plausible. They are simply not the same video, and
the cross-fade between two of them is a visible double exposure.

**That is not a blend artefact**, which the 2-frame overlap made the obvious
suspect. A longer cross-fade would smear the dissolve over more frames; it
cannot reconcile a gold-domed locomotive with a plain one. Audio agrees:
−14.5 dB mean and −0.5 dB peak against the reference's −18.5 and −5.4, i.e.
three independently scheduled streams summing hot to the edge of clipping.

### What this does and does not license

It does **not** condemn banding, and the difference is mechanical rather than a
hedge. Chunking removes cross-boundary information *entirely*; a band still
passes it transitively, and 50 layers of ±N frames reach far further than N.
This probe cannot say how much bandwidth that path needs — only that the path
matters.

What it does establish is that **this checkpoint fixes global scene identity
using long-range temporal context**, and that the failure mode to watch for is
not a seam at a boundary but the subject quietly becoming a different subject
across the clip. Aggregate statistics are blind to it; correlation and eyes are
not.

So: **build the kernel**, and when measuring it, re-measure the floor at the
target geometry, report correlation beside rel_L2, and look at whether the
subject is still the same object at the end of the clip. `vidfab_chunkprobe
stats` prints exactly that set.

## Performance

Measured on the RTX 5090, and the first number is the one that recalibrates
everything else:

**The sustained dense bf16 GEMM ceiling on this card is 216–222 TFLOP/s, not
the 419 on the spec sheet.** At 575 W it holds about 2.45 GHz, and cuBLAS on a
16384³ GEMM gets 220. Every efficiency claim below is against that.

| op | shape | time | achieved |
|---|---|---|---|
| attention, blocked | seq 37710, 56 heads, dim 128 | 589.6 ms | 69.1 TFLOP/s |
| **attention, fused** | seq 37710, 56 heads, dim 128 | **234.5 ms** | **173.9 TFLOP/s** |
| `qkv_proj` | `[21504, 5376]` × 37710 rows | 42.0 ms | 207 TFLOP/s |
| `attn.out_proj` | `[5376, 7168]` | 13.6 ms | 214 TFLOP/s |
| `mlp.fc1` | `[28672, 5376]` | 56.0 ms | 208 TFLOP/s |
| `mlp.fc2` | `[5376, 14336]` | 27.4 ms | 212 TFLOP/s |

**An earlier revision of this table was wrong, and the way it was wrong is worth
keeping.** It reported the four linears at 231–237 TFLOP/s — *above* the ceiling
declared two paragraphs earlier, in a file that insists every efficiency claim be
measured against it. Nobody noticed for months. The cause was the benchmark
filling both operands with `cudaMemset`, so every tensor-core lane multiplied the
same two 16-bit values every cycle: switching activity collapses, the most
power-dense kernel in the pipeline stops reaching the 575 W limit, and the card
boosts. Measured with real data the clock is **2527 MHz at 575 W**; with memsets
it was **2857–2865 MHz at 517–535 W**. In situ, during an actual generation, it
is **2505 MHz at 575 W** — which is the benchmark and the pipeline finally
agreeing.

Attention was flattered worse than a clock argument alone explains, because
constant `q` and `k` make every score identical, so the online softmax never
rescales and never moves its running maximum — **the fused kernel was being timed
on the one input that exercises none of its data-dependent work.**

The control that makes this specific rather than general: `nvfp4_dequant_timings`
kept its memsets deliberately and did not move (1414 / 2094 / 1419 / 1403 GB/s).
It is bandwidth-bound, so it never reaches the power limit, so it never had a
boost to lose.
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

So the four linear layers are not worth touching — `cublasLt` heuristic search,
a larger cuBLAS workspace and row alignment were all measured and buy nothing.
The precise version of that claim is better than "at the machine ceiling": at
the 2505 MHz the card actually holds under a generation, 170 SMs at 512
bf16 FLOP/clk give a roofline these four hit to **97.6–101%**. There is no
headroom in them at all, and the 1.7% that chunking into 8192-row passes costs
— measured by wave quantisation, and separately by the four in-situ TFLOP/s
figures landing within 1.5% of the isolated ones once both are clocked the same
— is the only part that is even structurally attributable.

Attention **was** the dominant cost of a denoising step and bandwidth-bound: the
blocked path moves ~1131 GB per layer while running at only 69.1 of 216
TFLOP/s, because the score tile round-trips through HBM at ~12 bytes per
element. Its own floor was ~682 ms, so tuning it further was pointless — the
round trip had to go, not shrink. (The 1.41 TB/s and ~682 ms figures here were
derived from the constant-operand timings and are correspondingly a few percent
optimistic; they are left as written because the argument they support —
delete the round trip rather than tune it — does not turn on the last digit,
and the path they describe no longer runs.)

The tile-budget sweep (192 MiB → 1010 ms, 640 MiB → 749 ms, 1536 MiB → 711 ms)
is explained by `t ≈ (12·H·S² + 16·H·D·S²/bk) / 1.4 TB/s`, where the second
term is the fp32 accumulator being read-modify-written twice per key block.
That model reproduces all three points to within 3%, and both of its terms are
exactly what the fused kernel deletes.

### The fused kernel

`AttentionBackend::kFused` keeps S and P in registers and never writes them
anywhere. **234.5 ms against the blocked path's 589.6, at 173.9 TFLOP/s — 78–80%
of the machine ceiling, and no workspace at all against 1.63 GiB.**

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

`cp.async` double-buffering of the K/V stage was tried and **reverted**: two
resident buffers and one barrier instead of two measured **7.3% slower** than
the single-buffered kernel it replaced, and `cp.async.ca` in place of `.cg`
refuted the L1-bypass explanation. The surviving hypothesis is that the
compiler's own hoisted loads already covered the latency, and an explicit
one-tile pipeline replaced a working schedule with a rigid one.

### Frame-banded attention — `--attn-band`, off by default

The generator exposes
`--attention none|flash2|sage2|sol|sol-experimental|exact`.
`sage2` is the default; `flash2` is the shipped BF16 fused path and `none` is
the unfused, memory-bounded cuBLAS reference. `exact` is a separate pinned
CUDA/Vulkan cooperative contract: CUDA transformer main blocks and the token
refiner call the deterministic H3 primitive directly, with no materialized
score tensor and no fallback through `flash2` or `none`. It intentionally does
not promise byte identity with the shipped fused implementation. On the
qualified RTX 5090 tuple its accepted current cost at S37727/H56/D128 is
2.525 s full or 1.261 s at the default +/-9 band on Vulkan, and 1.667/0.783 s
on CUDA; this is the fastest native exact implementation, not the default.
`sage2` is an explicitly lossy SageAttention2.2 path: smooth-K and per-warp
INT8 Q/K on every supported GPU. SM86 uses the upstream INT8-QK/FP16-PV kernel
with FP32 accumulation because Ampere has no FP8 tensor cores. Blackwell
retains the existing per-channel FP8 E4M3 V kernel. Workspace sizing follows
the selected device, so Blackwell's allocation and launch path are unchanged;
Ampere uses twice the packed-V storage and no V-scale preparation. Head
dimensions 64 and 128 are supported on compute capability 8.0 or newer. Frame
banding is accepted by `flash2` and `exact`; Sage2 is rejected rather than
silently falling back.
The vendored primitives retain Apache-2.0 notices under
`third_party/sageattention`.

`sol` is the scalar training-free block-routing reference. `sol-experimental`
selects the separate SM120 TMA/WMMA pipeline evaluated in
`docs/sol_quality_gate_2026-08-08.md`. Both are lossy: rejected 64-token blocks
use the released centroid/value correction rather than exact token attention.
The experimental spelling is intentional while its quality matrix is under
review. The declared two-seed quality matrix failed 11 of 12 numeric rows, so
this path is not production-approved and remains disabled by default; see
`docs/sol_quality_matrix_2026-08-08.md`. It requires SM120, head dimension 128, no more than 1024 physical
blocks, and TMA-compatible aligned tensors; an unsupported request fails
clearly and never silently runs the scalar kernel. Frame banding is incompatible
with either Sol mode.

A video row attends to ±N latent frames instead of the whole packed sequence.
Text and audio rows keep global attention, and every video row keeps the
text/audio prefix, so the conditioning path is untouched; what it drops is
distant video-to-video attention. At ±9 frames on the default geometry that is
**2.05× on the attention kernel and 1.44× on a denoising step** — 19.2 s to
13.2 s, roughly 16 minutes to 11 on a default run. The saving grows with request
length, because the band is a fixed number of frames and the sequence is not.

The prior is unusually good for a lossy method: MiniMax's own documentation says
H3 *"natively supports sparse-attention training and inference"*, introduced in
the final training stage. **These weights were fine-tuned to tolerate sparsity**;
the open release ships full attention only because their sparse path is not
published.

**The band is frame-granular, not a window over the packed row index.** Video
packs frame-major, so a latent frame is a contiguous run of 1008 rows and a naive
row-index band would cut *within* a frame — an anisotropic spatial prior that
would look like a banding artefact and get blamed on the method. Each query tile
therefore gets two key ranges, not one: its frame band **plus** the text/audio
prefix, which is 431 of 37 730 rows (1.14%) and sits at the *front* of the
sequence. A single window around a late frame would exclude it and cut every
video row off from the prompt.

**What it costs, measured against its own noise floor.** The floor is the
distance between two integrators on the same request — `--sampler euler` against
`ab2`, both band-off, same geometry and step count — i.e. the distance between
two equally valid samples. **It has to be measured where the comparison is
made:** it is 0.221 at 4 170 rows, 0.435 at 10 040, and 0.348 at 37 730 with 40
steps. Inheriting one of those instead of measuring it is worth a factor of two.

| steps | video floor | video band ±9 | ratio | audio floor | audio band ±9 | ratio |
|---|---|---|---|---|---|---|
| 6 | 0.2371 | 0.4640 | 1.96 | 0.7375 | 0.7118 | 0.97 |
| 24 | 0.3718 | 0.6129 | 1.65 | 0.4414 | 0.9213 | 2.09 |
| 40 | 0.3478 | 0.6321 | 1.82 | 0.7628 | 0.8570 | 1.12 |

**Video and audio must be reported separately and never combined.** Audio is 414
of 37 730 rows, so a combined figure reads 1.65× and buries a 2.09×.

**Video's cost is 1.6–2.0× the floor and shows no step-count trend.** Audio's is
**not characterised**, and the reason is worth stating because two points made it
look like one: measured at 6 and 24 steps alone, audio's ratio appeared to climb
from 0.97 to 2.09 while video's fell, with a tidy mechanism attached — audio
converging while the band's damage accumulated. The 40-step point destroyed it.
Neither series is monotonic, and the 24-step reading was an outlier in *both* of
its components rather than a point on a line.

The diagnostic that says which number to trust is the spread of each quantity
across the three step counts:

| | floor | band | ratio | |
|---|---|---|---|---|
| video | ×1.57 | ×1.36 | **×1.19** | normalising **absorbs** variance |
| audio | ×1.73 | ×1.29 | **×2.16** | normalising **amplifies** it |

For video the ratio is steadier than either of its parts, which is what a
working yardstick looks like. For audio, dividing by the floor produces something
*less* stable than the raw numbers — so the audio ratio is not a measurement, it
is two noisy numbers divided. Fixing that needs more **seeds**, not more step
counts; one seed per point leaves the floor moving as much as the effect.

**Is it worse than chunking the request?** Measured against its own floor, ±9
banding is 1.6–2.0× for video; a chunked-and-cross-faded probe was 1.79× against
its floor. Dividing those is the natural thing to do and it does not support a
conclusion — different geometries, different step counts, and the band's own
ratio moves ×1.19 across step counts with a single seed, comparable to the whole
gap being compared. What the data supports is narrower: **banding at ±9 costs
video roughly 1.6–2.0× the sampler noise floor, with audio uncharacterised.**
Whether that trade is worth taking is a judgement about output that wants eyes on
a set of samples, which is why it is a flag and not a default.

**One viewer has now looked**, at 40 steps, default geometry, same seed and
prompt, band off against band ±9: the banded clip's **camera movement differs,
and the quality is otherwise good**. That is the "different sample rather than
a degraded one" signature again, reported by eyes rather than inferred from
statistics — and it is what the whole table above is a proxy for, since no
rel_L2 answers "did the clip become a different clip". It is one person and one
pair of clips, so it does not settle the trade; it is recorded because it is
the only observation here of the kind the decision actually turns on.

All fifteen latent dumps behind this table were compared with `vidfab compare`,
whose `rel_L2` and correlation were cross-checked against an independently
written tool: the two agree to every digit quoted.

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

Ref2VA loads the conditioner checkpoint's vision tower on demand, runs all 27
visual blocks and the main/DeepStack mergers, then releases those weights after
the one conditioning pass. Text-only generation never pays that 1.19 GB cost.

The exact Vulkan conditioner now implements this same visual topology. It
streams one BF16 visual block or merger at a time, extracts DeepStack features
after visual blocks 8/16/24, and injects them after text decoder layers 0/1/2.
On the pinned 256x256 production-grid fixture, CUDA and Vulkan matched all 27
visual residual boundaries for the I8 authority, and all 50 multimodal decoder
boundaries plus the final FP32 prompt embedding byte-for-byte for both shipped
formats. The I8+ConvRot
final FNV64 is `A875C128AA7A0E9D` (CUDA/Vulkan 4.51/6.99 s, 771.8 MiB peak);
the NVFP4+AWQ final is `EBC9E36A30C843CD` (2.92/4.04 s, 569.9 MiB peak).
CUDA-disabled I8 replay took 8.24 s. The checkpoint authorities are SHA-256
`BC2CED0FBEA64757FA9ACDDCCFC0B3F4819D1DCF1DA6C124D690D368BE283923`
and `33E69E3EDAB846D52949BAFDB00378BD3F5A93F78124FC83D5EF109DC4A1FCBB`;
the normal-prompt tokenizer provenance remains
`A5D85B6DCC535E6B93115A9EF287E6132FDBF30270DA6218194BA742261173C7`.
At the trace-free production maximum (S16384 visual patches, L4100 decoder
rows), actual-SHA I8 and NV runs pin visual main/DeepStack FNV64 values
`63A7DD4533D82D51`, `DA837B598AE29C53`, `0EDF2B1389F62CE2`, and
`D8F970E5016F22B7`; the final I8/NV embeddings pin `EDE491026C069661` and
`4435938303E77287`. The runs took 60.79/54.96 s cold/warm for I8 and
54.52/51.65 s for NVFP4. Both formats reported a 2676.3 MiB logical and
non-staging allocator peak; maximum streamed
weights were 465.3/261.6 MiB, with stable repeat pool/reservation/descriptor
counts and a return to the staging-only used baseline after unload. The test
also verifies all 351 `visual.*` tensor names, dtypes, shapes, byte counts, and
payload bytes are identical between the two independently SHA-pinned archives.

The checked-in exact tensor shader provenance is source SHA-256
`67D485F5818226DAE9D95FCB7B9DD0D208F767DF60B014981AECEDB2FFC1EC7E`
and SPIR-V SHA-256
`08AB6BBA63C121EAAE8348CB62328F9C7337D21E9E6DE5C62292DA8BE47F1E9B`,
matching the mandatory configure-time checks in `CMakeLists.txt`.
Top-level reference generation still fails closed until the separate keyframe
video-VAE encoder is ported; no reference request crosses into CUDA silently.

## Licence

The code in this repository is the author's. Model weights, configuration files
and the reference material under `ref/` are covered by the MiniMax H3 Community
License and are not redistributed.
