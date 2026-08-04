# vidfab

A from-scratch C++/CUDA implementation of [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3),
targeting a single RTX 5090 with no Python at runtime.

**Status: everything downstream of the denoiser works end to end.** `generate
--synthetic-latents` produces a real MP4 — both VAEs, the colour transform and
the muxer, running against the real checkpoints. The text conditioner and the
transformer forward pass are the remaining gap. See [Roadmap](#roadmap).

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

The default CUDA architecture is `120` (Blackwell / RTX 50 series). Override
with `-DCMAKE_CUDA_ARCHITECTURES=90` for Hopper. The core library and CLI build
without CUDA; the decoder does not.

## Usage

```sh
# Resolve a request: canvas, frame alignment, packed sequence length, both
# sigma schedules. Reads no weights, so it is instant.
vidfab generate --prompt "..." --aspect 16:9 --frames 124 --steps 50 --dry-run

# Everything downstream of the denoiser, against the real checkpoints: seeded
# noise -> unpatchify -> video VAE -> audio VAE -> H.264/AAC in an MP4.
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
  **1719 checks.**
- **GPU kernel tests**: every kernel against independent CPU references written
  from the spec rather than from the kernel. **318 checks.** These exist because
  the failure modes here are silent — a wrong QKV de-interleave, a wrong
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

`docs/vae_decoder_spec.md`, `docs/transformer_spec.md` and
`docs/audio_vae_spec.md` document each stage layer by layer with citations
into the reference source; `docs/convrot_notes.md` covers the int8 rotation.
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
| H3-Omni-Transformer, 50 layers | in progress |
| Qwen3-VL-32B text encoder (int8 ConvRot, 50 layers) | spec in progress |
| Fused attention, native fp8/nvfp4/int4 GEMM | not started |

## Memory budget

The card is a 32 GB RTX 5090 and the stages do not fit together, which is what
forces the pipeline's shape. Resident weights, measured from the checkpoints:

| Stage | Device weights |
|---|---|
| Qwen3-VL conditioner, int8 | 24.4 GB (+1.6 GB embedding, kept on the host) |
| H3 transformer, fp8 | 19.3 GB |
| Video VAE decoder | 9.0 GB (fp16 on disk, widened to fp32) |
| Audio VAE | 0.6 GB |

So the conditioner and the transformer cannot co-exist, and the pipeline runs
strictly in sequence: encode the prompt, free the conditioner, load the
transformer, denoise, free it, decode. The vision tower in the conditioner
checkpoint (1.19 GB) is never loaded — it is reached only by the keyframe path,
which this port does not implement.

## Licence

The code in this repository is the author's. Model weights, configuration files
and the reference material under `ref/` are covered by the MiniMax H3 Community
License and are not redistributed.
