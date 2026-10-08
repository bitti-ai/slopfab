# Upscaling

For learned upscaling before video VAE decoding, see [H3 latent upscaling](latent_upscaling.md).
Use `generate --latent-upscale` with the Minimax H3 3D conv v1 checkpoint.

slopfab runs RealESRGAN_x4plus natively on CUDA or Vulkan, with no Python at
runtime. It supports the 23-block RRDBNet with 64 feature channels, 32 growth
channels, and RGB input/output. Inference and accumulation use FP32; F32, F16
and BF16 checkpoint tensors are accepted and validated before upload.

Download the [Comfy-Org safetensors repackaging](https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged)
to `weights/upscaler/RealESRGAN_x4plus.safetensors`:

```powershell
New-Item -ItemType Directory -Force weights/upscaler
curl.exe -L --fail -o weights/upscaler/RealESRGAN_x4plus.safetensors https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged/resolve/main/RealESRGAN_x4plus.safetensors
```

Weights are not included in the repository or downloaded by slopfab.
The original project is [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

[SeedVR2 3B](seedvr2.md) is also available through the same interface with
`--upscale-method seedvr2`. It runs on CUDA and restores temporal segments with
a tiled VAE. The selector works for standalone media and generated output.

## Standalone image

```sh
slopfab upscale --input image.png --out image-4x.ppm --upscale-method realesrgan
slopfab upscale --input image.png --out image-4x.ppm --inference-backend vulkan
```

The default backend is CUDA when compiled in, otherwise Vulkan. Choose a model
elsewhere with `--upscale-model FILE`. The default model path is relative to the
working directory. At least one GPU backend must be enabled at build time.
Input uses the normal reference-image decoder (PNG/JPEG and other supported
formats); with `.ppm` output, video input reads only its first frame. `.mp4` or `.mkv`
output instead streams and upscales the complete video, copying its audio. Output is an
8-bit RGB binary PPM. Alpha is not preserved. `--dump FILE` additionally writes
the float output as a `pixels` safetensor with shape `[3,1,H,W]`.

`--upscale-method` selects the algorithm independently of `--upscale-model` and
the GPU backend. It accepts `realesrgan` (the default) and `seedvr2`.
Unknown method names are rejected before loading weights. The same selector is
available on `generate`, and library callers use stable method identifiers.

## Generated video or still image

```sh
slopfab generate --prompt "A cat in warm lamplight" --resolution 256x256 --frames 22 --upscale-method realesrgan --upscale-model weights/upscaler/RealESRGAN_x4plus.safetensors --out video.mp4
```

The example generates at 256x256 and delivers 1024x1024 frames. Upscaling runs
after the VAEs release their GPU resources, on the selected inference backend.
Frame count, frame rate and audio are preserved. Saved latents and the generation
plan retain the original diffusion dimensions. Upscaling is disabled by default; selecting `--upscale-method` enables it with
the default checkpoint path, unless `--upscale-model` overrides that path.
Image editing rejects upscaling because resizing would violate exact preservation
outside the edit box.

Both commands accept:

| Option | Default | Meaning |
| --- | --- | --- |
| `--upscale-tile` | 128 | Tile side in input pixels; 0 processes the complete padded frame |
| `--upscale-tile-pad` | 10 | Input context on each tile edge, cropped from the result |
| `--upscale-pre-pad` | 10 | Reflect padding at the right and bottom image edges |

These follow the upstream RealESRGANer padding/cropping convention. Small images
use repeated reflection (a one-pixel axis repeats its sole pixel). Tile padding
reduces seams but does not guarantee equality to whole-frame inference. Increase
tile padding for difficult boundaries, or use tile size 0 for small images.
Accepted tile sizes are 0..512 and padding is 0..256. Actual limits also depend
on GPU storage and dispatch limits; reduce the tile size if a limit is exceeded.
An unsupported backend or allocation failure produces an error, with no silent
fallback. Decoded output uses 16 times the original pixel storage on the host.

## Real-ESRGAN video consistency

Real-ESRGAN x4plus has **no inference sampling seed**. Each frame passes through
the same deterministic convolution/residual network; it has no temporal state,
posterior sampling, diffusion noise, dropout or per-frame color normalization.
This also matches the upstream [RRDBNet architecture](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py)
and [frame-by-frame video inference](https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan_video.py).
The shared `seed`, `segment_frames` and `color_match` options apply to SeedVR2
only. Increasing the segment length does not add temporal context to Real-ESRGAN.

Real-checkpoint tests on CUDA and Vulkan verify bit-identical restored frames
for A/B/A sequences, separate versus multi-frame calls, changed SeedVR2-only
settings, intervening changes of dimensions/tiling, and fresh versus reused
models. The earlier SeedVR2 seed/conditioning defect does not apply here.
These checks fix the backend/build/settings; cross-backend or cross-version
floating-point bit identity is not guaranteed. Lossy input/output codecs can
also change decoded pixels even when the underlying scene is stationary.

The main measured avoidable inconsistency is insufficient tile context. As a
feature moves across the fixed tile grid, the model sees different surrounding
pixels. A controlled five-frame, two-input-pixel horizontal pan at 320x180 →
1280x720 gave the following results on this RTX 5090. The metric aligns motion
and excludes a 32-input-pixel border. All settings use the same checkpoint,
FP32 inference and default pre-padding of 10.

| Tile / context padding | Mean motion-aligned RGB RMSE | RMSE versus untiled output | Warm CUDA seconds/frame |
| --- | ---: | ---: | ---: |
| 128 / 10 (default) | 0.002638 | 0.006965 | 0.277 |
| 128 / 32 | 0.001131 | 0.003318 | 0.409 |
| 128 / 64 | 0.000401 | 0.000159 | 0.666 |
| Untiled | 0.000395 | 0 | 0.236 |

CUDA and Vulkan agree on the reported consistency values. Padding 64 reduces
motion-aligned error about 85% on this fixture, but is about 2.4 times slower
than padding 10. These are controlled translation measurements, not a general
perceptual-quality score; complex motion, compression noise and newly visible
objects can still change framewise predictions. No temporal averaging was added.

For videos where tile boundaries are visible, use complete frames when the
backend and memory permit, or increase context padding:

```sh
# Complete frames: removes artificial internal tile boundaries.
slopfab upscale --input input.mp4 --out output.mp4 --upscale-method realesrgan --upscale-tile 0
# Bounded spatial tiles with more surrounding context.
slopfab upscale --input input.mp4 --out output.mp4 --upscale-method realesrgan --upscale-tile 128 --upscale-tile-pad 64
```

Defaults are unchanged to preserve their runtime/memory tradeoff. The standalone
C streaming API exposes tile size, including 0, but not context padding;
the CLI and C++ API expose both.

The old complete-frame size check still assumed a full-spatial im2col allocation.
It now checks the actual largest activation (4x resolution, 64 channels), since
CUDA convolution scratch is chunked. This enables larger untiled CUDA frames
without changing model arithmetic. A real 960x540 → 3840x2160 run completed
three times at about 2.1–2.3 seconds/frame and a sampled global CUDA peak of
7,323 MiB, including a 1,613 MiB idle baseline. A 640x360 → 2560x1440 run also
passed. The 32-bit activation index limit remains enforced; Vulkan has additional
device storage/dispatch limits, and allocation failures still report errors.

The video CLI now follows FFmpeg autorotation when choosing decoded dimensions
and preserves Real-ESRGAN's source sample aspect ratio, including its inversion
after quarter-turn rotations. Previously, rotation-tagged portrait video could
be forced into landscape dimensions, and non-square pixels lost their display
aspect. Tests cover integer/fractional rotations and encoded aspect ratios on
Windows and Linux. A real-checkpoint CLI test restored a 32x16, SAR 2:1, 90-degree
source to 64x128, SAR 1:2, with no leftover rotation tag. SeedVR2's explicit target
dimensions and square-pixel output remain unchanged.

Reproduce the motion measurements with:

```sh
slopfab_realesrgan_temporalbench MODEL.safetensors cuda 320 180 0 10 5 pan full.f32
slopfab_realesrgan_temporalbench MODEL.safetensors cuda 320 180 128 10 5 pan pad10.f32
slopfab_realesrgan_temporalbench MODEL.safetensors cuda 320 180 128 64 5 pan pad64.f32
python tools/realesrgan_temporal_metrics.py 320 180 full.f32 pad10.f32 pad64.f32
```

Use `vulkan` to check the other backend or `static` for identical input frames.
These dumps contain interleaved RGB float32; they differ from the planar dumps
written by the performance benchmark below. NumPy is needed only for analysis.

## Measured Real-ESRGAN performance

Measured on Windows, RTX 5090 32 GiB, Release build, CUDA 12.8 compiler with
the dynamically selected CUDA 13 cuBLAS, using `RealESRGAN_x4plus.safetensors`.
The baseline is revision `1a09b89`; the benchmark was added in `8711ee4` before
runtime changes. Both versions use FP32 inference and accumulation without TF32.
These are local measurements, not performance guarantees for other GPUs.

The main workload is one 320x180 RGB frame upscaled to 1280x720, with default
10-pixel tile padding and reflection padding. Times are the median of nine warm
calls after one cold call, reusing the loaded model. They include tile preparation,
GPU transfers and output assembly, and exclude writing the optional float dump.

| Backend / input tile size | Before | After | Speedup | Sampled global peak GPU memory before / after |
| --- | ---: | ---: | ---: | ---: |
| CUDA / 128 | 1.299 s | 0.275 s | 4.73x | 2,569 / 2,029 MiB |
| CUDA / untiled | 0.450 s | 0.235 s | 1.92x | 4,387 / 2,545 MiB |
| Vulkan / 128 | 0.633 s | 0.477 s | 1.33x | 3,369–3,374 / 3,246 MiB |

CUDA memory uses `cudaMemGetInfo` every 10 ms. Its idle baseline was 1,612.6 MiB;
the incremental peak therefore fell from 956 to 416 MiB tiled, and from 2,774 to
932 MiB untiled. Warm retained global memory fell from 2,409 to 2,029 MiB tiled
and from 3,895 to 2,545 MiB untiled. Activations remain cached for model reuse.
Peak in the benchmark is cumulative across all calls. Sampling can miss brief
peaks, and global usage includes unrelated allocations.

Vulkan memory was measured separately with `nvidia-smi`, requesting 20 ms
sampling; its idle baseline was 2,716 MiB. Two alternating before/after runs
showed a 123–128 MiB lower peak. These counters differ from CUDA's counters,
so the memory columns should only be compared within each backend. The benchmark
itself reports `memory_available=0` for Vulkan; zero fields do not mean zero use.
A smaller 64x36 untiled Vulkan workload also improved from 56.2 to 30.1 ms
(median of three warm calls).

A complete 13-frame 320x180 H.264/AAC video, using the default CUDA tile size,
fell from 18.09 to 4.55 seconds including model initialization, FFmpeg decoding,
encoding and audio remuxing. Both outputs contain 13 frames at 1280x720, and the
copied audio packet SHA-256 matches the input. This is a separate single-run
end-to-end check; the frame benchmark above provides repeated timings.

The retained changes are:

- CUDA activation leases reuse buffers in default-stream order, removing
  repeated allocation/free synchronization. Persistent weights have separate
  allocations; outstanding outputs keep the activation pool alive.
- Convolution im2col processes at most 32,768 pixels at once. Scratch is bounded
  to 216 MiB at the widest 192-channel layer, instead of growing with the full
  output area. This preserves FP32 math and the original padding semantics.
- Vulkan records up to 16 operations per submission and collects completed
  resources immediately. A 64 MiB resource budget limits batching retention;
  a single larger operation is submitted immediately. Readback flushes pending
  work, preserving tile-level cancellation. `SLOPFAB_REALESRGAN_VULKAN_BATCH=1`
  restores per-operation submission for comparisons (valid range: 1–64).

Experiments were rejected when they did not help: an 8,192-pixel CUDA scratch
chunk slowed tiled inference about 10% compared with 32,768; fused nearest
upsampling showed no measurable end-to-end or peak-memory benefit; gathering
concatenated channels inside im2col slowed this workload roughly threefold.
Those fusion experiments are not enabled or retained in the implementation.

Maximum final RGB differences versus the original CUDA implementation were
`1.91e-6` tiled and `6.30e-5` untiled (RMSE `2.64e-7` and `1.01e-6`). Changing
GEMM chunk shapes can change floating-point rounding. The existing independent
PyTorch reference still passes its `2e-4` maximum-error requirement on both
backends. Vulkan before/after outputs are bit-for-bit identical at both benchmark
sizes. Windows and Linux tests cover reference parity, chunk boundaries, queued
buffer reuse, output lifetime, batching dependencies, resource bounds, tiling,
multiple frames and cancellation.

Reproduce the frame measurements with:

```powershell
cmake --build build --config Release --target slopfab_realesrganbench
build/Release/slopfab_realesrganbench.exe weights/upscaler/RealESRGAN_x4plus.safetensors cuda 320 180 1 128 10 output-prefix
# Use tile 0 for untiled inference, or replace cuda with vulkan.
# Use - instead of output-prefix to suppress raw float32 output files.
```

The positional arguments are checkpoint, backend, input width, input height,
frames, tile size, repeats and output prefix, optionally followed by tile context
padding (default 10). Dumps use planar RGB float32.
`tools/seedvr2_compare_outputs.py` also compares these raw float files despite its
name. Preserve separate baseline and candidate executables, keep the GPU free
of other workloads, and compare matching tile/padding settings: tiling changes
the model's context, so tiled and untiled outputs need not match each other.

## Library APIs

The C++ factory in `include/slopfab/upscale.h` returns the shared `Upscaler`
interface, with dispatch based on the requested method:

```cpp
auto upscaler = slopfab::make_upscaler(slopfab::UpscaleMethod::kRealEsrgan,
                                      model_path, slopfab::DeviceBackend::kCuda);
auto result = upscaler->upscale(pixels, frames, height, width);
```

It loads weights once and upscales planar float RGB clips
`[3,frames,height,width]`. `upscale_dimensions(height, width, method, options)` gives the output size.
Real-ESRGAN uses a fixed 4x factor; SeedVR2 accepts a target size and defaults to 4x. Output is clamped to `[0,1]`; non-finite input
is rejected. A progress callback can cancel between tiles by returning false.
Generation callers set `RunOptions::upscale_method` and `upscale_model_path`.

C API ABI 1.19 adds:

```c
slopfab_request_set_upscaler(request, SLOPFAB_UPSCALE_REALESRGAN, model_path, 128, 10, 10);
```

An empty model path disables postprocessing. Unknown method identifiers are
rejected without changing the request. The generation output dimensions
reflect the upscale; the plan describes the diffusion canvas. Progress uses
`SLOPFAB_STAGE_UPSCALING`, with completed/total tiles in `step`/`total_steps`.
Stage numeric values are stable and do not imply chronological order.

## Verification

`ctest --test-dir build -C Release -R "^upscale$" --output-on-failure` checks
GPU operators, full model execution with synthetic weights, non-square RGB
images, multiple frames, tiling, reflection, input validation and cancellation.

To compare both compiled backends with an independent PyTorch reference using
the actual checkpoint (requires development-only `torch` and `safetensors`):

```powershell
python tools/realesrgan_reference.py weights/upscaler/RealESRGAN_x4plus.safetensors weights/upscaler/reference.safetensors
$env:SLOPFAB_REALESRGAN_MODEL = (Resolve-Path weights/upscaler/RealESRGAN_x4plus.safetensors).Path
$env:SLOPFAB_REALESRGAN_REFERENCE = (Resolve-Path weights/upscaler/reference.safetensors).Path
ctest --test-dir build -C Release -R "^upscale_integration$" --output-on-failure
```

The reference covers full-frame and tiled inference, with two distinct frames.
It requires maximum absolute pixel error at most `2e-4`. Without these local
fixtures the integration suite explicitly skips.

The Vulkan shader is built with Khronos glslang 16.5.0:

```sh
glslang -V --target-env vulkan1.2 -S comp src/vulkan/upscale.comp -o src/vulkan/upscale.comp.spv
```

`src/vulkan/upscale.sha256` pins the LF-normalized source and binary. Normal
builds use the checked-in SPIR-V and need no shader compiler or Vulkan SDK.
