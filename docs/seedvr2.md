# SeedVR2 video restoration

`slopfab upscale --upscale-method seedvr2` restores decoded video using SeedVR2 3B on CUDA. The native
C++/CUDA implementation needs no Python, ComfyUI, text encoder, or H3 weights.
It reads the Comfy-Org safetensors directly, including their fixed positive
text conditioning. Model downloads remain separate from the application.

Download these files from [Comfy-Org/SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2/tree/main):

```text
weights/seedvr2/seedvr2_3b_fp16.safetensors   (diffusion_models/)
weights/seedvr2/ema_vae_fp16.safetensors     (vae/)
```

`seedvr2_ema_vae_fp16.safetensors` is the same VAE under another name; select it
with `--upscale-vae`. Custom locations use `--upscale-model FILE --upscale-vae FILE`.

```sh
slopfab upscale --input input.mp4 --upscale-method seedvr2 --upscale-resolution 1280x720 --out restored.mp4
```

The CLI requires `ffmpeg` and `ffprobe`, found beside the executable, in
`external/ffmpeg/bin`, or on PATH. It streams decoded RGB frames to restoration
and streams restored frames to H.264 output, copying the input audio streams.
Use `.mkv` if the source audio codec cannot be copied into MP4. Existing output
files are refused; a partial file is renamed only after both media processes
finish successfully. Ctrl+C cancels and removes that partial file.

## Clip length and memory

There is no total-frame or duration limit. Inference is bounded by one temporal
segment plus its overlapping boundary frame. The CLI also queues at most two
decoded and two restored frames, plus one active frame per media worker. A
15-second video does not need to fit on the GPU or in host RAM in full.

| Setting | Default | Effect |
| --- | --- | --- |
| `--upscale-segment-frames N` | 5 | Restore `4n+1` frames jointly; 1 permits independent image processing |
| `--upscale-tile N` | 256 | Bound VAE spatial activations; multiples of 16 from 128 to 2048, or 0 for untiled |
| `--upscale-seed N` | 666 | Separate restoration seed |
| `--upscale-device N` | 0 | CUDA device index |
| `--upscale-no-color-match` | off | Disable per-frame RGB mean/std color correction |

For a 32 GB RTX 5090 at 720p, an untiled VAE avoids spatial tile boundaries:

```sh
slopfab upscale --input input.mp4 --upscale-method seedvr2 --upscale-resolution 1280x720 --upscale-segment-frames 5 --upscale-tile 0 --out restored.mp4
```

For lower memory, retain the default VAE tile, reduce it to 128, or use
`--upscale-segment-frames 1`. A one-frame segment gives up joint temporal restoration.
Increase segments to 9 or 17 when memory permits. Explicit settings are honored;
allocation failure reports an error instead of silently changing quality settings.

Weights are memory mapped. A restorer caches converted weights across blocks
and segments within a budget: one third of initially free CUDA memory, capped
at 8 GiB, and disabled when less than 8 GiB is initially free. Uncached weights
retain their bounded stage/block lifetimes. Set `SLOPFAB_SEEDVR2_CACHE_MIB=0`
to disable residency, or a nonnegative MiB value to select a budget (clamped
to half the initially free memory). This changes memory use, not model settings.
The OS may also cache the checkpoint in host memory.

Activations and temporary buffers reuse separate pools. MLPs process at most
4096 token rows at a time. Convolution uses a reusable arena of at most 32 MiB,
with a direct GEMM path for pointwise kernels. Attention processes windows
without allocating a full-video attention matrix. VAE width-512 attention
uses the bounded blocked backend; DiT width-128 attention uses the fused backend.
GPU tile assembly is limited to segments containing at most 256 MiB of float
RGB and to staging that fits within one eighth of currently free CUDA memory;
larger requests use the CPU assembly path. FP8 checkpoint weights are uploaded
packed and converted on the GPU, while arithmetic and resident weights remain BF16.

Historical duration validation before the October 7 optimizations, measured on
this RTX 5090 (32 GB), using the published FP16 weights and five-frame segments:

| Test | VAE tile | Time | Peak total GPU use |
| --- | --- | --- | --- |
| 640x360 input, 1280x720 output, 359 frames / 15.000 seconds | 0 | 644 s | 8,886 MiB |
| 1280x720 output, five frames | 256 | 18.1 s | 3,454 MiB |

These totals include desktop baselines of 2,360 and 2,110 MiB respectively,
sampled with `nvidia-smi` every 0.5 seconds. The long output retained all 359
frames, exact duration and an identical copied-audio SHA-256. Smaller GPUs
have not been physically tested. Tiling is a memory/quality tradeoff, not a
claim of identical output. Clip length is not limited to the measured duration.

## Shared upscaling interface

Both algorithms use `--upscale-method`, `--upscale-model` and `--upscale-tile`.
Real-ESRGAN's tile is measured in input pixels; SeedVR2's tile is measured in
VAE output pixels. Its default CLI tile is 256. The two algorithms also share
video streaming, media probing, audio copying and `.ppm` still-image output.
Without `--upscale-resolution`, SeedVR2 defaults to 4x input dimensions.

Generation uses the same selector and options:

```sh
slopfab generate --prompt "A cat" --resolution 320x192 --upscale-method seedvr2 --upscale-resolution 1280x768 --upscale-tile 256 --out restored.mp4
```

Selecting a method enables upscaling and chooses its default model path;
`--upscale-model` can override that path. Generation releases VAE resources
before upscaling. Its existing host input/output buffers hold the generated
clip, while SeedVR2 GPU work remains segmented. The standalone video command
streams host buffers as well. Shared image/generation input is resized using
the existing 8-bit Lanczos image utility; the dedicated streaming C API accepts
already resized float pixels without this quantization.

## Processing contract

- Input is resized with Lanczos to the requested dimensions. Supply dimensions
  with the original aspect ratio to avoid stretching. MP4/MKV output dimensions
  must be even; the model pads spatially to multiples of 16 and crops afterward.
- Segments share one frame, averaged between adjacent restored outputs. The
  final segment repeats its last frame to reach `4n+1`; padding is removed before
  output. Frame count is exact for constant-rate input.
- Constant frame rates, including rational rates such as `30000/1001`, are kept.
  Variable-rate sources are explicitly normalized to their average frame rate.
- VAE tiles overlap by 64 output pixels and are feathered. Spatial tiling changes
  group normalization and attention context, so it is approximate and may show
  seams. Use `--upscale-tile 0` when memory permits.
- The VAE posterior is sampled, followed by one Euler diffusion evaluation at
  sigma 1 with CFG 1 and latent scaling 0.9152. Random draws are reproducible for
  a fixed native build/settings, but are not PyTorch's RNG sequence. Changing
  segment length changes restoration. Split at scene cuts for best boundaries.
- RGB is processed through 8-bit media input/output. HDR, alpha, frame
  interpolation, 7B/7B-sharp, packed INT8/MXFP8/NVFP4, and Vulkan are not supported.
  The loader accepts dense 3B FP16/BF16/FP32 and FP8 E4M3 tensors; the validated
  baseline is the published 3B FP16 checkpoint.

## Library

ABI 1.20 adds `slopfab_seedvr2_upscale` in `capi.h`. This is a synchronous
streaming API; call it on a host worker thread for asynchronous operation.
The input callback fills a packed `[height][width][3]` float RGB frame in
`[0,1]`, already resized to the target dimensions. The output callback receives
the same layout. Only callback-duration buffer access is valid. The host owns
media decoding, resizing, timestamps, audio and output encoding. There is no
FFmpeg dependency in this API.

Initialize `slopfab_seedvr2_options.struct_size` to `sizeof` the struct and
specify both checkpoint paths, geometry, segment length, tile size, CUDA device,
color matching and seed. Optional progress and cancellation callbacks run on
the calling thread. Cancellation is checked between VAE tiles and transformer
blocks. `frames_written` counts successfully delivered frames even on failure.
No existing generation request or output struct changes layout.

The shared `Upscaler` factory accepts `UpscaleMethod::kSeedVr2`, as does
`slopfab_request_set_upscaler(..., SLOPFAB_UPSCALE_SEEDVR2, ...)` with default
4x sizing and VAE path. `slopfab_request_set_seedvr2_options` copies a
`slopfab_seedvr2_options` to configure generation with custom paths, dimensions
and segment settings. Existing Real-ESRGAN identifiers and C struct layouts
remain unchanged; the ABI version is 1.20.

The internal C++ surface is `slopfab::seedvr2::Restorer` and `seedvr2::stream`.
Generated H3 samples use the shared upscaler after VAE GPU resources have been
released. Continuation latents remain H3 latents.

## Development validation

### Performance measurements, 2026-10-07

Windows Release, RTX 5090 32 GB, driver 610.88, dynamically selected CUDA 13
cuBLAS, published FP16 checkpoints. The baseline was commit `4db8c3b` with the
same benchmark harness added. Synthetic RGB input, seed 666, color matching
enabled, 1280x720 output, five-frame segments; settings were identical before
and after. Each process loads a fresh restorer. Warm timings exclude its first
restoration (two baseline samples, four optimized samples). Disk output is
outside the timer. These are measured workload results, not predictions for
other GPUs or resolutions.

| Workload | Baseline | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| Warm segment, VAE tile 256 | 14.02 s | 5.96 s | 2.35x |
| Warm segment, untiled VAE | 6.78 s | 3.24 s | 2.10x |
| 13-frame 12 FPS video, tile 256, including media I/O | 44.0 s | 21.5 s | 2.05x |

Weight residency trades memory for speed. Sampled CUDA-used memory for tile 256
rose from 2365 to 9363 MiB, and untiled from 7637 to 14725 MiB, including the
same 1613 MiB initial CUDA-used baseline. Sampling uses `cudaMemGetInfo` every
20 ms; these are sampled device-wide figures, not allocation-exact peaks or
the historical `nvidia-smi` measurements above. Disabling weight residency
measured 6.70 s per warm tiled segment at 2575 MiB sampled peak, still about
2.09x faster than the original low-memory implementation.

Independent measurements justify the individual changes:

| Change | Measurement |
| --- | --- |
| Reused device buffers | Disabling pooling regressed warm tiled segments from 5.96 to 9.82 s |
| GPU tile preparation and feathering | Disabling it regressed warm tiled segments from 5.96 to 6.18 s |
| Packed FP8 weight loading | 20.97 million finite values: 41.91 to 2.21 ms median, bit-identical; logical loader device peak 160 to 60 MiB, H2D 80 to 20 MiB, avoids 80 MiB CPU float buffer |
| Parallel spatial GroupNorm | 3.74x at 5x256x256x128; 15.32x at 1x360x640x128; retain original kernel for small groups |
| Pointwise convolution without im2col | 2.14-9.79x; bit-identical output |
| Specialized bounded 3x3x3 convolution | 1.10-1.22x for hidden channels, bit-identical; 512-channel scratch 54 to 27 MiB |
| Larger RGB-input convolution chunks | 3.23-3.66x; operator RMSE at most 9.3e-8 |
| Width-512 attention query tile 2048 | 1.09-1.11x at 4096/14400 spatial tokens, bit-identical; extra scratch 11/31 MiB |
| Fused biased linear projection | About 13% faster for representative 7200-row projections; removes 70-211 MiB FP32 intermediates; 12 shapes bit-identical on CUDA 12 and 13 |
| MLP chunks of 4096 rows | Activation pool peak 295.34 to 198.34 MiB for 7200x2560 input; 3.92 to 3.83 ms median over 30 runs; all 18,432,000 output floats bit-identical |
| Parallel CPU color matching | 57.59 to 10.82 ms median at 720p x 5; bit-identical |
| Bounded media worker queues | Synthetic three-stage throughput 2.32x Windows / 2.57x Linux; both platforms pass blocked-read/write and failure cleanup tests |

The allocation/weight-cache/GPU-tile changes alone were bit-identical to the
baseline. Parallel GroupNorm changes floating-point reduction order, so the
complete optimized model is not bit-identical: final RGB relative L2 error was
0.432% tiled and 0.492% untiled, RMSE 0.00249/0.00283, PSNR 52.08/50.97 dB;
maximum per-component differences were 0.0895/0.1017 in [0,1]. Repeated optimized
runs are deterministic. The encoded 13-frame test retained the exact frame
count/rate/duration and copied audio SHA-256; baseline-versus-optimized encoded
video PSNR was 46.50 dB. These comparisons use synthetic content and do not
establish a universal visual-quality bound.

Rejected experiments are also recorded: two-block asynchronous prefetch had no
warm throughput benefit (6.69-6.93 s versus 6.69-6.71 s) and cost another 624 MiB,
so its implementation was removed. Larger hidden-channel convolution chunks
increased numerical differences and were rejected. Smaller VAE attention query
tiles regressed. The existing fused attention geometry cannot support width 512
within the GPU's register/shared-memory limits; the measured blocked-kernel
tuning is used instead.

Build `slopfab_seedvr2bench` to reproduce the end-to-end measurement:

```sh
slopfab_seedvr2bench DIT.safetensors VAE.safetensors 1280 720 5 256 5 output-prefix
python tools/seedvr2_compare_outputs.py baseline-2.f32 output-prefix-4.f32
```

The final argument `-` skips raw float output. The probe prints per-run wall
time and sampled CUDA-used memory; stage times include preparation between
progress callbacks. Compare warm medians separately from first-use latency.
The comparison script requires NumPy. Operator probes are
`slopfab_seedvr2_perf`, `slopfab_seedvr2_linearbiasbench`,
`slopfab_seedvr2_colorbench`, `slopfab_seedvr2_mlpbench`, and
`slopfab_seedvr2_weightbench`. The FP8 loader measurement is isolated; the
end-to-end measurements above use FP16 checkpoints.

Diagnostic ablations are available through `SLOPFAB_SEEDVR2_POOL=0`,
`SLOPFAB_SEEDVR2_GPU_TILES=0`, `SLOPFAB_SEEDVR2_VAE_KERNELS=0`,
`SLOPFAB_SEEDVR2_FUSED_LINEAR=0`, and `SLOPFAB_SEEDVR2_MLP_ROWS=0` (unchunked).
They keep model geometry/RNG settings fixed. Smaller pool/scratch consumption
does not imply lower total VRAM when weight residency is enabled.

### Correctness checks

Host tests cover regular/shifted window partitions, overlap blending, settings,
partial final segments, and exact output counts for clips through 1001 frames.
`tools/seedvr2_goldens.py` creates independent PyTorch operator fixtures; set
`SLOPFAB_SEEDVR2_FIXTURE` to the resulting archive and run the `seedvr2_cuda` test.
PyTorch is a development dependency only. After rebasing onto Real-ESRGAN
support, the shared CLI was also checked with SeedVR2 still-image restoration,
15 frames at 30000/1001 FPS with a nonzero audio offset (preserved, including the
audio payload hash), and Real-ESRGAN video streaming. The existing Real-ESRGAN
real-checkpoint PyTorch comparison still passes. Linux CPU builds and host tests
pass; CUDA execution was measured on Windows.

Additional CUDA tests cover packed FP8/F16/BF16/F32 conversion, buffer/view
lifetimes, cache isolation/budgets, exact GPU tile preparation/feathering, and
cancellation followed by reuse of the same restorer. Set
`SLOPFAB_SEEDVR2_DIT` and `SLOPFAB_SEEDVR2_VAE` to the real checkpoints to run
the cancellation recovery test. The October 7 Windows run passed 823 checks in
the SeedVR2 executable, along with the host, media, upscaling CLI and C API suites.
The CUDA 12/13 C API dispatch tests also pass. Linux host/media suites and the
full SeedVR2 CUDA suite, including real-checkpoint cancellation recovery, pass
under WSL with CUDA 13.3.
The shared CLI also passed a three-frame Real-ESRGAN checkpoint smoke test
(16x16 input to 64x64 output) after the media-worker changes.

For numerical debugging, `SLOPFAB_SEEDVR2_CAPTURE_DIR` saves the first VAE tile's
input/moments/decoded tensors, DiT patches, timestep embedding, every block's
outputs (including text), and prediction as float safetensors. Later segments
overwrite these files; use a single segment when comparing references.

`tools/seedvr2_compare_upstream.py` checks these captures against the pinned
upstream architecture, using FP32 SDPA and the Apex RMSNorm rounding formula.
It checks every block in isolation as well as full inference with identical
captured conditioning/noise. A five-frame 128x128 fixture measured maximum
isolated-block relative L2 error 0.52%, VAE encode/decode error 0.26%/0.75%,
and full DiT prediction error 4.80% (correlation 0.9988). BF16 rounding differences
accumulate through the 32 blocks; this is not bit-exact PyTorch reproduction.

The downloaded baseline files were verified with SHA-256:

```text
98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660  seedvr2_3b_fp16.safetensors
20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1  ema_vae_fp16.safetensors
```

## Provenance

The architecture follows [ByteDance-Seed/SeedVR at e4de8c2](https://github.com/ByteDance-Seed/SeedVR/tree/e4de8c24441a67e1b7df56abea10645059bb1185),
linked by the [SeedVR2 project](https://github.com/IceClear/SeedVR2).
The output AdaSingle cache behavior was cross-checked against the native
[ComfyUI implementation](https://github.com/Comfy-Org/ComfyUI/blob/f1072eb0350638a3390ddb6afbcaa8c6b237c6fd/comfy/ldm/seedvr/model.py).
Apache-2.0 notices are included under `third_party/seedvr2/`. Model weights are
not distributed with slopfab.
