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

There is no total-frame or duration limit. Input, output and inference are
bounded by one temporal segment plus its overlapping boundary frame, so a
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

Transformer weights are memory mapped and uploaded one block at a time. VAE
and transformer weights have separate lifetimes. Convolution scratch is limited
to 2048 output positions, and attention processes windows without allocating a
full-video attention matrix. The OS may cache the checkpoint in host memory.

Measured on this RTX 5090 (32 GB), using the published FP16 weights and
five-frame segments:

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
