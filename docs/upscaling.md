# Upscaling

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
