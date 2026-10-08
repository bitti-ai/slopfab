# H3 latent upscaling

The [Minimax H3 3D conv v1 upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
runs natively on CUDA and Vulkan. It enlarges normalized 24-channel video latents
after denoising and before loading the video VAE. Frame count and audio are preserved.
Python is not needed at runtime.

Place `minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors` in `weights/`, then run:

```sh
slopfab generate --prompt "A cat in warm lamplight" --resolution 512x512 --frames 22 --latent-upscale --out video.mp4
```

This denoises at 512×512 and decodes at 1024×1024. The default scale is 2;
`--latent-upscale-scale 1.5` enables the stage with another scale in [1,4].
Output dimensions round to the nearest multiple of 32 pixels, with ties to even,
matching the reference node's default alignment. An unchanged size is a no-op.

Use `--latent-upscale-model FILE` to enable the stage with another local checkpoint
path. F16, BF16 and F32 safetensors of the exact 3D conv v1 architecture are accepted.
Weights are validated before generation; they are never downloaded automatically.
Inference uses FP32 weights, activations and accumulation, including when loading
the FP16 checkpoint. This costs more memory than the reference node's FP16 mode.

The default follows the [reference implementation](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/nodes/minimax_h3_latent_upscaler_3d.py):
clips longer than 32 latent frames use temporal segments with replicated context
and overlap blending. Group normalization depends on each segment, so this is an
approximation to processing the complete clip. `--latent-upscale-no-chunking`
processes the full clip with greater memory use. Spatial tiling is not applied.
Allocation and device-limit failures are reported; no backend fallback occurs.

Saved latents (`--save-latents`, `--dump-latents` and the C++ `on_latents` hook)
retain the original diffusion resolution. Continuation joins the original latents
before upscaling the cumulative clip for decoding. This stage performs learned
upscaling only; it does not add a second diffusion/refinement pass. The model author
recommends refinement when additional high-resolution detail is needed.

Pixel upscalers remain available independently: `--upscale-method realesrgan`
or `seedvr2` processes the decoded, already enlarged frames. Image editing rejects
latent upscaling because it cannot preserve original pixels outside the edit box.
The standalone `upscale` command accepts decoded media, not H3 latents.

C++ callers use `RunOptions::latent_upscale_model_path` and `latent_upscale`, or
`upscale_latents` from `slopfab/latent_upscale.h` for normalized `[24,T,H,W]` buffers.
The function applies and reverses the companion node's additional channel
normalization internally. The checkpoint expects this even though H3 sampler
latents are already normalized; omitting it produces colored tile artifacts.
Pass sampler latents, not raw VAE latents. The initial implementation omitted
this transform; this has been corrected.
The C API (1.24) exposes:

```c
slopfab_request_set_latent_upscaler(request,
    "weights/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors", 2.0f, 1);
```

An empty path disables the stage. Progress/cancellation uses
`SLOPFAB_STAGE_UPSCALING` before `SLOPFAB_STAGE_VIDEO_DECODE`.

For numerical regression tests, generate golden data with PyTorch and safetensors:

```sh
python tools/latent_upscale_reference.py /path/to/minimax_h3_latent_upscaler_3d.py weights/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors build/latent-upscale-golden.safetensors --chunked
```

Set `SLOPFAB_LATENT_UPSCALE_MODEL` and `SLOPFAB_LATENT_UPSCALE_GOLDEN` to absolute
paths, then run `ctest --test-dir build -C Release -R latent_upscale --output-on-failure`.
The real-checkpoint tests compare identity, still, fractional-scale video, chunked
and whole-clip video against the upstream FP32 node's `execute` method, including
its normalization, on each compiled backend. Regenerate older fixtures: bare-network
fixtures are rejected. To also test saved generation latents at production sizes,
append `--packed-latents FILE --latent-resolution 32x32` (for a 512x512 generation)
to the fixture command. This checks layout and CUDA convolution chunk boundaries.
Model weights are
licensed separately (Apache-2.0) and are not distributed with slopfab.
