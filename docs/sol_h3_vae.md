# Sol-H3-VAE video decoding

Slopfab supports the released [Sol-H3-VAE checkpoint](https://huggingface.co/Efficient-Large-Model/Sol-H3-VAE)
through native CUDA in the CLI and DLL (C API 1.44). Python, PyTorch, Triton and
FlashAttention Python packages are not runtime dependencies. Vulkan is currently
unsupported for this decoder and is rejected explicitly.

The checkpoint is detected from tensor names and validated against the released
24-layer, width-2048 architecture. The original F32 checkpoint works directly;
F16/BF16 copies with the same shapes also load. No conversion or H3 VAE config
file is needed at runtime: Slopfab supplies the released H3 normalization and
architecture constants. The downloaded release tested here has SHA-256
`8658f1e85b5997c4ba716406a2f148bfa36cf97d4d5fcd402c7bd341f54e77a1`.

## Selecting the decoder

For generation, keep the original video VAE available for encoding reference
images, video and image ending anchors, and override only the decoder:

```sh
slopfab generate --prompt "A continuous tracking shot" --resolution 768x1280 --frames 124 --vae weights/vae/minimax_h3_video_vae_fp16.safetensors --video-decoder C:/Users/NN/Downloads/sol_h3_vae.safetensors --out clip.mp4
```

In the DLL, use the existing model-path setter with the new selector:

```c
slopfab_request_set_model_path(request, SLOPFAB_MODEL_VIDEO_VAE, original_vae);
slopfab_request_set_model_path(request, SLOPFAB_MODEL_VIDEO_DECODER, sol_vae);
slopfab_request_set_inference_backend(request, SLOPFAB_INFERENCE_CUDA);
```

Setting `SLOPFAB_MODEL_VIDEO_DECODER` to an empty path clears the override.
Text-only generation can also select Sol directly as `VIDEO_VAE`/`--vae` when
no VAE encoding is required. Sol contains no encoder; using it to encode an
image/reference is rejected with instructions for selecting the override.

Decode an existing normalized latent archive without denoising again:

```sh
slopfab decode --latent clip.safetensors --vae C:/Users/NN/Downloads/sol_h3_vae.safetensors --out clip.y4m
```

The DLL's `slopfab_latent_upscale_decode_file_start` accepts the Sol checkpoint
as its `video_vae` argument. This also works for enlarged latent archives and
repeated upscaling. Audio continues to use the separately selected audio VAE.

## Behavior and limits

The implementation follows upstream revision
`6a8a33fc61e092281f32bf28cde236760e2caaa4`: bidirectional five-latent temporal
windows, 16x16 spatial windows, alternating shifts, and five global tokens per
native block. Global tokens attend across their own complete native block.
RoPE coordinates restart inside each shifted window. Three frames are removed
from the first latent of each block, yielding `4*T - 3*ceil(T/5)` frames.

Sol processes the complete latent clip. It does not use the original decoder's
overlapping spatial tiles, temporal recomputation or cross-fades. Still images
use one temporal latent and produce one frame. Attention matrices are bounded
by window size; feed-forward activations are processed in 512-row chunks.
Weights occupy about 3.03 GiB on the GPU, with additional memory proportional
to clip length and resolution. There is no automatic CPU fallback or temporal
chunking; an oversized decode reports an allocation error.

The transformer uses BF16 matrices/operands and FP32 residuals, matching the
upstream reference's mixed precision. Input embedding and output projection
use FP32. Sol's arithmetic is independent of the DiT attention setting;
`--attention exact` does not imply cross-backend bit parity for Sol.

Normalized latent archives, continuation anchors and audio are unaffected by
decoder selection. Sol is distilled, so its decoded pixels differ from the
original H3 decoder. Like upstream Sol, it is noncausal; appending latents can
change previously decoded pixels near the join even if those latents are fixed.

During Sol decoding, DLL `VIDEO_DECODE` callbacks report completed layers from
0 through 24. Cancellation is checked between layers and before delivery.

## Validation

Four real-checkpoint comparisons against upstream `engine="ref"` covered latent
shapes `(T,H,W)` of `(1,2,3)`, `(5,4,6)`, `(6,17,19)` and `(11,3,5)`.
These exercise stills, complete and partial temporal blocks, spatial edges,
shifted windows and cross-block global tokens. RGB PSNR was 58.31–61.02 dB;
frame counts matched and all pixels were finite. This compares implementations
of Sol, not Sol against the original H3 decoder.

On an RTX 5090, synthetic `(35,48,80)` latents produced 119 frames at 1280x768
in 3.94 seconds (3.90 seconds on the second decode), excluding a 2.28-second
checkpoint load. This is a local timing, not a general speed guarantee.

Reproduce the reference comparison with torch, safetensors and Diffusers in a
test environment and the downloaded upstream `code/sol_h3_vae` sources:

```sh
python tools/sol_h3_vae_reference.py --weights sol_h3_vae.safetensors --config ref/upstream-h3/vae/config.json --reference-code /path/to/code/sol_h3_vae --out build/sol-vae-validation --cli build/Release/slopfab.exe
python tools/sol_h3_vae_smoke.py build/Release/slopfab.dll sol_h3_vae.safetensors
python tools/latent_archive_upscale_smoke.py build/Release/slopfab.dll . cuda --video-vae sol_h3_vae.safetensors
```
