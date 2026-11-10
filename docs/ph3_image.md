# P-H3 Image

P-H3 Image supports native text-to-image generation on CUDA and portable Vulkan.
Use the `ph3_image` family, or supply a transformer checkpoint and let Slopfab
identify it from its tensors. Files do not need renaming or conversion.

The tested bundle is the upstream INT8 ConvRot transformer, INT8 ConvRot
Qwen3-VL-4B text encoder and INT8 image VAE. Keep all three files together,
or specify their paths:

```powershell
$weights = "$env:USERPROFILE/Downloads"
build/Release/slopfab.exe image generate --model ph3_image `
  --transformer "$weights/p-h3-image_int8_convrot.safetensors" `
  --text-encoder "$weights/p-h3-image_text_encoder_int8_convrot.safetensors" `
  --vae "$weights/p-h3-image_vae_int8_convrot.safetensors" `
  --inference-backend cuda --resolution 512x512 --seed 42 `
  --prompt 'A studio photograph of a red ceramic teapot on a pale wooden table. Soft natural window light, a plain cream background, detailed glazed ceramic texture, gentle shadows.' `
  --out teapot.png
```

For Vulkan, use `--inference-backend vulkan --vulkan-arithmetic portable`.
The CLI selects Flash2 attention. Omit `--resolution` for the published
2912 x 2912 canvas; the small canvas above is a quick execution check.
Dimensions must be divisible by 32 and are limited to 16 megapixels.
The CLI outputs PNG or PPM; the C API can deliver one RGB frame through its sample callback.

`--dry-run` also works before downloads finish, by omitting checkpoint paths.
Do not pass a partial `.crdownload` file. Configured `weight_paths` support
flat folders and the upstream `diffusion_models/`, `text_encoders/`, and `vae/`
layout. Multiple transformer precisions require an explicit path.
The optional prompt upsampler is never selected as the image text encoder.

Sampling follows the published eight-step Euler grid:

```text
1, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0
```

Video and audio shifts are both 1, and guidance is 1 without CFG or a negative
prompt. P-H3 samples two internal audio positions per channel alongside its
single image latent frame. Those audio rows are discarded before image delivery.
Different step counts or schedules are rejected.

The implementation carries the `[4097,16]` timestep table through both GPU
runners. Text conditioning captures hidden states 13, 19, 25, 31 and 36,
normalizes the final state, concatenates to 12800 channels, applies the FP32
connector and replaces the first token with the learned sink. The image VAE
decodes one temporal patch at the fixed RoPE coordinate `-6/7`, using the
checkpoint's latent statistics and the existing spatial tiling.

Current limits: text-to-image only, Flash2 attention, no LoRAs, references,
image editing, approximate caches, H3 latent archives, previews or upscaling.
Prompt upsampling is not implemented. W4A8 text encoders are unsupported;
other bundle precisions have not been validated end to end. CUDA and Vulkan
use different numerical paths, so their images are not expected to match
pixel for pixel. Matching an upstream ComfyUI render has not been established.

Validation commands:

```powershell
$env:SLOPFAB_PH3_IMAGE_CHECKPOINT = "$weights/p-h3-image_int8_convrot.safetensors"
$env:SLOPFAB_PH3_TEXT_ENCODER = "$weights/p-h3-image_text_encoder_int8_convrot.safetensors"
$env:SLOPFAB_TEST_FILTER = 'ph3_image'
build/Release/slopfab_tests.exe
$env:SLOPFAB_PH3_GPU = '1'
build/Release/slopfab_tensor_backend_tests.exe
```

Host tests cover planning, connector orientation/standardization/sink and VAE
temporal scheduling. Synthetic GPU tests embed all rank-8 modulation in
channels 9?16 and require the same trajectory on each backend. The optional
real-transformer test checks eight-step execution and finite outputs.

Sources: [model card and licensing](https://huggingface.co/PrunaAI/P-H3-Image-Comfy),
[published workflow](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/blob/main/workflows/p-h3-image_text_to_image.json),
[conditioning node](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py),
and [text connector](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/minimax.py).
