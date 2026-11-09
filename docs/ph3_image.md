# P-H3 Image: initial support

P-H3 Image currently supports checkpoint inspection and text-to-image dry-run
planning. **Rendering is not implemented yet.** The CLI and C API reject execution
before loading GPU weights. The model is listed as `planning_only` with no
available inference backends.

Use the `ph3_image` family, or supply a transformer checkpoint and let Slopfab
identify it from its tensors. Files do not need to be renamed or converted.

```powershell
build/Release/slopfab.exe image generate --model ph3_image --transformer "$env:USERPROFILE/Downloads/p-h3-image_int8_convrot.safetensors" --prompt 'A neon sign above a ramen shop in the rain' --dry-run
```

Planning also works before the download finishes, by omitting `--transformer`.
Do not pass a partial `.crdownload` file. When configuring `weight_paths`, both
flat folders and the upstream `diffusion_models/`, `text_encoders/`, and `vae/`
layout are supported. Multiple transformer precisions require an explicit path.
The optional prompt upsampler is never selected as the image text encoder.

The initial plan uses a 2912 x 2912 canvas, or an explicit `--resolution WxH`
with dimensions divisible by 32 (limited to 16 megapixels). It follows the
published eight-step Euler sigma grid:

```text
1, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0
```

Video and audio shifts are both 1, and guidance is 1 without CFG or a negative
prompt. Unlike the existing H3 still-image path, P-H3 samples two internal audio
latent positions per channel along with its single image latent frame. Audio
is not an output. Step counts or schedules that differ from this recipe are
rejected. References, image edits, LoRAs and approximate caches are not yet planned.

The host AdaLN lookup loads both the existing `[1025, 8]` table and P-H3's
`[4097, 16]` table. `lookup_code()` returns all channels; the fixed-width legacy
`lookup()` rejects a rank-16 table. Existing GPU runners remain restricted to
their supported layouts until the wider modulation buffers are implemented.

Remaining work for native generation:

- Carry rank 16 and 4097 rows through CUDA and Vulkan modulation, captures,
  and allocation paths; retain rank-8 adapter and cache compatibility checks.
- Implement the Qwen3-VL-4B text connector: hidden states 13, 19, 25, 31 and 36,
  final-state normalization, concatenation to 12800 channels, standardization,
  projection to 5120, output rescaling and first-token sink replacement.
- Load the image VAE's `decoder.*` layout and `latents_mean` / `latents_std`.
  The existing H3 video decoder's temporal still-image workaround does not
  establish compatibility with this VAE.
- Wire joint image/audio denoising, validate output against upstream, then add
  prompt upsampling and image editing.

Run host contract tests with a completed local transformer:

```powershell
$env:SLOPFAB_PH3_IMAGE_CHECKPOINT = "$env:USERPROFILE/Downloads/p-h3-image_int8_convrot.safetensors"
$env:SLOPFAB_TEST_FILTER = 'ph3_image'
build/Release/slopfab_tests.exe
```

Sources: [model card and licensing](https://huggingface.co/PrunaAI/P-H3-Image-Comfy),
[published workflow](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/blob/main/workflows/p-h3-image_text_to_image.json),
[conditioning node](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py),
and [text connector](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/minimax.py).
