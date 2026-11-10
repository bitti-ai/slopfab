# MiniMax-H3-Image

Native text-to-image generation and reference editing use the `h3_image` model family. Slopfab also
recognizes the downloaded transformer from its tensor shapes, without renaming
or conversion. This is separate from Pruna's `ph3_image` model.

Use the pruned INT8 ConvRot transformer, the shared H3 Qwen3-VL-32B text
encoder, and the MiniMax-H3-Image FP16 VAE. Existing H3 text encoder weights
can be reused. No audio VAE is needed.

```powershell
build/Release/slopfab.exe image generate --model h3_image `
  --transformer weights/minimax_h3_image_pruned_int8_convrot.safetensors `
  --text-encoder weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors `
  --vae weights/minimax_h3_image_vae_fp16.safetensors `
  --inference-backend cuda --resolution 1024x1024 --seed 42 `
  --prompt 'A studio photograph of a red ceramic teapot on a pale wooden table, soft window light.' `
  --out teapot.png
```

For Vulkan, select `--inference-backend vulkan --vulkan-arithmetic portable`.
The CLI selects Flash2 attention. Output is PNG or PPM; the C API delivers one
RGB frame. Omit `--resolution` for the upstream 4096×4096 default. Dimensions
must be divisible by 32, with a 20 Mi-pixel ceiling (20,971,520 pixels).
MiniMax describes an effective range of roughly 4–20 MP, with best results
around 16 MP. For example, 5120x4096 reaches Slopfab's ceiling. Small canvases
are useful for execution checks.

The download mod lists the matching checkpoints and URLs:

```sh
slopfab download h3_image
slopfab download h3_image transformer-int8
slopfab download h3_image text-nvfp4
slopfab download h3_image vae-fp16
```

`text-int8` is an alternative to `text-nvfp4`. Configure `weight_paths` to
discover existing files. Both flat folders and `diffusion_models/`,
`text_encoders/`, and `vae/` subdirectories work. If several compatible text
encoders or transformer precisions are present, pass explicit paths.

For editing, supply the source with `--edit-image`. It becomes `<Picture 1>`;
additional `--reference-image` arguments become `<Picture 2>`, `<Picture 3>`,
and so on, in command-line order. Up to 16 images are supported, including the
source. With only `--reference-image` arguments, the first is Picture 1.

```powershell
build/Release/slopfab.exe image generate --model h3_image `
  --transformer weights/minimax_h3_image_pruned_int8_convrot.safetensors `
  --text-encoder weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors `
  --vae weights/minimax_h3_image_vae_fp16.safetensors `
  --edit-image teapot.png --reference-image glaze-reference.png `
  --prompt 'Edit <Picture 1>. Give the teapot the blue glaze from <Picture 2>. Keep its shape and the background.' `
  --out edited.png
```

Use prompt instructions to describe the edit. This path generates a complete
image; it does not apply an inpainting mask or preserve unedited pixels exactly.
Omit `--edit-box`, `--edit-strength` and `--edit-feather`. The C API uses ordered
image-reference attachments for the same native editing operation.

Without an explicit resolution, editing inherits the first picture's scaled
VAE canvas. Its short edge is clamped to 1024–2048 pixels and dimensions are
rounded to multiples of 32. Qwen independently receives a 1024-short-edge copy
of each original. Reference aspect ratios must be between 1:4 and 4:1; large
collections must also fit the shared 32,768-token conditioning limit.
Reference order is preserved in both picture labels and latent blocks.

The sampler uses six Euler updates, guidance 1, and the published PDD grid:

```text
1, 0.9675236940, 0.8875485063, 0.75, 0.5327739120, 0.2320088744, 0
```

Other step counts, CFG, negative prompts, caches and H3 adapters are rejected.
The model runs without audio tokens. Each output projection combines the base
head with the residual for its trained interval. The 24-layer, width-1024 VAE
decodes one latent frame using 2D RoPE over 32 of each head's 64 channels.

## Masked inpainting

Use a rectangle or a grayscale mask to restrict a prompt edit:

```powershell
build/Release/slopfab.exe image generate --model h3_image `
  --transformer weights/minimax_h3_image_pruned_int8_convrot.safetensors `
  --text-encoder weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors `
  --vae weights/minimax_h3_image_vae_fp16.safetensors `
  --edit-image teapot.png --edit-mask mask.png `
  --prompt 'Edit <Picture 1>. Change the teapot to cobalt blue, keeping its shape and lighting.' `
  --out inpainted.png
```

The mask must match the source dimensions. Black preserves source pixels
exactly, white selects pixels to regenerate, and gray values blend source and
generated pixels. Masks may contain holes and disconnected regions. Use a
grayscale RGB image without relying on alpha; color masks and all-black masks
are rejected. The mask is not resized. Supply soft edges in the mask itself.

For a rectangle, replace `--edit-mask mask.png` with
`--edit-box x,y,width,height`; optional `--edit-feather N` blends inward without
changing pixels outside the rectangle. Do not combine a mask with a rectangle,
nonzero feathering or outpainting.

The delivered image keeps the original dimensions. Its internal canvas is
padded to multiples of 32; omit `--resolution` or match that padded size.
The source is `<Picture 1>`, followed by up to 15 additional references. Its
visual and VAE reference copies follow the normal H3 Image reference sizing.
Unselected latent cells are restored at each step and final pixel compositing
preserves every unselected source pixel. Sampling always uses all six PDD steps
at strength 1; reduced `--edit-strength` is rejected.

C API callers use `slopfab_request_set_image_edit_path` or
`slopfab_request_set_image_edit_rgb24` at strength 1, then optionally attach
`slopfab_request_set_image_edit_mask_u8`. Source and mask buffers are copied;
the existing asynchronous snapshot and ownership contracts apply. This uses
constrained sampling with the image model, rather than a separate inpainting
checkpoint.

## Outpainting

Expand an existing image without scaling its contents:

```powershell
build/Release/slopfab.exe image generate --model h3_image `
  --transformer weights/minimax_h3_image_pruned_int8_convrot.safetensors `
  --text-encoder weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors `
  --vae weights/minimax_h3_image_vae_fp16.safetensors `
  --edit-image teapot.png --outpaint 256,128,256,128 `
  --prompt 'Extend the scene in <Picture 1>. Continue the wooden table and background, matching perspective and window light.' `
  --out expanded.png
```

Margins are **left, top, right, bottom**, in pixels. They must be nonnegative
and at least one must be positive. The expanded canvas is internally padded to
32 pixels and cropped back on delivery. Omit `--resolution`, or supply that
padded size. The existing 20 Mi-pixel canvas limit applies, with at most 8192
pixels per axis. The source must contain a complete 16-pixel latent cell after
placement. No `--edit-box`, `--edit-strength`, `--edit-feather` or `--aspect` is
needed. Additional references remain available, starting at `<Picture 2>`;
the source counts toward the 16-picture limit.

The source supplies visual prompt context, locked target latents and a spatially
positioned latent anchor. Final compositing preserves the source interior.
`--outpaint-blend 9` (the default) blends up to eight pixels inward from edges
facing generated space. Use `--outpaint-blend 1` to preserve every source pixel
exactly. H3 Image retains its six Euler evaluations at strength 1 and disables
Langevin refinement; nonzero `--outpaint-langevin-steps` is rejected.

C API callers use the existing image-edit setters with the expanded canvas and
source rectangle, enable `slopfab_request_set_image_edit_invert_mask(request, 1)`,
then set `slopfab_request_set_outpaint_langevin_steps(request, 0)` before resolving
or starting generation. The general H3 default of five refinement evaluations
does not apply to this model. C++ callers can create the canvas with
`make_outpaint_edit(source, left, top, right, bottom)` and set `langevin_steps = 0`.

This is experimental constrained sampling using the existing image model;
seam quality depends on the source and prompt. It does not add a separately
trained outpainting checkpoint. The teapot regression sample preserves the
source correctly but has visible joins and repeated window details at its
expanded boundary. A passing preservation test does not guarantee seamless
scene continuation.

The optional prompt enhancer, video/audio
references, original sharded full-AdaLN weights, previews and H3 latent archives
are not supported.

Host tests cover checkpoint identity, incompatible requests and metadata,
schedule selection, PDD head fusion, 2D RoPE and single-frame decoding. Set
`SLOPFAB_H3_IMAGE_CHECKPOINT` to inspect a real transformer with the host tests.
GPU regression tests compare CUDA and Vulkan fused RoPE at both widths.
Reference tests cover all 16 ordered slots, pinned reference timesteps,
FP32 posterior sampling, reference sizing, cache separation and CLI editing.

Validated on an RTX 5090 with the INT8 transformer, NVFP4 H3 text encoder and
FP16 image VAE: CUDA at 512×512 and 4096×4096, and portable Vulkan at 512×512.
The 4096×4096 CUDA run took about 64 seconds including conditioning, loading,
six denoising steps and PNG delivery. Both backends produced complete images;
their full generation arithmetic is not expected to be pixel-identical.

The 16-case `tools/h3_image_render_smoke.py` suite also passed on both backends:
text generation, single-reference and two-reference editing, rectangular and
pixel-mask inpainting, and outpainting through the CLI and C API. The CLI edit
tests use 512x512 sources; outpainting expands these to 768x768. Tests verify
exact preservation outside selections and inside the outpaint source interior.
C API mask tests additionally use an unaligned 67x65 source, holes, soft coverage
and padded input strides, then release the caller buffers and request during
generation. All runs retain six model evaluations. Render success and source
preservation do not remove the outpainting seam limitation described above.

Sources: [original model and examples](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image),
[PDD schedule](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/pdd_config.json),
[VAE configuration](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/vae/config.json),
[Comfy checkpoints](https://huggingface.co/Comfy-Org/MiniMax-H3-Image), and
[Comfy's head implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py).
