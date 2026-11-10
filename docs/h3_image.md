# MiniMax-H3-Image

Native text-to-image generation uses the `h3_image` model family. Slopfab also
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
must be divisible by 32, with a 17 Mi-pixel ceiling that accommodates the
published aspect-ratio presets. Small canvases are useful for execution checks.

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

The sampler uses six Euler updates, guidance 1, and the published PDD grid:

```text
1, 0.9675236940, 0.8875485063, 0.75, 0.5327739120, 0.2320088744, 0
```

Other step counts, CFG, negative prompts, caches and H3 adapters are rejected.
The model runs without audio tokens. Each output projection combines the base
head with the residual for its trained interval. The 24-layer, width-1024 VAE
decodes one latent frame using 2D RoPE over 32 of each head's 64 channels.

This implementation supports text-to-image. Reference-image editing, the
optional prompt enhancer, original sharded full-AdaLN weights, previews and
H3 latent archives are not supported.

Host tests cover checkpoint identity, incompatible requests and metadata,
schedule selection, PDD head fusion, 2D RoPE and single-frame decoding. Set
`SLOPFAB_H3_IMAGE_CHECKPOINT` to inspect a real transformer with the host tests.

Sources: [original model and examples](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image),
[PDD schedule](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/pdd_config.json),
[VAE configuration](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/vae/config.json),
[Comfy checkpoints](https://huggingface.co/Comfy-Org/MiniMax-H3-Image), and
[Comfy's head implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py).
