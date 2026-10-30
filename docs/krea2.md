# Krea 2

Native CUDA text-to-image generation with [Krea 2](https://github.com/krea-ai/krea-2),
including the 28-block transformer, twelve-layer Qwen text fusion, Qwen3-VL-4B
conditioning and Qwen-Image VAE decoding. No Python is required at runtime.

## Local weights

The supplied Comfy INT8 ConvRot checkpoints load directly:

```text
weights/transformer/krea2_turbo_int8_convrot.safetensors
weights/text_encoder/qwen3vl_4b_int8_convrot.safetensors
weights/text_encoder/qwen_image_vae.safetensors
```

The VAE may also live in `weights/vae/`; discovery checks there first. The embedded
Qwen tokenizer is compatible. No downloads or checkpoint conversion are required.
INT4 Qwen checkpoints are unsupported. Explicit transformer paths select the family
by tensor structure, even if the file has been renamed.

## CLI

```sh
slopfab generate --model krea2 --prompt "A red apple on a wooden table" --resolution 1024x1024 --seed 42 --out apple.ppm
slopfab generate --model krea2 --prompt "A mountain lake at sunrise" --resolution 1536x1024 --steps 8 --out lake.ppm
slopfab generate --model krea2 --dry-run
```

Defaults match Turbo: eight Euler updates, CFG disabled, a fixed timestep-shift
`mu=1.15` (multiplicative shift `exp(1.15)`), and a 1024×1024 canvas. Dimensions must
be positive multiples of 16, at most 16 megapixels. Turbo is intended for roughly
1024–2048-pixel images; smaller canvases are useful for testing.

`--guidance` enables Krea's upstream convention:
`prediction = conditional + guidance * (conditional - unconditional)`.
`--negative-prompt` supplies the negative branch. The architecture also accepts
floating-point RAW checkpoints; explicitly select their sampling settings,
typically 52 updates and guidance 3.5. RAW's resolution-dependent shift can be set
with `--sampling-settings FILE`, using `video_sigma_shift = exp(mu)`, where
`mu = 0.5 + (image_tokens - 256) * 0.65 / (6400 - 256)` and
`image_tokens = (width / 16) * (height / 16)`. Variant defaults cannot be inferred
from tensor shapes; this family always defaults to Turbo settings. RAW weights
have not been tested locally.

The conditioner retains up to 507 prompt tokens plus five suffix tokens after the
34-token system prefix, matching upstream truncation. Padding is removed without
changing valid-token attention. Seeded native noise is reproducible but differs
from PyTorch's RNG. Transformer activations and Euler updates use BF16; VAE decoding
uses FP32.

## C API

Set family `krea2` (or leave it empty for checkpoint detection), inference backend
`SLOPFAB_INFERENCE_CUDA`, and attention `flash2`. Supply transformer, text encoder
and `VIDEO_VAE` paths using `slopfab_request_set_model_path`. The C API does not
discover files. Set prompt, resolution and seed, then start generation normally.
The result contains one frame of planar RGB floats in `[0,1]` and no audio.

Leave image guidance unset for Turbo. To enable CFG, call
`slopfab_request_set_image_guidance(request, scale, negative_prompt, nullptr, 0)`.
Progress and cancellation use the existing callbacks; cancellation is checked
between conditioner and transformer layers and pipeline stages. Transformer
blocks stream when available VRAM cannot hold the checkpoint plus working space.

Current limits: CUDA only; no image editing/references, video, LoRA, H3 caches,
sampling effects, LCS, upscaling or H3 latent archives. File output is PPM.
`--dump-latents` saves `image_latents` shaped `[H/16,W/16,64]`, with each pixel token
ordered `(channel, patch_y, patch_x)`. Precomputed `prompt_embedding` accepts
`[tokens,12,2560]`, containing only valid tokens and raw Qwen hidden states
2, 5, …, 35. CFG still needs a text encoder for the negative prompt.

## Validation

Device-free tests cover planning, schedule values, CLI discovery/routing, C API
selection and incompatible options. Development-only numerical comparisons use
local weights and the official inference modules:

```sh
python tools/krea_reference.py --mode transformer --model weights/transformer/krea2_turbo_int8_convrot.safetensors
python tools/krea_reference.py --mode fusion --model weights/transformer/krea2_turbo_int8_convrot.safetensors
python tools/krea_reference.py --mode text --model weights/text_encoder/qwen3vl_4b_int8_convrot.safetensors
python tools/krea_reference.py --mode vae --model weights/text_encoder/qwen_image_vae.safetensors
python tools/krea_generation_smoke.py --library build/Release/slopfab.dll
```

Build `slopfab_kreaprobe` for the comparisons. They require CUDA PyTorch,
Transformers, Diffusers, safetensors, einops and an upstream checkout in
`ref/krea-2` (override with `--source`). They never download weights. The DLL smoke
test checks real eight-step generation, output ownership and cancellation.

On the supplied INT8 checkpoints, comparisons measured 1.48% relative RMS for
conditioning, 1.02% for text fusion, 3.38% for the full transformer and 0.000094%
for VAE decoding. A real 1024?1024 eight-step Turbo run completed in about 54
seconds on the local RTX 5090. These are numerical checks, not pixel-identical
reproduction of the upstream sampler.
