# MOSS Image

slopfab supports native text-to-image generation with
[MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) on CUDA.
The runtime uses C++/CUDA throughout: Qwen3.5-9B conditioning, the 42-layer MOSS
transformer, and the FP32 FLUX.2 small VAE decoder. No Python runtime is needed.

## Local assets

The supplied single-file Comfy checkpoints can be used directly:

```text
weights/
  transformer/moss_image_sft_int8_convrot.safetensors
  text_encoder/qwen3.5_9b_int8_convrot.safetensors
  text_encoder/qwen3.5_tokenizer.json
  vae/flux2-vae_small.safetensors
```

Get the tokenizer from the model's
[tokenizer.json](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/resolve/main/tokenizer/tokenizer.json)
and save it under the name above, or pass `--tokenizer FILE`. H3's embedded
tokenizer has a different vocabulary and is rejected for MOSS. Inference never
downloads assets. Hugging Face sharded model directories are not currently a
supported checkpoint input; use the single-file checkpoints above.

`--model moss` discovers these components. Explicit `--transformer`,
`--text-encoder`, `--vae`, and `--tokenizer` paths override discovery. An explicit
MOSS transformer also selects MOSS by its tensor structure, even if renamed.
An explicit family that disagrees with the checkpoint is rejected.

## Generation

```sh
slopfab generate --model moss --prompt "A red apple on a wooden table, studio photograph" --resolution 512x512 --steps 50 --seed 42 --out apple.ppm
slopfab generate --model moss --prompt-file layout.json --negative-prompt "watermark" --guidance 3 --uncond-skip-layers 22 --out layout.ppm
slopfab generate --model moss --resolution 1024x1024 --dry-run
```

Defaults follow the [upstream pipeline](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/pipeline_moss_image.py):

| Setting | MOSS default |
| --- | --- |
| Canvas | 1024×1024; positive multiples of 16 |
| Sampler | Euler, 50 requested updates |
| Flow shift | 5, snapped to the 1000-point training grid |
| CFG | 3, empty negative prompt |
| Negative branch skipping | Layer 22 during the default SLG interval [0.01,1) |
| Conditioner | User chat turn with thinking prefix; final normalized Qwen3.5 hidden state |
| Prompt limit | 4096 tokens, truncated on the right, including chat tokens |
| Output | PPM file, or planar RGB pixels through the library |

`--uncond-skip-layers none` disables skipping. `--guidance 1` uses only the
positive branch. MOSS `--steps` counts requested Euler updates, unlike H3's count
of grid points including terminal zero. Snapping can remove duplicate steps.
Use sampling-settings `video_sigma_shift` to override the image shift. Native
seeded noise is reproducible but does not reproduce PyTorch's RNG.

The current MOSS implementation supports CUDA with `flash2` attention, which the
CLI selects by default for MOSS. It does not implement Vulkan, references/image
editing, LoRA, H3 caching/continuation/latent archives, or upscaling. These options
fail explicitly. The tokenizer must match Qwen3.5. Model reuse currently releases
MOSS stage weights after each stage. JSON layout prompts are passed verbatim as
text; slopfab does not interpret layout objects.

`--dump-latents FILE` saves FP32 `image_latents` with shape `[H/16,W/16,128]`, in
pixel-major packed-channel order. The same tensor can be supplied with
`--init-latents FILE`. `--synthetic-latents --init-latents FILE` tests just the
decoder. MOSS archives are deliberately distinct from H3's `video_rows` archives.
`--prompt-embedding FILE` accepts `[tokens,4096]` `prompt_embedding` and, when
CFG is enabled, `negative_prompt_embedding`. Each contains only valid tokens.

## C API

ABI 1.30 adds `slopfab_request_set_model_family` and
`slopfab_request_set_image_guidance`. Existing callers can also select MOSS by
supplying a MOSS transformer path. Set the transformer, Qwen3.5 text encoder,
Qwen3.5 tokenizer, and FLUX.2 VAE with the existing model-path setter; the VAE
uses `SLOPFAB_MODEL_VIDEO_VAE`. There is no audio VAE. Model paths are explicit
in the C API; it does not perform CLI file discovery.

Select CUDA and `flash2`, set resolution/prompt/seed, then use the ordinary
planning and asynchronous generation APIs. MOSS returns one RGB frame and no
audio, with the existing progress/cancellation and sample ownership contracts.
The model family determines still-image behavior without requiring the H3
still-image flag. For a one-step smoke run, select the family before calling
`slopfab_request_set_steps(request, 1)`.

## Validation and extension points

`ModelProfile` centralizes family identity, compatible asset discovery and
family policy. The shared planner dispatches to a family planner before applying
H3 geometry or sampling rules. The shared runner dispatches before H3-specific
conditioner, denoising or codec logic. H3's existing kernels remain H3-specific;
merely changing their dimensions is not treated as another model implementation.

The MOSS CUDA runtime composes checkpoint-backed projection, normalization,
activation and attention operations. It reuses the existing ConvRot readers,
quantized linear operations, attention primitives, noise generator and output
callbacks. Its bias-before-rounding behavior is separate from H3 arithmetic.
The Qwen3.5 conditioner and FLUX.2 decoder are separate from the MOSS transformer.

Build `slopfab_mossprobe` with tests enabled. The development-only
`tools/moss_reference.py` compares components against PyTorch equations and the
locally supplied upstream MOSS model source:

```sh
python tools/moss_reference.py text --probe build/Release/slopfab_mossprobe.exe
python tools/moss_reference.py transformer --probe build/Release/slopfab_mossprobe.exe --source /path/to/pipeline_moss_image.py
python tools/moss_reference.py vae --probe build/Release/slopfab_mossprobe.exe
```

These checks require PyTorch, safetensors, tokenizers and diffusers, and do not
download files. They compare the supplied INT8 ConvRot weights with matching
input rotation/dequantization. The transformer check uses real prompt features;
it tests a rectangular image grid and the full 42-layer graph. BF16 kernels
have different reduction/rounding orders: the measured relative RMS differences
on RTX 5090 were approximately 1.7% for Qwen, 2.4% for the transformer, and
0.013% for the FP32 VAE. The comparison limits are 3%, 3%, and 0.1% respectively.
These are component checks, not a claim of bitwise parity with Diffusers.
Tiny grids with random, untrained text features amplify BF16 differences and
are not used as a full-model equivalence criterion; zero- and one-layer probes
can isolate arithmetic differences with `--layers`.

`python tools/moss_generation_smoke.py --library build/Release/slopfab.dll`
exercises the real C API generation path, planar RGB delivery, and cancellation
with the local weights. It needs only Python's standard library.
