# Fibo 2.0 Edit Turbo

Native CUDA image editing with [Fibo 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo), available in the C API since 1.34. No Python runtime or online captioning service is required. The implementation includes the 44-layer transformer, Perceiver, Qwen3-VL-4B visual/text conditioner and FLUX.2 VAE encoder/decoder.

## Local weights

These supplied single-file Comfy checkpoints are supported directly:

```text
weights/transformer/fibo2_edit_turbo_int8_convrot.safetensors
weights/text_encoder/qwen3vl_4b_int8_convrot.safetensors
weights/vae/flux2-vae.safetensors
```

The embedded Qwen tokenizer is compatible. `--model fibo2` discovers the above files. Explicit transformer paths also select Fibo by tensor structure. The separate Qwen INT4 file is not supported; use the INT8 file. Hugging Face sharded directories and the unfused Diffusers transformer layout are not supported inputs.

## CLI

```sh
slopfab generate --model fibo2 --prompt "Make the car blue" --reference-image source.png --resolution 1024x1024 --steps 4 --seed 42 --out edited.ppm
slopfab generate --model fibo2 --prompt-file instruction.json --reference-image scene.png --reference-image object.png --resolution 1024x1024 --out combined.ppm
```

One to four reference images are accepted. Ordinary reference inputs are resized and center-cropped to the requested canvas. The default canvas is 1024x1024; explicit dimensions must be positive multiples of 16. Editing an in-memory image preserves its original output dimensions, with internal padding to multiples of 16.

Four Euler updates and guidance 1 are used by default, with an image-size-dependent flow shift. Turbo uses a single positive prediction; MOSS guidance and negative-prompt controls do not apply. Seeded native noise is reproducible but is not PyTorch's RNG. A prompt may be a plain instruction or JSON:

```json
{"edit_instruction":"Add a blue sign","structured_caption":{"short_description":"A blue sign on the wall","text_render":[{"text":"Welcome"}]}}
```

Structured captions are optional and supplied by the application. Windows builds render `text_render` glyph images with native Arial; glyph conditioning is currently Windows-only. The complete prompt, including image tokens, is limited to 4096 tokens and overlong prompts are rejected.

## DLL integration

Use `dist/windows/slopfab.dll` with its matching `capi.h`. The C ABI remains compatible with existing callers.

1. Set the transformer, text encoder and video VAE model paths with `slopfab_request_set_model_path`. No audio VAE is needed.
2. Leave the family empty for detection, or call `slopfab_request_set_model_family(request, "fibo2")`.
3. Select `SLOPFAB_INFERENCE_CUDA` and `slopfab_request_set_attention(request, "flash2")`.
4. Set an edit instruction and source using `slopfab_request_set_image_edit_rgb24` or `slopfab_request_set_image_edit_path`. A full-image rectangle without a pixel mask performs an ordinary edit.
5. For lasso editing, attach source-sized coverage bytes using `slopfab_request_set_image_edit_mask_u8`. Zero preserves, 255 replaces, and intermediate values blend at final compositing. Fibo greys pixels with coverage >=128 before VAE and vision conditioning. A full-white explicit mask therefore removes all source content from conditioning.
6. Start generation normally. The output contains one frame of planar float RGB and no audio. Unselected pixels are preserved exactly. Strength below 1 retains a suffix of the schedule and mixes original image latents with noise at its starting sigma.

Masked editing takes exactly one source image. Reference-image lists support multiple images without a mask. Cancellation is checked between text/transformer blocks and pipeline stages. Conditioner weights stream by layer; transformer blocks stream when available VRAM cannot hold the model plus working space. Model caches are released between requests.

Current limitations: CUDA only, no Vulkan, LoRA, H3 schedules/caches/LCS, video, rectangular outpaint, upscaling or H3 latent archives. `--dump-latents` saves packed FP32 `image_latents` shaped `[H/16,W/16,128]`. Precomputed `prompt_embedding` accepts `[tokens,46080]` with six consecutive 7680-wide bundles and must already include the intended reference-image conditioning.

## Validation

`tests/test_fibo.cpp` and the C API tests exercise planning, schedule values, mask dimensions, prompt formatting and unsupported options. Development-only checks use local weights:

```sh
python tools/fibo_generation_smoke.py --library dist/windows/slopfab.dll
slopfab_fiboprobe text weights/text_encoder/qwen3vl_4b_int8_convrot.safetensors build/fibo-support/source.ppm build/fibo-support/text-native.safetensors
python tools/fibo_reference.py --probe build/Release/slopfab_fiboprobe.exe --model weights/transformer/fibo2_edit_turbo_int8_convrot.safetensors --embedding build/fibo-support/text-native.safetensors
```

The reference tools require PyTorch, safetensors, Transformers/Diffusers and local fixtures under `build/fibo-support`; they never download models. The independent comparisons measured approximately 1.2% relative RMS for the full transformer, 1.4% for multimodal conditioning, 0.054% for VAE encoding and 0.009% for decoding. The DLL smoke test runs a real four-step masked edit and checks exact unselected pixels, output ownership and cancellation.
