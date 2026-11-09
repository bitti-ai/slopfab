# Open-H3-Regenerate-2K-IC

[Open-H3-Regenerate-2K-IC](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC)
refines an H3 draft through **Ref2VA**, keeping the draft in video context while
generating higher-resolution frames. Slopfab supports CUDA and Vulkan, with
the released IC LoRA (312 rank-384 projections, including both token-refiner
blocks), HyperFlow, and fixed silence latents.

Use `slopfab video generate --regenerate-2k up3` with one Y4M draft containing
at least 124 frames at 24 fps. The first 124 frames are used. Normalize or pad
shorter clips externally. Default output is **2560x1440**; `--resolution` can
select an aligned smaller canvas for testing or 768p refinement. Each target
axis must be at least the source axis and at most 4x larger, with both axes
multiples of 32. This implementation stops at 2K.

| Mode | Updates | Initialization | First video sigma |
|---|---:|---|---:|
| `up3` | 3 | Learned upscale of draft, blended with noise | ~0.834942 |
| `up4` | 4 | Learned upscale of draft, blended with noise | ~0.923077 |
| `hf8` | 8 | Pure noise | 1 |

All modes use CFG 1 and Euler on the exact HyperFlow grid (video/audio shifts
12/3). The shorter modes take its final intervals, preserving the trained
two-time modulation. Reference rows stay at 0.999 clean; target audio stays
at timestep 1. Alternate schedules, caches and other editing modes are rejected.

## Weights

Configure `weight_paths` or pass explicit model paths. The CLI discovers the
following files in those folders and their `loras/` subfolder:

- `open_h3_regenerate_2k_ic_lora.safetensors` (the original FP32 file).
- `minimax_h3_hyperflow_8step_v1.0.safetensors`, plus its prepared
  `.hyperflow-ref2va.safetensors` companion alongside it.
- `silence_audio_124f.safetensors` from the model's `assets/` directory.
- `minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors` for `up3`/`up4`.
  `--regenerate-upsampler FILE` also accepts the author's BF16 checkpoint.

Use an ordinary **pruned H3 Ref2VA** transformer, a full multimodal Qwen3-VL
encoder, and the standard H3 video/audio VAEs. ClipProj's text-only path is
incompatible. Explicit `--lora` entries override discovery for that adapter;
IC and HyperFlow should each have strength 1. Prepare HyperFlow once if needed:

```sh
python tools/prepare_hyperflow.py --adapter weights/loras/minimax_h3_hyperflow_8step_v1.0.safetensors --variant ref2va
```

The download catalog group is `h3-regenerate-2k`. Downloading does not prepare
the HyperFlow companion or change `weight_paths`.

## Example

Keep the original generation caption as `--prompt`; an empty caption is allowed.
Slopfab adds the author's enhancement instruction and encodes the draft's visual
context automatically. Do not substitute a text-only cached prompt embedding.

```sh
slopfab video generate --regenerate-2k up3 --reference-video draft.y4m --prompt "Original scene caption" --transformer weights/transformer/minimax_h3_ref2va_pruned_nvfp4.safetensors --text-encoder weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors --vae weights/vae/minimax_h3_video_vae_fp16.safetensors --audio-vae weights/vae/minimax_h3_audio_vae_fp32.safetensors --out output/PROJECT_NAME.y4m
```

Append `--inference-backend vulkan --vulkan-arithmetic portable --attention flash2`
for Vulkan. `--dry-run` validates geometry and companion headers without inference.
`--regenerate-silence FILE` selects the published silence asset explicitly.

The output WAV decodes the fixed silence latent. To retain a source soundtrack,
mux it back externally, for example:

```sh
ffmpeg -i output/PROJECT_NAME.y4m -i draft.mp4 -map 0:v:0 -map 1:a:0? -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest output/PROJECT_NAME.mp4
```

This native recipe uses the standard VAE and disables the author's optional LMS
Sharp branch (`--sharp 0` upstream). It does not implement that release's custom
Fast VAE or distributed multi-GPU execution. Quantization, VAE precision and
seeded noise differ from the upstream PyTorch implementation; results are not
bit-identical. A full 2K pass is substantially more expensive than draft generation.

The C++ API exposes `GenerateRequest::regenerate`; C API 1.52 adds
`slopfab_request_set_regenerate_2k`. Library callers supply paths and add both
LoRAs explicitly. JSON requests use `options.regenerate-2k`,
`options.regenerate-silence` and `options.regenerate-upsampler`.

## Validation

The Windows build passes all 28 release checks. Local tests load all 312
projections from the released FP32 IC adapter, check all three schedule modes,
the 2K default canvas, silence transposition and invalid requests. On an RTX
5090, CUDA `up3` and Vulkan portable `up4` both completed 124-frame smoke renders
from 160x96 to 320x192 using Ref2VA NVFP4, full Qwen, both LoRAs and the learned
latent upscaler; pinned audio remained unchanged. These verify execution, not
2K visual quality or performance. Full-resolution 2K rendering has not been
benchmarked here.
