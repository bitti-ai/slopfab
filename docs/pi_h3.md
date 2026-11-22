# pi-H3 DWD

Slopfab loads the released `pi_h3_2step_dwd_adapter_int8_convrot.safetensors`
and `pi_h3_4step_dwd_adapter_int8_convrot.safetensors` adapters on CUDA and
Vulkan. Video pixels come directly from the final transformer's hidden states
through the adapter's eight-layer pixel head. No video VAE decoder is loaded
or run. Audio still uses the standard H3 audio VAE.

These files are adapters, not complete transformers. Select a standard pruned
MiniMax H3 **FL2VA** base, its Qwen conditioner, and one adapter at strength 1:

```powershell
slopfab video generate --prompt "A cat playing piano, warm afternoon light" --transformer weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors --lora weights/pi_h3_2step_dwd_adapter_int8_convrot.safetensors --text-encoder weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors --audio-vae weights/vae/minimax_h3_audio_vae_fp32.safetensors --resolution 864x480 --frames 124 --out pi-h3.mp4
```

Choose the `4step` adapter to use four evaluations. The step count is inferred
from the adapter filename; a renamed adapter requires an explicit `--steps 2`
or `--steps 4` matching its training. Ref2VA, Hybrid DMAD-PDMD, FastH3 V2 and
OmniRef are not replacement bases for this FL2VA adapter.

For Vulkan, add `--inference-backend vulkan --vulkan-arithmetic portable`. The existing
C API adapter setter selects the same implementation. No Python dependencies
are needed.

The trained recipe uses Euler, video/audio shifts 12/3, initial noise scaled by
the first sigma, and these unshifted grids (with terminal zero):

| Adapter | Base grid |
| --- | --- |
| 2 steps | 0.93, 0.34875, 0 |
| 4 steps | 0.96, 0.6933333333, 0.4266666667, 0.16, 0 |

The adapter replaces all block AdaLN weights, the final heads, and selected
weights in blocks 46–49. Its rank-64 updates apply to blocks 0–45. These are
loaded as an in-memory overlay without modifying either checkpoint. Pixel-head
INT8 ConvRot matrices are materialized to BF16 for the existing Sol attention
operators; backbone matrices retain their native quantization. The head runs
after the backbone is unloaded.

Vulkan submits pi-H3's main transformer one layer at a time. This avoids the
device-loss observed with a single long submission on the tested Windows
NVIDIA driver, while preserving the computation. CUDA is faster on that setup.
The two-step smoke's latents, Y4M video and WAV audio were byte-identical before
and after this submission change.

Text-to-video, first-frame, last-frame and first+last-frame conditioning are
supported. Add these options to either adapter's generation command:

```powershell
--first-frame opening.png --last-frame ending.png --vae weights/vae/minimax_h3_video_vae_fp16.safetensors
```

Either image can be omitted. Images are resized to the target canvas and
encoded as fixed guides at its opening and ending rotary times. They guide
the generated endpoints rather than replacing output pixels. The video VAE
is used only for image encoding; DWD still produces all output video pixels.
`--start-anchor` and `--end-anchor` are aliases. C API 1.54 adds
`slopfab_request_set_start_anchor`; combine it with `slopfab_request_set_end_anchor`.
General multimedia Ref2VA and ordinary `--reference-image` slots are not FL2VA inputs.
The current integration rejects adapter stacking, alternate sampling grids,
approximate caches, latent continuation/bridges, latent effects/upscaling,
reusable latent archives and VAE previews. `--dump-latents` remains available
for numerical diagnostics, but those latents cannot reproduce DWD pixels.
Pixel-space postprocessing uses the normal output pipeline.

Validation includes both local INT8 ConvRot adapters, NVFP4/FP8 FL2VA adapter
loading, schedule/norm/ConvRot/pixel-layout tests, and actual CUDA and Vulkan
generation with no video VAE path. The four-step adapter was checked at
864x480 for 124 frames on both backends; the two-step smoke uses 128x128 for
22 frames. Use a trained canvas such as 864x480 for image quality: tiny smoke
resolutions can produce visible patch artifacts.

Sources: [pi-H3 model card](https://huggingface.co/Lakonik/pi-H3),
[pinned upstream implementation](https://github.com/Lakonik/ComfyUI-piFlow/tree/bcb6ffb8ff6191115e5803c4ddddef6a73b0b8a8).
