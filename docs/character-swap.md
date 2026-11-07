# H3 Character Swap V1.5

The original [akatz-ai Character Swap LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA)
loads directly on CUDA and Vulkan. Use a **Ref2VA** transformer, the full
multimodal Qwen conditioner, and H3 video/audio VAEs. FL2VA bases are rejected
when loading this adapter or planning generation; ClipProj cannot encode its
image/video references.

Use one source performance as `<Video 1>` and one replacement identity as
`<Picture 1>`. These are ordinary references, not first-frame anchors or the
Viggle Animate mode. With the other model components configured:

```sh
slopfab video generate --model h3 --transformer weights/transformer/minimax_h3_ref2va_pruned_nvfp4.safetensors --lora weights/h3_character_swap_v1.5.safetensors --lora-strength 1 --reference-video source.y4m --reference-image character.png --prompt "Replace the person wearing a green coat in <Video 1> with the robot in <Picture 1>. Keep the robot's face, jacket and proportions. Follow the source person's head turns, hand wave and walking pace. Retain the station platform and other passengers." --frames 73 --resolution 512x512 --out output/project.y4m
```

Replace `project` with the project's name. Add `--inference-backend vulkan`
and `--vulkan-arithmetic portable` for Vulkan. The CLI accepts native Y4M references;
convert compressed source videos first. Start with a short, continuous 24 fps
shot, match its aspect ratio and duration, and identify the intended person
explicitly when others are visible. Supported video frame counts follow `22 + 17n`.

The command uses the base sampling defaults. The author's eight-step recipe
also needs the separate `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`
adapter at strength 1. When that adapter is available, append its own `--lora`
and `--lora-strength 1`, plus `--steps 8 --sampler er_sde --schedule default`.
The recipe uses video/audio shifts 12/3 and CFG 1. Character Swap alone is not
an eight-step distillation adapter. Turbo's AdaLN tensors may require the
[local timestep grid](loras.md#turbo-adapters-with-adaln-updates).

Retrieve the embedded prompting recipe without loading any weights:

```sh
slopfab prompts get h3 character-swap
slopfab prompts select --model h3 --character-swap --reference-image character.png --reference-video source.y4m
```

`--character-swap` selects a prompting guide only; generation uses `--lora`.
For source audio conditioning, add `--driving-audio source.wav`
and `--preserve-driving-audio`. To retain the exact original soundtrack, mux it into
the result separately. Use [SyncNet](syncnet.md) to measure lip-sync alignment.

This integrates the adapter and ordinary Ref2VA conditioning. The author's
separate SAM masking and multi-window Relay workflow are not implemented here.
Unmasked generation can change the background, timing and identity; prompting
does not guarantee exact preservation.

The released file contains 208 rank-16 BF16 A/B pairs across 50 main blocks and
two text-refiner blocks. Its native fused QKV and SwiGLU order is preserved;
absent alpha uses rank 16, so strength 1 applies `B @ A`. The embedded V2 name
belongs to the published V1.5 file and does not require conversion. Tests load
all pairs unchanged and compare representative real projections against CPU,
CUDA and Vulkan. These checks verify adapter execution, not visual quality.

The download mod catalog lists it under `h3-character-swap`. The upstream
MiniMax H3 Community License applies.
