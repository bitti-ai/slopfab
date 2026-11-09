# slopfab

Generate videos, images and music through Claude Code, Codex, Hermes, Gemini CLI, OpenCode, GitHub Copilot CLI or your favourite agentic CLI. Supports local generation or remote generation over SSH. Runs natively on Windows and Linux with CUDA and Vulkan.

1. Extract the package.
2. Launch your favourite agent in that folder.
3. Tell it to use slopfab to generate video/image/music.

Ask your agent to create [mods](docs/mods.md): reusable workflows you can share with others for music videos, cinematic action sequences or animated short films.

[Documentation](docs/agent-cli.md)

Update to the latest GitHub release with `slopfab update`, or check availability
with `slopfab update --check`. Your weights, projects and settings are preserved.

## Models and capabilities

| Model | What it does |
| --- | --- |
| MiniMax H3 (FL2VA, Ref2VA, Singularity / v2) | Generates video with audio, or still images. Ref2VA supports image, video and audio references; Singularity variants use the Turbo LoRA. |
| [FastH3 V2](docs/fasth3_v2.md) / [FastH3 OmniRef](docs/omniref.md) | Eight-step H3 generation: V2 generates video/audio from text; OmniRef also accepts multimedia references. |
| [Viggle Remake](docs/viggle_remake.md) | Recasts characters or edits a scene from a source video, painted frames and a prompt; retains source audio. |
| [Krea 2](docs/krea2.md) (Turbo / Raw) | Text-to-image generation; Turbo is the default image model. |
| [MOSS Image](docs/moss.md) | Text-to-image generation with Qwen3.5 conditioning. |
| [P-H3 Image](docs/ph3_image.md) (initial support) | Checkpoint inspection and eight-step text-to-image planning; rendering is not implemented yet. |
| [Fibo 2.0](docs/fibo2.md) | Text-to-image generation and instruction-based image editing; Edit Turbo accepts up to four reference images. |
| [YuE2 / MiniMax Music3](docs/music-models.md) | Generates stereo songs from style descriptions and lyrics, or instrumental music. |
| [RealESRGAN x4plus](docs/upscaling.md) / [SeedVR2 3B](docs/seedvr2.md) | Image/video upscaling; SeedVR2 restores video in temporal segments. |
| [H3 latent upscaler](docs/latent_upscaling.md) | Enlarges H3 video before decoding, preserving frame count and audio. |
| [H3 Regenerate 2K](docs/regenerate_2k.md) | Refines a draft at up to 2560x1440 using the IC LoRA, Ref2VA and HyperFlow; 3, 4 or 8 steps. |
| [Sol-H3-VAE](docs/sol_h3_vae.md) / [TAEH3](docs/generation_previews.md) | Alternative H3 video decoding / lightweight generation previews. |
| Whisper (tiny through large, including large-v3-turbo) | Transcribes speech or song vocals, with subtitle export and optional word timestamps. |
| [HTDemucs](docs/separation.md) | Separates drums, bass, other instruments and vocals, or vocals and an instrumental track. |
| [SyncNet / S3FD](docs/syncnet.md) | Measures lip synchronization, with automatic face detection/tracking and per-face timing offsets. |

**Video:** Compatible H3 models support [multimedia references](docs/reference_media.md),
ending-image anchors, [long-form continuation and latent bridging](docs/continuation.md),
and [extending or bridging imported clips](docs/video-transitions.md). Optional
[LoRAs](docs/loras.md) include Turbo, TaoMate, LightX2V, DMAD, Pika and Hyperflow.
[Motions](docs/motions.md) adds scripted vector titles, graphics and overlays.
[Character Swap](docs/character-swap.md) combines an H3 Ref2VA source video with a replacement character image.

**Image editing:** [H3 inpainting and outpainting](docs/inpainting.md) replace selected
content or expand the canvas, with edit strength and edge blending. Fibo supports
prompted edits and combining references. The CLI provides rectangular H3 edits;
the C API also supports brush, lasso and soft masks for H3/Fibo, preserving pixels
outside the selection.

## Model licensing

MiniMax H3 model weights are licensed separately under the MiniMax H3 COMMUNITY LICENSE AGREEMENT.

Fibo 2.0 is released under CC BY-NC 4.0 for non-commercial use. Commercial use is subject to an agreement with BRIA.

YuE2 uses CC BY-NC 4.0; MiniMax Music3 uses Apache 2.0.
