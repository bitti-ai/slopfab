# slopfab

Native C++ video, image and music generation on CUDA and Vulkan. No Python is required at runtime.

## Features

- **Agent projects:** [Project folders](docs/projects.md) with an offline media review page, clip editing prompts, saved generation recipes and explicit file naming rules.
- **Agent workflows:** [Embedded JavaScript](docs/scripting.md) for files, HTTP downloads, FFmpeg and parallel processes; [mods](docs/mods.md) can add CLI commands. QuickJS is linked only into the executable, never the DLL.
- **Motions:** [JavaScript vector animations](docs/motions.md) rendered with embedded ThorVG, with transparent stills, video exports and overlays on generated video. ThorVG is linked only into the executable.
- **Video review:** [Single frames and downscaled grids](docs/video_frames.md) through `slopfab video frames` or JavaScript, including direct capture from generation frames in RAM.

- **Video with audio:** MiniMax H3 generation with configurable resolution, duration, seeds and batches.
- **Reference conditioning:** images, video and audio with Ref2VA; reusable [RefMods, mixed bundles and text embeddings](docs/refmods.md).
- **LoRA adapters:** stacking, TaoMate (3 evaluations), DMAD (4), and **Hyperflow (8)** with two-time conditioning and an automatically selected schedule. Hyperflow requires a prepared companion; see [LoRA setup](docs/loras.md).
- **Sampling controls:** Euler, AB2, re-noising and **ER-SDE**, beta scheduling, model/LoRA defaults, JSON overrides and experimental independent audio steps. See [sampling settings and compatibility](docs/sampling_settings.md).
- **Editing and continuity:** [masked editing and outpainting](docs/inpainting.md), [saved-latent continuation, bridging and ending anchors](docs/continuation.md), including persistent Keystone/Capstone context and optional context noise.
- **Color and tone:** [latent color controls and writable sampling callbacks](docs/lcs.md), with masks, sharpness and reusable VAE calibration.
- **Low-VRAM H3 text encoding:** [ClipProj](docs/clipproj.md) with a Qwen3-VL-4B/8B student, streaming GPU layers and a CPU projection.
- **Memory and speed:** quantized checkpoints, [CUDA offloading](docs/denoising_memory.md), [FastH3 V2](docs/fasth3_v2.md), [OmniRef](docs/omniref.md), [VEDA](docs/veda.md) and optional [MotionCache](docs/motioncache.md). [Viggle-Animate loading](docs/viggle_animate.md) has separate workflow limits.
- **Decoding and upscaling:** [Sol H3 VAE](docs/sol_h3_vae.md), [H3 latent upscaling](docs/latent_upscaling.md) and [Real-ESRGAN](docs/upscaling.md) on CUDA/Vulkan; [SeedVR2 restoration](docs/seedvr2.md) on CUDA.
- **Application integration:** asynchronous C API, cancellation, progress, [generation previews](docs/generation_previews.md), [isolated session caches](docs/conditioning_settings.md) and model reuse.
- **Media and tools:** Native PNG images and Y4M/WAV output, checkpoint inspection, tensor comparison and standalone media/text encoding.

## Build

Linux builds support CUDA, Vulkan, or both, including the C API and an embedded
tokenizer. See [Linux build, packaging and WSL instructions](docs/linux.md).

For Windows builds, install:

- CMake 3.31 or newer for the default CUDA architectures, and Visual Studio 2022 C++ build tools.
- CUDA 12.8 for compilation and CUDA 13.0 headers for the cuBLAS compatibility checks.
- The model tokenizer at `ref/text_encoder/tokenizer.json` for embedding in the executable and DLL.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T "cuda=$($env:CUDA_PATH_V12_8.Replace('\', '/'))"
cmake --build build --config Release
```

Build and test the Windows/Linux CLI and libraries together with `.\build.cmd` (Linux uses WSL).
Use `-Target windows|linux`, `-Distro Ubuntu-22.04` or `-Jobs 4` to narrow the build;
`.\build.cmd -Help` lists all options. Executables, libraries, headers and checksums go to
`dist/windows/` and `dist/linux/`. Linux releases target glibc 2.35 and default
to WSL Ubuntu-22.04; the Linux guide also provides a container build. GPU runtimes remain external.

The default CUDA targets are RTX 30-series and RTX 50-series. Windows selects
cuBLAS from an installed CUDA 13 or CUDA 12 toolkit; Linux links its build toolkit.
Vulkan requires a compatible Vulkan 1.2 device and driver.

Common CMake options (all default to `ON`):

| Option | Purpose |
| --- | --- |
| `SLOPFAB_ENABLE_CUDA` / `SLOPFAB_ENABLE_VULKAN` | Enable either or both inference backends |
| `SLOPFAB_BUILD_C_API` | Build the shared library when a GPU backend is enabled |
| `SLOPFAB_EMBED_TOKENIZER` | Embed the tokenizer; override its path with `SLOPFAB_TOKENIZER_FILE` |
| `SLOPFAB_BUILD_TESTS` | Build tests |

```powershell
ctest --test-dir build -C Release --output-on-failure
.\package.cmd
```

Windows packaging uses `build/` and includes the CLI, DLL, C header, import
library under `dist/`. Linux uses
`bash package.sh --backend vulkan` or `--backend both`, with external GPU runtimes. Packages exclude model weights and CUDA libraries.

## Models

| Family | Output | Backends | Setup |
| --- | --- | --- | --- |
| MiniMax H3 | Video with audio; still images | CUDA, Vulkan | [References](docs/reference_media.md), [LoRAs](docs/loras.md) |
| Krea 2 | Images | CUDA, Vulkan | [Krea](docs/krea2.md) |
| MOSS Image SFT | Images | CUDA | [MOSS](docs/moss.md) |
| Fibo 2.0 Edit Turbo | Image editing | CUDA | [Fibo](docs/fibo2.md) |
| YuE2 / MiniMax Music3 | Music | CUDA, Vulkan | [Music models](docs/music-models.md) |

Select `--model h3|moss|fibo2|krea2|yue2|minimax_music3`, or let an explicit
transformer checkpoint determine the family. H3 is the default.

Place compatible checkpoints in `weights/text_encoder/`, `weights/transformer/`
and `weights/vae/`; adapters conventionally go in `weights/loras/`. Register the
root with `slopfab settings set weight_paths "D:/models/weights"` (multiple paths
are supported). The CLI searches configured paths for conditioner, transformer
and VAE files, or accepts explicit
`--text-encoder`, `--transformer`, `--vae` and `--audio-vae` paths.
Quantization is detected from each file, including [H3 W4A8](docs/w4a8.md),
[W6A8](docs/w6a8.md) and Comfy INT8 ConvRot video VAEs.

H3 text-to-video uses FL2VA weights; reference inputs require Ref2VA and the
appropriate VAE encoders. Sol H3 VAE is a decoder override (`--video-decoder`),
so retain the original VAE for reference encoding. On Vulkan, select the FP32
H3 audio VAE explicitly if discovery chooses an NF4 checkpoint.

## Usage

See the [agent CLI guide](docs/agent-cli.md) for JSON requests, `--agent` NDJSON
responses and progress, capability discovery, configuration, and persistent jobs.
Use `slopfab mods list` to find [shareable agent workflows](docs/mods.md), and
`slopfab mods load NAME` to bring one into agent context.
Run `slopfab` for compact instructions, or `slopfab help` to list commands.
Native outputs are PNG, Y4M, and WAV. Use a separate FFmpeg installation to encode
MP4 or other compressed formats; neither the executable nor DLL depends on FFmpeg.


The Windows executable is `build/Release/slopfab.exe`; the Linux executable is
`slopfab` inside its chosen build directory. The examples below assume `slopfab` is on your `PATH`.

Generate a video, or a Krea image with ER-SDE and beta scheduling:

```sh
slopfab generate --prompt "A cat playing a piano in warm lamplight" --frames 22 --resolution 512x512 --seed 11 --out video.y4m
slopfab generate --model krea2 --sampler er_sde --schedule beta --steps 8 --prompt "A red apple" --out apple.png
```

Read a longer prompt from a UTF-8 file, or inspect a request without loading model weights:

```sh
slopfab generate --prompt-file prompts/scene.txt --out scene.y4m
slopfab generate --prompt "A moonlit forest" --frames 124 --dry-run
```

Common generation options:

| Option | Purpose |
| --- | --- |
| `--inference-backend cuda\|vulkan` | Select the inference backend |
| `--vulkan-arithmetic portable\|exact` | Vulkan arithmetic policy; Linux defaults to portable |
| `--steps N` | Requested denoising updates (`N` updates; terminal zero added separately) |
| `--sampler KIND` / `--schedule beta` | Select the solver and optional beta grid |
| `--sampling-settings FILE` | Load sampling JSON overrides |
| `--audio-steps N` | Experimental independent H3 audio grid length |
| `--count N` | Generate multiple variations |
| `--motion-cache` | Enable approximate motion-aware denoising reuse |
| `--reference-image FILE` | Add an image reference; repeat for multiple images |
| `--reference-video FILE` / `--reference-audio FILE` | Add video or audio references |
| `--lora FILE --lora-strength VALUE` | Apply a LoRA adapter |
| `--save-latents FILE` / `--continue-from FILE` | Save or extend a generation |
| `--raw` | Write Y4M video and WAV audio |

Fixed/distilled recipes determine their own grids. For Hyperflow, prepare its
[companion](docs/loras.md#hyperflow-h3-eight-evaluations), pass `--lora FILE` and omit `--steps`.

Without `--out`, videos are written to unique run directories under `output/`. Use `slopfab generate --help` for the complete option list.

Inspect checkpoints and compare tensor archives:

```sh
slopfab inspect model.safetensors --list
slopfab compare reference.safetensors actual.safetensors --abs-tol 1e-3
slopfab devices
```

## C API

[`include/slopfab/capi.h`](include/slopfab/capi.h) defines the C interface for applications written in C, C++, Rust, C#, Python and other languages with C bindings.

The API supports request validation, asynchronous generation, progress callbacks, cancellation, reusable model caches, still-image generation and latent continuation. Results are returned as decoded video pixels and audio samples; hosts can also supply decoded reference frames and PCM audio directly.

Optional [TAEH3 previews](docs/generation_previews.md) expose small, silent RGB previews during H3 video and still-image denoising on CUDA and Vulkan.

## Licensing

MiniMax H3 model weights are licensed separately under their own **MiniMax H3 COMMUNITY LICENSE AGREEMENT**.

Model weights are not distributed in this repository. Bundled third-party components retain their own licenses: [SageAttention](third_party/sageattention/LICENSE) and [Vulkan headers](third_party/vulkan/LICENSE.md).
