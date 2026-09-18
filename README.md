# slopfab

Native C++ video and audio generation with [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), using CUDA or Vulkan. No Python is required at runtime.

## Features

- **Text-to-video with audio**, configurable resolution, frame count, sampling steps, seeds and batch generation.
- **FastH3 V2 on CUDA**, with learned VSA-H3 sparse attention and automatic eight-step scheduling. See [FastH3 V2](docs/fasth3_v2.md).
- **Image, video and audio references** with compatible Ref2VA models, including reusable [refmods](docs/refmods.md).
- **CUDA and Vulkan inference** with native text and reference conditioning, plus selectable attention backends.
- **LoRA adapters**, adapter stacking and the [TaoMate three-step schedule](docs/loras.md).
- **Viggle-Animate checkpoint and adapter loading**, with [fixed-conditioning and driving-audio options](docs/viggle_animate.md).
- **Video continuation** through [saved video and audio latents](docs/continuation.md).
- **Quantized checkpoints** and [automatic CUDA transformer offloading](docs/denoising_memory.md) to manage GPU memory.
- **A C API** for asynchronous video and still-image generation, progress callbacks, cancellation and model reuse.
- **MP4 output through FFmpeg**, raw Y4M/WAV output, checkpoint inspection and tensor comparison tools.

## Build

For Windows builds, install:

- CMake 3.24 or newer and Visual Studio 2022 C++ build tools.
- CUDA 12.8 for compilation and CUDA 13.0 headers for the cuBLAS compatibility checks.
- The model tokenizer at `ref/text_encoder/tokenizer.json` for embedding in the executable and DLL.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T "cuda=$($env:CUDA_PATH_V12_8.Replace('\', '/'))"
cmake --build build --config Release
```

The default CUDA build targets NVIDIA RTX 30-series and RTX 50-series GPUs. At runtime, it selects cuBLAS from an installed CUDA 13 or CUDA 12 toolkit. Vulkan inference requires a compatible Vulkan 1.2 device and driver.

Use `build/` for all configurations. Reconfigure it with CMake options as needed:

| Option | Default | Purpose |
| --- | --- | --- |
| `SLOPFAB_ENABLE_CUDA` | `ON` | CUDA inference |
| `SLOPFAB_ENABLE_VULKAN` | `ON` | Vulkan inference and output conversion |
| `SLOPFAB_WITH_FFMPEG` | `ON` | MP4 output and FFmpeg media input |
| `SLOPFAB_BUILD_C_API` | `ON` when CUDA is enabled | Build `slopfab.dll` |
| `SLOPFAB_BUILD_TESTS` | `ON` | Build tests |
| `SLOPFAB_BUILD_DEV_TOOLS` | `OFF` | Build additional profiling and diagnostic tools |

Run tests or create a Windows package:

```powershell
ctest --test-dir build -C Release --output-on-failure
.\package.cmd
```

Packaging uses the same build directory and writes an archive under `dist/`. It includes the CLI, DLL, C header, import library and FFmpeg runtime. Model weights and CUDA libraries are not included.

## Models

Generation uses a Qwen3-VL text encoder, an H3 transformer, a video VAE and an audio VAE. Place compatible checkpoints under:

```text
weights/
  text_encoder/
  transformer/
  vae/
```

The CLI discovers checkpoints in these folders. On Windows, it can download missing defaults automatically. To choose files explicitly, use `--text-encoder`, `--transformer`, `--vae` and `--audio-vae`. Checkpoint quantization is detected from the file.

Text-to-video uses FL2VA weights. Reference conditioning requires compatible Ref2VA weights; video and audio references also require VAE encoder weights. See [reference media support](docs/reference_media.md) for formats and requirements.

## Usage

The Windows executable is `build/Release/slopfab.exe`. The examples below assume `slopfab` is on your `PATH`.

Generate a video:

```sh
slopfab generate --prompt "A cat playing a piano in warm lamplight" --frames 22 --resolution 512x512 --seed 11 --out video.mp4
```

Read a longer prompt from a UTF-8 file, or inspect a request without loading model weights:

```sh
slopfab generate --prompt-file prompts/scene.txt --out scene.mp4
slopfab generate --prompt "A moonlit forest" --frames 124 --dry-run
```

Common generation options:

| Option | Purpose |
| --- | --- |
| `--inference-backend cuda\|vulkan` | Select the inference backend |
| `--steps N` | Set the sigma schedule length; performs `N - 1` evaluations |
| `--count N` | Generate multiple variations |
| `--reference-image FILE` | Add an image reference; repeat for multiple images |
| `--reference-video FILE` / `--reference-audio FILE` | Add video or audio references |
| `--lora FILE --lora-strength VALUE` | Apply a LoRA adapter |
| `--save-latents FILE` / `--continue-from FILE` | Save or extend a generation |
| `--raw` | Write Y4M video and WAV audio |

Without `--out`, videos are written to timestamped files under `output/`. Use `slopfab generate --help` for the complete option list.

Inspect checkpoints and compare tensor archives:

```sh
slopfab inspect model.safetensors --list
slopfab compare reference.safetensors actual.safetensors --abs-tol 1e-3
slopfab devices
```

## C API

[`include/slopfab/capi.h`](include/slopfab/capi.h) defines the C interface for applications written in C, C++, Rust, C#, Python and other languages with C bindings.

The API supports request validation, asynchronous generation, progress callbacks, cancellation, reusable model caches, still-image generation and latent continuation. Results are returned as decoded video pixels and audio samples; hosts can also supply decoded reference frames and PCM audio directly.

## Licensing

MiniMax H3 model weights are licensed separately under their own **MiniMax H3 COMMUNITY LICENSE AGREEMENT**.

Model weights are not distributed in this repository. Bundled third-party components retain their own licenses: [FFmpeg](external/ffmpeg/LICENSE), [SageAttention](third_party/sageattention/LICENSE) and [Vulkan headers](third_party/vulkan/LICENSE.md).
