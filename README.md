# slopfab

Native C++ video and audio generation with [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), using CUDA or Vulkan. No Python is required at runtime.

## Features

- **Text-to-video with audio**, configurable resolution, frame count, sampling steps, seeds and batch generation.
- **Image, video and audio references** with compatible Ref2VA models, including reusable [refmods and mixed bundles](docs/refmods.md).
- **Raw-media RefMod export** via `encode-refmod` or C API `slopfab_export_refmod` (ABI 1.18), using only the required VAEs.
- **Standalone text embeddings** via [`encode-text`](docs/refmods.md#standalone-text-embeddings), reusable alongside media bundles.
- **CUDA and Vulkan inference** with native text and reference conditioning, plus selectable attention backends.
- **LoRA adapters**, adapter stacking and the [TaoMate three-step schedule](docs/loras.md).
- **Model and LoRA sampling settings**, with [metadata defaults and JSON overrides](docs/sampling_settings.md).
- **Independent audio steps** (experimental) through `--audio-steps` or the C API, while retaining a LoRA's video grid. See [sampling settings](docs/sampling_settings.md#independent-audio-steps-experimental).
- **Configurable conditioning and isolated session caches**, with [model/LoRA recipes and library APIs](docs/conditioning_settings.md).
- **Video continuation** through [saved video and audio latents](docs/continuation.md).
- **Latent bridging** between retained clips, with persistent reference conditioning and editable margins on both sides. See [bridging](docs/continuation.md#bridging-retained-clips-c-api-126-experimental).
- **Quantized checkpoints** and [automatic CUDA transformer offloading](docs/denoising_memory.md) to manage GPU memory.
- **A C API** for asynchronous video and still-image generation, progress callbacks, cancellation and model reuse.
- **Bounding-box image editing** on CUDA and Vulkan, with edit strength, inward feathering and exact preservation outside the box. See [inpainting](docs/inpainting.md).
- **MP4 output through FFmpeg**, raw Y4M/WAV output, checkpoint inspection and tensor comparison tools.
- **Real-ESRGAN 4x upscaling** on CUDA and Vulkan, for standalone images and generated frames. See [upscaling](docs/upscaling.md).
- **H3 latent upscaling** on CUDA and Vulkan, enlarging video latents before VAE decoding. See [latent upscaling](docs/latent_upscaling.md).
- **SeedVR2 3B video restoration on CUDA**, with streamed segments, tiled VAE processing and direct [Comfy-Org safetensors](https://huggingface.co/Comfy-Org/SeedVR2) loading. See [SeedVR2](docs/seedvr2.md).

## Misc

- **FastH3 V2 on CUDA and Vulkan**, with learned VSA-H3 sparse attention and automatic eight-step scheduling. See [FastH3 V2](docs/fasth3_v2.md).
- **Optional MotionCache on CUDA and Vulkan**, reusing motion-aware video/audio residuals to reduce transformer calls. See [MotionCache](docs/motioncache.md).
- **Viggle-Animate checkpoint and adapter loading**, with [fixed-conditioning and driving-audio options](docs/viggle_animate.md).

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

To build the current checkout's Windows and Linux libraries together, run:

```powershell
.\build.cmd
```

This builds Release CUDA/Vulkan libraries, runs the C API and embedded-tokenizer
tests for each platform, and writes:

- `dist/windows/slopfab.dll`, `slopfab_c.lib` and `capi.h`.
- `dist/linux/libslopfab.so`, its versioned copies and `capi.h`.
- `SHA256SUMS` and `build-info.txt` in both folders.

The Linux build runs in the default WSL distribution, which needs the
[Linux build prerequisites](docs/linux.md#build) and a Linux CUDA toolkit.
Windows uses `build/`; Linux uses `$HOME/slopfab-build-linux` on its native
filesystem. Existing build directories are reused, including settings such as
FFmpeg; CUDA, Vulkan, the C API, tests and the full tokenizer are enabled
explicitly. No source, models or dependencies are downloaded. A failed build or
test stops the script with a nonzero exit code before publishing that platform.

Use `-Target windows` or `-Target linux` to build just one platform, `-Distro Ubuntu`
to select WSL, or `-Jobs 4` to change the default two compilation jobs.
`-WindowsBuildDir`, `-LinuxBuildDir` and `-LinuxCudaCompiler` override the build
directories and Linux compiler. Linux overrides are absolute Linux paths.
Run `.\build.cmd -Help` for all options. Libraries use external GPU runtimes;
use the packaging commands below when you also need a CLI distribution.

The default CUDA build targets NVIDIA RTX 30-series and RTX 50-series GPUs. On Windows, it selects cuBLAS from an installed CUDA 13 or CUDA 12 toolkit at runtime; Linux links the toolkit selected when building. Vulkan inference requires a compatible Vulkan 1.2 device and driver.

Use `build/` for all configurations. Reconfigure it with CMake options as needed:

| Option | Default | Purpose |
| --- | --- | --- |
| `SLOPFAB_ENABLE_CUDA` | `ON` | CUDA inference |
| `SLOPFAB_ENABLE_VULKAN` | `ON` | Vulkan inference and output conversion |
| `SLOPFAB_WITH_FFMPEG` | `ON` | MP4 output and FFmpeg media input |
| `SLOPFAB_BUILD_C_API` | `ON` when either GPU backend is enabled | Build `slopfab.dll` or `libslopfab.so` |
| `SLOPFAB_EMBED_TOKENIZER` | `ON` | Embed the tokenizer in the CLI and C API |
| `SLOPFAB_TOKENIZER_FILE` | `ref/text_encoder/tokenizer.json` | Tokenizer JSON to embed |
| `SLOPFAB_BUILD_TESTS` | `ON` | Build tests |
| `SLOPFAB_BUILD_DEV_TOOLS` | `OFF` | Build additional profiling and diagnostic tools |

Run tests or create a Windows package:

```powershell
ctest --test-dir build -C Release --output-on-failure
.\package.cmd
```

Packaging uses the same build directory and writes an archive under `dist/`. It includes the CLI, DLL, C header, import library and FFmpeg runtime. Model weights and CUDA libraries are not included.

On Linux, `bash package.sh --backend vulkan` or `bash package.sh --backend both`
builds, tests and packages the CLI, shared library, C header, documentation and
licenses. Linux packages use external GPU and media runtimes.

## Models

Generation uses a Qwen3-VL text encoder, an H3 transformer, a video VAE and an audio VAE. Place compatible checkpoints under:

```text
weights/
  text_encoder/
  transformer/
  vae/
```

The CLI discovers local checkpoints in these folders. Supply model files yourself; slopfab does not download missing weights. To choose files explicitly, use `--text-encoder`, `--transformer`, `--vae` and `--audio-vae`. Checkpoint quantization is detected from the file.

The video VAE supports Comfy INT8 ConvRot checkpoints, including `minimax_h3_video_vae_int8_convrot.safetensors`, for decoding and reference-image encoding during image or video generation. CUDA keeps quantized decoder weights compressed and expands one matrix at a time; Vulkan and exact CUDA decoding expand them to FP16 when loading. The encoder's FP32 biases and normalization parameters are converted to FP16 on both backends. No manual quantization or shift setting is needed.

Text-to-video uses FL2VA weights. Reference conditioning requires compatible Ref2VA weights; video and audio references also require VAE encoder weights. See [reference media support](docs/reference_media.md) for formats and requirements.

## Usage

The Windows executable is `build/Release/slopfab.exe`; the Linux executable is
`slopfab` inside its chosen build directory. The examples below assume `slopfab` is on your `PATH`.

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
| `--vulkan-arithmetic portable\|exact` | Vulkan arithmetic policy; Linux defaults to portable |
| `--steps N` | Set the sigma schedule length; performs `N - 1` evaluations |
| `--count N` | Generate multiple variations |
| `--motion-cache` | Enable approximate motion-aware denoising reuse |
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
