# Linux

slopfab builds a native Linux CLI and `libslopfab.so` with CUDA, Vulkan, or both.
Vulkan-only generation and the C API do not require a CUDA toolkit. A build with
both backends disabled still supports host tools such as checkpoint inspection.
The initial distribution target is x86-64 Linux.

## Build

Install CMake 3.24 or newer, a C++17 compiler and a build tool. On Ubuntu:

```sh
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build libpng-dev libjpeg-dev
```

Supply the matching Qwen tokenizer as `ref/text_encoder/tokenizer.json`, or pass
`-DSLOPFAB_TOKENIZER_FILE=/absolute/path/tokenizer.json` when configuring. The
tokenizer is embedded in the CLI and shared library on Linux, so the JSON file
is only required at build time. To deliberately build without it, configure
`-DSLOPFAB_EMBED_TOKENIZER=OFF` and supply `--tokenizer FILE` when generating.
Model weights and LoRA adapters must also be supplied locally; slopfab never
downloads them.

Build Vulkan without CUDA:

```sh
cmake -S . -B build/linux-vulkan -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSLOPFAB_ENABLE_CUDA=OFF -DSLOPFAB_ENABLE_VULKAN=ON
cmake --build build/linux-vulkan --parallel 2
build/linux-vulkan/slopfab devices
build/linux-vulkan/slopfab generate --inference-backend vulkan \
  --prompt "A cat playing a piano" --frames 22 --resolution 512x512 --raw
```

Vulkan headers and compiled shaders are included in the source tree; building
does not require a Vulkan SDK or shader compiler. Inference needs the Linux
Vulkan loader (`libvulkan.so.1`) and a compatible GPU driver. The runtime checks
the device's Vulkan features before generation. Software Vulkan drivers are
useful for limited tests but do not substitute for GPU generation validation.

Linux generation defaults to `--vulkan-arithmetic portable`. This permits
devices with the required shader capabilities without requiring the driver to
be on the CUDA bit-parity qualification list. Results are not guaranteed to
match CUDA byte for byte. `--vulkan-arithmetic exact` keeps the stricter
device/driver qualification for normalization and pointwise operations;
unqualified tuples are rejected. Windows retains its existing exact default.
This arithmetic policy is separate from `--attention`: choosing exact
attention alone does not promise that every decoder operation is bit-identical
to CUDA.

C API callers can override the platform default with
`slopfab_request_set_vulkan_arithmetic` (added in ABI 1.15); see the arithmetic
enum in `capi.h`. The choice is part of the request and session cache identity.
Native Linux GPU generation still needs model and driver qualification beyond
the host and software Vulkan tests run here.

For CUDA, install a CUDA toolkit supporting your Linux distribution and a
compatible NVIDIA driver, and make `nvcc` available on `PATH`. CUDA 12.8 or newer
is needed for the default architecture list `86;120a`, covering
the project's RTX 30-series and RTX 50-series kernels. Build both backends:

```sh
cmake -S . -B build/linux-both -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSLOPFAB_ENABLE_CUDA=ON -DSLOPFAB_ENABLE_VULKAN=ON
cmake --build build/linux-both --parallel 2
build/linux-both/slopfab generate --inference-backend cuda \
  --prompt "A cat playing a piano" --frames 22 --resolution 512x512 --raw
```

Use CMake 3.31 or newer for the default architecture list: older releases can
reject the `120a` suffix. Host and Vulkan builds retain the CMake 3.24 minimum.
Adjust the compiler path for your toolkit. CUDA 12.8's headers are incompatible
with newer glibc declarations on Ubuntu 26.04; use a newer toolkit supported on
that distribution or build in a compatible older distribution. CUDA compilation can consume
substantial memory; increase parallelism only when the machine has room.
Linux links cuBLAS from the selected build toolkit. Its corresponding shared
libraries must be discoverable by the dynamic linker at runtime. Linux does
not use Windows' automatic CUDA 12/13 toolkit switching. To build CUDA alone,
set `SLOPFAB_ENABLE_VULKAN=OFF`.

## Media dependencies

`SLOPFAB_WITH_FFMPEG=ON` compiles media support without requiring FFmpeg headers
or libraries at build time. Image decoding and MP4 output load FFmpeg 8.x shared
libraries at runtime: `libavcodec.so.62`, `libavformat.so.62`, `libavutil.so.60`
and `libswscale.so.9`, plus dependencies of the FFmpeg build (such as
`libswresample.so.6`). A different ABI is rejected with a version diagnostic.
The distribution's default FFmpeg package may provide an older ABI.

For a separately installed compatible FFmpeg build:

```sh
export SLOPFAB_FFMPEG_DIR=/opt/ffmpeg-8/lib
export LD_LIBRARY_PATH="$SLOPFAB_FFMPEG_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/opt/ffmpeg-8/bin:$PATH"
```

`LD_LIBRARY_PATH` also makes dependencies of those libraries discoverable. A
system-wide installation registered with `ldconfig` needs no override. Video
and audio reference files additionally need `ffmpeg` and `ffprobe` executables
on `PATH` or beside the slopfab executable; these subprocesses are independent
of the shared-library ABI requirement.

Use `--raw` for Y4M/WAV output without MP4 encoding. With
`SLOPFAB_WITH_FFMPEG=OFF`, Linux image input supports PNG, JPEG and PPM through
the native fallback; video/audio file decoding is disabled. C API callers can supply decoded pixels
and PCM directly.

## Install and package

```sh
cmake --install build/linux-vulkan --prefix "$HOME/.local"
bash package.sh --backend vulkan
bash package.sh --backend both -- -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

`package.sh` configures, builds and runs host/C API/reference-decode tests, then
creates `dist/slopfab-VERSION-linux-x86_64-BACKEND.tar.gz`. It uses a separate
directory under `build/` for each backend; `--build-dir DIR` overrides it.
Additional CMake arguments follow `--`. Set `CMAKE_BUILD_PARALLEL_LEVEL` to
change the default of two compilation jobs. Packaging stages files under Linux
`/tmp` before copying the archive to `dist/`, so WSL checkouts on Windows drives
do not need to support Linux permissions or shared-library symlinks.

Archives contain `bin/slopfab`, the versioned C API library and SONAME symlinks
under `lib/`, `include/slopfab/capi.h`, documentation and licenses. The tokenizer
and Vulkan shaders are compiled in. Model weights, adapters, NVIDIA/Vulkan
drivers, CUDA libraries, libpng/libjpeg and FFmpeg runtimes are external dependencies. Extract
the archive and add its `bin/` directory to `PATH`. C API consumers may load
`lib/libslopfab.so` by absolute path or install it in their loader search path.
The installed executable and shared library include relative `$ORIGIN` library
search paths so the installation can be relocated.

## Validation and CI

Run tests without model weights or a GPU:

```sh
ctest --test-dir build/linux-vulkan --output-on-failure \
  -E '^(vulkan|kernels|tensor_backends)($|_)'
```

On a GPU host, run `ctest --test-dir build/linux-vulkan --output-on-failure` or
use the CUDA build directory. Suites without required hardware or checkpoints
report skipped tests. Full validation also requires real generation, reference
input, LoRA, continuation and output tests with the desired model files.

The Linux workflow builds host-only and Vulkan configurations with media support
enabled and disabled, exercises host and C API tests, and checks installation
and packaging. It embeds a small test tokenizer; its uploaded archives are test
builds and cannot tokenize real model prompts correctly. A manually enabled
CUDA job compiles CLI and C API against CUDA 12.8 for SM86. Hosted CI does not
claim GPU execution, native SM120a coverage or successful model generation.

Validation recorded on 2026-09-28 used WSL Ubuntu 26.04, GCC 15, CMake 4.2 and
CUDA 13.3:

- The CUDA + Vulkan configuration built all targets and passed nine host, C API
  and CUDA dispatch tests. CUDA-only CLI/C API builds with FFmpeg disabled also
  passed device enumeration and embedded-tokenizer smoke checks.
- Real CUDA video/audio VAE decoding of synthetic latents produced a 22-frame,
  256x256 MP4; probing confirmed H.264 video and AAC audio. This validates
  decoding and output, not the complete text-to-video pipeline.
- The Vulkan-only configuration passed seven of eight CTest suites; the model
  integration suite skipped because WSL exposed only llvmpipe software Vulkan.
  Portable arithmetic tests executed 33 checks through actual Vulkan shaders.
  No physical Vulkan GPU generation was validated. Full H3 inference also
  requires the supported BF16/FP16 cooperative-matrix operations and subgroup
  size; a Vulkan device listing alone does not establish those capabilities.
- The packaged Vulkan CLI and C API passed all five packaging checks. After
  extraction elsewhere, the CLI tokenized with the embedded full vocabulary
  and the versioned shared library loaded and accepted both arithmetic policies.
  SONAME symlinks and relative library search paths were verified.
- Linux host tests exercised FFmpeg 8 image decoding, MP4 output and reference
  decoding. Four Windows host regression tests passed as well.

## WSL

Run the same Linux commands inside the WSL distribution, for example from
`/mnt/d/Projects/slopfab`. Keep Linux and Windows build directories separate.
Use a build directory on the Linux filesystem, such as
`-B /tmp/slopfab-linux-vulkan`, when the source checkout is on a Windows drive;
CMake may be unable to set generated-file permissions on that mounted drive.
Check GPU availability inside WSL using `slopfab devices`; CUDA availability
does not imply that a usable Vulkan device is exposed. Use the toolkit installed
inside the Linux distribution for compilation and the WSL-compatible Windows
NVIDIA driver for CUDA device access.
