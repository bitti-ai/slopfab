# Linux

slopfab builds a native Linux CLI named `slopfab` (the counterpart of Windows
`slopfab.exe`) and `libslopfab.so` with CUDA, Vulkan, or both.
Vulkan-only generation and the C API do not require a CUDA toolkit. A build with
both backends disabled still supports host tools such as checkpoint inspection.
The initial distribution target is x86-64 Linux.

## Release compatibility

Release libraries and packages target **glibc 2.35** (Ubuntu 22.04). Build on
Ubuntu 22.04, including its system libraries; building on a newer distribution
and copying the binary cannot lower its glibc requirements. Development builds
through CMake can still use the host distribution. The publishing helpers reject
newer build hosts and check ELF version requirements before copying artifacts.

For an isolated CUDA 13 / Vulkan release build from Linux or WSL with Docker:

```sh
docker build -t slopfab-linux-glibc235 -f tools/linux-build.Dockerfile tools
docker run --rm -v "$PWD:/src" -v slopfab-linux-glibc235:/build \
  slopfab-linux-glibc235
```

The tokenizer must already exist in the checkout. The `slopfab` executable,
shared library, headers and checksums go to `dist/linux/`.
The container does not need GPU access for the build and host C API tests.
For native WSL builds, install `Ubuntu-22.04` and its compiler/toolkit dependencies;
`build.cmd` now selects that distribution by default. Use CMake 3.31 or newer
(for example `python3 -m pip install cmake==3.31.6`), Python 3.10+ and binutils.
Do not reuse build directories from a newer distribution. The default release
build directories include `glibc235` to separate them from earlier outputs.

`python3 tools/check_glibc.py FILE_OR_DIRECTORY...` checks all ELF files,
including shared libraries and weak version requirements. When integrating
`libslopfab.so` into Slopus, rebuild the app and all AppImage dependencies on the
same baseline and scan the complete extracted AppImage. Replacing slopfab alone
cannot fix newer requirements in GTK, GLib or WebKitGTK. glibc compatibility does
not replace the GPU driver, CUDA, media or webview requirements.

## Build

Install CMake 3.24 or newer, a C++17 compiler and a build tool. On Ubuntu:

```sh
sudo apt-get update
sudo apt-get install build-essential ninja-build libpng-dev libjpeg-dev libcurl4-openssl-dev python3-pip binutils
python3 -m pip install --user cmake==3.31.6
export PATH="$HOME/.local/bin:$PATH"
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

The H3 exact-attention shader declares about 99 KiB of shared arrays. Its
48 KiB device-limit threshold relies on shared-memory reuse observed in NVIDIA's
compiler; successful pipeline creation remains a final compatibility check.
Portable arithmetic does not guarantee that every feature-compatible GPU can
compile this shader.

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

## Agent CLI on Linux

The Linux executable supports the same settings, mods, NDJSON output and job
commands as Windows. No Windows compatibility layer is needed. To build just
the executable after configuring either backend:

```sh
cmake --build build/linux-vulkan --target slopfab --parallel 2
build/linux-vulkan/slopfab settings set weight_paths /path/to/weights
build/linux-vulkan/slopfab mods list --agent
build/linux-vulkan/slopfab help generate
```

Global settings default to `$XDG_CONFIG_HOME/slopfab/config.json`, falling back
to `~/.config/slopfab/config.json`. Share mods as `mods/NAME/MOD.md` beside that
file, beside the executable, or in the working directory. See the [agent CLI
guide](agent-cli.md) and [mod format](mods.md). Use a supported GPU backend for
actual generation; host-only builds support discovery and request validation.

## Media dependencies

The executable and shared library no longer load FFmpeg libraries or launch
FFmpeg processes. Linux still-image input uses libpng/libjpeg. Native CLI output
is PNG/PPM, Y4M video, and WAV audio; native reference input includes Y4M/WAV.
Use a separately installed FFmpeg yourself for compressed formats. DLL callers
continue supplying decoded pixels and PCM directly. See [agent CLI](agent-cli.md).

## Install and package

```sh
cmake --install build/linux-vulkan --prefix "$HOME/.local"
bash package.sh --backend vulkan --windows-dir dist/windows
bash package.sh --backend both --windows-dir dist/windows -- -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

`package.sh` configures, builds and runs host/C API/reference-decode tests, then
creates `dist/slopfab-VERSION-windows-linux-x64.zip` and `.tar.gz`. Both archives
contain both executables; a prebuilt Windows executable is required under
`dist/windows` or `--windows-dir DIR`. It uses a separate
directory `build/linux-glibc235-BACKEND`; `--build-dir DIR` overrides it.
Additional CMake arguments follow `--`. Set `CMAKE_BUILD_PARALLEL_LEVEL` to
change the default of two compilation jobs. Packaging stages files under Linux
`/tmp` before copying the archive to `dist/`, so WSL checkouts on Windows drives
do not need to support Linux permissions or shared-library symlinks.

Archives contain only `slopfab`, `slopfab.exe` and `mods/` at the top level,
without an enclosing directory. The download and licenses commands are installed
under `mods/download/` and `mods/licenses/`. Run `./slopfab licenses` to print
the license and notice texts included in the licenses mod.
`tools/package_bundle.py` (or `package.cmd` on Windows) packages
existing `build/Release` and `build/linux-cli` outputs without rebuilding; use
`--windows-dir dist/windows --linux-dir dist/linux` for `build.cmd` outputs.
The package version comes from the root `VERSION` file, currently `0.1.0`.
It refuses single-platform packages.
The tokenizer
and Vulkan shaders are compiled in. Model weights, adapters, NVIDIA/Vulkan
drivers, CUDA libraries, libpng/libjpeg runtimes are external dependencies. The
CLI also needs libcurl for workflow HTTP requests (Ubuntu: `libcurl4`); QuickJS
itself is embedded only in the executable. Extract
the archive and add the extraction directory to `PATH`. C API libraries and
headers are available from the build outputs or `cmake --install` separately.
The installed executable and shared library include relative `$ORIGIN` library
search paths so the installation can be relocated.

## Validation and CI

The agent CLI update was built on WSL Ubuntu 26.04.1 with GCC 15.2 and CMake
4.2.3 in both host-only and Vulkan configurations. All 12 host-only and 13 Vulkan
CLI checks passed, covering settings, mods, background jobs, request validation,
startup and native delivery. A CUDA 13.3 + Vulkan build for `86;120a` also
compiled the executable and passed all 12 CLI command suites. Five corresponding
Windows regression suites passed. This is development-build validation, separate from
the glibc 2.35 release baseline and from GPU inference qualification.

Release baseline validation on 2026-11-02 used an isolated Ubuntu 22.04
filesystem with glibc 2.35, GCC 11.4 and CMake 3.31.6. The Vulkan build passed
17 host tests with three model/GPU integration tests skipped; its installed CLI
and shared library passed the ELF compatibility check. A CUDA 13.3 + Vulkan
build for `86;120a`, with the full tokenizer, passed the C API and tokenizer
tests and the same glibc check. That library was copied into Slopus. CUDA driver
stubs supported host tests only; this did not validate GPU generation.

Run tests without model weights or a GPU:

```sh
ctest --test-dir build/linux-vulkan --output-on-failure \
  -E '^(vulkan|kernels|tensor_backends)($|_)'
```

On a GPU host, run `ctest --test-dir build/linux-vulkan --output-on-failure` or
use the CUDA build directory. Suites without required hardware or checkpoints
report skipped tests. Full validation also requires real generation, reference
input, LoRA, continuation and output tests with the desired model files.

The Linux workflow builds host-only and Vulkan configurations, exercises CLI
(including settings, mods and agent jobs), host and C API tests, and checks
installation and packaging. FFmpeg is only an external conversion tool; there
is no FFmpeg build option or linked dependency. It embeds a small test tokenizer; its uploaded archives are test
builds and cannot tokenize real model prompts correctly. A manually enabled
CUDA job compiles CLI and C API against CUDA 12.8 for SM86. Hosted CI does not
claim GPU execution, native SM120a coverage or successful model generation.

Validation recorded on 2026-09-28 used WSL Ubuntu 26.04, GCC 15, CMake 4.2 and
CUDA 13.3:

- The CUDA + Vulkan configuration built all targets and passed nine host, C API
  and CUDA dispatch tests. CUDA-only CLI/C API builds with FFmpeg disabled also
  passed device enumeration and embedded-tokenizer smoke checks.
- Both synthetic-latent decoding and a full CUDA text-to-video smoke run
  produced 22-frame, 256x256 MP4 files with H.264 video and AAC audio. The full
  run used the embedded tokenizer, local NVFP4 Qwen/transformer checkpoints,
  video/audio VAEs and one denoising evaluation; it checks execution and output,
  not generation quality.
- The CUDA synthetic kernel suite completed 18,700 checks with one failure:
  `transformer_row_chunks_match_full_rows`. On this Linux CUDA 13.3 build,
  changing GEMM row dimensions changes a few Q/K projection values, reaching
  one BF16 step in the second refiner block and exceeding the final output
  tolerance. Identical checkpoint bytes pass on Windows. Restricting cuBLAS
  reductions did not resolve it; the original assertion remains unchanged.
  This toolchain's row-chunk numerical equivalence is not qualified.
- The Vulkan-only configuration passed seven of eight CTest suites; the model
  integration suite skipped because WSL exposed only llvmpipe software Vulkan.
  Portable arithmetic tests executed 33 checks through actual Vulkan shaders.
  No physical Vulkan GPU generation was validated. Full H3 inference also
  requires the supported BF16/FP16 cooperative-matrix operations and
  attention-specific subgroup/workgroup limits; a Vulkan device listing alone does not establish those capabilities.
- The Vulkan-only and combined CUDA/Vulkan packages passed all five packaging checks. After
  extraction elsewhere, the CLI tokenized with the embedded full vocabulary
  and the versioned shared library loaded and accepted both arithmetic policies.
  SONAME symlinks and relative library search paths were verified.
- Linux host tests exercised FFmpeg 8 image decoding, MP4 output and reference
  decoding. Four Windows host regression tests passed as well.

## WSL

Run the same Linux commands inside the WSL distribution, for example from
`/mnt/d/Projects/slopfab`. Keep Linux and Windows build directories separate.
From Windows, `build.cmd` builds the CLI and C API libraries into `dist/windows`
and `dist/linux`; `build.cmd -Target linux -Distro Ubuntu-22.04` builds only the
Linux executable and library. Release builds use Ubuntu 22.04 (pass `-Distro Ubuntu-22.04`). The build
directory defaults to `$HOME/slopfab-build-linux-glibc235` inside WSL and reuses its
cached CUDA compiler, or discovers `/usr/local/cuda/bin/nvcc` for a new build.
Override these with `-LinuxBuildDir /absolute/build/path` and
`-LinuxCudaCompiler /absolute/path/to/nvcc`. The same helper can be run natively
as `bash tools/build_linux_library.sh`. Matching headers, SHA-256 manifests and
build information accompany the executables and shared libraries; no model weights or runtime
dependencies are copied.

Use a build directory on the Linux filesystem, such as
`-B /tmp/slopfab-linux-vulkan`, when the source checkout is on a Windows drive;
CMake may be unable to set generated-file permissions on that mounted drive.
Check GPU availability inside WSL using `slopfab devices`; CUDA availability
does not imply that a usable Vulkan device is exposed. Use the toolkit installed
inside the Linux distribution for compilation and the WSL-compatible Windows
NVIDIA driver for CUDA device access.
