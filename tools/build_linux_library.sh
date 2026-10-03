#!/usr/bin/env bash
# WSL half of build.cmd. Also callable directly from a Linux checkout.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$HOME/slopfab-build-linux"
cuda_compiler=
jobs=2
check_only=false
while (($#)); do
  case "$1" in
    --check) check_only=true; shift ;;
    --jobs|--build-dir|--cuda-compiler)
      if (($# < 2)); then printf 'build: %s needs a value\n' "$1" >&2; exit 2; fi
      case "$1" in
        --jobs) jobs=$2 ;;
        --build-dir) build_dir=$2 ;;
        --cuda-compiler) cuda_compiler=$2 ;;
      esac
      shift 2 ;;
    *) printf 'build: unknown Linux option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! $jobs =~ ^[1-9][0-9]*$ ]] || ((jobs > 64)); then
  printf 'build: --jobs must be 1..64\n' >&2; exit 2
fi
if [[ $build_dir != /* ]]; then
  printf 'build: --build-dir must be an absolute Linux path\n' >&2; exit 2
fi
for tool in cmake ctest c++; do
  command -v "$tool" >/dev/null || { printf 'build: install Linux %s\n' "$tool" >&2; exit 1; }
done
if [[ ! -f "$root/ref/text_encoder/tokenizer.json" ]]; then
  printf 'build: missing ref/text_encoder/tokenizer.json\n' >&2; exit 1
fi
if [[ -z $cuda_compiler && -f "$build_dir/CMakeCache.txt" ]]; then
  cuda_compiler=$(sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' "$build_dir/CMakeCache.txt")
fi
if [[ -z $cuda_compiler ]]; then
  cuda_compiler=${CUDACXX:-/usr/local/cuda/bin/nvcc}
  if [[ ! -x $cuda_compiler ]]; then cuda_compiler=$(command -v nvcc || true); fi
fi
if [[ $cuda_compiler != /* || ! -x $cuda_compiler ]]; then
  printf 'build: Linux CUDA compiler missing; pass -LinuxCudaCompiler /absolute/path/to/nvcc\n' >&2
  exit 1
fi
if $check_only; then
  printf 'build: Linux prerequisites found; CUDA compiler %s\n' "$cuda_compiler"
  exit 0
fi

generator=()
if [[ ! -f "$build_dir/CMakeCache.txt" ]] && command -v ninja >/dev/null; then
  generator=(-G Ninja)
fi
printf 'build: configuring Linux CUDA/Vulkan library in %s...\n' "$build_dir"
cmake -S "$root" -B "$build_dir" "${generator[@]}" \
  -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$cuda_compiler" \
  '-DCMAKE_CUDA_ARCHITECTURES=86;120a' \
  -DSLOPFAB_ENABLE_CUDA=ON -DSLOPFAB_ENABLE_VULKAN=ON \
  -DSLOPFAB_BUILD_C_API=ON -DSLOPFAB_BUILD_TESTS=ON \
  -DSLOPFAB_EMBED_TOKENIZER=ON "-DSLOPFAB_TOKENIZER_FILE=$root/ref/text_encoder/tokenizer.json"
cmake --build "$build_dir" --config Release --parallel "$jobs" \
  --target slopfab_c slopfab_capi_tests slopfab_embedded_tokenizer_tests
ctest --test-dir "$build_dir" -C Release --output-on-failure -R '^(capi|embedded_tokenizer)$'

major=$(awk '/^#define SLOPFAB_CAPI_VERSION_MAJOR / {print $3}' "$root/include/slopfab/capi.h" | tr -d '\r')
minor=$(awk '/^#define SLOPFAB_CAPI_VERSION_MINOR / {print $3}' "$root/include/slopfab/capi.h" | tr -d '\r')
patch=$(awk '/^#define SLOPFAB_CAPI_VERSION_PATCH / {print $3}' "$root/include/slopfab/capi.h" | tr -d '\r')
version="$major.$minor.$patch"
if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  printf 'build: cannot read the C API version\n' >&2; exit 1
fi
library="$build_dir/libslopfab.so.$version"
if [[ ! -f $library ]]; then printf 'build: missing library %s\n' "$library" >&2; exit 1; fi
destination="$root/dist/linux"
mkdir -p -- "$destination"
pending=
trap 'if [[ -n $pending ]]; then rm -f -- "$pending"; fi' EXIT
publish() {
  # Regular copies work on Windows-mounted drives without Linux symlink support.
  # Replacing the directory entry also leaves older hard-linked files intact.
  pending="$destination/.$2.pending-$$"
  cp -- "$1" "$pending"
  mv -f -- "$pending" "$destination/$2"
  pending=
}
publish "$library" "libslopfab.so.$version"
publish "$library" "libslopfab.so.$major"
publish "$library" libslopfab.so
publish "$root/include/slopfab/capi.h" capi.h
(
  cd -- "$destination"
  sha256sum "libslopfab.so.$version" "libslopfab.so.$major" libslopfab.so capi.h > SHA256SUMS
)
{
  printf 'slopfab C API %s\n' "$version"
  printf 'Linux x86_64 Release; CUDA and Vulkan enabled\n'
  printf 'CUDA targets: sm_86 and sm_120a; full tokenizer embedded\n'
  printf 'Source: %s\nBuild directory: %s\nCUDA compiler: %s\n' "$root" "$build_dir" "$cuda_compiler"
  printf 'Built UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Tests passed: capi, embedded_tokenizer\n'
  printf 'CUDA driver and matching toolkit shared libraries are external dependencies.\n'
} > "$destination/build-info.txt"
printf 'build: wrote %s/libslopfab.so (ABI %s)\n' "$destination" "$version"
