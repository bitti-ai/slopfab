#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
backend=both
build_dir=
cmake_args=()
while (($#)); do
  case "$1" in
    --backend|--build-dir)
      if (($# < 2)); then printf 'package: %s needs a value\n' "$1" >&2; exit 2; fi
      if [[ $1 == --backend ]]; then backend=$2; else build_dir=$2; fi
      shift 2 ;;
    --help|-h)
      printf '%s\n' 'Usage: bash package.sh [--backend both|cuda|vulkan] [--build-dir DIR] [-- CMAKE_ARGS...]'
      exit 0 ;;
    --) shift; cmake_args=("$@"); break ;;
    *) printf 'package: unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$backend" in
  both) cuda=ON; vulkan=ON ;;
  cuda) cuda=ON; vulkan=OFF ;;
  vulkan) cuda=OFF; vulkan=ON ;;
  *) printf 'package: invalid backend: %s\n' "$backend" >&2; exit 2 ;;
esac
if [[ $(uname -s) != Linux ]]; then
  printf 'package: run this script on Linux (including WSL).\n' >&2
  exit 2
fi
build_dir=${build_dir:-"$root/build/linux-$backend"}
cmake -S "$root" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib \
  -DSLOPFAB_ENABLE_CUDA="$cuda" -DSLOPFAB_ENABLE_VULKAN="$vulkan" \
  -DSLOPFAB_BUILD_C_API=ON -DSLOPFAB_BUILD_TESTS=ON "${cmake_args[@]}"
if [[ $cuda == ON ]] && grep -Eq '^CMAKE_CUDA_COMPILER:[^=]*=(|.*NOTFOUND)$' "$build_dir/CMakeCache.txt"; then
  printf 'package: CUDA packaging requires a working CUDA compiler; set CMAKE_CUDA_COMPILER.\n' >&2
  exit 1
fi
cmake --build "$build_dir" --config Release --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
# Model and GPU integration suites are run separately on a suitable host.
ctest --test-dir "$build_dir" -C Release --output-on-failure -R '^(unit|capi|generation_backends|embedded_tokenizer|reference_media_decode)$'
mkdir -p -- "$root/dist"
cpack --config "$build_dir/CPackConfig.cmake" -C Release -B "$root/dist"
printf 'package: wrote Linux archive under %s/dist (model weights and GPU/media runtimes are external).\n' "$root"
