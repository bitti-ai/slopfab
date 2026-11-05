#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
backend=both
build_dir=
windows_dir="$root/dist/windows"
cmake_args=()
while (($#)); do
  case "$1" in
    --backend|--build-dir|--windows-dir)
      if (($# < 2)); then printf 'package: %s needs a value\n' "$1" >&2; exit 2; fi
      if [[ $1 == --backend ]]; then backend=$2; elif [[ $1 == --windows-dir ]]; then windows_dir=$2; else build_dir=$2; fi
      shift 2 ;;
    --help|-h)
      printf '%s\n' 'Usage: bash package.sh [--backend both|cuda|vulkan] [--build-dir DIR] [--windows-dir DIR] [-- CMAKE_ARGS...]' 'A prebuilt Windows slopfab.exe is required; every release contains both platforms.'
      exit 0 ;;
    --) shift; cmake_args=("$@"); break ;;
    *) printf 'package: unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! -f "$windows_dir/slopfab.exe" ]]; then
  printf 'package: both platforms are required; provide --windows-dir containing slopfab.exe.\n' >&2
  exit 1
fi
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
python3 "$root/tools/check_glibc.py" --check-host
build_dir=${build_dir:-"$root/build/linux-glibc235-$backend"}
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
# WSL checkouts can live on a Windows drive where CMake cannot install ELF
# permissions or shared-library symlinks. Stage on the Linux filesystem and
# copy only the completed archive back to the checkout.
package_dir=$(mktemp -d /tmp/slopfab-package.XXXXXXXX)
trap 'rm -rf -- "$package_dir"' EXIT
cmake --install "$build_dir" --config Release --prefix "$package_dir/check"
python3 "$root/tools/check_glibc.py" "$package_dir/check"
mkdir -p -- "$package_dir/linux"
cp -- "$package_dir/check/bin/slopfab" "$package_dir/linux/slopfab"
cp -L -- "$package_dir/check/lib/"libslopfab.so* "$package_dir/linux/"
cp -- "$root/include/slopfab/capi.h" "$package_dir/linux/capi.h"
python3 "$root/tools/package_bundle.py" --windows-dir "$windows_dir" --linux-dir "$package_dir/linux"
