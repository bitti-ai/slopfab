cmake_minimum_required(VERSION 3.24)

# CUDA is enabled lazily: the core loader and CLI build without it so the
# project stays useful on machines without a toolkit, and so early phases can
# be developed and tested before any kernels exist.
project(slopfab LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SLOPFAB_ENABLE_CUDA "Build CUDA kernels" ON)
option(SLOPFAB_ENABLE_VULKAN "Build the standalone Vulkan runtime foundation" ON)
option(SLOPFAB_BUILD_TESTS "Build unit tests" ON)
option(SLOPFAB_BUILD_DEV_TOOLS "Build developer profiling/probe executables" OFF)

# FFmpeg is loaded at runtime and never linked, so this option does not change
# what the binary depends on to *start* — it changes what is compiled in at
# all. Off, the muxer is replaced by a stub, MP4 output is gone, and reference
# images are decoded by the platform (WIC on Windows) instead of by libavcodec.
# `slopfab` then writes .y4m + .wav, which is the same fallback it already takes
# on a machine where FFmpeg is not installed.
#
# It is one switch for the whole build tree rather than per target, and that is
# a real limitation worth stating: slopfab_cuda links slopfab_core, so an
# FFmpeg-free DLL beside an MP4-capable exe would need a second copy of both
# libraries, and slopfab_cuda costs minutes to compile. The C API does not need
# the distinction anyway — it hands the caller pixels and never reaches the
# muxer in either configuration — so the choice only decides what `slopfab.exe`
# can write. Configure with -DSLOPFAB_WITH_FFMPEG=OFF to build the DLL with no
# FFmpeg code in the process at all.
option(SLOPFAB_WITH_FFMPEG "Compile MP4 muxing and FFmpeg-based image decoding" ON)

include(cmake/Core.cmake)
include(cmake/Cuda.cmake)
include(cmake/Vulkan.cmake)
include(cmake/Cli.cmake)
include(cmake/CApi.cmake)
include(cmake/Tools.cmake)
include(cmake/Tests.cmake)
include(cmake/Install.cmake)
