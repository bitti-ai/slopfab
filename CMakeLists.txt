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
option(SLOPFAB_TEST_BENCHMARKS "Register opt-in performance measurements with CTest" OFF)

# Kept as a compile-time compatibility macro for existing C++ callers/tests.
# Media conversion belongs to an external tool; no FFmpeg code is built.
set(SLOPFAB_WITH_FFMPEG OFF)

include(cmake/Core.cmake)
include(cmake/Tokenizer.cmake)
include(cmake/Cuda.cmake)
include(cmake/Vulkan.cmake)
include(cmake/Generation.cmake)
include(cmake/Whisper.cmake)
include(cmake/Demucs.cmake)
include(cmake/Cli.cmake)
include(cmake/CApi.cmake)
include(cmake/Tools.cmake)
include(cmake/Tests.cmake)
include(cmake/Install.cmake)
