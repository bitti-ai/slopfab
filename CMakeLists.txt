cmake_minimum_required(VERSION 3.24)

# CUDA is enabled lazily: the core loader and CLI build without it so the
# project stays useful on machines without a toolkit, and so early phases can
# be developed and tested before any kernels exist.
project(vidfab LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(VIDFAB_ENABLE_CUDA "Build CUDA kernels" ON)
option(VIDFAB_BUILD_TESTS "Build unit tests" ON)

# --- core -------------------------------------------------------------------

add_library(vidfab_core STATIC
  src/core/json.cpp
  src/core/safetensors.cpp
  src/core/tensor_convert.cpp
  src/core/safetensors_write.cpp
  src/dit/packing.cpp
  src/pipeline.cpp
  src/sampler/scheduler.cpp
  src/text/tokenizer.cpp
  src/video/y4m.cpp
  src/video/mux.cpp
  src/audio/wav.cpp
)

target_include_directories(vidfab_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
  # /utf-8 keeps string literals well-defined regardless of the system codepage.
  target_compile_options(vidfab_core PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(vidfab_core PRIVATE -Wall -Wextra)
endif()

# --- cuda -------------------------------------------------------------------

if(VIDFAB_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Must be set before enable_language(CUDA), which otherwise picks a
    # conservative default. 90 = Hopper, 120 = Blackwell (RTX 50 series), which
    # is the primary target and the only one of the two with native fp4.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "vidfab: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")

    add_library(vidfab_cuda STATIC
      src/cuda/device.cu
      src/cuda/workspace.cu
      src/cuda/nn_kernels.cu
      src/cuda/linear.cu
      src/cuda/attention.cu
      src/cuda/vae_kernels.cu
      src/cuda/vit_decoder.cu
      src/cuda/audio_vae_kernels.cu
      # Host-side orchestration lives here rather than in vidfab_core so that
      # the dependency runs one way: cuda -> core, never back.
      src/vae/decode_pipeline.cpp
      src/vae/audio_decoder.cpp
    )
    target_include_directories(vidfab_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(vidfab_cuda PUBLIC CUDA::cublas)
    # Whole-program compilation: no device code crosses translation units, and
    # -rdc=true would otherwise require an explicit device-link step.
    set_target_properties(vidfab_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_link_libraries(vidfab_cuda PUBLIC vidfab_core)

    target_compile_definitions(vidfab_core PUBLIC VIDFAB_WITH_CUDA=1)
  else()
    message(WARNING "vidfab: no CUDA compiler found; building CPU-only")
    set(VIDFAB_ENABLE_CUDA OFF)
  endif()
endif()

# --- cli --------------------------------------------------------------------

add_executable(vidfab src/main.cpp)
target_link_libraries(vidfab PRIVATE vidfab_core)
if(VIDFAB_ENABLE_CUDA)
  target_link_libraries(vidfab PRIVATE vidfab_cuda)
endif()

if(MSVC)
  target_compile_options(vidfab PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(vidfab PRIVATE -Wall -Wextra)
endif()

# --- tests ------------------------------------------------------------------

if(VIDFAB_BUILD_TESTS)
  enable_testing()
  # Host tests. Each area contributes its own translation unit and registers
  # its cases with the shared harness, so adding tests never touches a file
  # someone else is editing.
  add_executable(vidfab_tests
    tests/harness.cpp
    tests/test_main.cpp
    tests/test_packing.cpp
    tests/test_pipeline.cpp
    tests/test_tokenizer.cpp
    tests/test_output.cpp
  )
  target_link_libraries(vidfab_tests PRIVATE vidfab_core)
  if(MSVC)
    target_compile_options(vidfab_tests PRIVATE /W4 /permissive- /utf-8)
  else()
    target_compile_options(vidfab_tests PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME unit COMMAND vidfab_tests)

  if(VIDFAB_ENABLE_CUDA)
    add_executable(vidfab_kernel_tests
      tests/harness.cpp
      tests/test_kernels.cu
      tests/test_nn_kernels.cu
      tests/test_audio_vae.cu
    )
    target_link_libraries(vidfab_kernel_tests PRIVATE vidfab_cuda)
    add_test(NAME kernels COMMAND vidfab_kernel_tests)
  endif()
endif()


