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
  src/core/nf4.cpp
  src/core/safetensors.cpp
  src/core/tensor_convert.cpp
  src/core/image.cpp
  src/core/safetensors_write.cpp
  src/dit/adaln.cpp
  src/dit/chunking.cpp
  src/dit/checkpoint.cpp
  src/dit/packing.cpp
  src/dit/ref2va.cpp
  src/dit/step_cache.cpp
  src/vae/keyframe_encoder.cpp
  src/pipeline.cpp
  src/sampler/noise.cpp
  src/sampler/scheduler.cpp
  src/text/tokenizer.cpp
  src/text/qwen_vision.cpp
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
    #
    # **120a, not 120.** The trailing `a` selects the architecture-specific
    # target, and it is required rather than preferred: the block-scaled fp4
    # tensor-core instruction this project needs for nvfp4 weights,
    #
    #   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale
    #       .scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
    #
    # is rejected by ptxas on plain `sm_120` with "Feature '.block_scale' not
    # supported on .target 'sm_120'", and assembles on `sm_120a`. `scale_vec::4X`
    # with `.ue4m3` scales is exactly the shipped checkpoint layout: 16 elements
    # per block, E4M3 scale, which is NVFP4 rather than MXFP4's 32-element
    # blocks and UE8M0 scales.
    #
    # The cost is that an `a` target emits no forward-compatible PTX, so the
    # binary runs on sm_120 hardware and nothing later. That is the right trade
    # here -- this project targets one RTX 5090 -- but it is a real constraint,
    # not a free upgrade.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "vidfab: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")

    add_library(vidfab_cuda STATIC
      src/cuda/device.cu
      src/cuda/profile.cpp
      src/cuda/workspace.cu
      src/cuda/nn_kernels.cu
      src/cuda/linear.cu
      src/cuda/nf4_weight.cu
      src/cuda/nvfp4_gemm.cu
      src/cuda/attention.cu
      src/cuda/sage_attention.cu
      src/cuda/vae_kernels.cu
      src/cuda/vit_decoder.cu
      src/cuda/audio_vae_kernels.cu
      src/cuda/dit_kernels.cu
      src/cuda/encoder_kernels.cu
      src/cuda/keyframe_encoder.cu
      src/cuda/qwen_vision.cu
      src/cuda/qwen_vision_encoder.cu
      # Host-side orchestration lives here rather than in vidfab_core so that
      # the dependency runs one way: cuda -> core, never back.
      src/vae/decode_pipeline.cpp
      src/vae/keyframe_cuda.cpp
      src/generate.cpp
      src/vae/audio_decoder.cpp
      src/dit/transformer.cpp
      src/dit/denoise.cpp
      src/text/encoder.cpp
    )
    target_include_directories(vidfab_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(vidfab_cuda PUBLIC CUDA::cublas)
    # Whole-program compilation: no device code crosses translation units, and
    # -rdc=true would otherwise require an explicit device-link step.
    set_target_properties(vidfab_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

    # The CUDA runtime is linked statically, which is deliberate: it keeps the
    # binary's runtime dependencies to the driver alone. The cost is a
    # cosmetic `LNK4098: defaultlib 'LIBCMT' conflicts` on MSVC, because
    # cudart_static.lib is built against the static CRT while this project uses
    # the dynamic one. The linker resolves it correctly and NVIDIA's own
    # samples emit the same warning; it is left visible rather than suppressed
    # with /NODEFAULTLIB, which would hide a real CRT conflict if one ever
    # appeared. Switching to CUDA_RUNTIME_LIBRARY=Shared would silence it at
    # the price of shipping cudart64_*.dll — the wrong trade for this project.
    target_link_libraries(vidfab_cuda PUBLIC vidfab_core)

    add_executable(vidfab_qwenvisionprobe tools/qwenvisionprobe.cpp)
    target_link_libraries(vidfab_qwenvisionprobe PRIVATE vidfab_cuda)
    add_executable(vidfab_qwenconditionerprobe tools/qwenconditionerprobe.cpp)
    target_link_libraries(vidfab_qwenconditionerprobe PRIVATE vidfab_cuda)

    target_compile_definitions(vidfab_core PUBLIC VIDFAB_WITH_CUDA=1)
  else()
    message(WARNING "vidfab: no CUDA compiler found; building CPU-only")
    set(VIDFAB_ENABLE_CUDA OFF)
  endif()
endif()

# --- cli --------------------------------------------------------------------

if(WIN32)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ref/text_encoder/tokenizer.json")
    message(FATAL_ERROR "vidfab: ref/text_encoder/tokenizer.json is required to embed the tokenizer")
  endif()
  set(VIDFAB_LICENSE_RESOURCE "${CMAKE_CURRENT_BINARY_DIR}/generated/minimax_h3_LICENSE")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
  file(DOWNLOAD
    "https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE"
    "${VIDFAB_LICENSE_RESOURCE}"
    STATUS VIDFAB_LICENSE_DOWNLOAD_STATUS
    TLS_VERIFY ON)
  list(GET VIDFAB_LICENSE_DOWNLOAD_STATUS 0 VIDFAB_LICENSE_DOWNLOAD_CODE)
  if(NOT VIDFAB_LICENSE_DOWNLOAD_CODE EQUAL 0)
    list(GET VIDFAB_LICENSE_DOWNLOAD_STATUS 1 VIDFAB_LICENSE_DOWNLOAD_MESSAGE)
    message(FATAL_ERROR "vidfab: could not download MiniMax H3 license: ${VIDFAB_LICENSE_DOWNLOAD_MESSAGE}")
  endif()
  file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ref/text_encoder/tokenizer.json"
       VIDFAB_TOKENIZER_RESOURCE)
  file(TO_CMAKE_PATH "${VIDFAB_LICENSE_RESOURCE}" VIDFAB_LICENSE_RESOURCE)
  configure_file(src/vidfab_resources.rc.in generated/vidfab_resources.rc @ONLY)
  add_executable(vidfab src/main.cpp "${CMAKE_CURRENT_BINARY_DIR}/generated/vidfab_resources.rc")
else()
  add_executable(vidfab src/main.cpp)
endif()
target_link_libraries(vidfab PRIVATE vidfab_core)
if(VIDFAB_ENABLE_CUDA)
  target_link_libraries(vidfab PRIVATE vidfab_cuda)
endif()

if(MSVC)
  target_compile_options(vidfab PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(vidfab PRIVATE -Wall -Wextra)
endif()
if(WIN32)
  target_link_libraries(vidfab PRIVATE winhttp)

  # FFmpeg remains runtime-loaded (and therefore replaceable), but the checked-in
  # distribution is the default used by local builds. Keep only the libraries
  # mux.cpp loads, plus swresample which avcodec depends on in this FFmpeg build.
  set(VIDFAB_FFMPEG_BIN "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin")
  set(VIDFAB_FFMPEG_RUNTIME_DLLS
    avcodec-62.dll
    avformat-62.dll
    avutil-60.dll
    swresample-6.dll
    swscale-9.dll)
  foreach(VIDFAB_FFMPEG_DLL IN LISTS VIDFAB_FFMPEG_RUNTIME_DLLS)
    if(NOT EXISTS "${VIDFAB_FFMPEG_BIN}/${VIDFAB_FFMPEG_DLL}")
      message(FATAL_ERROR
        "vidfab: required FFmpeg runtime is missing: ${VIDFAB_FFMPEG_BIN}/${VIDFAB_FFMPEG_DLL}")
    endif()
    add_custom_command(TARGET vidfab POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${VIDFAB_FFMPEG_BIN}/${VIDFAB_FFMPEG_DLL}"
        "$<TARGET_FILE_DIR:vidfab>/${VIDFAB_FFMPEG_DLL}"
      VERBATIM)
  endforeach()
endif()

# --- tools ------------------------------------------------------------------

# Disk-side load probe. Deliberately a separate binary rather than a `vidfab`
# subcommand: it links vidfab_core only, so it never initialises CUDA and can
# be run while the card is busy with something else.
add_executable(vidfab_loadprobe tools/loadprobe.cpp)
target_link_libraries(vidfab_loadprobe PRIVATE vidfab_core)
if(MSVC)
  target_compile_options(vidfab_loadprobe PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(vidfab_loadprobe PRIVATE -Wall -Wextra)
endif()

# Temporal-chunking driver for the frame-banding quality probe. Separate binary
# for the same reason as the load probe: it links vidfab_core only, so it
# initialises no CUDA context and reads no checkpoint, and it can be run while
# the card is busy with the generations it is preparing or analysing.
add_executable(vidfab_chunkprobe tools/chunkprobe.cpp)
target_link_libraries(vidfab_chunkprobe PRIVATE vidfab_core)
if(MSVC)
  target_compile_options(vidfab_chunkprobe PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(vidfab_chunkprobe PRIVATE -Wall -Wextra)
endif()

add_executable(vidfab_vaeprobe tools/vaeprobe.cpp)
target_link_libraries(vidfab_vaeprobe PRIVATE vidfab_core)
if(MSVC)
  target_compile_options(vidfab_vaeprobe PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(vidfab_vaeprobe PRIVATE -Wall -Wextra)
endif()

if(VIDFAB_ENABLE_CUDA)
  add_executable(vidfab_vaecudaprobe tools/vaecudaprobe.cpp)
  target_link_libraries(vidfab_vaecudaprobe PRIVATE vidfab_cuda)
  if(MSVC)
    target_compile_options(vidfab_vaecudaprobe PRIVATE /W4 /permissive- /utf-8)
  else()
    target_compile_options(vidfab_vaecudaprobe PRIVATE -Wall -Wextra)
  endif()
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
    tests/test_adaln.cpp
    tests/test_chunking.cpp
    tests/test_compare_metrics.cpp
    tests/test_image.cpp
    tests/test_noise.cpp
    tests/test_packing.cpp
    tests/test_ref2va.cpp
    tests/test_pipeline.cpp
    tests/test_sampler.cpp
    tests/test_step_cache.cpp
    tests/test_tokenizer.cpp
    tests/test_qwen_vision.cpp
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
      tests/test_transformer.cu
      tests/test_encoder.cu
      tests/test_keyframe_encoder.cu
    )
    target_link_libraries(vidfab_kernel_tests PRIVATE vidfab_cuda)
    add_test(NAME kernels COMMAND vidfab_kernel_tests)
  endif()
endif()


