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
option(VIDFAB_ENABLE_VULKAN "Build the standalone Vulkan runtime foundation" ON)
option(VIDFAB_BUILD_TESTS "Build unit tests" ON)
option(VIDFAB_BUILD_DEV_TOOLS "Build developer profiling/probe executables" OFF)

# FFmpeg is loaded at runtime and never linked, so this option does not change
# what the binary depends on to *start* — it changes what is compiled in at
# all. Off, the muxer is replaced by a stub, MP4 output is gone, and reference
# images are decoded by the platform (WIC on Windows) instead of by libavcodec.
# `vidfab` then writes .y4m + .wav, which is the same fallback it already takes
# on a machine where FFmpeg is not installed.
#
# It is one switch for the whole build tree rather than per target, and that is
# a real limitation worth stating: vidfab_cuda links vidfab_core, so an
# FFmpeg-free DLL beside an MP4-capable exe would need a second copy of both
# libraries, and vidfab_cuda costs minutes to compile. The C API does not need
# the distinction anyway — it hands the caller pixels and never reaches the
# muxer in either configuration — so the choice only decides what `vidfab.exe`
# can write. Configure with -DVIDFAB_WITH_FFMPEG=OFF to build the DLL with no
# FFmpeg code in the process at all.
option(VIDFAB_WITH_FFMPEG "Compile MP4 muxing and FFmpeg-based image decoding" ON)

# --- core -------------------------------------------------------------------

add_library(vidfab_core STATIC
  src/core/json.cpp
  src/core/nf4.cpp
  src/core/safetensors.cpp
  src/core/tensor_convert.cpp
  src/core/device_tensor.cpp
  src/core/image.cpp
  src/core/image_platform.cpp
  src/core/safetensors_write.cpp
  src/dit/adaln.cpp
  src/dit/block_cache.cpp
  src/dit/chunking.cpp
  src/dit/checkpoint.cpp
  src/dit/packing.cpp
  src/dit/ref2va.cpp
  src/dit/rope.cpp
  src/dit/step_cache.cpp
  src/vae/keyframe_encoder.cpp
  src/pipeline.cpp
  src/sampler/noise.cpp
  src/sampler/scheduler.cpp
  src/text/tokenizer.cpp
  src/text/qwen_vision.cpp
  src/video/y4m.cpp
  src/video/y4m_compare.cpp
  src/audio/wav.cpp
)

# The muxer, or the stub that stands in for it. Exactly one of the two is
# compiled: they define the same four functions, so every call site links
# against whichever this build has and none of them needs an #if.
if(VIDFAB_WITH_FFMPEG)
  target_sources(vidfab_core PRIVATE src/video/mux.cpp)
else()
  target_sources(vidfab_core PRIVATE src/video/mux_disabled.cpp)
endif()

# PUBLIC because headers branch on it: media.h only declares the demuxer where
# it exists, and image.cpp picks its decoder with it. A consumer that saw a
# different value than this library was built with would disagree about which
# functions exist.
target_compile_definitions(vidfab_core PUBLIC
  VIDFAB_WITH_FFMPEG=$<IF:$<BOOL:${VIDFAB_WITH_FFMPEG}>,1,0>)

target_include_directories(vidfab_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# The platform image decoder. Compiled in every configuration — so the two
# decoders can be compared and the code cannot rot unbuilt — which is why
# these are linked whether or not FFmpeg is on. Both ship with Windows.
if(WIN32)
  target_link_libraries(vidfab_core PRIVATE windowscodecs ole32)
endif()

# The RGB -> YUV 4:2:0 conversion in src/video/y4m.cpp is threaded. MSVC needs
# nothing for std::thread, but every other toolchain wants -pthread both to
# compile and to link, and without this the non-MSVC build breaks. Requested
# here as well as in the CUDA block below because vidfab_core must build on its
# own, on a machine with no CUDA toolkit at all.
find_package(Threads REQUIRED)
target_link_libraries(vidfab_core PUBLIC Threads::Threads)

if(MSVC)
  # /utf-8 keeps string literals well-defined regardless of the system codepage.
  target_compile_options(vidfab_core PRIVATE /W4 /permissive- /utf-8 /EHsc)
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
    # The VAE decode's final de-normalise is the project's first std::thread
    # use. MSVC needs nothing for it, but every other toolchain wants -pthread
    # both to compile and to link, and without this the non-MSVC build breaks.
    find_package(Threads REQUIRED)
    message(STATUS "vidfab: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")

    add_library(vidfab_cuda STATIC
      src/cuda/device.cu
      src/cuda/diagnostics.cu
      src/cuda/profile.cpp
      src/cuda/workspace.cu
      src/cuda/nn_kernels.cu
      src/cuda/linear.cu
      src/cuda/nf4_weight.cu
      src/cuda/nvfp4_gemm.cu
      src/cuda/attention.cu
      src/cuda/sage_attention.cu
      src/cuda/sol_attention.cu
      src/cuda/sol_attention_pipeline.cu
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
      # src/capi/capi.cpp is deliberately NOT here. It calls run_generate, so
      # it depends on this target — but compiling it *into* this target as well
      # as into vidfab_c builds every C entry point twice: once with capi.h
      # seen as dllimport (here, MSVC C4273 "inconsistent dll linkage") and
      # once as dllexport (there). The link happens to pick the right copy, so
      # the failure is silent, and it would leave the C ABI exported from a
      # library that is otherwise pure C++. It belongs to vidfab_c alone; see
      # the c api section below.
      src/vae/audio_decoder.cpp
      src/dit/transformer.cpp
      src/dit/denoise.cpp
      src/text/encoder.cpp
    )
    target_include_directories(vidfab_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(vidfab_cuda PUBLIC CUDA::cublas CUDA::cuda_driver Threads::Threads)
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
    #
    # CUDA::cudart_static is public because plain-C++ consumers such as
    # qwenvisionprobe call the runtime API without compiling a CUDA source of
    # their own and therefore do not get its link dependency automatically.
    target_link_libraries(vidfab_cuda PUBLIC vidfab_core CUDA::cudart_static)

    add_executable(vidfab_qwenvisionprobe tools/qwenvisionprobe.cpp)
    target_link_libraries(vidfab_qwenvisionprobe PRIVATE vidfab_cuda)
    add_executable(vidfab_qwenconditionerprobe tools/qwenconditionerprobe.cpp)
    target_link_libraries(vidfab_qwenconditionerprobe PRIVATE vidfab_cuda)
    if(VIDFAB_BUILD_DEV_TOOLS)
      add_executable(vidfab_solbench tools/solbench.cu)
      target_link_libraries(vidfab_solbench PRIVATE vidfab_cuda)
    endif()

    target_compile_definitions(vidfab_core PUBLIC VIDFAB_WITH_CUDA=1)
  else()
    message(WARNING "vidfab: no CUDA compiler found; building CPU-only")
    set(VIDFAB_ENABLE_CUDA OFF)
  endif()
endif()

# --- vulkan -----------------------------------------------------------------

# Vulkan is independent of CUDA and implements the core frame-converter
# interface. The Khronos core ABI headers are pinned in-tree, while the
# platform loader is opened dynamically by runtime.cpp. Consequently enabling
# this target needs no Vulkan SDK at build time or Vulkan loader at process
# startup.
if(VIDFAB_ENABLE_VULKAN)
  set(VIDFAB_YUV_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/rgb_to_yuv.comp")
  set(VIDFAB_YUV_SHADER_SPV "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/rgb_to_yuv.comp.spv")
  set(VIDFAB_TENSOR_OPS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_ops.comp")
  set(VIDFAB_TENSOR_OPS_SPV "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_ops.comp.spv")
  set(VIDFAB_TENSOR_OPS_DENORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_ops_denorm.comp.spv")
  set(VIDFAB_TENSOR_NORM_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_norm.comp")
  set(VIDFAB_TENSOR_RMS_NORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_rms_norm.comp.spv")
  set(VIDFAB_TENSOR_LAYER_NORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_layer_norm.comp.spv")
  set(VIDFAB_TENSOR_SHARED_NORM_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_shared_norm.comp")
  set(VIDFAB_TENSOR_BF16_RMS_BLOCK_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_bf16_rms_block.comp.spv")
  set(VIDFAB_TENSOR_BF16_RMS_NARROW_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_bf16_rms_narrow.comp.spv")
  set(VIDFAB_TENSOR_BF16_LAYER_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_bf16_layer.comp.spv")
  set(VIDFAB_TENSOR_BF16_MOD_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_bf16_mod.comp.spv")
  set(VIDFAB_TENSOR_FP32_MOD_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_fp32_mod.comp.spv")
  set(VIDFAB_TENSOR_GROUP_NORM_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_group_norm.comp")
  set(VIDFAB_TENSOR_GROUP_NORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_group_norm.comp.spv")
  set(VIDFAB_TENSOR_ROPE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_rope.comp")
  set(VIDFAB_TENSOR_ROPE_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_rope.comp.spv")
  set(VIDFAB_TENSOR_ROPE_DENORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_rope_denorm.comp.spv")
  file(READ "${VIDFAB_YUV_SHADER_SOURCE}" VIDFAB_YUV_SHADER_TEXT)
  string(REPLACE "\r\n" "\n" VIDFAB_YUV_SHADER_TEXT "${VIDFAB_YUV_SHADER_TEXT}")
  string(SHA256 VIDFAB_YUV_SHADER_SOURCE_HASH "${VIDFAB_YUV_SHADER_TEXT}")
  file(SHA256 "${VIDFAB_YUV_SHADER_SPV}" VIDFAB_YUV_SHADER_SPV_HASH)
  if(NOT VIDFAB_YUV_SHADER_SOURCE_HASH STREQUAL
      "77fa686e8c2ccca0b62551211defb4d47dd941dd617627efce56df35637d2c37" OR
     NOT VIDFAB_YUV_SHADER_SPV_HASH STREQUAL
      "c4cee251c15f54371387d2dbacb4637c76c0dbc5dc6c0663c1e1269c703e71e6")
    message(FATAL_ERROR
      "vidfab: RGB-to-YUV shader/source hash mismatch; regenerate with glslang 16.5.0 "
      "and update both pinned hashes")
  endif()
  file(READ "${VIDFAB_YUV_SHADER_SPV}" VIDFAB_YUV_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," VIDFAB_YUV_SPV_BYTES
    "${VIDFAB_YUV_SPV_HEX}")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan")
  configure_file(src/vulkan/embedded_yuv_spv.h.in
    generated/vulkan/embedded_yuv_spv.h @ONLY)
  file(READ "${VIDFAB_TENSOR_OPS_SOURCE}" VIDFAB_TENSOR_OPS_TEXT)
  string(REPLACE "\r\n" "\n" VIDFAB_TENSOR_OPS_TEXT "${VIDFAB_TENSOR_OPS_TEXT}")
  string(SHA256 VIDFAB_TENSOR_OPS_SOURCE_HASH "${VIDFAB_TENSOR_OPS_TEXT}")
  file(SHA256 "${VIDFAB_TENSOR_OPS_SPV}" VIDFAB_TENSOR_OPS_SPV_HASH)
  file(SHA256 "${VIDFAB_TENSOR_OPS_DENORM_SPV}" VIDFAB_TENSOR_OPS_DENORM_SPV_HASH)
  if(NOT VIDFAB_TENSOR_OPS_SOURCE_HASH STREQUAL
      "a5e10d2ff909dc131e0e38e4bf5927df1bc3a9fbb02f283bf5118942dfcaa806" OR
     NOT VIDFAB_TENSOR_OPS_SPV_HASH STREQUAL
      "a11b87b00658e97b8a3b53a24618045eadbbfa929a44ad9bc08997f958fa2106" OR
     NOT VIDFAB_TENSOR_OPS_DENORM_SPV_HASH STREQUAL
      "e35519ece4af1021bf5056b100fae32d8c3a8757fc0d439c9eaeda9613f92aa7")
    message(FATAL_ERROR "vidfab: tensor-ops shader/source hash mismatch")
  endif()
  file(READ "${VIDFAB_TENSOR_NORM_SOURCE}" VIDFAB_TENSOR_NORM_TEXT)
  string(REPLACE "\r\n" "\n" VIDFAB_TENSOR_NORM_TEXT "${VIDFAB_TENSOR_NORM_TEXT}")
  string(SHA256 VIDFAB_TENSOR_NORM_SOURCE_HASH "${VIDFAB_TENSOR_NORM_TEXT}")
  file(SHA256 "${VIDFAB_TENSOR_RMS_NORM_SPV}" VIDFAB_TENSOR_RMS_NORM_SPV_HASH)
  file(SHA256 "${VIDFAB_TENSOR_LAYER_NORM_SPV}" VIDFAB_TENSOR_LAYER_NORM_SPV_HASH)
  if(NOT VIDFAB_TENSOR_NORM_SOURCE_HASH STREQUAL
      "f82042e4d335da14e442abd2dde6df877b832bccc8c14033d2aa22f7ed03c325" OR
     NOT VIDFAB_TENSOR_RMS_NORM_SPV_HASH STREQUAL
      "160b9d0f6260bc912379c26171dd8d0c60ad6e4d9b81da57a413b831e513229c" OR
     NOT VIDFAB_TENSOR_LAYER_NORM_SPV_HASH STREQUAL
      "516ecc94d93cd829f6f6b208acfb3ff647861daae9db3017b31729528d8724e5")
    message(FATAL_ERROR "vidfab: tensor-norm shader/source hash mismatch")
  endif()
  file(READ "${VIDFAB_TENSOR_SHARED_NORM_SOURCE}" VIDFAB_TENSOR_SHARED_NORM_TEXT)
  string(REPLACE "\r\n" "\n" VIDFAB_TENSOR_SHARED_NORM_TEXT
    "${VIDFAB_TENSOR_SHARED_NORM_TEXT}")
  string(SHA256 VIDFAB_TENSOR_SHARED_NORM_SOURCE_HASH
    "${VIDFAB_TENSOR_SHARED_NORM_TEXT}")
  file(SHA256 "${VIDFAB_TENSOR_BF16_RMS_BLOCK_SPV}"
    VIDFAB_TENSOR_BF16_RMS_BLOCK_SPV_HASH)
  file(SHA256 "${VIDFAB_TENSOR_BF16_RMS_NARROW_SPV}"
    VIDFAB_TENSOR_BF16_RMS_NARROW_SPV_HASH)
  file(SHA256 "${VIDFAB_TENSOR_BF16_LAYER_SPV}" VIDFAB_TENSOR_BF16_LAYER_SPV_HASH)
  file(SHA256 "${VIDFAB_TENSOR_BF16_MOD_SPV}" VIDFAB_TENSOR_BF16_MOD_SPV_HASH)
  file(SHA256 "${VIDFAB_TENSOR_FP32_MOD_SPV}" VIDFAB_TENSOR_FP32_MOD_SPV_HASH)
  if(NOT VIDFAB_TENSOR_SHARED_NORM_SOURCE_HASH STREQUAL
      "6435b32e3c0272ea0f20695ca38e993e01bbe8b16688258c2c79bc5845cfa972" OR
     NOT VIDFAB_TENSOR_BF16_RMS_BLOCK_SPV_HASH STREQUAL
      "5784286fc645492fe7f9b68e292f47e1d8b6efd5ff3bd213bef003559f5eb49f" OR
     NOT VIDFAB_TENSOR_BF16_RMS_NARROW_SPV_HASH STREQUAL
      "5473b833ff74234c38817535f57b33de46534aa678e5dcfe0f562a0b4f1779a2" OR
     NOT VIDFAB_TENSOR_BF16_LAYER_SPV_HASH STREQUAL
      "5f33ab2d86663d2611d64cde7a62f27b13471256c808d067b80cf2baeac5c96a" OR
     NOT VIDFAB_TENSOR_BF16_MOD_SPV_HASH STREQUAL
      "7c6634df12153142fc0c17d9a7ec89275e9bfde45ec7c0249e8353b7685f16ae" OR
     NOT VIDFAB_TENSOR_FP32_MOD_SPV_HASH STREQUAL
      "c744ee45acf2e34c017a1d1ec83be5bc83042ba29aac95c70ae864d55198f5ec")
    message(FATAL_ERROR "vidfab: shared-norm shader/source hash mismatch")
  endif()
  file(READ "${VIDFAB_TENSOR_GROUP_NORM_SOURCE}" VIDFAB_TENSOR_GROUP_NORM_TEXT)
  string(REPLACE "\r\n" "\n" VIDFAB_TENSOR_GROUP_NORM_TEXT
    "${VIDFAB_TENSOR_GROUP_NORM_TEXT}")
  string(SHA256 VIDFAB_TENSOR_GROUP_NORM_SOURCE_HASH
    "${VIDFAB_TENSOR_GROUP_NORM_TEXT}")
  file(SHA256 "${VIDFAB_TENSOR_GROUP_NORM_SPV}"
    VIDFAB_TENSOR_GROUP_NORM_SPV_HASH)
  if(NOT VIDFAB_TENSOR_GROUP_NORM_SOURCE_HASH STREQUAL
      "988faec12f82a744a49cbb5e46aae35b5563f408a7809e72652e4b67ed1457ca" OR
     NOT VIDFAB_TENSOR_GROUP_NORM_SPV_HASH STREQUAL
      "57aed42788d1d8ca9a3f6ede8d6130b2c32fda2d62c2261fe00f1d3ff2e98d33")
    message(FATAL_ERROR "vidfab: group-norm shader/source hash mismatch")
  endif()
  file(READ "${VIDFAB_TENSOR_ROPE_SOURCE}" VIDFAB_TENSOR_ROPE_TEXT)
  string(REPLACE "\r\n" "\n" VIDFAB_TENSOR_ROPE_TEXT "${VIDFAB_TENSOR_ROPE_TEXT}")
  string(SHA256 VIDFAB_TENSOR_ROPE_SOURCE_HASH "${VIDFAB_TENSOR_ROPE_TEXT}")
  file(SHA256 "${VIDFAB_TENSOR_ROPE_SPV}" VIDFAB_TENSOR_ROPE_SPV_HASH)
  file(SHA256 "${VIDFAB_TENSOR_ROPE_DENORM_SPV}"
    VIDFAB_TENSOR_ROPE_DENORM_SPV_HASH)
  if(NOT VIDFAB_TENSOR_ROPE_SOURCE_HASH STREQUAL
      "c4189ad9924d0ce8ba3d44a80c6a4852bd5e6f3860fcee464b188851e4276858" OR
     NOT VIDFAB_TENSOR_ROPE_SPV_HASH STREQUAL
      "0138044bf306ca344965dffefba8271fc88cae8ae8f387a01159aa04d7d0a47c" OR
     NOT VIDFAB_TENSOR_ROPE_DENORM_SPV_HASH STREQUAL
      "8e9540823b776e5e5dc92c5244aceba9cb2921da1c9816f8cb9c2477d78e3006")
    message(FATAL_ERROR "vidfab: RoPE shader/source hash mismatch")
  endif()
  file(READ "${VIDFAB_TENSOR_OPS_SPV}" VIDFAB_TENSOR_OPS_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," VIDFAB_TENSOR_OPS_SPV_BYTES
    "${VIDFAB_TENSOR_OPS_SPV_HEX}")
  file(READ "${VIDFAB_TENSOR_OPS_DENORM_SPV}" VIDFAB_TENSOR_OPS_DENORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    VIDFAB_TENSOR_OPS_DENORM_SPV_BYTES "${VIDFAB_TENSOR_OPS_DENORM_SPV_HEX}")
  file(READ "${VIDFAB_TENSOR_RMS_NORM_SPV}" VIDFAB_TENSOR_RMS_NORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," VIDFAB_TENSOR_RMS_NORM_SPV_BYTES
    "${VIDFAB_TENSOR_RMS_NORM_SPV_HEX}")
  file(READ "${VIDFAB_TENSOR_LAYER_NORM_SPV}" VIDFAB_TENSOR_LAYER_NORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    VIDFAB_TENSOR_LAYER_NORM_SPV_BYTES "${VIDFAB_TENSOR_LAYER_NORM_SPV_HEX}")
  foreach(VARIANT BF16_RMS_BLOCK BF16_RMS_NARROW BF16_LAYER BF16_MOD FP32_MOD)
    file(READ "${VIDFAB_TENSOR_${VARIANT}_SPV}"
      VIDFAB_TENSOR_${VARIANT}_SPV_HEX HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
      VIDFAB_TENSOR_${VARIANT}_SPV_BYTES "${VIDFAB_TENSOR_${VARIANT}_SPV_HEX}")
  endforeach()
  file(READ "${VIDFAB_TENSOR_GROUP_NORM_SPV}"
    VIDFAB_TENSOR_GROUP_NORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    VIDFAB_TENSOR_GROUP_NORM_SPV_BYTES "${VIDFAB_TENSOR_GROUP_NORM_SPV_HEX}")
  file(READ "${VIDFAB_TENSOR_ROPE_SPV}" VIDFAB_TENSOR_ROPE_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    VIDFAB_TENSOR_ROPE_SPV_BYTES "${VIDFAB_TENSOR_ROPE_SPV_HEX}")
  file(READ "${VIDFAB_TENSOR_ROPE_DENORM_SPV}"
    VIDFAB_TENSOR_ROPE_DENORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    VIDFAB_TENSOR_ROPE_DENORM_SPV_BYTES "${VIDFAB_TENSOR_ROPE_DENORM_SPV_HEX}")
  configure_file(src/vulkan/embedded_tensor_spv.h.in
    generated/vulkan/embedded_tensor_spv.h @ONLY)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${VIDFAB_YUV_SHADER_SOURCE}" "${VIDFAB_YUV_SHADER_SPV}"
    "${VIDFAB_TENSOR_OPS_SOURCE}"
    "${VIDFAB_TENSOR_OPS_SPV}" "${VIDFAB_TENSOR_OPS_DENORM_SPV}"
    "${VIDFAB_TENSOR_NORM_SOURCE}" "${VIDFAB_TENSOR_RMS_NORM_SPV}"
    "${VIDFAB_TENSOR_LAYER_NORM_SPV}" "${VIDFAB_TENSOR_SHARED_NORM_SOURCE}"
    "${VIDFAB_TENSOR_BF16_RMS_BLOCK_SPV}" "${VIDFAB_TENSOR_BF16_RMS_NARROW_SPV}"
    "${VIDFAB_TENSOR_BF16_LAYER_SPV}" "${VIDFAB_TENSOR_BF16_MOD_SPV}"
    "${VIDFAB_TENSOR_FP32_MOD_SPV}" "${VIDFAB_TENSOR_GROUP_NORM_SOURCE}"
    "${VIDFAB_TENSOR_GROUP_NORM_SPV}" "${VIDFAB_TENSOR_ROPE_SOURCE}"
    "${VIDFAB_TENSOR_ROPE_SPV}" "${VIDFAB_TENSOR_ROPE_DENORM_SPV}")

  add_library(vidfab_vulkan STATIC
    src/vulkan/runtime.cpp
    src/vulkan/tensor.cpp
    src/vulkan/yuv_converter.cpp)
  target_include_directories(vidfab_vulkan
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/include
            ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan)
  target_link_libraries(vidfab_vulkan PUBLIC vidfab_core)
  target_compile_definitions(vidfab_vulkan PRIVATE VK_NO_PROTOTYPES)
  if(CMAKE_DL_LIBS)
    target_link_libraries(vidfab_vulkan PRIVATE ${CMAKE_DL_LIBS})
  endif()
  if(MSVC)
    target_compile_options(vidfab_vulkan PRIVATE /W4 /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(vidfab_vulkan PRIVATE -Wall -Wextra)
  endif()
endif()

# --- cli --------------------------------------------------------------------

if(WIN32)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ref/text_encoder/tokenizer.json")
    message(FATAL_ERROR "vidfab: ref/text_encoder/tokenizer.json is required to embed the tokenizer")
  endif()
  set(VIDFAB_LICENSE_RESOURCE "${CMAKE_CURRENT_BINARY_DIR}/generated/minimax_h3_LICENSE")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
  # Fetched once per build tree and then kept. Embedding the licence is a
  # compliance mechanism and stays; re-downloading it on every configure only
  # made `cmake -S . -B build` require the network and block on a third-party
  # host, every time, forever.
  #
  # Caching it, though, has to come with a content check, because `file(DOWNLOAD)`
  # does not set CURLOPT_FAILONERROR: an HTTP 404, 403 or 502 from the host is a
  # *successful transfer of the error body*. STATUS comes back 0 and a perfectly
  # non-empty page of HTML lands at the destination. Re-downloading every time
  # used to paper over that — the next configure fetched the real document — so
  # a size-only guard would have pinned HuggingFace's error page in place as the
  # MiniMax H3 licence for the life of the build tree, silently. Of all the ways
  # this can fail, that is the one it must not.
  #
  # The check is the document's first line rather than an EXPECTED_HASH. A hash
  # would be stricter, but the file carries its own "release date/License date"
  # line and upstream may legitimately revise the text; pinning it would turn a
  # lawful upstream edit into a build break that only a source change can clear.
  # The marker catches the actual failure mode without that.
  set(VIDFAB_LICENSE_MARKER "MiniMax H3 COMMUNITY LICENSE AGREEMENT")

  function(vidfab_check_license path out_var)
    set(${out_var} FALSE PARENT_SCOPE)
    if(NOT EXISTS "${path}")
      return()
    endif()
    file(SIZE "${path}" license_bytes)
    if(license_bytes EQUAL 0)
      return()
    endif()
    file(READ "${path}" license_head LIMIT 4096)
    string(FIND "${license_head}" "${VIDFAB_LICENSE_MARKER}" license_marker_at)
    if(license_marker_at EQUAL -1)
      return()
    endif()
    set(${out_var} TRUE PARENT_SCOPE)
  endfunction()

  vidfab_check_license("${VIDFAB_LICENSE_RESOURCE}" VIDFAB_LICENSE_PRESENT)
  if(NOT VIDFAB_LICENSE_PRESENT)
    file(DOWNLOAD
      "https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE"
      "${VIDFAB_LICENSE_RESOURCE}"
      STATUS VIDFAB_LICENSE_DOWNLOAD_STATUS
      TLS_VERIFY ON)
    list(GET VIDFAB_LICENSE_DOWNLOAD_STATUS 0 VIDFAB_LICENSE_DOWNLOAD_CODE)
    # Applied to what actually arrived, not just to the transport result, so an
    # error page is rejected on the way in as well as out of the cache.
    vidfab_check_license("${VIDFAB_LICENSE_RESOURCE}" VIDFAB_LICENSE_PRESENT)
    if(NOT VIDFAB_LICENSE_DOWNLOAD_CODE EQUAL 0 OR NOT VIDFAB_LICENSE_PRESENT)
      if(VIDFAB_LICENSE_DOWNLOAD_CODE EQUAL 0)
        # The transfer succeeded and the body is not the licence. Usually that
        # means an error page, but it is also what a lawful upstream retitle
        # looks like from here, and the reader has no way to tell which without
        # being told where the expectation lives.
        string(CONCAT VIDFAB_LICENSE_DOWNLOAD_MESSAGE
          "the server returned something that is not the MiniMax H3 license. "
          "The transfer itself succeeded, so this is either an error page or a "
          "document whose text no longer contains VIDFAB_LICENSE_MARKER "
          "(\"${VIDFAB_LICENSE_MARKER}\"). Fetch the URL by hand to tell which; "
          "if upstream legitimately retitled the document, update that marker "
          "in CMakeLists.txt to match")
      else()
        list(GET VIDFAB_LICENSE_DOWNLOAD_STATUS 1 VIDFAB_LICENSE_DOWNLOAD_MESSAGE)
      endif()
      file(REMOVE "${VIDFAB_LICENSE_RESOURCE}")
      message(FATAL_ERROR "vidfab: could not download MiniMax H3 license: ${VIDFAB_LICENSE_DOWNLOAD_MESSAGE}")
    endif()
  endif()
  file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ref/text_encoder/tokenizer.json"
       VIDFAB_TOKENIZER_RESOURCE)
  file(TO_CMAKE_PATH "${VIDFAB_LICENSE_RESOURCE}" VIDFAB_LICENSE_RESOURCE)
  configure_file(src/vidfab_tokenizer.rc.in generated/vidfab_tokenizer.rc @ONLY)
  configure_file(src/vidfab_license.rc.in generated/vidfab_license.rc @ONLY)

  # A static library cannot usefully carry a .res — the linker has no reference
  # to pull it in with — so the executable embeds both resources itself.
  set(VIDFAB_EXE_RESOURCES
    "${CMAKE_CURRENT_BINARY_DIR}/generated/vidfab_tokenizer.rc"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/vidfab_license.rc")
  add_executable(vidfab src/main.cpp ${VIDFAB_EXE_RESOURCES})
else()
  add_executable(vidfab src/main.cpp)
endif()
target_link_libraries(vidfab PRIVATE vidfab_core)
target_compile_definitions(vidfab PRIVATE
  VIDFAB_WITH_VULKAN=$<IF:$<BOOL:${VIDFAB_ENABLE_VULKAN}>,1,0>)
if(VIDFAB_ENABLE_VULKAN)
  target_link_libraries(vidfab PRIVATE vidfab_vulkan)
endif()
if(VIDFAB_ENABLE_CUDA)
  target_link_libraries(vidfab PRIVATE vidfab_cuda)
endif()

if(MSVC)
  target_compile_options(vidfab PRIVATE /W4 /permissive- /utf-8 /EHsc)
else()
  target_compile_options(vidfab PRIVATE -Wall -Wextra)
endif()
if(WIN32)
  target_link_libraries(vidfab PRIVATE winhttp)
endif()

# Staging the FFmpeg runtime beside the executable, which only a build that
# compiled the muxer has any use for. Guarded rather than left unconditional
# because the check below is a FATAL_ERROR: without this, configuring with
# -DVIDFAB_WITH_FFMPEG=OFF would still refuse to proceed without the very
# distribution the option exists to do without.
if(WIN32 AND VIDFAB_WITH_FFMPEG)
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

# --- c api ------------------------------------------------------------------

# vidfab_c builds vidfab.dll, the stable C ABI with a hand-written export set, for
# Rust, C#, Python and anything else with an FFI. See include/vidfab/capi.h.
#
# Dependent on CUDA rather than merely checked against it, because the C API
# exists to run generations and `run_generate` is on the CUDA side. Making it a
# plain option defaulting to ON meant that a machine with no toolkit — the case
# the lazy-CUDA design at the top of this file exists to support — stopped
# configuring at all, since the default would then contradict a hard error two
# hundred lines earlier. This forces it off in that situation instead, and says
# so, and an explicit -DVIDFAB_BUILD_C_API=ON cannot resurrect it.
include(CMakeDependentOption)
cmake_dependent_option(VIDFAB_BUILD_C_API
  "Build vidfab_c, the stable C ABI shared library" ON
  "VIDFAB_ENABLE_CUDA" OFF)

if(NOT VIDFAB_BUILD_C_API AND NOT VIDFAB_ENABLE_CUDA)
  message(STATUS "vidfab: no CUDA, so vidfab_c (the C ABI DLL) is not built")
endif()

if(VIDFAB_BUILD_C_API)
  # The internal core and CUDA libraries are static and link into this target.
  # Consequently vidfab.dll is the only project DLL, and its exports are exactly
  # the C entry points — no unstable C++ ABI and no companion vidfab DLLs.
  #
  # It is not, however, dependency-free, and it is worth being exact about that
  # because the mistake is invisible on a developer machine. The CUDA *runtime*
  # is embedded (see the cudart note above), but cuBLAS is not: vidfab.dll
  # carries a load-time import of cublas64_<major>.dll, which in turn pulls
  # cublasLt64_<major>.dll. On a box with the toolkit installed those resolve
  # off PATH and nothing looks wrong; on a consumer's machine the process fails
  # at LoadLibrary with no useful message. Linking them statically is not the
  # answer — cublasLt_static alone is ~456 MB — so they are named in the
  # install rule below instead, and a build tree simply relies on PATH.
  add_library(vidfab_c SHARED src/capi/capi.cpp)
  # vidfab_cuda carries vidfab_core with it as a PUBLIC dependency, so naming
  # the one target is naming both.
  target_link_libraries(vidfab_c PRIVATE vidfab_cuda)

  # PRIVATE: consumers of the DLL include capi.h and nothing else, and the
  # install rule below places it for them. Anything in this build tree that
  # needs the header asks for the include directory itself.
  target_include_directories(vidfab_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # Switches capi.h from dllimport to dllexport. It is the only definition the
  # C API needs: the version it reports is stated in the header next to the
  # constants a consumer compiles against, so the two cannot drift.
  target_compile_definitions(vidfab_c PRIVATE VIDFAB_C_BUILD=1)

  # The static libraries end up inside a shared object here, which on ELF
  # targets requires every object in them to be position independent. Harmless
  # and ignored on Windows.
  set_property(TARGET vidfab_core PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET vidfab_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)

  # The export set is the point of this target, so it is stated rather than
  # inherited. WINDOWS_EXPORT_ALL_SYMBOLS is off explicitly because a build
  # that sets CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS globally would otherwise dump
  # the whole C++ surface into this DLL's export table alongside the C one, and
  # the resulting module would still work — silently defeating the boundary
  # instead of failing.
  #
  # The version is the *C ABI's*, not the project's. They are different numbers
  # answering different questions — capi.h declares 1.0.0 and tells a binding to
  # refuse a different major, while PROJECT_VERSION is 0.1.0 and moves whenever
  # anything in the project does. Using the latter here would produce a
  # libvidfab.so.0 whose soname says 0 for an ABI that says 1, and would bump
  # it for releases that did not touch the C interface at all.
  set(VIDFAB_CAPI_VERSION_MAJOR 1)
  set(VIDFAB_CAPI_VERSION_MINOR 0)
  set(VIDFAB_CAPI_VERSION_PATCH 0)
  set(VIDFAB_CAPI_VERSION
    "${VIDFAB_CAPI_VERSION_MAJOR}.${VIDFAB_CAPI_VERSION_MINOR}.${VIDFAB_CAPI_VERSION_PATCH}")

  set_target_properties(vidfab_c PROPERTIES
    OUTPUT_NAME vidfab
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${VIDFAB_CAPI_VERSION}
    SOVERSION ${VIDFAB_CAPI_VERSION_MAJOR})

  # The visibility presets above are *compile* properties, so they only cover
  # capi.cpp. vidfab_core and vidfab_cuda are compiled with default visibility,
  # and on ELF their entire C++ symbol table would otherwise be re-exported
  # from libvidfab.so — precisely what WINDOWS_EXPORT_ALL_SYMBOLS OFF
  # prevents on the other platform. --exclude-libs makes the two agree.
  if(NOT WIN32 AND NOT APPLE)
    target_link_options(vidfab_c PRIVATE "LINKER:--exclude-libs,ALL")
  endif()

  # capi.h states the same three numbers for the compiler; if they ever
  # disagree, the soname and the number a binding checks would describe
  # different interfaces. Cheaper to catch here than in a consumer.
  #
  # Matched per name rather than by position. Reading the three numbers in file
  # order and joining them would compare 0.1.0 against 1.0.0 — and abort with a
  # message blaming the version rather than the reordering — if someone ever
  # rearranged three `#define`s that C is perfectly happy to see in any order.
  foreach(VIDFAB_CAPI_PART MAJOR MINOR PATCH)
    unset(VIDFAB_CAPI_HEADER_${VIDFAB_CAPI_PART})
    file(STRINGS include/vidfab/capi.h VIDFAB_CAPI_HEADER_LINE
      REGEX "^#define[ \t]+VIDFAB_CAPI_VERSION_${VIDFAB_CAPI_PART}[ \t]+[0-9]+")
    if(NOT VIDFAB_CAPI_HEADER_LINE)
      message(FATAL_ERROR
        "vidfab: capi.h has no VIDFAB_CAPI_VERSION_${VIDFAB_CAPI_PART} to check against")
    endif()
    string(REGEX MATCH "[0-9]+$" VIDFAB_CAPI_HEADER_${VIDFAB_CAPI_PART}
      "${VIDFAB_CAPI_HEADER_LINE}")
  endforeach()
  set(VIDFAB_CAPI_HEADER_VERSION
    "${VIDFAB_CAPI_HEADER_MAJOR}.${VIDFAB_CAPI_HEADER_MINOR}.${VIDFAB_CAPI_HEADER_PATCH}")
  if(NOT VIDFAB_CAPI_HEADER_VERSION STREQUAL VIDFAB_CAPI_VERSION)
    message(FATAL_ERROR
      "vidfab: capi.h declares ABI ${VIDFAB_CAPI_HEADER_VERSION} but CMakeLists says "
      "${VIDFAB_CAPI_VERSION}; update both.")
  endif()
  # `file(STRINGS)` creates no dependency of its own, so without this the check
  # would go stale at exactly the moment it exists for: editing the header's
  # version macros would not re-run CMake, and the soname would keep describing
  # the previous ABI.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/vidfab/capi.h")

  # `Tokenizer::load_embedded` looks the resource up in the module holding its
  # own code. That code is linked into vidfab.dll, so the 7 MB tokenizer has to
  # be here too or a C caller that leaves the tokenizer path empty gets
  # "embedded resource is missing".
  if(WIN32)
    target_sources(vidfab_c PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/vidfab_tokenizer.rc")
  endif()

  if(MSVC)
    target_compile_options(vidfab_c PRIVATE /W4 /permissive- /utf-8)
  else()
    target_compile_options(vidfab_c PRIVATE -Wall -Wextra)
  endif()
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
    tests/test_safetensors.cpp
    tests/test_pipeline.cpp
    tests/test_sampler.cpp
    tests/test_step_cache.cpp
    tests/test_block_cache.cpp
    tests/test_tile_merge.cpp
    tests/test_tokenizer.cpp
    tests/test_tokenizer_json.cpp
    tests/test_qwen_vision.cpp
    tests/test_output.cpp
    tests/test_device_tensor.cpp
  )
  target_link_libraries(vidfab_tests PRIVATE vidfab_core)
  if(MSVC)
    target_compile_options(vidfab_tests PRIVATE /W4 /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(vidfab_tests PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME unit COMMAND vidfab_tests)

  if(VIDFAB_ENABLE_VULKAN)
    add_executable(vidfab_vulkan_tests tests/harness.cpp tests/test_vulkan.cpp)
    target_link_libraries(vidfab_vulkan_tests PRIVATE vidfab_vulkan)
    set(VIDFAB_AFFINE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/affine.comp")
    set(VIDFAB_AFFINE_SPV "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/affine.comp.spv")
    file(READ "${VIDFAB_AFFINE_SOURCE}" VIDFAB_AFFINE_SOURCE_TEXT)
    string(REPLACE "\r\n" "\n" VIDFAB_AFFINE_SOURCE_TEXT "${VIDFAB_AFFINE_SOURCE_TEXT}")
    string(SHA256 VIDFAB_AFFINE_SOURCE_HASH "${VIDFAB_AFFINE_SOURCE_TEXT}")
    file(SHA256 "${VIDFAB_AFFINE_SPV}" VIDFAB_AFFINE_SPV_HASH)
    if(NOT VIDFAB_AFFINE_SOURCE_HASH STREQUAL
        "5ff2705b9c300adbdb476d393226f0dda0a3e23dd7b1eda9ac8095a9165345cd" OR
       NOT VIDFAB_AFFINE_SPV_HASH STREQUAL
        "75c8343ac68d52af2c5908db040af9768f6abbce9df0a82268d131afbfc2ede5")
      message(FATAL_ERROR
        "vidfab: affine.comp or its checked-in SPIR-V changed. Regenerate with the "
        "documented glslang command, verify the shader test, and update both hashes")
    endif()
    target_compile_definitions(vidfab_vulkan_tests PRIVATE
      VIDFAB_TEST_AFFINE_SPV_PATH="affine.comp.spv")
    add_custom_command(TARGET vidfab_vulkan_tests POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${VIDFAB_AFFINE_SPV}" "$<TARGET_FILE_DIR:vidfab_vulkan_tests>/affine.comp.spv"
      VERBATIM)
    if(MSVC)
      target_compile_options(vidfab_vulkan_tests PRIVATE /W4 /permissive- /utf-8 /EHsc)
    else()
      target_compile_options(vidfab_vulkan_tests PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME vulkan COMMAND vidfab_vulkan_tests)
    set_tests_properties(vulkan PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vidfab_vulkan_tests>")
  endif()

  if(VIDFAB_ENABLE_CUDA AND VIDFAB_ENABLE_VULKAN)
    add_executable(vidfab_tensor_backend_tests
      tests/harness.cpp tests/test_tensor_backends.cu)
    target_link_libraries(vidfab_tensor_backend_tests PRIVATE
      vidfab_cuda vidfab_vulkan)
    if(MSVC)
      target_compile_options(vidfab_tensor_backend_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/utf-8;/EHsc>)
    endif()
    add_test(NAME tensor_backends COMMAND vidfab_tensor_backend_tests)
  endif()

  if(VIDFAB_ENABLE_CUDA)
    add_executable(vidfab_kernel_tests
      tests/harness.cpp
      tests/test_kernels.cu
      tests/test_nn_kernels.cu
      tests/test_audio_vae.cu
      tests/test_transformer.cu
      tests/test_encoder.cu
      tests/test_keyframe_encoder.cu
      tests/test_vit_decoder.cu
    )
    target_link_libraries(vidfab_kernel_tests PRIVATE vidfab_cuda)
    add_test(NAME kernels COMMAND vidfab_kernel_tests)
  endif()

  # The C ABI is tested through the DLL rather than by compiling capi.cpp into
  # the test binary, which is the only way the interesting failures show up:
  # a missing export, a mangled name, a calling-convention mismatch and a
  # struct the two sides lay out differently are all link- or run-time
  # properties of the module, invisible to a test that includes the source.
  if(VIDFAB_BUILD_C_API)
    # Its own main, not tests/test_main.cpp: that file carries C++ test cases
    # of its own and would drag vidfab_core into a binary whose entire point is
    # linking nothing but the DLL.
    add_executable(vidfab_capi_tests tests/harness.cpp tests/test_capi.cpp)
    target_link_libraries(vidfab_capi_tests PRIVATE vidfab_c)
    target_include_directories(vidfab_capi_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(MSVC)
      target_compile_options(vidfab_capi_tests PRIVATE /W4 /permissive- /utf-8)
    else()
      target_compile_options(vidfab_capi_tests PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME capi COMMAND vidfab_capi_tests)
  endif()
endif()

# --- install ----------------------------------------------------------------

# The DLL installs one header rather than the whole include tree: capi.h is the
# supported surface and the C++ headers beside it are internal implementation.
if(VIDFAB_BUILD_C_API)
  include(GNUInstallDirs)
  install(TARGETS vidfab_c
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES include/vidfab/capi.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vidfab)

  # The cuBLAS redistributables vidfab.dll imports at load time. Without
  # these an installed DLL fails at LoadLibrary on any machine that has no CUDA
  # toolkit on PATH — which is every machine this is shipped to, and the reason
  # it looks fine here. They are installed rather than copied beside every
  # build because they are half a gigabyte and a build tree can rely on PATH.
  #
  # Located by name rather than through $<TARGET_FILE:CUDA::cublas>, whose
  # imported location is the import library on Windows. Both layouts CUDA has
  # used are searched, and a miss is a warning rather than an error: it makes
  # the install incomplete, not the build wrong, and saying which file was not
  # found is more useful than refusing to install anything.
  if(WIN32)
    foreach(VIDFAB_CUDA_DLL cublas cublasLt)
      set(VIDFAB_CUDA_DLL_NAME "${VIDFAB_CUDA_DLL}64_${CUDAToolkit_VERSION_MAJOR}.dll")
      unset(VIDFAB_CUDA_DLL_PATH)
      foreach(VIDFAB_CUDA_DLL_DIR "${CUDAToolkit_BIN_DIR}/x64" "${CUDAToolkit_BIN_DIR}")
        if(NOT DEFINED VIDFAB_CUDA_DLL_PATH AND
           EXISTS "${VIDFAB_CUDA_DLL_DIR}/${VIDFAB_CUDA_DLL_NAME}")
          set(VIDFAB_CUDA_DLL_PATH "${VIDFAB_CUDA_DLL_DIR}/${VIDFAB_CUDA_DLL_NAME}")
        endif()
      endforeach()
      if(DEFINED VIDFAB_CUDA_DLL_PATH)
        install(FILES "${VIDFAB_CUDA_DLL_PATH}" DESTINATION ${CMAKE_INSTALL_BINDIR})
      else()
        message(WARNING
          "vidfab: ${VIDFAB_CUDA_DLL_NAME} was not found under ${CUDAToolkit_BIN_DIR}; "
          "an installed vidfab_c will not load unless it is supplied separately")
      endif()
    endforeach()
  endif()
endif()
