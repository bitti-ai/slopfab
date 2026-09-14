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

# --- core -------------------------------------------------------------------

add_library(slopfab_core STATIC
  src/core/json.cpp
  src/core/nf4.cpp
  src/core/w4a8.cpp
  src/core/lora.cpp
  src/core/safetensors.cpp
  src/core/tensor_convert.cpp
  src/core/device_tensor.cpp
  src/core/image.cpp
  src/core/reference_media.cpp
  src/core/refmod.cpp
  src/core/reference_conditioning.cpp
  src/core/image_platform.cpp
  src/core/safetensors_write.cpp
  src/core/sha256.cpp
  src/dit/adaln.cpp
  src/dit/block_cache.cpp
  src/dit/chunking.cpp
  src/dit/checkpoint.cpp
  src/dit/packing.cpp
  src/dit/ref2va.cpp
  src/dit/rope.cpp
  src/dit/step_cache.cpp
  src/vae/keyframe_encoder.cpp
  src/vae/audio_primitives.cpp
  src/vae/vit_block.cpp
  src/vae/decode_pipeline.cpp
  src/pipeline.cpp
  src/sampler/noise.cpp
  src/sampler/scheduler.cpp
  src/text/tokenizer.cpp
  src/text/encoder.cpp
  src/text/layer_capture.cpp
  src/text/qwen_vision.cpp
  src/video/y4m.cpp
  src/video/y4m_compare.cpp
  src/audio/wav.cpp
)

# The muxer, or the stub that stands in for it. Exactly one of the two is
# compiled: they define the same four functions, so every call site links
# against whichever this build has and none of them needs an #if.
if(SLOPFAB_WITH_FFMPEG)
  target_sources(slopfab_core PRIVATE src/video/mux.cpp)
else()
  target_sources(slopfab_core PRIVATE src/video/mux_disabled.cpp)
endif()

# PUBLIC because headers branch on it: media.h only declares the demuxer where
# it exists, and image.cpp picks its decoder with it. A consumer that saw a
# different value than this library was built with would disagree about which
# functions exist.
target_compile_definitions(slopfab_core PUBLIC
  SLOPFAB_WITH_FFMPEG=$<IF:$<BOOL:${SLOPFAB_WITH_FFMPEG}>,1,0>)

target_include_directories(slopfab_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# The platform image decoder. Compiled in every configuration — so the two
# decoders can be compared and the code cannot rot unbuilt — which is why
# these are linked whether or not FFmpeg is on. Both ship with Windows.
if(WIN32)
  target_link_libraries(slopfab_core PRIVATE windowscodecs ole32 bcrypt)
endif()

# The RGB -> YUV 4:2:0 conversion in src/video/y4m.cpp is threaded. MSVC needs
# nothing for std::thread, but every other toolchain wants -pthread both to
# compile and to link, and without this the non-MSVC build breaks. Requested
# here as well as in the CUDA block below because slopfab_core must build on its
# own, on a machine with no CUDA toolkit at all.
find_package(Threads REQUIRED)
target_link_libraries(slopfab_core PUBLIC Threads::Threads)

if(MSVC)
  # /utf-8 keeps string literals well-defined regardless of the system codepage.
  target_compile_options(slopfab_core PRIVATE /W4 /permissive- /utf-8 /EHsc)
else()
  target_compile_options(slopfab_core PRIVATE -Wall -Wextra)
endif()

# --- cuda -------------------------------------------------------------------

if(SLOPFAB_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Must be set before enable_language(CUDA), which otherwise picks a
    # conservative default. 86 = Ampere (RTX 30 series), while 120 = Blackwell
    # (RTX 50 series) and is the only supported family with native fp4.
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
    # image runs on sm_120 hardware and nothing later. The separate sm_86 image
    # serves RTX 30-series cards; later architectures need their own image and
    # runtime dispatch entry rather than inheriting Blackwell's native-fp4 path.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      # One toolkit-specific executable serves both supported GPU families.
      # Runtime dispatch keeps Ampere on SM86 kernels and Blackwell on the
      # architecture-specific SM120a image required by native NVFP4.
      set(CMAKE_CUDA_ARCHITECTURES "86;120a" CACHE STRING "CUDA architectures" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    if(WIN32)
      # The shipped CUDA core is deliberately built with 12.8: static cudart
      # and the 86/120a SASS use the driver ABI and therefore run on a newer
      # driver, while cuBLAS itself is selected independently at runtime.
      if(NOT CUDAToolkit_VERSION MATCHES "^12\\.8(\\.|$)")
        message(FATAL_ERROR
          "slopfab: Windows release/core builds require CUDA 12.8; got ${CUDAToolkit_VERSION}")
      endif()

      try_compile(SLOPFAB_CUBLAS12_ABI_OK
        "${CMAKE_CURRENT_BINARY_DIR}/cublas12-abi-probe"
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cublas_abi_probe.cpp"
        CMAKE_FLAGS
          "-DINCLUDE_DIRECTORIES=${CUDAToolkit_INCLUDE_DIRS}"
          "-DCMAKE_CXX_STANDARD=17"
          "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
        OUTPUT_VARIABLE SLOPFAB_CUBLAS12_ABI_OUTPUT)
      if(NOT SLOPFAB_CUBLAS12_ABI_OK)
        message(FATAL_ERROR
          "slopfab: CUDA 12.8 cuBLAS ABI does not match the late-bound adapter:\n${SLOPFAB_CUBLAS12_ABI_OUTPUT}")
      endif()

      # NVIDIA guarantees ABI compatibility within a major, not across one.
      # Our eight-call adapter supports the exact locally validated CUDA 13.0
      # ABI, so compile its neutral signature/enum contract against 13.0's own
      # headers instead of blindly casting CUDA 12 declarations.
      set(SLOPFAB_CUDA13_ROOT "$ENV{CUDA_PATH_V13_0}" CACHE PATH
        "CUDA 13.0 toolkit used to validate the late-bound cuBLAS ABI")
      if(NOT SLOPFAB_CUDA13_ROOT)
        set(SLOPFAB_CUDA13_ROOT
          "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
      endif()
      if(NOT EXISTS "${SLOPFAB_CUDA13_ROOT}/include/cublas_v2.h")
        message(FATAL_ERROR
          "slopfab: CUDA 13.0 headers are required to validate dual-major cuBLAS dispatch; set SLOPFAB_CUDA13_ROOT")
      endif()
      try_compile(SLOPFAB_CUBLAS13_ABI_OK
        "${CMAKE_CURRENT_BINARY_DIR}/cublas13-abi-probe"
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cublas_abi_probe.cpp"
        CMAKE_FLAGS
          "-DINCLUDE_DIRECTORIES=${SLOPFAB_CUDA13_ROOT}/include"
          "-DCMAKE_CXX_STANDARD=17"
          "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
        OUTPUT_VARIABLE SLOPFAB_CUBLAS13_ABI_OUTPUT)
      if(NOT SLOPFAB_CUBLAS13_ABI_OK)
        message(FATAL_ERROR
          "slopfab: CUDA 13.0 cuBLAS ABI does not match the late-bound adapter:\n${SLOPFAB_CUBLAS13_ABI_OUTPUT}")
      endif()
      message(STATUS "slopfab: validated late-bound cuBLAS ABI against CUDA 12.8 and 13.0 headers")
    endif()
    # The VAE decode's final de-normalise is the project's first std::thread
    # use. MSVC needs nothing for it, but every other toolchain wants -pthread
    # both to compile and to link, and without this the non-MSVC build breaks.
    find_package(Threads REQUIRED)
    message(STATUS "slopfab: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")

    add_library(slopfab_cuda STATIC
      src/cuda/device.cu
      src/cuda/cublas_dispatch.cpp
      src/cuda/diagnostics.cu
      src/cuda/profile.cpp
      src/cuda/workspace.cu
      src/cuda/nn_kernels.cu
      src/cuda/lora.cu
      src/cuda/linear.cu
      src/cuda/deterministic_gemm.cu
      src/cuda/deterministic_attention.cu
      src/cuda/nf4_weight.cu
      src/cuda/w4a8.cu
      src/cuda/nvfp4_gemm.cu
      src/cuda/attention.cu
      src/cuda/sage_attention.cu
      src/cuda/sol_attention.cu
      src/cuda/sol_attention_pipeline.cu
      src/cuda/vae_kernels.cu
      src/cuda/vae_vit_block.cu
      src/cuda/vit_decoder.cu
      src/cuda/audio_vae_kernels.cu
      src/cuda/dit_kernels.cu
      src/cuda/encoder_kernels.cu
      src/cuda/keyframe_encoder.cu
      src/cuda/reference_encoder.cu
      src/cuda/qwen_vision.cu
      src/cuda/qwen_vision_encoder.cu
      src/vae/keyframe_cuda.cpp
      src/vae/audio_encoder.cpp
      src/generate.cpp
      # src/capi/capi.cpp is deliberately NOT here. It calls run_generate, so
      # it depends on this target — but compiling it *into* this target as well
      # as into slopfab_c builds every C entry point twice: once with capi.h
      # seen as dllimport (here, MSVC C4273 "inconsistent dll linkage") and
      # once as dllexport (there). The link happens to pick the right copy, so
      # the failure is silent, and it would leave the C ABI exported from a
      # library that is otherwise pure C++. It belongs to slopfab_c alone; see
      # the c api section below.
      src/vae/audio_decoder.cpp
      src/dit/transformer.cpp
      src/dit/denoise.cpp
      src/cuda/encoder_checkpoint_upload.cpp
    )
    target_include_directories(slopfab_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(slopfab_cuda PUBLIC CUDA::cuda_driver Threads::Threads)
    if(NOT WIN32)
      # Windows resolves CUDA 12/13 cuBLAS in-process. Other platforms retain
      # the toolkit selected at link time until an equivalent loader exists.
      target_link_libraries(slopfab_cuda PUBLIC CUDA::cublas)
    endif()
    # Whole-program compilation: no device code crosses translation units, and
    # -rdc=true would otherwise require an explicit device-link step.
    set_target_properties(slopfab_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

    # The CUDA runtime is linked statically, which is deliberate: it avoids an
    # additional cudart DLL dependency. cuBLAS remains a dynamic dependency
    # supplied by the selected installed toolkit. The cost is a
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
    target_link_libraries(slopfab_cuda PUBLIC slopfab_core CUDA::cudart_static)
    if(SLOPFAB_ENABLE_VULKAN)
      target_link_libraries(slopfab_cuda PUBLIC slopfab_vulkan)
      target_compile_definitions(slopfab_cuda PRIVATE SLOPFAB_WITH_VULKAN=1)
    else()
      target_compile_definitions(slopfab_cuda PRIVATE SLOPFAB_WITH_VULKAN=0)
    endif()

    add_executable(slopfab_qwenvisionprobe tools/qwenvisionprobe.cpp)
    target_link_libraries(slopfab_qwenvisionprobe PRIVATE slopfab_cuda)
    add_executable(slopfab_qwenconditionerprobe tools/qwenconditionerprobe.cpp)
    target_link_libraries(slopfab_qwenconditionerprobe PRIVATE slopfab_cuda)
    if(SLOPFAB_BUILD_DEV_TOOLS)
      add_executable(slopfab_solbench tools/solbench.cu)
      target_link_libraries(slopfab_solbench PRIVATE slopfab_cuda)
    endif()

    target_compile_definitions(slopfab_core PUBLIC SLOPFAB_WITH_CUDA=1)
  else()
    message(WARNING "slopfab: no CUDA compiler found; building CPU-only")
    set(SLOPFAB_ENABLE_CUDA OFF)
  endif()
endif()

# --- vulkan -----------------------------------------------------------------

# Vulkan is independent of CUDA and implements the core frame-converter
# interface. The Khronos core ABI headers are pinned in-tree, while the
# platform loader is opened dynamically by runtime.cpp. Consequently enabling
# this target needs no Vulkan SDK at build time or Vulkan loader at process
# startup.
if(SLOPFAB_ENABLE_VULKAN)
  set(SLOPFAB_YUV_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/rgb_to_yuv.comp")
  set(SLOPFAB_YUV_SHADER_SPV "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/rgb_to_yuv.comp.spv")
  set(SLOPFAB_TENSOR_OPS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_ops.comp")
  set(SLOPFAB_TENSOR_OPS_SPV "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_ops.comp.spv")
  set(SLOPFAB_TENSOR_OPS_DENORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_ops_denorm.comp.spv")
  set(SLOPFAB_TENSOR_NORM_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_norm.comp")
  set(SLOPFAB_TENSOR_RMS_NORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_rms_norm.comp.spv")
  set(SLOPFAB_TENSOR_LAYER_NORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_layer_norm.comp.spv")
  set(SLOPFAB_TENSOR_SHARED_NORM_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_shared_norm.comp")
  set(SLOPFAB_TENSOR_BF16_RMS_BLOCK_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_bf16_rms_block.comp.spv")
  set(SLOPFAB_TENSOR_BF16_RMS_NARROW_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_bf16_rms_narrow.comp.spv")
  set(SLOPFAB_TENSOR_BF16_LAYER_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_bf16_layer.comp.spv")
  set(SLOPFAB_TENSOR_BF16_MOD_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_bf16_mod.comp.spv")
  set(SLOPFAB_TENSOR_FP32_MOD_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_fp32_mod.comp.spv")
  set(SLOPFAB_TENSOR_GROUP_NORM_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_group_norm.comp")
  set(SLOPFAB_TENSOR_GROUP_NORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_group_norm.comp.spv")
  set(SLOPFAB_TENSOR_VAE_POINTWISE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_vae_pointwise.comp")
  set(SLOPFAB_TENSOR_VAE_RESIDUAL_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_vae_residual.comp.spv")
  set(SLOPFAB_TENSOR_VAE_SWIGLU_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_vae_swiglu.comp.spv")
  set(SLOPFAB_TENSOR_VAE_DENORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_vae_denorm.comp.spv")
  set(SLOPFAB_TENSOR_AUDIO_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_audio.hlsl")
  set(SLOPFAB_TENSOR_AUDIO_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_audio.comp.spv")
  set(SLOPFAB_TENSOR_KEYFRAME_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_keyframe.hlsl")
  set(SLOPFAB_TENSOR_KEYFRAME_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_keyframe.comp.spv")
  set(SLOPFAB_TENSOR_DIT_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_dit.hlsl")
  set(SLOPFAB_TENSOR_DIT_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_dit.comp.spv")
  set(SLOPFAB_TENSOR_ROPE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_rope.comp")
  set(SLOPFAB_TENSOR_ROPE_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_rope.comp.spv")
  set(SLOPFAB_TENSOR_ROPE_DENORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_rope_denorm.comp.spv")
  set(SLOPFAB_TENSOR_VAE_ROPE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_vae_rope.comp")
  set(SLOPFAB_TENSOR_VAE_ROPE_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_vae_rope.comp.spv")
  set(SLOPFAB_TENSOR_WEIGHT_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_weight.comp")
  set(SLOPFAB_TENSOR_WEIGHT_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_weight.comp.spv")
  set(SLOPFAB_TENSOR_WEIGHT_DENORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_weight_denorm.comp.spv")
  set(SLOPFAB_TENSOR_GEMM_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm.hlsl")
  set(SLOPFAB_TENSOR_GEMM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm.comp.spv")
  set(SLOPFAB_TENSOR_GEMM_DENORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm_denorm.comp.spv")
  set(SLOPFAB_TENSOR_GEMM_COOP_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm_coop.comp")
  set(SLOPFAB_TENSOR_GEMM_COOP_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm_coop.comp.spv")
  set(SLOPFAB_TENSOR_GEMM_COOP_DENORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm_coop_denorm.comp.spv")
  set(SLOPFAB_TENSOR_GEMM_PREPARE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm_prepare.hlsl")
  set(SLOPFAB_TENSOR_GEMM_PREPARE_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm_prepare.comp.spv")
  set(SLOPFAB_TENSOR_GEMM_COOP_F16_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm_coop_f16.comp")
  set(SLOPFAB_TENSOR_GEMM_COOP_F16_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm_coop_f16.comp.spv")
  set(SLOPFAB_TENSOR_GEMM_COOP_F16_DENORM_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_gemm_coop_f16_denorm.comp.spv")
  set(SLOPFAB_TENSOR_ATTENTION_BLOCKED_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_attention_blocked.comp")
  set(SLOPFAB_TENSOR_ATTENTION_BLOCKED_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_attention_blocked.comp.spv")
  set(SLOPFAB_TENSOR_ATTENTION_PREPARE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_attention_prepare.comp")
  set(SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_attention_prepare.comp.spv")
  set(SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_attention_causal_gqa.comp")
  set(SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_attention_causal_gqa.comp.spv")
  set(SLOPFAB_TENSOR_ATTENTION_H3_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_attention_h3.comp")
  set(SLOPFAB_TENSOR_ATTENTION_H3_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_attention_h3.comp.spv")
  set(SLOPFAB_TENSOR_ATTENTION_H3_BANDED_SPV
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_attention_h3_banded.comp.spv")
  file(READ "${SLOPFAB_YUV_SHADER_SOURCE}" SLOPFAB_YUV_SHADER_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_YUV_SHADER_TEXT "${SLOPFAB_YUV_SHADER_TEXT}")
  string(SHA256 SLOPFAB_YUV_SHADER_SOURCE_HASH "${SLOPFAB_YUV_SHADER_TEXT}")
  file(SHA256 "${SLOPFAB_YUV_SHADER_SPV}" SLOPFAB_YUV_SHADER_SPV_HASH)
  if(NOT SLOPFAB_YUV_SHADER_SOURCE_HASH STREQUAL
      "77fa686e8c2ccca0b62551211defb4d47dd941dd617627efce56df35637d2c37" OR
     NOT SLOPFAB_YUV_SHADER_SPV_HASH STREQUAL
      "c4cee251c15f54371387d2dbacb4637c76c0dbc5dc6c0663c1e1269c703e71e6")
    message(FATAL_ERROR
      "slopfab: RGB-to-YUV shader/source hash mismatch; regenerate with glslang 16.5.0 "
      "and update both pinned hashes")
  endif()
  file(READ "${SLOPFAB_YUV_SHADER_SPV}" SLOPFAB_YUV_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," SLOPFAB_YUV_SPV_BYTES
    "${SLOPFAB_YUV_SPV_HEX}")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan")
  configure_file(src/vulkan/embedded_yuv_spv.h.in
    generated/vulkan/embedded_yuv_spv.h @ONLY)
  file(READ "${SLOPFAB_TENSOR_OPS_SOURCE}" SLOPFAB_TENSOR_OPS_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_OPS_TEXT "${SLOPFAB_TENSOR_OPS_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_OPS_SOURCE_HASH "${SLOPFAB_TENSOR_OPS_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_OPS_SPV}" SLOPFAB_TENSOR_OPS_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_OPS_DENORM_SPV}" SLOPFAB_TENSOR_OPS_DENORM_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_OPS_SOURCE_HASH STREQUAL
      "a5e10d2ff909dc131e0e38e4bf5927df1bc3a9fbb02f283bf5118942dfcaa806" OR
     NOT SLOPFAB_TENSOR_OPS_SPV_HASH STREQUAL
      "a11b87b00658e97b8a3b53a24618045eadbbfa929a44ad9bc08997f958fa2106" OR
     NOT SLOPFAB_TENSOR_OPS_DENORM_SPV_HASH STREQUAL
      "e35519ece4af1021bf5056b100fae32d8c3a8757fc0d439c9eaeda9613f92aa7")
    message(FATAL_ERROR "slopfab: tensor-ops shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_NORM_SOURCE}" SLOPFAB_TENSOR_NORM_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_NORM_TEXT "${SLOPFAB_TENSOR_NORM_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_NORM_SOURCE_HASH "${SLOPFAB_TENSOR_NORM_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_RMS_NORM_SPV}" SLOPFAB_TENSOR_RMS_NORM_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_LAYER_NORM_SPV}" SLOPFAB_TENSOR_LAYER_NORM_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_NORM_SOURCE_HASH STREQUAL
      "f82042e4d335da14e442abd2dde6df877b832bccc8c14033d2aa22f7ed03c325" OR
     NOT SLOPFAB_TENSOR_RMS_NORM_SPV_HASH STREQUAL
      "160b9d0f6260bc912379c26171dd8d0c60ad6e4d9b81da57a413b831e513229c" OR
     NOT SLOPFAB_TENSOR_LAYER_NORM_SPV_HASH STREQUAL
      "516ecc94d93cd829f6f6b208acfb3ff647861daae9db3017b31729528d8724e5")
    message(FATAL_ERROR "slopfab: tensor-norm shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_SHARED_NORM_SOURCE}" SLOPFAB_TENSOR_SHARED_NORM_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_SHARED_NORM_TEXT
    "${SLOPFAB_TENSOR_SHARED_NORM_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_SHARED_NORM_SOURCE_HASH
    "${SLOPFAB_TENSOR_SHARED_NORM_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_BF16_RMS_BLOCK_SPV}"
    SLOPFAB_TENSOR_BF16_RMS_BLOCK_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_BF16_RMS_NARROW_SPV}"
    SLOPFAB_TENSOR_BF16_RMS_NARROW_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_BF16_LAYER_SPV}" SLOPFAB_TENSOR_BF16_LAYER_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_BF16_MOD_SPV}" SLOPFAB_TENSOR_BF16_MOD_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_FP32_MOD_SPV}" SLOPFAB_TENSOR_FP32_MOD_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_SHARED_NORM_SOURCE_HASH STREQUAL
      "6435b32e3c0272ea0f20695ca38e993e01bbe8b16688258c2c79bc5845cfa972" OR
     NOT SLOPFAB_TENSOR_BF16_RMS_BLOCK_SPV_HASH STREQUAL
      "5784286fc645492fe7f9b68e292f47e1d8b6efd5ff3bd213bef003559f5eb49f" OR
     NOT SLOPFAB_TENSOR_BF16_RMS_NARROW_SPV_HASH STREQUAL
      "5473b833ff74234c38817535f57b33de46534aa678e5dcfe0f562a0b4f1779a2" OR
     NOT SLOPFAB_TENSOR_BF16_LAYER_SPV_HASH STREQUAL
      "5f33ab2d86663d2611d64cde7a62f27b13471256c808d067b80cf2baeac5c96a" OR
     NOT SLOPFAB_TENSOR_BF16_MOD_SPV_HASH STREQUAL
      "7c6634df12153142fc0c17d9a7ec89275e9bfde45ec7c0249e8353b7685f16ae" OR
     NOT SLOPFAB_TENSOR_FP32_MOD_SPV_HASH STREQUAL
      "c744ee45acf2e34c017a1d1ec83be5bc83042ba29aac95c70ae864d55198f5ec")
    message(FATAL_ERROR "slopfab: shared-norm shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_GROUP_NORM_SOURCE}" SLOPFAB_TENSOR_GROUP_NORM_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_GROUP_NORM_TEXT
    "${SLOPFAB_TENSOR_GROUP_NORM_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_GROUP_NORM_SOURCE_HASH
    "${SLOPFAB_TENSOR_GROUP_NORM_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_GROUP_NORM_SPV}"
    SLOPFAB_TENSOR_GROUP_NORM_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_GROUP_NORM_SOURCE_HASH STREQUAL
      "988faec12f82a744a49cbb5e46aae35b5563f408a7809e72652e4b67ed1457ca" OR
     NOT SLOPFAB_TENSOR_GROUP_NORM_SPV_HASH STREQUAL
      "57aed42788d1d8ca9a3f6ede8d6130b2c32fda2d62c2261fe00f1d3ff2e98d33")
    message(FATAL_ERROR "slopfab: group-norm shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_VAE_POINTWISE_SOURCE}"
    SLOPFAB_TENSOR_VAE_POINTWISE_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_VAE_POINTWISE_TEXT
    "${SLOPFAB_TENSOR_VAE_POINTWISE_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_VAE_POINTWISE_SOURCE_HASH
    "${SLOPFAB_TENSOR_VAE_POINTWISE_TEXT}")
  foreach(VARIANT RESIDUAL SWIGLU DENORM)
    file(SHA256 "${SLOPFAB_TENSOR_VAE_${VARIANT}_SPV}"
      SLOPFAB_TENSOR_VAE_${VARIANT}_SPV_HASH)
  endforeach()
  if(NOT SLOPFAB_TENSOR_VAE_POINTWISE_SOURCE_HASH STREQUAL
      "c0d3ca5a113ab83c08a12d2f06a63d127d2858fe72d96df8e58f4f6af559b83d" OR
     NOT SLOPFAB_TENSOR_VAE_RESIDUAL_SPV_HASH STREQUAL
      "8b64d04cb5b579f787148fb546d77afb1ed6b1a1782bde8d6573b28ea9408c92" OR
     NOT SLOPFAB_TENSOR_VAE_SWIGLU_SPV_HASH STREQUAL
      "e4f51a55b55d7888fc264c1fa2f547bbc8930d9d7db5f627d0356220b92cb453" OR
     NOT SLOPFAB_TENSOR_VAE_DENORM_SPV_HASH STREQUAL
      "e295edcad9058d8581007b4776f0bc7db95a10c99aa9df4fb149947aae9519f2")
    message(FATAL_ERROR "slopfab: VAE pointwise shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_AUDIO_SOURCE}" SLOPFAB_TENSOR_AUDIO_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_AUDIO_TEXT
    "${SLOPFAB_TENSOR_AUDIO_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_AUDIO_SOURCE_HASH "${SLOPFAB_TENSOR_AUDIO_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_AUDIO_SPV}" SLOPFAB_TENSOR_AUDIO_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_AUDIO_SOURCE_HASH STREQUAL
      "c5d8f8d334d64ae5cad62e8f725bad1bb818d25718ce3d1b4d723004f738dbec" OR
     NOT SLOPFAB_TENSOR_AUDIO_SPV_HASH STREQUAL
      "51c0834880118790b70eccbd5fef7267c8dfd8bb98256ee2f67d3250a1e90ab5")
    message(FATAL_ERROR
      "slopfab: audio shader/source hash mismatch; regenerate with Microsoft DXC "
      "1.9.2607 using -spirv -fspv-target-env=vulkan1.2 -T cs_6_6 -E main "
      "-O3 -Gis and update both pinned hashes")
  endif()
  file(READ "${SLOPFAB_TENSOR_KEYFRAME_SOURCE}" SLOPFAB_TENSOR_KEYFRAME_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_KEYFRAME_TEXT
    "${SLOPFAB_TENSOR_KEYFRAME_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_KEYFRAME_SOURCE_HASH
    "${SLOPFAB_TENSOR_KEYFRAME_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_KEYFRAME_SPV}"
    SLOPFAB_TENSOR_KEYFRAME_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_KEYFRAME_SOURCE_HASH STREQUAL
      "2f9e12838e3f8ac93110cadce002466c4e66ef7c54f4a4082822bd9056066775" OR
     NOT SLOPFAB_TENSOR_KEYFRAME_SPV_HASH STREQUAL
      "acbab36d4016a4ff4c8ba90992e8c38622de86e44f19a28ebb2df4ffc8e1974a")
    message(FATAL_ERROR
      "slopfab: keyframe shader/source hash mismatch; regenerate with Microsoft "
      "DXC 1.9.2607 using -spirv -fspv-target-env=vulkan1.2 -T cs_6_6 "
      "-E main -O3 -Gis and update both pinned hashes")
  endif()
  file(READ "${SLOPFAB_TENSOR_DIT_SOURCE}" SLOPFAB_TENSOR_DIT_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_DIT_TEXT
    "${SLOPFAB_TENSOR_DIT_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_DIT_SOURCE_HASH "${SLOPFAB_TENSOR_DIT_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_DIT_SPV}" SLOPFAB_TENSOR_DIT_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_DIT_SOURCE_HASH STREQUAL
      "67d485f5818226dae9d95fcb7b9dd0d208f767df60b014981aecedb2ffc1ec7e" OR
     NOT SLOPFAB_TENSOR_DIT_SPV_HASH STREQUAL
      "08ab6bba63c121eaae8348cb62328f9c7337d21e9e6de5c62292da8be47f1e9b")
    message(FATAL_ERROR
      "slopfab: DiT shader/source hash mismatch; regenerate with Microsoft DXC "
      "1.9.2607 using -spirv -fspv-target-env=vulkan1.2 -T cs_6_6 -E main "
      "-O3 -Gis and update both pinned hashes")
  endif()
  file(READ "${SLOPFAB_TENSOR_ROPE_SOURCE}" SLOPFAB_TENSOR_ROPE_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_ROPE_TEXT "${SLOPFAB_TENSOR_ROPE_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_ROPE_SOURCE_HASH "${SLOPFAB_TENSOR_ROPE_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_ROPE_SPV}" SLOPFAB_TENSOR_ROPE_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_ROPE_DENORM_SPV}"
    SLOPFAB_TENSOR_ROPE_DENORM_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_ROPE_SOURCE_HASH STREQUAL
      "b98e96debc46548ac25065f4ba75037ebda95b92f43303edc625e50e4fdc6347" OR
     NOT SLOPFAB_TENSOR_ROPE_SPV_HASH STREQUAL
      "fb36a5238c64dffd756d65471e0bb492b4fe59910bb676f1d8a4109715391a90" OR
     NOT SLOPFAB_TENSOR_ROPE_DENORM_SPV_HASH STREQUAL
      "e15674264a7cf898c286cc4e9110f8d7a5918732682120010ee02272d73e1734")
    message(FATAL_ERROR "slopfab: RoPE shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_VAE_ROPE_SOURCE}" SLOPFAB_TENSOR_VAE_ROPE_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_VAE_ROPE_TEXT
    "${SLOPFAB_TENSOR_VAE_ROPE_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_VAE_ROPE_SOURCE_HASH
    "${SLOPFAB_TENSOR_VAE_ROPE_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_VAE_ROPE_SPV}" SLOPFAB_TENSOR_VAE_ROPE_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_VAE_ROPE_SOURCE_HASH STREQUAL
      "57f4761647d8f8f1775cd8edce0156ff5a89b6e19adf99b51330d78056b9c8fd" OR
     NOT SLOPFAB_TENSOR_VAE_ROPE_SPV_HASH STREQUAL
      "e51ae5f500a65b87b5fd1af440e0b60db375a75dc7021620fae40379812a8196")
    message(FATAL_ERROR "slopfab: VAE RoPE shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_WEIGHT_SOURCE}" SLOPFAB_TENSOR_WEIGHT_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_WEIGHT_TEXT
    "${SLOPFAB_TENSOR_WEIGHT_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_WEIGHT_SOURCE_HASH "${SLOPFAB_TENSOR_WEIGHT_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_WEIGHT_SPV}" SLOPFAB_TENSOR_WEIGHT_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_WEIGHT_DENORM_SPV}"
    SLOPFAB_TENSOR_WEIGHT_DENORM_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_WEIGHT_SOURCE_HASH STREQUAL
      "55fadb68982abcdb0079c8d9b0a84e77f64d268e63e7227f2a2bc770ed60307f" OR
     NOT SLOPFAB_TENSOR_WEIGHT_SPV_HASH STREQUAL
      "da6454d969f13f02f6571eaa5850e7cd757e9ab0df68e005557b277c4d73f30e" OR
     NOT SLOPFAB_TENSOR_WEIGHT_DENORM_SPV_HASH STREQUAL
      "d67d8f272b694181a350760c29c4adaccaff692b043e6fa8a619d0a8cb807dae")
    message(FATAL_ERROR "slopfab: tensor-weight shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_GEMM_SOURCE}" SLOPFAB_TENSOR_GEMM_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_GEMM_TEXT
    "${SLOPFAB_TENSOR_GEMM_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_GEMM_SOURCE_HASH "${SLOPFAB_TENSOR_GEMM_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_GEMM_SPV}" SLOPFAB_TENSOR_GEMM_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_GEMM_DENORM_SPV}"
    SLOPFAB_TENSOR_GEMM_DENORM_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_GEMM_SOURCE_HASH STREQUAL
      "86e608b81ba51dd0f827aa53a07d91cc8b2c0542c764dec3d16c937afcdb2f31" OR
     NOT SLOPFAB_TENSOR_GEMM_SPV_HASH STREQUAL
      "f91563d2de7553c84059a416812e2a4241c3f500206d57fa1655af0e3d3061b1" OR
     NOT SLOPFAB_TENSOR_GEMM_DENORM_SPV_HASH STREQUAL
      "d606a8e6ee510b8d32de4341af977d478121e0101cb8274c23640feabe1c83f9")
    message(FATAL_ERROR "slopfab: tensor-gemm shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_GEMM_COOP_SOURCE}" SLOPFAB_TENSOR_GEMM_COOP_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_GEMM_COOP_TEXT
    "${SLOPFAB_TENSOR_GEMM_COOP_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_GEMM_COOP_SOURCE_HASH
    "${SLOPFAB_TENSOR_GEMM_COOP_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_GEMM_COOP_SPV}"
    SLOPFAB_TENSOR_GEMM_COOP_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_GEMM_COOP_DENORM_SPV}"
    SLOPFAB_TENSOR_GEMM_COOP_DENORM_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_GEMM_COOP_SOURCE_HASH STREQUAL
      "76589b0446b567f2fdad655cc337ee96cdc2e22f909994815891f84a3e4c5691" OR
     NOT SLOPFAB_TENSOR_GEMM_COOP_SPV_HASH STREQUAL
      "b9719969312c4962006cd0877dbdadb2aab8e4c8232be84d593f4267f7597cf9" OR
     NOT SLOPFAB_TENSOR_GEMM_COOP_DENORM_SPV_HASH STREQUAL
      "b5b80ac23262aab3285aac2f763910d56d6aaed256da920274fd6d7778ac4ea9")
    message(FATAL_ERROR "slopfab: cooperative GEMM shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_GEMM_PREPARE_SOURCE}"
    SLOPFAB_TENSOR_GEMM_PREPARE_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_GEMM_PREPARE_TEXT
    "${SLOPFAB_TENSOR_GEMM_PREPARE_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_GEMM_PREPARE_SOURCE_HASH
    "${SLOPFAB_TENSOR_GEMM_PREPARE_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_GEMM_PREPARE_SPV}"
    SLOPFAB_TENSOR_GEMM_PREPARE_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_GEMM_PREPARE_SOURCE_HASH STREQUAL
      "c8914cf9427fd5a904b7c503de1f5f1b63e2bb9ec37dca46aa6823b96acc8063" OR
     NOT SLOPFAB_TENSOR_GEMM_PREPARE_SPV_HASH STREQUAL
      "d716981e9734554c9e18e9587959fa70770e768df94f1aa88478192b4727aa59")
    message(FATAL_ERROR "slopfab: GEMM prepare shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_GEMM_COOP_F16_SOURCE}"
    SLOPFAB_TENSOR_GEMM_COOP_F16_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_GEMM_COOP_F16_TEXT
    "${SLOPFAB_TENSOR_GEMM_COOP_F16_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_GEMM_COOP_F16_SOURCE_HASH
    "${SLOPFAB_TENSOR_GEMM_COOP_F16_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_GEMM_COOP_F16_SPV}"
    SLOPFAB_TENSOR_GEMM_COOP_F16_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_GEMM_COOP_F16_DENORM_SPV}"
    SLOPFAB_TENSOR_GEMM_COOP_F16_DENORM_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_GEMM_COOP_F16_SOURCE_HASH STREQUAL
      "11b75eefefaef0eff8266bae7d5e4ce0d1b6ebaa43c29f2d0cb9da5b12e6203f" OR
     NOT SLOPFAB_TENSOR_GEMM_COOP_F16_SPV_HASH STREQUAL
      "e76c1a114f177a4eaa48d69a1df7c102c06e1109ea4512c04b4335e99fea537d" OR
     NOT SLOPFAB_TENSOR_GEMM_COOP_F16_DENORM_SPV_HASH STREQUAL
      "9b51be611b7f6477f6f5d4c82daba6e1f900f29a06c9811f987126bda9d23c9a")
    message(FATAL_ERROR "slopfab: cooperative F16 GEMM shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_ATTENTION_BLOCKED_SOURCE}"
    SLOPFAB_TENSOR_ATTENTION_BLOCKED_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_ATTENTION_BLOCKED_TEXT
    "${SLOPFAB_TENSOR_ATTENTION_BLOCKED_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_ATTENTION_BLOCKED_SOURCE_HASH
    "${SLOPFAB_TENSOR_ATTENTION_BLOCKED_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_ATTENTION_BLOCKED_SPV}"
    SLOPFAB_TENSOR_ATTENTION_BLOCKED_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_ATTENTION_BLOCKED_SOURCE_HASH STREQUAL
      "c27a8133ad290d086e1aa0c03418dd87b5f3eda924e69f0959ac3d9562f840b7" OR
     NOT SLOPFAB_TENSOR_ATTENTION_BLOCKED_SPV_HASH STREQUAL
      "9c1339b2fd44b9f453bd3974f720e635682130ce9f808411823c3657a97098f2")
    message(FATAL_ERROR "slopfab: blocked-attention shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_ATTENTION_PREPARE_SOURCE}"
    SLOPFAB_TENSOR_ATTENTION_PREPARE_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_ATTENTION_PREPARE_TEXT
    "${SLOPFAB_TENSOR_ATTENTION_PREPARE_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_ATTENTION_PREPARE_SOURCE_HASH
    "${SLOPFAB_TENSOR_ATTENTION_PREPARE_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV}"
    SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_ATTENTION_PREPARE_SOURCE_HASH STREQUAL
      "786295c4e33eedc7f67317b9ecf6b1bdea0108b319ae5b5e8d57b6b3cea4677d" OR
     NOT SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV_HASH STREQUAL
      "56dc48503f296776cd1c105d0dc44d34e5f8768b35a8fee6ec73efaa9d3ff8f4")
    message(FATAL_ERROR "slopfab: attention-prepare shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SOURCE}"
    SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_TEXT
    "${SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SOURCE_HASH
    "${SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SPV}"
    SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SOURCE_HASH STREQUAL
      "dd700e2fdc18ed483973b2e161aea3f1f43e8f2db18fc8796f800be766a79930" OR
     NOT SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SPV_HASH STREQUAL
      "9f8b4480179c01cc26a3e467d1f0915d606594388dab8b5c0856770b2e7e3778")
    message(FATAL_ERROR "slopfab: causal-GQA attention shader/source hash mismatch")
  endif()
  file(READ "${SLOPFAB_TENSOR_ATTENTION_H3_SOURCE}"
    SLOPFAB_TENSOR_ATTENTION_H3_TEXT)
  string(REPLACE "\r\n" "\n" SLOPFAB_TENSOR_ATTENTION_H3_TEXT
    "${SLOPFAB_TENSOR_ATTENTION_H3_TEXT}")
  string(SHA256 SLOPFAB_TENSOR_ATTENTION_H3_SOURCE_HASH
    "${SLOPFAB_TENSOR_ATTENTION_H3_TEXT}")
  file(SHA256 "${SLOPFAB_TENSOR_ATTENTION_H3_SPV}"
    SLOPFAB_TENSOR_ATTENTION_H3_SPV_HASH)
  file(SHA256 "${SLOPFAB_TENSOR_ATTENTION_H3_BANDED_SPV}"
    SLOPFAB_TENSOR_ATTENTION_H3_BANDED_SPV_HASH)
  if(NOT SLOPFAB_TENSOR_ATTENTION_H3_SOURCE_HASH STREQUAL
      "b9e51135b436dff97ce6463f4731965e7466530dc1b1e69d3fd7444b3cb6ba2b" OR
     NOT SLOPFAB_TENSOR_ATTENTION_H3_SPV_HASH STREQUAL
      "4fd87fbdbe7ae6ec6c40c67aad6fd39a0b9ef05f6e3cc1a5ed688a5b17fbb37f" OR
     NOT SLOPFAB_TENSOR_ATTENTION_H3_BANDED_SPV_HASH STREQUAL
      "9564933f40b33b8c6077cb073f6e0cb78ad40627cd0c973ff16897cab1764efb")
    message(FATAL_ERROR "slopfab: H3 attention shader/source hash mismatch")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    src/vulkan/tensor_reference.comp src/vulkan/tensor_reference.comp.spv
    src/vulkan/reference.sha256)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_reference.comp" REFERENCE_SOURCE)
  string(REPLACE "\r\n" "\n" REFERENCE_SOURCE "${REFERENCE_SOURCE}")
  string(SHA256 REFERENCE_SOURCE_HASH "${REFERENCE_SOURCE}")
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_reference.comp.spv" REFERENCE_SPV_HASH)
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/reference.sha256" REFERENCE_HASHES)
  list(GET REFERENCE_HASHES 0 EXPECTED_REFERENCE_SOURCE)
  list(GET REFERENCE_HASHES 1 EXPECTED_REFERENCE_SPV)
  if(NOT REFERENCE_SOURCE_HASH STREQUAL EXPECTED_REFERENCE_SOURCE OR
     NOT REFERENCE_SPV_HASH STREQUAL EXPECTED_REFERENCE_SPV)
    message(FATAL_ERROR "slopfab: reference shader/source hash mismatch")
  endif()
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_reference.comp.spv" REFERENCE_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," SLOPFAB_TENSOR_REFERENCE_SPV_BYTES "${REFERENCE_HEX}")
  file(READ "${SLOPFAB_TENSOR_OPS_SPV}" SLOPFAB_TENSOR_OPS_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," SLOPFAB_TENSOR_OPS_SPV_BYTES
    "${SLOPFAB_TENSOR_OPS_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_OPS_DENORM_SPV}" SLOPFAB_TENSOR_OPS_DENORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_OPS_DENORM_SPV_BYTES "${SLOPFAB_TENSOR_OPS_DENORM_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_RMS_NORM_SPV}" SLOPFAB_TENSOR_RMS_NORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," SLOPFAB_TENSOR_RMS_NORM_SPV_BYTES
    "${SLOPFAB_TENSOR_RMS_NORM_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_LAYER_NORM_SPV}" SLOPFAB_TENSOR_LAYER_NORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_LAYER_NORM_SPV_BYTES "${SLOPFAB_TENSOR_LAYER_NORM_SPV_HEX}")
  foreach(VARIANT BF16_RMS_BLOCK BF16_RMS_NARROW BF16_LAYER BF16_MOD FP32_MOD)
    file(READ "${SLOPFAB_TENSOR_${VARIANT}_SPV}"
      SLOPFAB_TENSOR_${VARIANT}_SPV_HEX HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
      SLOPFAB_TENSOR_${VARIANT}_SPV_BYTES "${SLOPFAB_TENSOR_${VARIANT}_SPV_HEX}")
  endforeach()
  file(READ "${SLOPFAB_TENSOR_GROUP_NORM_SPV}"
    SLOPFAB_TENSOR_GROUP_NORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_GROUP_NORM_SPV_BYTES "${SLOPFAB_TENSOR_GROUP_NORM_SPV_HEX}")
  foreach(VARIANT RESIDUAL SWIGLU DENORM)
    file(READ "${SLOPFAB_TENSOR_VAE_${VARIANT}_SPV}"
      SLOPFAB_TENSOR_VAE_${VARIANT}_SPV_HEX HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
      SLOPFAB_TENSOR_VAE_${VARIANT}_SPV_BYTES
      "${SLOPFAB_TENSOR_VAE_${VARIANT}_SPV_HEX}")
  endforeach()
  file(READ "${SLOPFAB_TENSOR_AUDIO_SPV}" SLOPFAB_TENSOR_AUDIO_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_AUDIO_SPV_BYTES "${SLOPFAB_TENSOR_AUDIO_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_KEYFRAME_SPV}"
    SLOPFAB_TENSOR_KEYFRAME_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_KEYFRAME_SPV_BYTES "${SLOPFAB_TENSOR_KEYFRAME_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_DIT_SPV}" SLOPFAB_TENSOR_DIT_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_DIT_SPV_BYTES "${SLOPFAB_TENSOR_DIT_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_ROPE_SPV}" SLOPFAB_TENSOR_ROPE_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_ROPE_SPV_BYTES "${SLOPFAB_TENSOR_ROPE_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_ROPE_DENORM_SPV}"
    SLOPFAB_TENSOR_ROPE_DENORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_ROPE_DENORM_SPV_BYTES "${SLOPFAB_TENSOR_ROPE_DENORM_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_VAE_ROPE_SPV}" SLOPFAB_TENSOR_VAE_ROPE_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_VAE_ROPE_SPV_BYTES "${SLOPFAB_TENSOR_VAE_ROPE_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_WEIGHT_SPV}" SLOPFAB_TENSOR_WEIGHT_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_WEIGHT_SPV_BYTES "${SLOPFAB_TENSOR_WEIGHT_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_WEIGHT_DENORM_SPV}"
    SLOPFAB_TENSOR_WEIGHT_DENORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_WEIGHT_DENORM_SPV_BYTES "${SLOPFAB_TENSOR_WEIGHT_DENORM_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_GEMM_SPV}" SLOPFAB_TENSOR_GEMM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_GEMM_SPV_BYTES "${SLOPFAB_TENSOR_GEMM_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_GEMM_DENORM_SPV}"
    SLOPFAB_TENSOR_GEMM_DENORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_GEMM_DENORM_SPV_BYTES "${SLOPFAB_TENSOR_GEMM_DENORM_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_GEMM_COOP_SPV}"
    SLOPFAB_TENSOR_GEMM_COOP_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_GEMM_COOP_SPV_BYTES "${SLOPFAB_TENSOR_GEMM_COOP_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_GEMM_COOP_DENORM_SPV}"
    SLOPFAB_TENSOR_GEMM_COOP_DENORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_GEMM_COOP_DENORM_SPV_BYTES
    "${SLOPFAB_TENSOR_GEMM_COOP_DENORM_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_GEMM_PREPARE_SPV}"
    SLOPFAB_TENSOR_GEMM_PREPARE_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_GEMM_PREPARE_SPV_BYTES
    "${SLOPFAB_TENSOR_GEMM_PREPARE_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_GEMM_COOP_F16_SPV}"
    SLOPFAB_TENSOR_GEMM_COOP_F16_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_GEMM_COOP_F16_SPV_BYTES
    "${SLOPFAB_TENSOR_GEMM_COOP_F16_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_GEMM_COOP_F16_DENORM_SPV}"
    SLOPFAB_TENSOR_GEMM_COOP_F16_DENORM_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_GEMM_COOP_F16_DENORM_SPV_BYTES
    "${SLOPFAB_TENSOR_GEMM_COOP_F16_DENORM_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_ATTENTION_BLOCKED_SPV}"
    SLOPFAB_TENSOR_ATTENTION_BLOCKED_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_ATTENTION_BLOCKED_SPV_BYTES
    "${SLOPFAB_TENSOR_ATTENTION_BLOCKED_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV}"
    SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV_BYTES
    "${SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SPV}"
    SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SPV_BYTES
    "${SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_ATTENTION_H3_SPV}"
    SLOPFAB_TENSOR_ATTENTION_H3_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_ATTENTION_H3_SPV_BYTES
    "${SLOPFAB_TENSOR_ATTENTION_H3_SPV_HEX}")
  file(READ "${SLOPFAB_TENSOR_ATTENTION_H3_BANDED_SPV}"
    SLOPFAB_TENSOR_ATTENTION_H3_BANDED_SPV_HEX HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1,"
    SLOPFAB_TENSOR_ATTENTION_H3_BANDED_SPV_BYTES
    "${SLOPFAB_TENSOR_ATTENTION_H3_BANDED_SPV_HEX}")
  # Fast attention is checked in like the exact modules. Verify normalized
  # sources and binaries together; rebuild with tools/build_vulkan_attention.py.
  set(SLOPFAB_FAST_ATTENTION_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/attention_fast.sha256")
  file(STRINGS "${SLOPFAB_FAST_ATTENTION_MANIFEST}" SLOPFAB_FAST_ATTENTION_HASHES)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SLOPFAB_FAST_ATTENTION_MANIFEST}")
  foreach(ENTRY IN LISTS SLOPFAB_FAST_ATTENTION_HASHES)
    string(REGEX MATCH "^([0-9a-f]+)  (.+)$" MATCHED "${ENTRY}")
    if(NOT MATCHED)
      message(FATAL_ERROR "slopfab: malformed fast attention shader manifest")
    endif()
    set(EXPECTED_HASH "${CMAKE_MATCH_1}")
    set(SHADER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/${CMAKE_MATCH_2}")
    if(SHADER_FILE MATCHES "\\.comp$")
      file(READ "${SHADER_FILE}" SOURCE_TEXT)
      string(REPLACE "\r\n" "\n" SOURCE_TEXT "${SOURCE_TEXT}")
      string(SHA256 ACTUAL_HASH "${SOURCE_TEXT}")
    else()
      file(SHA256 "${SHADER_FILE}" ACTUAL_HASH)
    endif()
    if(NOT ACTUAL_HASH STREQUAL EXPECTED_HASH)
      message(FATAL_ERROR "slopfab: fast attention shader/source hash mismatch: ${SHADER_FILE}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SHADER_FILE}")
  endforeach()
  set(SLOPFAB_SAGE_VARIANT_DECLARATIONS "")
  foreach(VARIANT FLASH FLASH_BANDED SAGE SAGE_BANDED SAGE_PREPARE
      SAGE_FULL SAGE_FULL_BANDED SAGE_WIDE SAGE_WIDE_BANDED SAGE_WIDE_FULL SAGE_WIDE_FULL_BANDED
      SAGE_SG64 SAGE_SG64_BANDED SAGE_FULL_SG64 SAGE_FULL_SG64_BANDED
      SAGE_WIDE_SG64 SAGE_WIDE_SG64_BANDED SAGE_WIDE_FULL_SG64 SAGE_WIDE_FULL_SG64_BANDED)
    string(TOLOWER "${VARIANT}" LOWER_VARIANT)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/tensor_attention_${LOWER_VARIANT}.comp.spv" SHADER_HEX HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," SLOPFAB_TENSOR_ATTENTION_${VARIANT}_SPV_BYTES "${SHADER_HEX}")
    if(NOT VARIANT MATCHES "^(FLASH|FLASH_BANDED|SAGE|SAGE_BANDED|SAGE_PREPARE)$")
      string(APPEND SLOPFAB_SAGE_VARIANT_DECLARATIONS
        "inline constexpr uint8_t kTensorAttention_${VARIANT}[] = {${SLOPFAB_TENSOR_ATTENTION_${VARIANT}_SPV_BYTES}};\n")
    endif()
  endforeach()
  configure_file(src/vulkan/embedded_tensor_spv.h.in
    generated/vulkan/embedded_tensor_spv.h @ONLY)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${SLOPFAB_YUV_SHADER_SOURCE}" "${SLOPFAB_YUV_SHADER_SPV}"
    "${SLOPFAB_TENSOR_OPS_SOURCE}"
    "${SLOPFAB_TENSOR_OPS_SPV}" "${SLOPFAB_TENSOR_OPS_DENORM_SPV}"
    "${SLOPFAB_TENSOR_NORM_SOURCE}" "${SLOPFAB_TENSOR_RMS_NORM_SPV}"
    "${SLOPFAB_TENSOR_LAYER_NORM_SPV}" "${SLOPFAB_TENSOR_SHARED_NORM_SOURCE}"
    "${SLOPFAB_TENSOR_BF16_RMS_BLOCK_SPV}" "${SLOPFAB_TENSOR_BF16_RMS_NARROW_SPV}"
    "${SLOPFAB_TENSOR_BF16_LAYER_SPV}" "${SLOPFAB_TENSOR_BF16_MOD_SPV}"
    "${SLOPFAB_TENSOR_FP32_MOD_SPV}" "${SLOPFAB_TENSOR_GROUP_NORM_SOURCE}"
    "${SLOPFAB_TENSOR_GROUP_NORM_SPV}" "${SLOPFAB_TENSOR_VAE_POINTWISE_SOURCE}"
    "${SLOPFAB_TENSOR_VAE_RESIDUAL_SPV}" "${SLOPFAB_TENSOR_VAE_SWIGLU_SPV}"
    "${SLOPFAB_TENSOR_VAE_DENORM_SPV}" "${SLOPFAB_TENSOR_AUDIO_SOURCE}"
    "${SLOPFAB_TENSOR_AUDIO_SPV}" "${SLOPFAB_TENSOR_KEYFRAME_SOURCE}"
    "${SLOPFAB_TENSOR_KEYFRAME_SPV}" "${SLOPFAB_TENSOR_DIT_SOURCE}"
    "${SLOPFAB_TENSOR_DIT_SPV}" "${SLOPFAB_TENSOR_ROPE_SOURCE}"
    "${SLOPFAB_TENSOR_ROPE_SPV}" "${SLOPFAB_TENSOR_ROPE_DENORM_SPV}"
    "${SLOPFAB_TENSOR_VAE_ROPE_SOURCE}" "${SLOPFAB_TENSOR_VAE_ROPE_SPV}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${SLOPFAB_TENSOR_WEIGHT_SOURCE}" "${SLOPFAB_TENSOR_WEIGHT_SPV}"
    "${SLOPFAB_TENSOR_WEIGHT_DENORM_SPV}" "${SLOPFAB_TENSOR_GEMM_SOURCE}"
    "${SLOPFAB_TENSOR_GEMM_SPV}" "${SLOPFAB_TENSOR_GEMM_DENORM_SPV}"
    "${SLOPFAB_TENSOR_GEMM_COOP_SOURCE}" "${SLOPFAB_TENSOR_GEMM_COOP_SPV}"
    "${SLOPFAB_TENSOR_GEMM_COOP_DENORM_SPV}"
    "${SLOPFAB_TENSOR_GEMM_PREPARE_SOURCE}" "${SLOPFAB_TENSOR_GEMM_PREPARE_SPV}"
    "${SLOPFAB_TENSOR_GEMM_COOP_F16_SOURCE}" "${SLOPFAB_TENSOR_GEMM_COOP_F16_SPV}"
    "${SLOPFAB_TENSOR_GEMM_COOP_F16_DENORM_SPV}"
    "${SLOPFAB_TENSOR_ATTENTION_BLOCKED_SOURCE}"
    "${SLOPFAB_TENSOR_ATTENTION_BLOCKED_SPV}"
    "${SLOPFAB_TENSOR_ATTENTION_PREPARE_SOURCE}"
    "${SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV}"
    "${SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SOURCE}"
    "${SLOPFAB_TENSOR_ATTENTION_CAUSAL_GQA_SPV}"
    "${SLOPFAB_TENSOR_ATTENTION_H3_SOURCE}"
    "${SLOPFAB_TENSOR_ATTENTION_H3_SPV}"
    "${SLOPFAB_TENSOR_ATTENTION_H3_BANDED_SPV}")

  add_library(slopfab_vulkan STATIC
    src/vulkan/runtime.cpp
    src/vulkan/tensor.cpp
    src/vulkan/text_layer.cpp
    src/vulkan/text_encoder.cpp
    src/vulkan/vision_stage.cpp
    src/vulkan/lora.cpp
    src/vulkan/dit_block.cpp
    src/vulkan/dit_denoise.cpp
    src/vulkan/dit_graph.cpp
    src/vulkan/dit_transformer.cpp
    src/vulkan/audio_decoder.cpp
    src/vulkan/keyframe_encoder.cpp
    src/vulkan/reference_encoder.cpp
    src/vulkan/vae_decoder.cpp
    src/vulkan/vae_vit_block.cpp
    src/vulkan/yuv_converter.cpp)
  target_include_directories(slopfab_vulkan
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/include
            ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan)
  target_link_libraries(slopfab_vulkan PUBLIC slopfab_core)
  if(SLOPFAB_BUILD_DEV_TOOLS)
    add_executable(slopfab_attentionbench tools/attentionbench.cpp)
    target_link_libraries(slopfab_attentionbench PRIVATE slopfab_vulkan)
    if(TARGET slopfab_cuda)
      target_link_libraries(slopfab_attentionbench PRIVATE slopfab_cuda)
    endif()
  endif()
  target_compile_definitions(slopfab_vulkan PRIVATE VK_NO_PROTOTYPES)
  if(CMAKE_DL_LIBS)
    target_link_libraries(slopfab_vulkan PRIVATE ${CMAKE_DL_LIBS})
  endif()
  if(MSVC)
    target_compile_options(slopfab_vulkan PRIVATE /W4 /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(slopfab_vulkan PRIVATE -Wall -Wextra)
  endif()
endif()

# --- cli --------------------------------------------------------------------

if(WIN32)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ref/text_encoder/tokenizer.json")
    message(FATAL_ERROR "slopfab: ref/text_encoder/tokenizer.json is required to embed the tokenizer")
  endif()
  set(SLOPFAB_LICENSE_RESOURCE "${CMAKE_CURRENT_BINARY_DIR}/generated/minimax_h3_LICENSE")
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
  set(SLOPFAB_LICENSE_MARKER "MiniMax H3 COMMUNITY LICENSE AGREEMENT")

  function(slopfab_check_license path out_var)
    set(${out_var} FALSE PARENT_SCOPE)
    if(NOT EXISTS "${path}")
      return()
    endif()
    file(SIZE "${path}" license_bytes)
    if(license_bytes EQUAL 0)
      return()
    endif()
    file(READ "${path}" license_head LIMIT 4096)
    string(FIND "${license_head}" "${SLOPFAB_LICENSE_MARKER}" license_marker_at)
    if(license_marker_at EQUAL -1)
      return()
    endif()
    set(${out_var} TRUE PARENT_SCOPE)
  endfunction()

  slopfab_check_license("${SLOPFAB_LICENSE_RESOURCE}" SLOPFAB_LICENSE_PRESENT)
  if(NOT SLOPFAB_LICENSE_PRESENT)
    file(DOWNLOAD
      "https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE"
      "${SLOPFAB_LICENSE_RESOURCE}"
      STATUS SLOPFAB_LICENSE_DOWNLOAD_STATUS
      TLS_VERIFY ON)
    list(GET SLOPFAB_LICENSE_DOWNLOAD_STATUS 0 SLOPFAB_LICENSE_DOWNLOAD_CODE)
    # Applied to what actually arrived, not just to the transport result, so an
    # error page is rejected on the way in as well as out of the cache.
    slopfab_check_license("${SLOPFAB_LICENSE_RESOURCE}" SLOPFAB_LICENSE_PRESENT)
    if(NOT SLOPFAB_LICENSE_DOWNLOAD_CODE EQUAL 0 OR NOT SLOPFAB_LICENSE_PRESENT)
      if(SLOPFAB_LICENSE_DOWNLOAD_CODE EQUAL 0)
        # The transfer succeeded and the body is not the licence. Usually that
        # means an error page, but it is also what a lawful upstream retitle
        # looks like from here, and the reader has no way to tell which without
        # being told where the expectation lives.
        string(CONCAT SLOPFAB_LICENSE_DOWNLOAD_MESSAGE
          "the server returned something that is not the MiniMax H3 license. "
          "The transfer itself succeeded, so this is either an error page or a "
          "document whose text no longer contains SLOPFAB_LICENSE_MARKER "
          "(\"${SLOPFAB_LICENSE_MARKER}\"). Fetch the URL by hand to tell which; "
          "if upstream legitimately retitled the document, update that marker "
          "in CMakeLists.txt to match")
      else()
        list(GET SLOPFAB_LICENSE_DOWNLOAD_STATUS 1 SLOPFAB_LICENSE_DOWNLOAD_MESSAGE)
      endif()
      file(REMOVE "${SLOPFAB_LICENSE_RESOURCE}")
      message(FATAL_ERROR "slopfab: could not download MiniMax H3 license: ${SLOPFAB_LICENSE_DOWNLOAD_MESSAGE}")
    endif()
  endif()
  file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ref/text_encoder/tokenizer.json"
       SLOPFAB_TOKENIZER_RESOURCE)
  file(TO_CMAKE_PATH "${SLOPFAB_LICENSE_RESOURCE}" SLOPFAB_LICENSE_RESOURCE)
  configure_file(src/slopfab_tokenizer.rc.in generated/slopfab_tokenizer.rc @ONLY)
  configure_file(src/slopfab_license.rc.in generated/slopfab_license.rc @ONLY)

  # A static library cannot usefully carry a .res — the linker has no reference
  # to pull it in with — so the executable embeds both resources itself.
  set(SLOPFAB_EXE_RESOURCES
    "${CMAKE_CURRENT_BINARY_DIR}/generated/slopfab_tokenizer.rc"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/slopfab_license.rc")
  set(SLOPFAB_CLI_SOURCES src/main.cpp ${SLOPFAB_EXE_RESOURCES})
else()
  set(SLOPFAB_CLI_SOURCES src/main.cpp)
endif()

# The CLI and C API use the same in-process CUDA core. On Windows cuBLAS is
# late-bound, so neither module has a fixed CUDA-major import and no launcher
# or toolkit-specific backend executable is needed.
add_executable(slopfab ${SLOPFAB_CLI_SOURCES})
target_sources(slopfab PRIVATE src/cli/reference_decode.cpp)
set(SLOPFAB_CLI_TARGET slopfab)

target_link_libraries(${SLOPFAB_CLI_TARGET} PRIVATE slopfab_core)
target_compile_definitions(${SLOPFAB_CLI_TARGET} PRIVATE
  SLOPFAB_WITH_VULKAN=$<IF:$<BOOL:${SLOPFAB_ENABLE_VULKAN}>,1,0>)
if(SLOPFAB_ENABLE_VULKAN)
  target_link_libraries(${SLOPFAB_CLI_TARGET} PRIVATE slopfab_vulkan)
endif()
if(SLOPFAB_ENABLE_CUDA)
  target_link_libraries(${SLOPFAB_CLI_TARGET} PRIVATE slopfab_cuda)
endif()

if(MSVC)
  target_compile_options(${SLOPFAB_CLI_TARGET} PRIVATE /W4 /permissive- /utf-8 /EHsc)
else()
  target_compile_options(${SLOPFAB_CLI_TARGET} PRIVATE -Wall -Wextra)
endif()
if(WIN32)
  target_link_libraries(${SLOPFAB_CLI_TARGET} PRIVATE winhttp)
endif()

# Staging the FFmpeg runtime beside the executable, which only a build that
# compiled the muxer has any use for. Guarded rather than left unconditional
# because the check below is a FATAL_ERROR: without this, configuring with
# -DSLOPFAB_WITH_FFMPEG=OFF would still refuse to proceed without the very
# distribution the option exists to do without.
if(WIN32 AND SLOPFAB_WITH_FFMPEG)
  # FFmpeg remains runtime-loaded (and therefore replaceable), but the checked-in
  # distribution is the default used by local builds. Keep only the libraries
  # mux.cpp loads, plus swresample which avcodec depends on in this FFmpeg build.
  set(SLOPFAB_FFMPEG_BIN "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin")
  set(SLOPFAB_FFMPEG_RUNTIME_DLLS
    avcodec-62.dll
    avformat-62.dll
    avutil-60.dll
    swresample-6.dll
    swscale-9.dll)
  foreach(SLOPFAB_FFMPEG_DLL IN LISTS SLOPFAB_FFMPEG_RUNTIME_DLLS)
    if(NOT EXISTS "${SLOPFAB_FFMPEG_BIN}/${SLOPFAB_FFMPEG_DLL}")
      message(FATAL_ERROR
        "slopfab: required FFmpeg runtime is missing: ${SLOPFAB_FFMPEG_BIN}/${SLOPFAB_FFMPEG_DLL}")
    endif()
    add_custom_command(TARGET ${SLOPFAB_CLI_TARGET} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SLOPFAB_FFMPEG_BIN}/${SLOPFAB_FFMPEG_DLL}"
        "$<TARGET_FILE_DIR:slopfab>/${SLOPFAB_FFMPEG_DLL}"
      VERBATIM)
  endforeach()
endif()

# --- c api ------------------------------------------------------------------

# slopfab_c builds slopfab.dll, the stable C ABI with a hand-written export set, for
# Rust, C#, Python and anything else with an FFI. See include/slopfab/capi.h.
#
# Dependent on CUDA rather than merely checked against it, because the C API
# exists to run generations and `run_generate` is on the CUDA side. Making it a
# plain option defaulting to ON meant that a machine with no toolkit — the case
# the lazy-CUDA design at the top of this file exists to support — stopped
# configuring at all, since the default would then contradict a hard error two
# hundred lines earlier. This forces it off in that situation instead, and says
# so, and an explicit -DSLOPFAB_BUILD_C_API=ON cannot resurrect it.
include(CMakeDependentOption)
cmake_dependent_option(SLOPFAB_BUILD_C_API
  "Build slopfab_c, the stable C ABI shared library" ON
  "SLOPFAB_ENABLE_CUDA" OFF)

if(NOT SLOPFAB_BUILD_C_API AND NOT SLOPFAB_ENABLE_CUDA)
  message(STATUS "slopfab: no CUDA, so slopfab_c (the C ABI DLL) is not built")
endif()

if(SLOPFAB_BUILD_C_API)
  # The internal core and CUDA libraries are static and link into this target.
  # Consequently slopfab.dll is the only project DLL, and its exports are exactly
  # the C entry points — no unstable C++ ABI and no companion slopfab DLLs.
  #
  # Static cudart is embedded (see above), while the shared in-process loader
  # resolves CUDA 13/12 cuBLAS by absolute toolkit path on first use. Thus the
  # DLL and CLI have identical selection/error behavior and neither has a
  # fixed cublas64_<major>.dll import.
  add_library(slopfab_c SHARED src/capi/capi.cpp)
  # slopfab_cuda carries slopfab_core with it as a PUBLIC dependency, so naming
  # the one target is naming both.
  target_link_libraries(slopfab_c PRIVATE slopfab_cuda)

  # PRIVATE: consumers of the DLL include capi.h and nothing else, and the
  # install rule below places it for them. Anything in this build tree that
  # needs the header asks for the include directory itself.
  target_include_directories(slopfab_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # Switches capi.h from dllimport to dllexport. It is the only definition the
  # C API needs: the version it reports is stated in the header next to the
  # constants a consumer compiles against, so the two cannot drift.
  target_compile_definitions(slopfab_c PRIVATE SLOPFAB_C_BUILD=1)

  # The static libraries end up inside a shared object here, which on ELF
  # targets requires every object in them to be position independent. Harmless
  # and ignored on Windows.
  set_property(TARGET slopfab_core PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET slopfab_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)

  # The export set is the point of this target, so it is stated rather than
  # inherited. WINDOWS_EXPORT_ALL_SYMBOLS is off explicitly because a build
  # that sets CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS globally would otherwise dump
  # the whole C++ surface into this DLL's export table alongside the C one, and
  # the resulting module would still work — silently defeating the boundary
  # instead of failing.
  #
  # The version is the *C ABI's*, not the project's. They are different numbers
  # answering different questions — capi.h declares its own version and tells
  # a binding to refuse a different major, while PROJECT_VERSION is 0.1.0 and
  # moves whenever anything in the project does. Using the latter here would
  # produce a libslopfab.so.0 whose soname says 0 for an ABI that says 1, and
  # would bump it for releases that did not touch the C interface at all.
  set(SLOPFAB_CAPI_VERSION_MAJOR 1)
  set(SLOPFAB_CAPI_VERSION_MINOR 8)
  set(SLOPFAB_CAPI_VERSION_PATCH 0)
  set(SLOPFAB_CAPI_VERSION
    "${SLOPFAB_CAPI_VERSION_MAJOR}.${SLOPFAB_CAPI_VERSION_MINOR}.${SLOPFAB_CAPI_VERSION_PATCH}")

  set_target_properties(slopfab_c PROPERTIES
    OUTPUT_NAME slopfab
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${SLOPFAB_CAPI_VERSION}
    SOVERSION ${SLOPFAB_CAPI_VERSION_MAJOR})
  if(WIN32)
    # The DLL and CLI intentionally share the runtime basename, but an MSVC
    # executable with exported symbols can also emit slopfab.lib. Give the DLL
    # import archive its own name so parallel aggregate builds cannot overwrite
    # the library C-API tests are about to link.
    set_target_properties(slopfab_c PROPERTIES ARCHIVE_OUTPUT_NAME slopfab_c)
  endif()

  # The visibility presets above are *compile* properties, so they only cover
  # capi.cpp. slopfab_core and slopfab_cuda are compiled with default visibility,
  # and on ELF their entire C++ symbol table would otherwise be re-exported
  # from libslopfab.so — precisely what WINDOWS_EXPORT_ALL_SYMBOLS OFF
  # prevents on the other platform. --exclude-libs makes the two agree.
  if(NOT WIN32 AND NOT APPLE)
    target_link_options(slopfab_c PRIVATE "LINKER:--exclude-libs,ALL")
  endif()

  # capi.h states the same three numbers for the compiler; if they ever
  # disagree, the soname and the number a binding checks would describe
  # different interfaces. Cheaper to catch here than in a consumer.
  #
  # Matched per name rather than by position. Reading the three numbers in file
  # order and joining them would compare 0.1.0 against 1.0.0 — and abort with a
  # message blaming the version rather than the reordering — if someone ever
  # rearranged three `#define`s that C is perfectly happy to see in any order.
  foreach(SLOPFAB_CAPI_PART MAJOR MINOR PATCH)
    unset(SLOPFAB_CAPI_HEADER_${SLOPFAB_CAPI_PART})
    file(STRINGS include/slopfab/capi.h SLOPFAB_CAPI_HEADER_LINE
      REGEX "^#define[ \t]+SLOPFAB_CAPI_VERSION_${SLOPFAB_CAPI_PART}[ \t]+[0-9]+")
    if(NOT SLOPFAB_CAPI_HEADER_LINE)
      message(FATAL_ERROR
        "slopfab: capi.h has no SLOPFAB_CAPI_VERSION_${SLOPFAB_CAPI_PART} to check against")
    endif()
    string(REGEX MATCH "[0-9]+$" SLOPFAB_CAPI_HEADER_${SLOPFAB_CAPI_PART}
      "${SLOPFAB_CAPI_HEADER_LINE}")
  endforeach()
  set(SLOPFAB_CAPI_HEADER_VERSION
    "${SLOPFAB_CAPI_HEADER_MAJOR}.${SLOPFAB_CAPI_HEADER_MINOR}.${SLOPFAB_CAPI_HEADER_PATCH}")
  if(NOT SLOPFAB_CAPI_HEADER_VERSION STREQUAL SLOPFAB_CAPI_VERSION)
    message(FATAL_ERROR
      "slopfab: capi.h declares ABI ${SLOPFAB_CAPI_HEADER_VERSION} but CMakeLists says "
      "${SLOPFAB_CAPI_VERSION}; update both.")
  endif()
  # `file(STRINGS)` creates no dependency of its own, so without this the check
  # would go stale at exactly the moment it exists for: editing the header's
  # version macros would not re-run CMake, and the soname would keep describing
  # the previous ABI.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/slopfab/capi.h")

  # `Tokenizer::load_embedded` looks the resource up in the module holding its
  # own code. That code is linked into slopfab.dll, so the 7 MB tokenizer has to
  # be here too or a C caller that leaves the tokenizer path empty gets
  # "embedded resource is missing".
  if(WIN32)
    target_sources(slopfab_c PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/slopfab_tokenizer.rc")
  endif()

  if(MSVC)
    target_compile_options(slopfab_c PRIVATE /W4 /permissive- /utf-8)
  else()
    target_compile_options(slopfab_c PRIVATE -Wall -Wextra)
  endif()
endif()

# --- tools ------------------------------------------------------------------

# Disk-side load probe. Deliberately a separate binary rather than a `slopfab`
# subcommand: it links slopfab_core only, so it never initialises CUDA and can
# be run while the card is busy with something else.
add_executable(slopfab_loadprobe tools/loadprobe.cpp)
target_link_libraries(slopfab_loadprobe PRIVATE slopfab_core)
if(MSVC)
  target_compile_options(slopfab_loadprobe PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(slopfab_loadprobe PRIVATE -Wall -Wextra)
endif()

# Temporal-chunking driver for the frame-banding quality probe. Separate binary
# for the same reason as the load probe: it links slopfab_core only, so it
# initialises no CUDA context and reads no checkpoint, and it can be run while
# the card is busy with the generations it is preparing or analysing.
add_executable(slopfab_chunkprobe tools/chunkprobe.cpp)
target_link_libraries(slopfab_chunkprobe PRIVATE slopfab_core)
if(MSVC)
  target_compile_options(slopfab_chunkprobe PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(slopfab_chunkprobe PRIVATE -Wall -Wextra)
endif()

add_executable(slopfab_vaeprobe tools/vaeprobe.cpp)
target_link_libraries(slopfab_vaeprobe PRIVATE slopfab_core)
if(MSVC)
  target_compile_options(slopfab_vaeprobe PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(slopfab_vaeprobe PRIVATE -Wall -Wextra)
endif()

if(SLOPFAB_ENABLE_CUDA)
  add_executable(slopfab_vaecudaprobe tools/vaecudaprobe.cpp)
  target_link_libraries(slopfab_vaecudaprobe PRIVATE slopfab_cuda)
  if(MSVC)
    target_compile_options(slopfab_vaecudaprobe PRIVATE /W4 /permissive- /utf-8)
  else()
    target_compile_options(slopfab_vaecudaprobe PRIVATE -Wall -Wextra)
  endif()
endif()

# --- tests ------------------------------------------------------------------
if(SLOPFAB_ENABLE_VULKAN AND SLOPFAB_BUILD_DEV_TOOLS)
  add_executable(slopfab_referenceprobe_vulkan tools/referenceprobe.cpp)
  target_compile_definitions(slopfab_referenceprobe_vulkan PRIVATE SLOPFAB_REFERENCEPROBE_VULKAN=1)
  target_link_libraries(slopfab_referenceprobe_vulkan PRIVATE slopfab_vulkan)
endif()

if(SLOPFAB_ENABLE_CUDA AND SLOPFAB_BUILD_DEV_TOOLS)
  add_executable(slopfab_stillprobe tools/stillprobe.cpp)
  add_executable(slopfab_referenceprobe tools/referenceprobe.cpp)
  target_link_libraries(slopfab_referenceprobe PRIVATE slopfab_cuda)
  target_link_libraries(slopfab_stillprobe PRIVATE slopfab_cuda)
  if(MSVC)
    target_compile_options(slopfab_stillprobe PRIVATE /W4 /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(slopfab_stillprobe PRIVATE -Wall -Wextra)
  endif()
endif()

if(SLOPFAB_BUILD_TESTS)
  enable_testing()
  # Host tests. Each area contributes its own translation unit and registers
  # its cases with the shared harness, so adding tests never touches a file
  # someone else is editing.
  add_executable(slopfab_tests
    tests/harness.cpp
    tests/test_main.cpp
    tests/test_adaln.cpp
    tests/test_audio_primitives.cpp
    tests/test_chunking.cpp
    tests/test_compare_metrics.cpp
    tests/test_decode_pipeline.cpp
    tests/test_stillprobe.cpp
    tests/test_image.cpp
    tests/test_reference_media.cpp
    tests/test_refmod.cpp
    tests/test_noise.cpp
    tests/test_packing.cpp
    tests/test_ref2va.cpp
    tests/test_safetensors.cpp
    tests/test_pipeline.cpp
    tests/test_lora.cpp
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
  target_link_libraries(slopfab_tests PRIVATE slopfab_core)
  if(MSVC)
    target_compile_options(slopfab_tests PRIVATE /W4 /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(slopfab_tests PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME unit COMMAND slopfab_tests)
  if(SLOPFAB_WITH_FFMPEG)
    find_program(SLOPFAB_TEST_FFMPEG ffmpeg HINTS "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin")
    find_program(SLOPFAB_TEST_FFPROBE ffprobe HINTS "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin")
    if(SLOPFAB_TEST_FFMPEG AND SLOPFAB_TEST_FFPROBE)
      add_executable(slopfab_reference_decode_tests tests/test_reference_decode.cpp src/cli/reference_decode.cpp)
      target_link_libraries(slopfab_reference_decode_tests PRIVATE slopfab_core)
      add_test(NAME reference_media_decode COMMAND ${CMAKE_COMMAND}
        "-DPROBE=$<TARGET_FILE:slopfab_reference_decode_tests>" "-DFFMPEG=${SLOPFAB_TEST_FFMPEG}"
        "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/reference_media_fixture"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_reference_decode.cmake")
      set_tests_properties(reference_media_decode PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 60)
    endif()
  endif()

  if(SLOPFAB_ENABLE_VULKAN)
    add_executable(slopfab_vulkan_tests tests/harness.cpp tests/test_vulkan.cpp)
    target_link_libraries(slopfab_vulkan_tests PRIVATE slopfab_vulkan)
    set(SLOPFAB_AFFINE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/affine.comp")
    set(SLOPFAB_AFFINE_SPV "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/affine.comp.spv")
    file(READ "${SLOPFAB_AFFINE_SOURCE}" SLOPFAB_AFFINE_SOURCE_TEXT)
    string(REPLACE "\r\n" "\n" SLOPFAB_AFFINE_SOURCE_TEXT "${SLOPFAB_AFFINE_SOURCE_TEXT}")
    string(SHA256 SLOPFAB_AFFINE_SOURCE_HASH "${SLOPFAB_AFFINE_SOURCE_TEXT}")
    file(SHA256 "${SLOPFAB_AFFINE_SPV}" SLOPFAB_AFFINE_SPV_HASH)
    if(NOT SLOPFAB_AFFINE_SOURCE_HASH STREQUAL
        "5ff2705b9c300adbdb476d393226f0dda0a3e23dd7b1eda9ac8095a9165345cd" OR
       NOT SLOPFAB_AFFINE_SPV_HASH STREQUAL
        "75c8343ac68d52af2c5908db040af9768f6abbce9df0a82268d131afbfc2ede5")
      message(FATAL_ERROR
        "slopfab: affine.comp or its checked-in SPIR-V changed. Regenerate with the "
        "documented glslang command, verify the shader test, and update both hashes")
    endif()
    target_compile_definitions(slopfab_vulkan_tests PRIVATE
      SLOPFAB_TEST_AFFINE_SPV_PATH="affine.comp.spv"
      SLOPFAB_TEST_ATTENTION_PREPARE_SPV_PATH="tensor_attention_prepare.comp.spv"
      SLOPFAB_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_custom_command(TARGET slopfab_vulkan_tests POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SLOPFAB_AFFINE_SPV}" "$<TARGET_FILE_DIR:slopfab_vulkan_tests>/affine.comp.spv"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV}"
        "$<TARGET_FILE_DIR:slopfab_vulkan_tests>/tensor_attention_prepare.comp.spv"
      VERBATIM)
    if(MSVC)
      target_compile_options(slopfab_vulkan_tests PRIVATE /W4 /permissive- /utf-8 /EHsc)
    else()
      target_compile_options(slopfab_vulkan_tests PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME vulkan COMMAND slopfab_vulkan_tests)
    set_tests_properties(vulkan PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:slopfab_vulkan_tests>")
  endif()

  if(SLOPFAB_ENABLE_CUDA AND SLOPFAB_ENABLE_VULKAN)
    add_executable(slopfab_tensor_backend_tests
      tests/harness.cpp tests/allocation_guard.cpp
      tests/test_tensor_backends.cu tests/test_audio_vulkan.cu tests/test_lora_backends.cu)
    target_link_libraries(slopfab_tensor_backend_tests PRIVATE
      slopfab_cuda slopfab_vulkan)
    target_compile_definitions(slopfab_tensor_backend_tests PRIVATE
      SLOPFAB_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    if(WIN32)
      target_link_libraries(slopfab_tensor_backend_tests PRIVATE bcrypt)
    endif()
    if(MSVC)
      target_compile_options(slopfab_tensor_backend_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/utf-8;/EHsc>)
    endif()
    add_test(NAME tensor_backends COMMAND slopfab_tensor_backend_tests)
  endif()

  if(SLOPFAB_ENABLE_CUDA)
    add_executable(slopfab_cublas_dispatch_tests tests/test_cublas_dispatch.cpp)
    target_link_libraries(slopfab_cublas_dispatch_tests PRIVATE slopfab_cuda)
    if(WIN32)
      add_test(NAME cublas_cuda12 COMMAND slopfab_cublas_dispatch_tests 12)
      set_tests_properties(cublas_cuda12 PROPERTIES
        ENVIRONMENT "SLOPFAB_CUDA_VERSION=12")
      add_test(NAME cublas_cuda13 COMMAND slopfab_cublas_dispatch_tests 13)
      set_tests_properties(cublas_cuda13 PROPERTIES
        ENVIRONMENT "SLOPFAB_CUDA_VERSION=13")
      add_test(NAME cublas_auto_prefer13 COMMAND slopfab_cublas_dispatch_tests auto)
      set_tests_properties(cublas_auto_prefer13 PROPERTIES
        ENVIRONMENT "SLOPFAB_CUDA_VERSION=auto")
      add_test(NAME cublas_auto_load_fallback
        COMMAND slopfab_cublas_dispatch_tests auto-fallback)
      add_test(NAME cublas_explicit_load_failure
        COMMAND slopfab_cublas_dispatch_tests broken-13)
    else()
      add_executable(slopfab_cublas_linked_dispatch_tests
        tests/test_cublas_linked_dispatch.cpp)
      target_link_libraries(slopfab_cublas_linked_dispatch_tests PRIVATE slopfab_cuda)
      add_test(NAME cublas_linked_auto
        COMMAND slopfab_cublas_linked_dispatch_tests auto success)
      add_test(NAME cublas_linked_matching
        COMMAND slopfab_cublas_linked_dispatch_tests
          ${CUDAToolkit_VERSION_MAJOR} success)
      if(CUDAToolkit_VERSION_MAJOR EQUAL 12)
        set(SLOPFAB_CUBLAS_MISMATCH_MAJOR 13)
      else()
        set(SLOPFAB_CUBLAS_MISMATCH_MAJOR 12)
      endif()
      add_test(NAME cublas_linked_mismatch
        COMMAND slopfab_cublas_linked_dispatch_tests
          ${SLOPFAB_CUBLAS_MISMATCH_MAJOR} failure)
      add_test(NAME cublas_linked_environment_matching
        COMMAND slopfab_cublas_linked_dispatch_tests environment success)
      set_tests_properties(cublas_linked_environment_matching PROPERTIES
        ENVIRONMENT "SLOPFAB_CUDA_VERSION=${CUDAToolkit_VERSION_MAJOR}")
      add_test(NAME cublas_linked_environment_mismatch
        COMMAND slopfab_cublas_linked_dispatch_tests environment failure)
      set_tests_properties(cublas_linked_environment_mismatch PROPERTIES
        ENVIRONMENT "SLOPFAB_CUDA_VERSION=${SLOPFAB_CUBLAS_MISMATCH_MAJOR}")
    endif()

    add_executable(slopfab_kernel_tests
      tests/harness.cpp
      tests/test_kernels.cu
      tests/test_nn_kernels.cu
      tests/test_audio_vae.cu
      tests/test_transformer.cu
      tests/test_encoder.cu
      tests/test_keyframe_encoder.cu
      tests/test_vit_decoder.cu
    )
    target_link_libraries(slopfab_kernel_tests PRIVATE slopfab_cuda)
    add_test(NAME kernels COMMAND slopfab_kernel_tests)

    if(WIN32)
      add_test(NAME cuda_version_option
        COMMAND slopfab --cuda-version=12 --help)
      add_test(NAME cuda_version_invalid
        COMMAND slopfab --cuda-version=invalid --help)
      set_tests_properties(cuda_version_invalid PROPERTIES WILL_FAIL TRUE)
    endif()
  endif()

  # The C ABI is tested through the DLL rather than by compiling capi.cpp into
  # the test binary, which is the only way the interesting failures show up:
  # a missing export, a mangled name, a calling-convention mismatch and a
  # struct the two sides lay out differently are all link- or run-time
  # properties of the module, invisible to a test that includes the source.
  if(SLOPFAB_BUILD_C_API)
    # Its own main, not tests/test_main.cpp: that file carries C++ test cases
    # of its own and would drag slopfab_core into a binary whose entire point is
    # linking nothing but the DLL.
    add_executable(slopfab_capi_tests tests/harness.cpp tests/test_capi.cpp)
    target_link_libraries(slopfab_capi_tests PRIVATE slopfab_c)
    target_include_directories(slopfab_capi_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(MSVC)
      target_compile_options(slopfab_capi_tests PRIVATE /W4 /permissive- /utf-8)
    else()
      target_compile_options(slopfab_capi_tests PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME capi COMMAND slopfab_capi_tests)

    add_executable(slopfab_capi_cuda_dispatch_tests tests/test_capi_cuda_dispatch.cpp)
    target_link_libraries(slopfab_capi_cuda_dispatch_tests PRIVATE slopfab_c)
    target_include_directories(slopfab_capi_cuda_dispatch_tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(MSVC)
      target_compile_options(slopfab_capi_cuda_dispatch_tests PRIVATE
        /W4 /permissive- /utf-8)
    else()
      target_compile_options(slopfab_capi_cuda_dispatch_tests PRIVATE -Wall -Wextra)
    endif()
    if(WIN32)
      add_test(NAME capi_cublas_cuda12 COMMAND slopfab_capi_cuda_dispatch_tests 12)
      add_test(NAME capi_cublas_cuda13 COMMAND slopfab_capi_cuda_dispatch_tests 13)
    endif()
  endif()
endif()

# --- install ----------------------------------------------------------------

# The DLL installs one header rather than the whole include tree: capi.h is the
# supported surface and the C++ headers beside it are internal implementation.
if(SLOPFAB_BUILD_C_API)
  include(GNUInstallDirs)
  install(TARGETS slopfab_c
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES include/slopfab/capi.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slopfab)

  # CUDA and cuBLAS DLLs are intentionally not installed. slopfab.dll discovers
  # the selected installed toolkit itself; consumers do not modify PATH.
endif()
