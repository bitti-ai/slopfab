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
      src/cuda/vsa_attention.cu
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
    src/dit/weight_metadata.cpp
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

