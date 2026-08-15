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
option(VIDFAB_BUILD_DEV_TOOLS "Build developer profiling/probe executables" OFF)
option(VIDFAB_BUILD_SHARED "Build vidfab_core/vidfab_cuda as shared libraries" OFF)

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

# --- shared library configuration -------------------------------------------

# The two libraries are built as one pair, static or shared together. Splitting
# them the other way round was considered and dropped: vidfab_cuda holds host
# orchestration that calls straight into vidfab_core, so a shared cuda over a
# static core would bake a second copy of the core into the DLL and leave the
# executable's copy running alongside it, with two of everything that has file
# scope. The pair also keeps the one-way dependency intact — a consumer that
# wants the loaders and no CUDA context links vidfab_core alone, exactly as
# vidfab_loadprobe does.
if(VIDFAB_BUILD_SHARED)
  set(VIDFAB_LIBRARY_TYPE SHARED)
else()
  set(VIDFAB_LIBRARY_TYPE STATIC)
endif()

# Symbols are exported wholesale rather than through a VIDFAB_API macro on ~40
# headers. That is a real trade and not just less typing: the export set becomes
# whatever the headers happen to declare, so nothing here defines a supported
# ABI surface, and a symbol disappearing is a link error at the consumer rather
# than a compile error here. It is the right trade only because the alternative
# is annotating a codebase that has no such annotations today, and because
# nothing in include/ exports mutable data — every constant is `constexpr` and
# every singleton (`StepProfiler::instance`, `PhaseProfiler::instance`) is
# defined out of line in exactly one translation unit. Auto-export handles
# functions correctly; *data* would need `dllimport` on the consumer side to
# work at all, so that last property is what makes this safe rather than merely
# convenient. Check it before adding an `extern` to a header.
function(vidfab_configure_library target)
  if(NOT VIDFAB_BUILD_SHARED)
    return()
  endif()
  set_target_properties(${target} PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
endfunction()

# Nothing copies DLLs beside the binaries, which is worth stating because the
# usual `TARGET_RUNTIME_DLLS` POST_BUILD step is the reflex here and was
# written before being deleted. On Windows a DLL is a RUNTIME artefact, so
# vidfab_core.dll and vidfab_cuda.dll are already written to the same directory
# as every executable that links them, and `ctest` therefore runs the test
# binaries out of the build tree unaided. The CUDA runtime stays embedded in
# each module rather than becoming a DLL — see the runtime-linkage note in the
# CUDA section — so it needs no copying either. FFmpeg is handled separately
# and for its own licensing reasons, further down.
#
# A build that overrides CMAKE_RUNTIME_OUTPUT_DIRECTORY per target would break
# that assumption. Nothing here does.

# --- core -------------------------------------------------------------------

add_library(vidfab_core ${VIDFAB_LIBRARY_TYPE}
  src/core/json.cpp
  src/core/nf4.cpp
  src/core/safetensors.cpp
  src/core/tensor_convert.cpp
  src/core/image.cpp
  src/core/image_platform.cpp
  src/core/safetensors_write.cpp
  src/dit/adaln.cpp
  src/dit/block_cache.cpp
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
vidfab_configure_library(vidfab_core)

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
    # The VAE decode's final de-normalise is the project's first std::thread
    # use. MSVC needs nothing for it, but every other toolchain wants -pthread
    # both to compile and to link, and without this the non-MSVC build breaks.
    find_package(Threads REQUIRED)
    message(STATUS "vidfab: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")

    add_library(vidfab_cuda ${VIDFAB_LIBRARY_TYPE}
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
    vidfab_configure_library(vidfab_cuda)
    # Whole-program compilation: no device code crosses translation units, and
    # -rdc=true would otherwise require an explicit device-link step. It is also
    # what keeps the shared build simple — separable compilation would need
    # CUDA_RESOLVE_DEVICE_SYMBOLS and a device-link step per shared object, and
    # it interacts badly with whole-target symbol export.
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
    # The shared build keeps that choice, which was not the first answer. The
    # argument for flipping it there is real: `tests/*.cu`, `tools/solbench.cu`
    # and `tools/qwenvisionprobe.cpp` call cudart directly *and* link
    # vidfab_cuda, so a static runtime puts one instance in vidfab_cuda.dll and
    # another in each of those binaries, and two cudart instances in a process
    # share neither error state nor kernel registration.
    #
    # It was tried and it does nothing on CUDA 13. `CUDA_RUNTIME_LIBRARY=Shared`
    # emits `nvcc warning: 'shared' is a deprecated value for option --cudart`
    # and produces a vidfab_cuda.dll that imports no cudart64_13.dll, references
    # that name nowhere in its image, and carries the runtime embedded exactly
    # as the static setting does. So the setting buys a deprecation warning and
    # a gratuitous difference between the two configurations, and no shared
    # runtime. Measured on 13.0.48 with dumpbin; re-measure before reaching for
    # it again on a newer toolkit.
    #
    # What the attempt did surface is a real defect, fixed on the next line.
    # A consumer with no CUDA *sources* — qwenvisionprobe.cpp is plain C++ that
    # calls cudaMemGetInfo — used to pick the runtime up from vidfab_cuda's
    # static archive by accident. Across a DLL boundary it no longer can, and
    # the link fails on an unresolved cudart symbol. Naming the dependency
    # PUBLIC states what was previously incidental, and is correct in both
    # configurations rather than only in the static one.
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

  # The two resources were one script until the shared build split them, and
  # they are split because they belong to different modules. `Tokenizer::
  # load_embedded` looks the resource up in the module holding its own code, so
  # in a shared build the 7 MB tokenizer has to be in vidfab_core.dll; the
  # licence is read by src/main.cpp and so is always in the executable. Putting
  # both in both would work and would cost 7 MB of duplicate payload.
  #
  # A static library cannot usefully carry a .res — the linker has no reference
  # to pull it in with — so in the static build the tokenizer stays in the
  # executable, which is the module `load_embedded` resolves to there anyway.
  if(VIDFAB_BUILD_SHARED)
    target_sources(vidfab_core PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/vidfab_tokenizer.rc")
    set(VIDFAB_EXE_RESOURCES "${CMAKE_CURRENT_BINARY_DIR}/generated/vidfab_license.rc")
  else()
    set(VIDFAB_EXE_RESOURCES
      "${CMAKE_CURRENT_BINARY_DIR}/generated/vidfab_tokenizer.rc"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/vidfab_license.rc")
  endif()
  add_executable(vidfab src/main.cpp ${VIDFAB_EXE_RESOURCES})
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

# vidfab_c is the stable C ABI: one DLL with a hand-written export set, for
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
  # SHARED unconditionally, and independent of VIDFAB_BUILD_SHARED. Those two
  # options answer different questions: VIDFAB_BUILD_SHARED asks whether the
  # *C++* libraries are DLLs, which is a choice about this build tree, while a
  # C ABI that is not a DLL cannot be loaded by the callers it exists for.
  #
  # The default configuration (VIDFAB_BUILD_SHARED=OFF) is the interesting one:
  # static vidfab_core and vidfab_cuda link *into* vidfab_c.dll, so all of this
  # project's own code is in one module whose exports are exactly the C entry
  # points — no C++ symbols, no second vidfab DLL, and no way for a consumer to
  # reach the unstable interface by accident. With VIDFAB_BUILD_SHARED=ON it
  # instead links against vidfab_core.dll and vidfab_cuda.dll and must ship
  # beside them.
  #
  # It is not, however, dependency-free, and it is worth being exact about that
  # because the mistake is invisible on a developer machine. The CUDA *runtime*
  # is embedded (see the cudart note above), but cuBLAS is not: vidfab_c.dll
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
  # libvidfab_c.so.0 whose soname says 0 for an ABI that says 1, and would bump
  # it for releases that did not touch the C interface at all.
  set(VIDFAB_CAPI_VERSION_MAJOR 1)
  set(VIDFAB_CAPI_VERSION_MINOR 0)
  set(VIDFAB_CAPI_VERSION_PATCH 0)
  set(VIDFAB_CAPI_VERSION
    "${VIDFAB_CAPI_VERSION_MAJOR}.${VIDFAB_CAPI_VERSION_MINOR}.${VIDFAB_CAPI_VERSION_PATCH}")

  set_target_properties(vidfab_c PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${VIDFAB_CAPI_VERSION}
    SOVERSION ${VIDFAB_CAPI_VERSION_MAJOR})

  # The visibility presets above are *compile* properties, so they only cover
  # capi.cpp. vidfab_core and vidfab_cuda are compiled with default visibility,
  # and on ELF their entire C++ symbol table would otherwise be re-exported
  # from libvidfab_c.so — precisely what WINDOWS_EXPORT_ALL_SYMBOLS OFF
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
  # own code. In the static configuration that code is linked into
  # vidfab_c.dll, so the 7 MB tokenizer has to be here too or a C caller that
  # leaves the tokenizer path empty gets "embedded resource is missing". When
  # VIDFAB_BUILD_SHARED is on, that code lives in vidfab_core.dll, which
  # already carries the resource — adding it here as well would duplicate the
  # payload and be looked up by nobody.
  if(WIN32 AND NOT VIDFAB_BUILD_SHARED)
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

# Only for the shared build, because that is the configuration whose point is
# being consumed from outside this build tree. The static build has never
# installed anything and nothing here asks it to start.
#
# Headers are installed wholesale rather than as a curated public subset. That
# is honest about the current state — `include/` is the API, there is no
# internal/ split — and it matches the export decision above: this ships a C++
# ABI with no stability promise, so a consumer must be built with the same
# compiler and, on MSVC, the same CRT. `std::string` and `std::vector` cross
# every one of these signatures and exceptions propagate out of them, neither
# of which survives a mismatched runtime.
if(VIDFAB_BUILD_SHARED)
  include(GNUInstallDirs)
  set(VIDFAB_INSTALL_TARGETS vidfab_core)
  if(VIDFAB_ENABLE_CUDA)
    list(APPEND VIDFAB_INSTALL_TARGETS vidfab_cuda)
  endif()
  install(TARGETS ${VIDFAB_INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY include/vidfab DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# vidfab_c installs on its own terms, and unlike the block above it is not
# conditioned on VIDFAB_BUILD_SHARED. The reason is the same one that makes it
# unconditionally SHARED: its whole purpose is being consumed from outside this
# build tree, and in the default static configuration it is the *only* artefact
# that can be. It also installs one header rather than the directory, because
# here the public subset genuinely is a subset — capi.h is the supported
# surface and the C++ headers beside it are not.
if(VIDFAB_BUILD_C_API)
  include(GNUInstallDirs)
  install(TARGETS vidfab_c
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  # Skipped when the block above already installed the whole directory, which
  # would otherwise place this same file twice.
  if(NOT VIDFAB_BUILD_SHARED)
    install(FILES include/vidfab/capi.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vidfab)
  endif()

  # The cuBLAS redistributables vidfab_c.dll imports at load time. Without
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


