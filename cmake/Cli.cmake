# --- cli --------------------------------------------------------------------

set(SLOPFAB_CLI_SOURCES src/main.cpp ${SLOPFAB_TOKENIZER_RESOURCES})

# The CLI and C API use the same in-process CUDA core. On Windows cuBLAS is
# late-bound, so neither module has a fixed CUDA-major import and no launcher
# or toolkit-specific backend executable is needed.
add_executable(slopfab ${SLOPFAB_CLI_SOURCES})
target_sources(slopfab PRIVATE src/cli/reference_decode.cpp)
target_sources(slopfab PRIVATE src/cli/assets.cpp)
target_sources(slopfab PRIVATE
  src/cli/model_assets.cpp src/cli/help.cpp src/cli/inspect.cpp
  src/cli/decode.cpp src/cli/tokenize.cpp src/cli/generate.cpp src/cli/devices.cpp)
if(TARGET slopfab_generation)
  target_sources(slopfab PRIVATE src/cli/upscale.cpp)
endif()
set(SLOPFAB_CLI_TARGET slopfab)

target_link_libraries(${SLOPFAB_CLI_TARGET} PRIVATE slopfab_core)
target_compile_definitions(${SLOPFAB_CLI_TARGET} PRIVATE
  SLOPFAB_WITH_VULKAN=$<IF:$<BOOL:${SLOPFAB_ENABLE_VULKAN}>,1,0>)
if(SLOPFAB_ENABLE_VULKAN)
  target_link_libraries(${SLOPFAB_CLI_TARGET} PRIVATE slopfab_vulkan)
endif()
if(TARGET slopfab_generation)
  target_link_libraries(${SLOPFAB_CLI_TARGET} PRIVATE slopfab_generation)
endif()

if(MSVC)
  target_compile_options(${SLOPFAB_CLI_TARGET} PRIVATE /W4 /permissive- /utf-8 /EHsc)
else()
  target_compile_options(${SLOPFAB_CLI_TARGET} PRIVATE -Wall -Wextra)
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

