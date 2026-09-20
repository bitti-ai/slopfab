# --- cli --------------------------------------------------------------------

if(WIN32)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ref/text_encoder/tokenizer.json")
    message(FATAL_ERROR "slopfab: ref/text_encoder/tokenizer.json is required to embed the tokenizer")
  endif()
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
  file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ref/text_encoder/tokenizer.json"
       SLOPFAB_TOKENIZER_RESOURCE)
  configure_file(src/slopfab_tokenizer.rc.in generated/slopfab_tokenizer.rc @ONLY)

  # A static library cannot usefully carry a .res — the linker has no reference
  # to pull it in with — so the executable embeds the tokenizer itself.
  set(SLOPFAB_EXE_RESOURCES
    "${CMAKE_CURRENT_BINARY_DIR}/generated/slopfab_tokenizer.rc")
  set(SLOPFAB_CLI_SOURCES src/main.cpp ${SLOPFAB_EXE_RESOURCES})
else()
  set(SLOPFAB_CLI_SOURCES src/main.cpp)
endif()

# The CLI and C API use the same in-process CUDA core. On Windows cuBLAS is
# late-bound, so neither module has a fixed CUDA-major import and no launcher
# or toolkit-specific backend executable is needed.
add_executable(slopfab ${SLOPFAB_CLI_SOURCES})
target_sources(slopfab PRIVATE src/cli/reference_decode.cpp)
target_sources(slopfab PRIVATE
  src/cli/model_assets.cpp src/cli/help.cpp src/cli/inspect.cpp
  src/cli/decode.cpp src/cli/tokenize.cpp src/cli/generate.cpp src/cli/devices.cpp)
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

