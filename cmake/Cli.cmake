# --- cli --------------------------------------------------------------------

set(SLOPFAB_CLI_SOURCES src/main.cpp src/cli/agent.cpp src/cli/request.cpp src/cli/jobs.cpp ${SLOPFAB_TOKENIZER_RESOURCES})

# The CLI and C API use the same in-process CUDA core. On Windows cuBLAS is
# late-bound, so neither module has a fixed CUDA-major import and no launcher
# or toolkit-specific backend executable is needed.
add_executable(slopfab ${SLOPFAB_CLI_SOURCES})
target_sources(slopfab PRIVATE src/cli/reference_decode.cpp)
target_sources(slopfab PRIVATE src/cli/assets.cpp)
target_sources(slopfab PRIVATE src/cli/upscale_latents.cpp)
target_sources(slopfab PRIVATE src/cli/upscale.cpp src/cli/upscale_options.cpp src/cli/native_media.cpp)
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
if(TARGET slopfab_generation)
  target_link_libraries(${SLOPFAB_CLI_TARGET} PRIVATE slopfab_generation)
endif()

if(MSVC)
  target_compile_options(${SLOPFAB_CLI_TARGET} PRIVATE /W4 /permissive- /utf-8 /EHsc)
else()
  target_compile_options(${SLOPFAB_CLI_TARGET} PRIVATE -Wall -Wextra)
endif()

