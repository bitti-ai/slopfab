# --- cli --------------------------------------------------------------------

set(SLOPFAB_CLI_SOURCES src/main.cpp src/cli/startup.cpp src/cli/settings.cpp src/cli/mods.cpp src/cli/agent.cpp src/cli/request.cpp src/cli/jobs.cpp ${SLOPFAB_TOKENIZER_RESOURCES})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/templates/project.html")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/templates/project.html" SLOPFAB_PROJECT_TEMPLATE)
# MSVC limits each string token to 16 KiB. Keep the offline template embedded,
# but use adjacent raw string literals for its individual lines.
string(REPLACE "\n" "\n)SLOPFAB_HTML\"\nR\"SLOPFAB_HTML(" SLOPFAB_PROJECT_TEMPLATE "${SLOPFAB_PROJECT_TEMPLATE}")
configure_file(src/cli/project_template.h.in generated/project_template.h @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/templates/AGENTS.md")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/templates/AGENTS.md" SLOPFAB_AGENT_INSTRUCTIONS)
configure_file(src/cli/agent_instructions_template.h.in generated/agent_instructions_template.h @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/resources/prompting.json")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/resources/prompting.json" SLOPFAB_PROMPTING_CATALOG)
string(REPLACE "\n" "\n)SLOPFAB_GUIDES\"\nR\"SLOPFAB_GUIDES(" SLOPFAB_PROMPTING_CATALOG "${SLOPFAB_PROMPTING_CATALOG}")
configure_file(src/cli/prompting_catalog.h.in generated/prompting_catalog.h @ONLY)

# The CLI and C API use the same in-process CUDA core. On Windows cuBLAS is
# late-bound, so neither module has a fixed CUDA-major import and no launcher
# or toolkit-specific backend executable is needed.
add_executable(slopfab ${SLOPFAB_CLI_SOURCES})
target_sources(slopfab PRIVATE src/cli/projects.cpp)
target_sources(slopfab PRIVATE src/cli/agent_instructions.cpp)
target_sources(slopfab PRIVATE src/cli/prompting.cpp)
target_include_directories(slopfab PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_sources(slopfab PRIVATE src/cli/reference_decode.cpp)
target_sources(slopfab PRIVATE src/cli/assets.cpp)
target_sources(slopfab PRIVATE src/cli/upscale_latents.cpp)
target_sources(slopfab PRIVATE src/cli/upscale.cpp src/cli/upscale_options.cpp src/cli/native_media.cpp)
target_sources(slopfab PRIVATE
  src/cli/model_assets.cpp src/cli/help.cpp src/cli/inspect.cpp
  src/cli/decode.cpp src/cli/tokenize.cpp src/cli/generate.cpp src/cli/devices.cpp)
set(SLOPFAB_CLI_TARGET slopfab)

target_link_libraries(${SLOPFAB_CLI_TARGET} PRIVATE slopfab_core ${CMAKE_DL_LIBS})
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

