# Backend availability is public because RunOptions chooses its default here.
target_compile_definitions(slopfab_core PUBLIC
  SLOPFAB_WITH_VULKAN=$<BOOL:${SLOPFAB_ENABLE_VULKAN}>)

# Shared generation orchestration. Neither backend owns the host pipeline;
# a Vulkan-only build must not include or link CUDA headers or libraries.
if(TARGET slopfab_cuda OR TARGET slopfab_vulkan)
  add_library(slopfab_generation STATIC
    src/generate.cpp
    src/generation/helpers.cpp
    src/generation/decode.cpp
    src/generation/session.cpp
    src/generation/prompt.cpp)
  target_link_libraries(slopfab_generation PUBLIC slopfab_core)
  set_target_properties(slopfab_generation PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(TARGET slopfab_cuda)
    target_link_libraries(slopfab_generation PUBLIC slopfab_cuda)
  endif()
  if(TARGET slopfab_vulkan)
    target_link_libraries(slopfab_generation PUBLIC slopfab_vulkan)
  endif()
  if(MSVC)
    target_compile_options(slopfab_generation PRIVATE /W4 /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(slopfab_generation PRIVATE -Wall -Wextra)
  endif()
endif()
