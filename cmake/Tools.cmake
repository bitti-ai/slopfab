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


if(SLOPFAB_ENABLE_VULKAN)
  if(SLOPFAB_BUILD_DEV_TOOLS)
    add_executable(slopfab_attentionbench tools/attentionbench.cpp)
    target_link_libraries(slopfab_attentionbench PRIVATE slopfab_vulkan)
    if(TARGET slopfab_cuda)
      target_link_libraries(slopfab_attentionbench PRIVATE slopfab_cuda)
    endif()
  endif()
endif()
