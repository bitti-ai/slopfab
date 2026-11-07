# Native SyncNet and S3FD. Legacy weights are hash-locked raw tensor ranges.
add_library(slopfab_syncnet STATIC src/syncnet/weights.cpp src/syncnet/network.cpp
  src/syncnet/features.cpp src/syncnet/detector.cpp src/syncnet/tracking.cpp)
target_link_libraries(slopfab_syncnet PUBLIC slopfab_core)
target_compile_definitions(slopfab_syncnet PRIVATE
  SLOPFAB_WITH_VULKAN=$<BOOL:${SLOPFAB_ENABLE_VULKAN}>)
if(SLOPFAB_ENABLE_VULKAN)
  target_link_libraries(slopfab_syncnet PUBLIC slopfab_vulkan)
endif()
if(MSVC)
  target_compile_options(slopfab_syncnet PRIVATE /W4 /utf-8 /EHsc)
else()
  target_compile_options(slopfab_syncnet PRIVATE -Wall -Wextra)
endif()
