# Native HTDemucs; no Python, LibTorch or pickle deserialization at runtime.
add_library(slopfab_demucs STATIC src/demucs/model.cpp src/demucs/spectral.cpp
  src/demucs/separate.cpp)
target_link_libraries(slopfab_demucs PUBLIC slopfab_core)
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(slopfab_demucs PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
  target_compile_options(slopfab_demucs PRIVATE /W4 /utf-8 /EHsc)
else()
  target_compile_options(slopfab_demucs PRIVATE -Wall -Wextra)
endif()
