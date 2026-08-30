#include <cstdlib>
#include <exception>
#include <string>

#include "vidfab/cuda/cublas_dispatch.h"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const std::string requested = argv[1];
  const bool should_succeed = std::string(argv[2]) == "success";
  try {
    if (requested != "environment")
      vidfab::cuda::set_cublas_version_request(requested);
    const int loaded = vidfab::cuda::cublas_loaded_major();
    if (!should_succeed) return 3;
    return loaded == CUDART_VERSION / 1000 ? 0 : 4;
  } catch (const std::exception& error) {
    if (should_succeed) return 5;
    return std::string(error.what()).find("non-Windows build is linked to CUDA") !=
                   std::string::npos
               ? 0
               : 6;
  }
}
