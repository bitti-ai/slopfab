#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "slopfab/capi.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const int expected = std::atoi(argv[1]);
  if (expected != 12 && expected != 13) return 3;

  if (slopfab_cuda_set_version("invalid") != SLOPFAB_ERR_INVALID_ARGUMENT) return 4;
  if (slopfab_cuda_loaded_major(nullptr) != SLOPFAB_ERR_INVALID_ARGUMENT) return 5;
  if (slopfab_cuda_set_version(argv[1]) != SLOPFAB_OK) {
    std::fprintf(stderr, "%s\n", slopfab_last_error());
    return 6;
  }
  if (slopfab_last_error()[0] != '\0') return 10;

  // Seed another thread-local failure so the successful query has stale
  // state to retire independently of the successful setter above.
  if (slopfab_cuda_loaded_major(nullptr) != SLOPFAB_ERR_INVALID_ARGUMENT) return 11;
  if (slopfab_last_error()[0] == '\0') return 12;

  int32_t loaded = 0;
  if (slopfab_cuda_loaded_major(&loaded) != SLOPFAB_OK) {
    std::fprintf(stderr, "%s\n", slopfab_last_error());
    return 7;
  }
  if (slopfab_last_error()[0] != '\0') return 13;
  if (loaded != expected) return 8;

  // Even an identical request is rejected after initialization: silently
  // accepting it would make a racing host believe it controlled selection.
  if (slopfab_cuda_set_version(argv[1]) != SLOPFAB_ERR_RUNTIME) return 9;
  std::printf("slopfab.dll selected CUDA %d cuBLAS\n", loaded);
  return 0;
}
