#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "vidfab/capi.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const int expected = std::atoi(argv[1]);
  if (expected != 12 && expected != 13) return 3;

  if (vidfab_cuda_set_version("invalid") != VIDFAB_ERR_INVALID_ARGUMENT) return 4;
  if (vidfab_cuda_loaded_major(nullptr) != VIDFAB_ERR_INVALID_ARGUMENT) return 5;
  if (vidfab_cuda_set_version(argv[1]) != VIDFAB_OK) {
    std::fprintf(stderr, "%s\n", vidfab_last_error());
    return 6;
  }
  if (vidfab_last_error()[0] != '\0') return 10;

  // Seed another thread-local failure so the successful query has stale
  // state to retire independently of the successful setter above.
  if (vidfab_cuda_loaded_major(nullptr) != VIDFAB_ERR_INVALID_ARGUMENT) return 11;
  if (vidfab_last_error()[0] == '\0') return 12;

  int32_t loaded = 0;
  if (vidfab_cuda_loaded_major(&loaded) != VIDFAB_OK) {
    std::fprintf(stderr, "%s\n", vidfab_last_error());
    return 7;
  }
  if (vidfab_last_error()[0] != '\0') return 13;
  if (loaded != expected) return 8;

  // Even an identical request is rejected after initialization: silently
  // accepting it would make a racing host believe it controlled selection.
  if (vidfab_cuda_set_version(argv[1]) != VIDFAB_ERR_RUNTIME) return 9;
  std::printf("vidfab.dll selected CUDA %d cuBLAS\n", loaded);
  return 0;
}
