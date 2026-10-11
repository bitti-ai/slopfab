#pragma once

#include "slopfab/cuda/device.h"
#include <cstdio>
#include <limits>

namespace slopfab::dit {

// Each retry must choose a strictly smaller device residency. The planner
// includes transfer slots, so offloading one or two blocks is not necessarily
// enough to save memory. No retry changes weights, precision or sequence size.
template <class Load, class Release>
void load_with_memory_retry(bool automatic, int max_retries, Load load, Release release) {
  size_t ceiling = std::numeric_limits<size_t>::max();
  for (int attempt = 0;; ++attempt) {
    size_t planned = std::numeric_limits<size_t>::max();
    try {
      load(ceiling, planned);
      return;
    } catch (const cuda::Error& error) {
      release();
      if (!automatic || error.status() != cudaErrorMemoryAllocation ||
          attempt >= max_retries || planned == 0 || planned > ceiling ||
          planned == std::numeric_limits<size_t>::max())
        throw;
      // Consume the handled allocation error before another kernel launch.
      (void)cudaGetLastError();
      ceiling = planned - 1;
      std::fprintf(stderr, "slopfab: transformer load ran out of CUDA memory; "
                           "retrying with fewer resident blocks (weight budget < %zu bytes)\n",
                   planned);
    } catch (...) {
      release();
      throw;
    }
  }
}

} // namespace slopfab::dit
