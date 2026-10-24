#pragma once

#include <cstdint>
#include <vector>

#include "slopfab/dit/packing.h"

namespace slopfab::dit {

// FastH3 V2: segment-pure prefix tiles, followed by 4x4x4 video tiles.
// Valid rows are packed at the start of each 64-slot tile; padding maps to -1.
struct VsaTiles {
  int prefix_tiles = 0;
  std::vector<int32_t> rows;
  std::vector<int32_t> sizes;
  std::vector<int32_t> row_tiles;
  int tile_size = 64;
  int keep_denominator = 5; // exact ceil(region_tiles / denominator)
  std::vector<int32_t> region_ends; // sparse regions after the dense prefix
};

VsaTiles build_vsa_tiles(const SequenceLayout& layout, int tile_size = 64, int keep_denominator = 5);

} // namespace slopfab::dit
