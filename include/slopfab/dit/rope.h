#pragma once

#include <cstdint>
#include <vector>

namespace slopfab::dit {

// Canonical serialized H3 MM-RoPE tables. `positions` is row-major interleaved
// [row,T/H/W] fp64 coordinates. The builder performs the
// reference's double-to-float position cast before the fp32 multiply, then
// stores [T frequencies | H frequencies | W frequencies] twice.
struct H3RopeTables {
  uint32_t rows = 0;
  uint32_t frequency_dim = 0;
  std::vector<float> cosine;
  std::vector<float> sine;
};

H3RopeTables build_h3_rope_tables(const std::vector<double>& positions, float theta,
                                  uint32_t frequency_dim);

} // namespace slopfab::dit
