#include "vidfab/dit/rope.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace vidfab::dit {

H3RopeTables build_h3_rope_tables(const std::vector<double>& positions,
                                  float theta, uint32_t frequency_dim) {
  if (positions.empty() || positions.size() % 3 != 0 ||
      !std::isnormal(theta) || theta < 1.0f || frequency_dim == 0) {
    throw std::invalid_argument("H3 RoPE: invalid table parameters");
  }
  const size_t rows = positions.size() / 3;
  if (rows > std::numeric_limits<uint32_t>::max() ||
      frequency_dim > std::numeric_limits<uint32_t>::max() / 6u ||
      rows > std::numeric_limits<size_t>::max() / (6ull * frequency_dim)) {
    throw std::overflow_error("H3 RoPE: table dimensions overflow");
  }
  for (double position : positions) {
    if (!std::isfinite(position) ||
        !std::isfinite(static_cast<float>(position))) {
      throw std::invalid_argument("H3 RoPE: position is outside fp32 range");
    }
  }
  const size_t half = 3ull * frequency_dim;
  const size_t width = 2ull * half;
  std::vector<float> inverse(frequency_dim);
  for (uint32_t k = 0; k < frequency_dim; ++k) {
    inverse[k] = static_cast<float>(
        1.0 / std::pow(static_cast<double>(theta),
                       static_cast<double>(k) / frequency_dim));
  }
  H3RopeTables result;
  result.rows = static_cast<uint32_t>(rows);
  result.frequency_dim = frequency_dim;
  result.cosine.resize(rows * width);
  result.sine.resize(rows * width);
  for (size_t row = 0; row < rows; ++row) {
    for (uint32_t axis = 0; axis < 3; ++axis) {
      const float position = static_cast<float>(positions[row * 3 + axis]);
      for (uint32_t k = 0; k < frequency_dim; ++k) {
        const float angle = position * inverse[k];
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const size_t at = row * width + axis * frequency_dim + k;
        result.cosine[at] = result.cosine[at + half] = cosine;
        result.sine[at] = result.sine[at + half] = sine;
      }
    }
  }
  return result;
}

}  // namespace vidfab::dit
