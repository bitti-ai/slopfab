#pragma once

#include <cstdint>
#include <cstring>

namespace vidfab {

// Exact attention deliberately accepts only the three production head widths
// and these serialized fp32 1/sqrt(D) bits. This keeps the score domain shared
// by CUDA and Vulkan; callers do not recompute the scale with a host libm.
inline uint32_t exact_attention_scale_bits(uint32_t head_dim) noexcept {
  return head_dim == 64 ? 0x3e000000u
      : head_dim == 72 ? 0x3df15befu
      : head_dim == 128 ? 0x3db504f3u : 0u;
}

inline float exact_attention_scale(uint32_t head_dim) noexcept {
  const uint32_t bits = exact_attention_scale_bits(head_dim);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline bool is_exact_attention_scale(uint32_t head_dim, float value) noexcept {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits != 0 && bits == exact_attention_scale_bits(head_dim);
}

}  // namespace vidfab
