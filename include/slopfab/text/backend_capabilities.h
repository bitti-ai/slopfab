#pragma once

#include "slopfab/attention_descriptor.h"

namespace slopfab::text {

// The exact CUDA and Vulkan text layers share this shipped arithmetic/shape
// contract. Wider support needs a new implementation, not a dimension override.
struct ExactTextLayerShape {
  int tokens = 0;
  int hidden = 5120;
  int query_heads = 64;
  int key_value_heads = 8;
  int head_dim = 128;
  int intermediate = 25600;
  float rms_norm_epsilon = 1.0e-6f;
};

inline bool supports_exact_text_layer(const ExactTextLayerShape& shape) noexcept {
  if (shape.tokens <= 0 || shape.hidden != 5120 || shape.intermediate <= 0 ||
      shape.rms_norm_epsilon != 1.0e-6f) return false;
  AttentionDescriptor attention{static_cast<uint32_t>(shape.tokens),
      static_cast<uint32_t>(shape.tokens), static_cast<uint32_t>(shape.query_heads),
      static_cast<uint32_t>(shape.key_value_heads), static_cast<uint32_t>(shape.head_dim),
      AttentionLayout::kTokensHeadsChannels, AttentionMask::kCausal,
      AttentionArithmetic::kExact, exact_attention_scale(shape.head_dim)};
  return supports_exact_text_attention(attention, UINT32_MAX);
}

}  // namespace slopfab::text
