#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "slopfab/attention.h"

namespace slopfab {

enum class AttentionLayout { kTokensHeadsChannels };
enum class AttentionMask { kFull, kCausal, kFrameBanded };
// Exact means the pinned operation order/scale contract, not just the same
// real-number equation. Existing fused/blocked CUDA attention is equivalent.
enum class AttentionArithmetic { kExact, kEquivalent, kApproximate };

struct AttentionDescriptor {
  uint32_t query_tokens = 0;
  uint32_t key_value_tokens = 0;
  uint32_t query_heads = 0;
  uint32_t key_value_heads = 0;
  uint32_t head_dim = 0;
  AttentionLayout layout = AttentionLayout::kTokensHeadsChannels;
  AttentionMask mask = AttentionMask::kFull;
  AttentionArithmetic arithmetic = AttentionArithmetic::kEquivalent;
  // Zero selects the implementation's default. Exact contracts require the
  // explicit serialized scale, preventing accidental host-libm differences.
  float scale = 0.0f;
};

inline void validate_attention_descriptor(const AttentionDescriptor& desc) {
  if (!desc.query_tokens || !desc.key_value_tokens || !desc.query_heads ||
      !desc.key_value_heads || !desc.head_dim ||
      desc.query_heads % desc.key_value_heads != 0)
    throw std::invalid_argument("attention: positive dimensions and divisible query/KV heads required");
  if (desc.layout != AttentionLayout::kTokensHeadsChannels)
    throw std::invalid_argument("attention: unsupported layout");
  if (desc.mask != AttentionMask::kFull && desc.mask != AttentionMask::kCausal &&
      desc.mask != AttentionMask::kFrameBanded)
    throw std::invalid_argument("attention: unknown mask");
  if (desc.arithmetic != AttentionArithmetic::kExact &&
      desc.arithmetic != AttentionArithmetic::kEquivalent &&
      desc.arithmetic != AttentionArithmetic::kApproximate)
    throw std::invalid_argument("attention: unknown arithmetic contract");
  if (!std::isfinite(desc.scale) || desc.scale < 0.0f)
    throw std::invalid_argument("attention: scale must be finite and nonnegative");
  if (desc.arithmetic == AttentionArithmetic::kExact &&
      !is_exact_attention_scale(desc.head_dim, desc.scale))
    throw std::invalid_argument("attention: unsupported exact scale/head width");
  // All current operators use signed 32-bit row strides.
  if (uint64_t(desc.query_heads) * desc.head_dim > uint64_t(INT32_MAX) ||
      uint64_t(desc.key_value_heads) * desc.head_dim > uint64_t(INT32_MAX) ||
      desc.query_tokens > uint32_t(INT32_MAX) ||
      desc.key_value_tokens > uint32_t(INT32_MAX))
    throw std::invalid_argument("attention: dimensions exceed kernel indexing limits");
}

// Shared shipped text-attention capability. Device/driver and storage limits
// remain backend checks; this deliberately does not promise arbitrary GQA.
inline bool supports_exact_text_attention(const AttentionDescriptor& desc,
                                          uint32_t max_tokens) noexcept {
  return desc.query_tokens > 0 && desc.query_tokens <= max_tokens &&
      desc.query_tokens == desc.key_value_tokens && desc.query_heads == 64 &&
      desc.key_value_heads == 8 && desc.head_dim == 128 &&
      desc.layout == AttentionLayout::kTokensHeadsChannels &&
      desc.mask == AttentionMask::kCausal &&
      desc.arithmetic == AttentionArithmetic::kExact &&
      is_exact_attention_scale(desc.head_dim, desc.scale);
}

}  // namespace slopfab
