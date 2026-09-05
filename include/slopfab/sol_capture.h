#pragma once

#include <cstdint>

namespace slopfab {

// Little-endian on-disk header followed by q, k, then v. Each tensor contains
// seq_len*num_heads*head_dim raw BF16 words in token/head/channel order.
struct SolCaptureHeader {
  char magic[8];                 // "VFSOLQKV"
  uint32_t version;              // 1
  uint32_t header_bytes;         // sizeof(SolCaptureHeader)
  uint32_t seq_len;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t exact_prefix;
  int32_t denoise_step;
  int32_t layer;
  uint64_t tensor_elements;      // elements in each of q, k and v
  uint64_t reserved[2];
};

static_assert(sizeof(SolCaptureHeader) == 64, "stable Sol capture header");

}  // namespace slopfab
