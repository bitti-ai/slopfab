#pragma once

#include <cstdint>

namespace vidfab::dit {

#pragma pack(push, 1)
struct H3BlockCaptureHeader {
  char magic[8];                 // "VFH3BLK\0"
  uint32_t version;             // 1
  uint32_t header_bytes;
  uint32_t sequence;
  uint32_t hidden;
  uint32_t heads;
  uint32_t head_dim;
  uint32_t ffn;
  uint32_t timesteps;
  uint32_t modalities;
  uint32_t adaln_rank;
  int32_t denoise_step;
  int32_t layer;
  uint32_t range_values;
  uint32_t reserved0;
  uint64_t residual_elements;
  uint64_t qkv_elements;
  uint64_t rope_elements;
  uint64_t input_fnv64;
  uint64_t qkv_fnv64;
  uint64_t attention_fnv64;
  uint64_t attention_residual_fnv64;
  uint64_t final_fnv64;
};
#pragma pack(pop)

static_assert(sizeof(H3BlockCaptureHeader) == 128,
              "H3 block capture header is a durable 128-byte record");

}  // namespace vidfab::dit
