#pragma once

#include <cstdint>

namespace vidfab::dit {

constexpr uint32_t kH3MainCaptureLayers = 50;

#pragma pack(push, 1)
struct H3MainGraphCaptureHeader {
  char magic[8];                 // "VFH3GRF\0"
  uint32_t version;
  uint32_t header_bytes;
  uint32_t sequence;
  uint32_t hidden;
  uint32_t heads;
  uint32_t head_dim;
  uint32_t ffn;
  uint32_t timesteps;
  uint32_t modalities;
  uint32_t adaln_rank;
  uint32_t layers;
  uint32_t range_values;
  int32_t denoise_step;
  uint32_t reserved0;
  uint64_t residual_elements;
  uint64_t rope_elements;
  uint64_t input_fnv64;
  uint64_t final_fnv64;
  uint64_t boundary_fnv64[kH3MainCaptureLayers];
  uint64_t reserved[2];
};
#pragma pack(pop)

static_assert(sizeof(H3MainGraphCaptureHeader) == 512,
              "H3 graph capture header is a durable 512-byte record");

}  // namespace vidfab::dit
