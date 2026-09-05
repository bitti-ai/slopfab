#pragma once

#include <cstddef>
#include <cstdint>

namespace slopfab {

// Immutable, backend-neutral checkpoint storage metadata. This describes the
// bytes on disk/device, not a GEMM algorithm or a backend handle.
enum class LinearWeightFormat {
  kFloat32,
  kFloat16,
  kBFloat16,
  kFloat8E4M3,
  kInt8,
  kNVFloat4,
  kNF4,
};

struct LinearWeightUpload {
  LinearWeightFormat format = LinearWeightFormat::kBFloat16;
  uint32_t out_features = 0;
  uint32_t in_features = 0;
  const void* data = nullptr;
  uint64_t data_bytes = 0;

  // F8: one fp32 scalar. I8: out_features fp32 values.
  const float* weight_scale = nullptr;
  uint64_t weight_scale_count = 0;

  // FP8 activation calibration is presence-sensitive. An absent scalar and a
  // numeric zero are not interchangeable. The checkpoint's native-matrix flag
  // is likewise retained verbatim for later GEMM plan selection.
  bool has_fp8_input_scale = false;
  float fp8_input_scale = 0.0f;
  bool full_precision_matrix_mult = false;

  // NVFP4: one raw E4M3 byte per 16 weights in the checkpoint's 128x4
  // swizzle, plus the host second-level multiplier.
  const uint8_t* block_scale = nullptr;
  uint64_t block_scale_count = 0;
  float global_scale = 1.0f;

  // NF4 double-quant state. Codes use high-even nibble order.
  const uint8_t* nf4_absmax = nullptr;
  uint64_t nf4_absmax_count = 0;
  const float* nf4_quant_map = nullptr;
  uint64_t nf4_quant_map_count = 0;  // exactly 16
  const float* nf4_nested_quant_map = nullptr;
  uint64_t nf4_nested_quant_map_count = 0;  // exactly 256
  const float* nf4_nested_absmax = nullptr;
  uint64_t nf4_nested_absmax_count = 0;
  uint32_t nf4_block_size = 64;
  uint32_t nf4_nested_block_size = 256;
  float nf4_nested_offset = 0.0f;

  // Optional AWQ activation scale and ConvRot metadata are persistent weight
  // properties, but the activation transformations record separately.
  const uint16_t* pre_quant_scale_bf16 = nullptr;
  uint64_t pre_quant_scale_count = 0;
  bool convrot = false;
  uint32_t convrot_group = 256;
};

}  // namespace slopfab
