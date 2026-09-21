#pragma once

#include <cstdint>

namespace slopfab {

// Exact dense row-major NT contraction modes shared by CUDA and Vulkan.
// C[M,N] = A[M,K] * W[N,K]^T. Each output consumes K in ascending order
// through one fp32 fused multiply-add per element.
enum class DenseGemmMode : uint32_t {
  kBFloat16 = 0,   // BF16 A/W, BF16 output
  kFloat16Vae = 1, // fp32 A narrowed to fp16, fp16 W, fp32 output
  kFloat32 = 2,    // fp32 A/W/output
};

enum class DenseGemmBias : uint32_t {
  kNone = 0,
  kFloat32 = 1,
  kBFloat16 = 2,
};

struct DenseGemmPlanDesc {
  uint32_t max_rows = 0;
  uint32_t out_features = 0;
  uint32_t in_features = 0;
  DenseGemmMode mode = DenseGemmMode::kBFloat16;
  DenseGemmBias bias = DenseGemmBias::kNone;
  // Preserve strict ascending-K fp32 FMA semantics even for full cooperative
  // tiles. Required when the value domain can produce subnormal intermediate
  // products, whose cooperative-matrix handling is not portable.
  bool force_scalar_order = false;
};

} // namespace slopfab
