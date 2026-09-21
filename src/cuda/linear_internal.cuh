#pragma once
#include "slopfab/cuda/linear.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/nvfp4_gemm.cuh"

namespace slopfab::cuda::linear_detail {
constexpr int kThreads = 256;

inline int grid_1d(size_t n, int block) {
  return static_cast<int>((n + block - 1) / block);
}

inline size_t align_up(size_t n) {
  return (n + 255) / 256 * 256;
}

inline bool convrot_applies(const QuantWeight& w) {
  return w.convrot && w.convrot_group > 0 && w.in_features % w.convrot_group == 0;
}

inline size_t element_bytes(QuantFormat f) {
  switch (f) {
  case QuantFormat::kF32:
    return 4;
  case QuantFormat::kF16:
  case QuantFormat::kBF16:
    return 2;
  case QuantFormat::kF8E4M3:
  case QuantFormat::kI8:
    return 1;
  case QuantFormat::kNVFP4:
    // Half a byte. Callers that need a size must go through `stored_bytes`,
    // which knows the packing; returning 0 or 1 here would be a silent
    // half-or-double on every offset computed from it.
    throw std::runtime_error("linear: nvfp4 has no whole-byte element size");
  }
  return 0;
}

void add_bias(void* y, bool y_is_f32, const QuantWeight& weight, int rows, cudaStream_t stream);
const __nv_bfloat16* materialise_bf16(const QuantWeight&, Workspace&, cudaStream_t);
void pre_quant_scale_f32(const float*, const __nv_bfloat16*, float*, int, int, cudaStream_t);
} // namespace slopfab::cuda::linear_detail
