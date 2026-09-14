#pragma once
#include <map>
#include <vector>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "slopfab/cuda/device.h"
#include "slopfab/lora.h"

namespace slopfab::cuda {
// Factors persist; activation scratch is bounded to 256 rows and shared by
// every projection. The caller supplies the original, pre-ConvRot activation.
class LoraRunner {
 public:
  void attach(const void* key, const std::vector<LoraFactors>& factors,
              int row_offset, int out);
  void apply(const void* key, const __nv_bfloat16* input, int rows,
             __nv_bfloat16* output, cublasHandle_t blas,
             cudaStream_t stream, bool exact);
  size_t weight_bytes() const;
  size_t scratch_bytes() const { return hidden_.nbytes() + delta_.nbytes(); }
 private:
  struct Factors {
    int rank = 0, in = 0, out = 0;
    DeviceBuffer<__nv_bfloat16> a, b;
  };
  std::map<const void*, std::vector<Factors>> factors_;
  DeviceBuffer<__nv_bfloat16> hidden_, delta_;
};
}  // namespace slopfab::cuda
