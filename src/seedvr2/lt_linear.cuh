#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <memory>

namespace slopfab::seedvr2 {
// Optional cuBLASLt bias epilogue. A false result leaves output untouched and
// permits the caller's original FP32-accumulator implementation to run.
class LtLinear {
public:
  LtLinear();
  ~LtLinear();
  LtLinear(const LtLinear&) = delete;
  LtLinear& operator=(const LtLinear&) = delete;
  bool forward(const __nv_bfloat16* x, const __nv_bfloat16* weights, const __nv_bfloat16* bias,
               __nv_bfloat16* output, int rows, int input_channels, int output_channels,
               cudaStream_t stream = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
