#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "slopfab/dit/veda.h"

namespace slopfab::cuda {

void veda_sparse_forward(cudaStream_t stream, const __nv_bfloat16* q, const __nv_bfloat16* k,
                         const __nv_bfloat16* v, __nv_bfloat16* out, int heads, int dim, int head,
                         const int32_t* rows, const int32_t* sizes, const uint8_t* mask, int tiles);

class VedaAttention {
public:
  explicit VedaAttention(std::shared_ptr<dit::VedaRequest> request);
  ~VedaAttention();
  VedaAttention(const VedaAttention&) = delete;
  VedaAttention& operator=(const VedaAttention&) = delete;
  void prepare(cudaStream_t stream);
  // Includes geometry, one layer of streamed weights and one head of scratch.
  size_t device_bytes() const;
  void forward(cudaStream_t stream, int layer, const __nv_bfloat16* q, const __nv_bfloat16* k,
               const __nv_bfloat16* v, __nv_bfloat16* out);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace slopfab::cuda
