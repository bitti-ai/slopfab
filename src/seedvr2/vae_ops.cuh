#pragma once
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/workspace.cuh"
#include <cuda_bf16.h>

namespace slopfab::seedvr2 {
// Helpers own the scratch cursor for the duration of a default-stream operation.
void vae_groupnorm(const __nv_bfloat16* x, __nv_bfloat16* y, const __nv_bfloat16* weight,
                   const __nv_bfloat16* bias, int frames, int spatial, int channels, bool silu,
                   cuda::Workspace& scratch);
void vae_conv(cublasHandle_t blas, const __nv_bfloat16* x, const __nv_bfloat16* weight,
              __nv_bfloat16* y, int t, int h, int w, int ci, int co, int kt, int kh, int kw,
              bool down, int temporal_stride, cuda::Workspace& scratch);
} // namespace slopfab::seedvr2
