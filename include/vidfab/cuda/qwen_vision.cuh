#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "vidfab/cuda/workspace.cuh"

namespace vidfab::cuda {

// Splits the fused [S,3*H] QKV projection, applies the vision tower's 2-D
// half-split RoPE to Q/K, and runs unmasked bidirectional self-attention.
// Scratch q/k/v/out are each [S,H].
void qwen_vision_attention(cublasHandle_t handle, cudaStream_t stream,
                           const __nv_bfloat16* qkv, const float* cos,
                           const float* sin, __nv_bfloat16* q,
                           __nv_bfloat16* k, __nv_bfloat16* v,
                           __nv_bfloat16* out, int rows, int heads,
                           int head_dim, Workspace& ws);

}  // namespace vidfab::cuda
