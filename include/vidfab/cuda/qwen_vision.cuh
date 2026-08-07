#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "vidfab/cuda/workspace.cuh"
#include "vidfab/cuda/linear.cuh"

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

struct QwenVisionBlockWeights {
  const __nv_bfloat16* norm1_weight = nullptr;
  const __nv_bfloat16* norm1_bias = nullptr;
  const __nv_bfloat16* norm2_weight = nullptr;
  const __nv_bfloat16* norm2_bias = nullptr;
  QuantWeight qkv, attention_out, mlp_fc1, mlp_fc2;
};

struct QwenVisionBlockScratch {
  __nv_bfloat16* normed = nullptr; // [S,1152]
  __nv_bfloat16* qkv = nullptr;    // [S,3456]
  __nv_bfloat16* q = nullptr;      // four [S,1152] buffers
  __nv_bfloat16* k = nullptr;
  __nv_bfloat16* v = nullptr;
  __nv_bfloat16* branch = nullptr;
  __nv_bfloat16* mlp = nullptr;    // [S,4304]
};

void qwen_vision_block_forward(cublasHandle_t handle, cudaStream_t stream,
                               LinearRunner& linear, const QwenVisionBlockWeights& weights,
                               const float* cos, const float* sin, __nv_bfloat16* x,
                               int rows, QwenVisionBlockScratch scratch, Workspace& ws,
                               float layernorm_eps = 1e-6f);

}  // namespace vidfab::cuda
