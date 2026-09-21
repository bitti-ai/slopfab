#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "slopfab/cuda/workspace.cuh"
#include "slopfab/cuda/linear.cuh"

namespace slopfab::cuda {

// Canonical exact-mode pointwise/layout seams shared with the Vulkan vision
// stage. The shipped tower functions below retain their historical fast path.
void qwen_vision_add_positions_exact(__nv_bfloat16* x, const __nv_bfloat16* position_table,
                                     const int32_t* position_index, int rows, int hidden,
                                     cudaStream_t stream);
void qwen_vision_split_qkv_exact(const __nv_bfloat16* fused, __nv_bfloat16* query,
                                 __nv_bfloat16* key, __nv_bfloat16* value, int rows, int hidden,
                                 cudaStream_t stream);
void qwen_vision_scatter_add_exact(const __nv_bfloat16* source, const int32_t* row_index,
                                   __nv_bfloat16* destination, int rows, int hidden,
                                   cudaStream_t stream);

// Splits the fused [S,3*H] QKV projection, applies the vision tower's 2-D
// half-split RoPE to Q/K, and runs unmasked bidirectional self-attention.
// Scratch q/k/v/out are each [S,H].
void qwen_vision_attention(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* qkv,
                           const float* cos, const float* sin, __nv_bfloat16* q, __nv_bfloat16* k,
                           __nv_bfloat16* v, __nv_bfloat16* out, int rows, int heads, int head_dim,
                           Workspace& ws);

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
  __nv_bfloat16* mlp = nullptr; // [S,4304]
};

struct QwenVisionMergerWeights {
  const __nv_bfloat16* norm_weight = nullptr;
  const __nv_bfloat16* norm_bias = nullptr;
  QuantWeight fc1, fc2;
  bool norm_before_merge = false; // true for main merger, false for DeepStack
};

void qwen_vision_block_forward(cublasHandle_t handle, cudaStream_t stream, LinearRunner& linear,
                               const QwenVisionBlockWeights& weights, const float* cos,
                               const float* sin, __nv_bfloat16* x, int rows,
                               QwenVisionBlockScratch scratch, Workspace& ws,
                               float layernorm_eps = 1e-6f);

// Canonical exact-mode authority paired with ExactQwenVisionBlockStage. It
// uses deterministic dense GEMM/blocked attention/GELU/residual arithmetic;
// the existing shipped cuBLAS/native-tanh path above remains the default.
void qwen_vision_block_forward_exact(cudaStream_t stream, const QwenVisionBlockWeights& weights,
                                     const float* cos, const float* sin, __nv_bfloat16* x, int rows,
                                     QwenVisionBlockScratch scratch, float layernorm_eps = 1e-6f);

void qwen_vision_patch_embed(LinearRunner& linear, const QuantWeight& projection,
                             const __nv_bfloat16* pixel_rows, const __nv_bfloat16* position_table,
                             const int32_t* position_index, __nv_bfloat16* x, int rows,
                             Workspace& ws, cudaStream_t stream);
void qwen_vision_patch_embed_exact(cudaStream_t stream, const QuantWeight& projection,
                                   const __nv_bfloat16* pixel_rows,
                                   const __nv_bfloat16* position_table,
                                   const int32_t* position_index, __nv_bfloat16* output, int rows);

void qwen_vision_merger_forward(cudaStream_t stream, LinearRunner& linear,
                                const QwenVisionMergerWeights& weights, const __nv_bfloat16* x,
                                __nv_bfloat16* normed, __nv_bfloat16* merged, __nv_bfloat16* hidden,
                                __nv_bfloat16* output, int rows, Workspace& ws,
                                float layernorm_eps = 1e-6f);
void qwen_vision_merger_forward_exact(cudaStream_t stream, const QwenVisionMergerWeights& weights,
                                      const __nv_bfloat16* x, __nv_bfloat16* normed,
                                      __nv_bfloat16* merged, __nv_bfloat16* hidden,
                                      __nv_bfloat16* output, int rows, float layernorm_eps = 1e-6f);

// Runs all 27 blocks. DeepStack outputs correspond to completed visual blocks
// 8, 16 and 24 (zero-based indexes), exactly as the checkpoint configuration.
void qwen_vision_tower_forward(cublasHandle_t handle, cudaStream_t stream, LinearRunner& linear,
                               const QwenVisionBlockWeights* blocks,
                               const QwenVisionMergerWeights& main_merger,
                               const QwenVisionMergerWeights* deepstack_mergers, const float* cos,
                               const float* sin, __nv_bfloat16* x, int rows,
                               QwenVisionBlockScratch block_scratch, __nv_bfloat16* merger_normed,
                               __nv_bfloat16* merged, __nv_bfloat16* merger_hidden,
                               __nv_bfloat16* output, __nv_bfloat16** deepstack_outputs,
                               Workspace& ws);

} // namespace slopfab::cuda
