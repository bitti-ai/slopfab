#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/workspace.cuh"

namespace vidfab::cuda {

size_t sol_attention_workspace_bytes(const AttentionConfig& cfg);

void sol_attention_forward(cudaStream_t stream, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v,
                           __nv_bfloat16* out, const AttentionConfig& cfg,
                           Workspace& ws);

// Internal fast path. sol_attention_forward validates that the current device
// is exactly SM120 before entering this pipeline.
bool sol_pipeline_forward(cudaStream_t stream, const __nv_bfloat16* q,
                          const __nv_bfloat16* k, const __nv_bfloat16* v,
                          const __nv_bfloat16* km, const __nv_bfloat16* vm,
                          const float* vs, const float* k_residual,
                          const float* v_residual,
                          const float* tau, __nv_bfloat16* out,
                          const AttentionConfig& cfg);

}  // namespace vidfab::cuda
