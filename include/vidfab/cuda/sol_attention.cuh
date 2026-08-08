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

bool sol_pipeline_forward(cudaStream_t stream, const __nv_bfloat16* q,
                          const __nv_bfloat16* k, const __nv_bfloat16* v,
                          const __nv_bfloat16* km, const float* vs,
                          const float* tau, __nv_bfloat16* out,
                          const AttentionConfig& cfg);

}  // namespace vidfab::cuda
