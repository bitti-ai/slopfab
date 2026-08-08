#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace vidfab::cuda {

struct AttentionConfig;
class Workspace;

bool sage2_supported(const AttentionConfig& cfg, int device, const char** reason = nullptr);
size_t sage2_workspace_bytes(const AttentionConfig& cfg, int num_kv_heads);
void sage2_attention_forward(cudaStream_t stream, const __nv_bfloat16* q,
                             const __nv_bfloat16* k, const __nv_bfloat16* v,
                             __nv_bfloat16* out, const AttentionConfig& cfg,
                             int num_kv_heads, Workspace& ws);

}  // namespace vidfab::cuda
