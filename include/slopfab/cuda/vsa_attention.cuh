#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "slopfab/cuda/workspace.cuh"

namespace slopfab::cuda {

// Device maps describe packed 64-token tiles. No padded Q/K/V copies are needed.
struct VsaConfig {
  int tiles = 0, prefix_tiles = 0, heads = 0, head_dim = 128;
  const int32_t* rows = nullptr;
  const int32_t* sizes = nullptr;
  const int32_t* row_tiles = nullptr;
};

size_t vsa_attention_workspace_bytes(int tiles, int heads, int head_dim);
// compressed is [tiles, heads, head_dim], BF16. Sparse out keeps packed order.
void vsa_attention_forward(cudaStream_t stream, const __nv_bfloat16* q, const __nv_bfloat16* k,
                           const __nv_bfloat16* v, __nv_bfloat16* out, __nv_bfloat16* compressed,
                           const VsaConfig& cfg, Workspace& workspace);
// BF16 multiply, then BF16 add, matching the checkpoint's eager gate branch.
void vsa_add_compression(cudaStream_t stream, __nv_bfloat16* output, const __nv_bfloat16* gate,
                         const __nv_bfloat16* compressed, int offset, int count,
                         const VsaConfig& cfg);

} // namespace slopfab::cuda
