#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace vidfab::cuda {

// Exact reference paired with Vulkan BlockedAttentionPlan. Inputs/output are
// token-major BF16 [sequence,heads,head_dim], head_dim is 64/72/128, and scale
// is supplied as an already-pinned finite-normal host bit pattern.
void launch_deterministic_blocked_attention(
    cudaStream_t stream, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* output, uint32_t sequence, uint32_t heads,
    uint32_t head_dim, float scale, uint32_t query_row_offset = 0,
    uint32_t rows = 0, uint32_t output_row_offset = 0);

}  // namespace vidfab::cuda
