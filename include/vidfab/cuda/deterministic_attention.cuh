#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace vidfab::cuda {

bool deterministic_attention_grid_fits(uint64_t rows, uint64_t heads,
                                       uint64_t max_grid_x,
                                       uint64_t max_grid_y) noexcept;

// Converts Q/K/V once per invocation. Multiple row-range launches may consume
// the resulting FP16 tensors without repeating the conversion.
void launch_prepare_deterministic_attention_inputs(
    cudaStream_t stream, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __half* prepared_query, __half* prepared_key, __half* prepared_value,
    uint64_t elements);

void launch_deterministic_blocked_attention_f16(
    cudaStream_t stream, const __half* query, const __half* key,
    const __half* value, __nv_bfloat16* output, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, float scale,
    uint32_t query_row_offset = 0, uint32_t rows = 0,
    uint32_t output_row_offset = 0);

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
