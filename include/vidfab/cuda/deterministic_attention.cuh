#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace vidfab::cuda {

// Pure reusable qualification predicate for the checked SM120a cooperative
// H3 artifact. Physical UUID is deliberately not part of eligibility; the
// parameter exists so regression tests pin that board identities are ignored.
bool deterministic_h3_cuda_tuple_fits(
    int major, int minor, const char* model, int driver_api_version,
    int runtime_version, uint32_t max_threads_per_block,
    uint32_t max_shared_bytes_per_block,
    const unsigned char physical_uuid[16]) noexcept;

bool deterministic_attention_grid_fits(uint64_t rows, uint64_t heads,
                                       uint64_t max_grid_x,
                                       uint64_t max_grid_y) noexcept;

// Runtime qualification for the pinned cooperative H3 implementation. This
// includes the reusable CUDA device/runtime/artifact tuple and its launch
// resources, and is intentionally available before any model weights load.
bool deterministic_h3_attention_available();

// These belong to the exact contract, not the independent Flash2 kernel.
// A band table built for exact attention must not borrow another backend's
// implementation constants even when the current values happen to match.
constexpr uint32_t deterministic_h3_query_tile() noexcept { return 128; }
constexpr uint32_t deterministic_h3_key_align() noexcept { return 64; }

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

// Exact-mode H3 reference paired with Vulkan H3AttentionPlan. Q/K/V and output
// are token-major BF16 [sequence,heads,head_dim], D is 64 or 128. A non-null
// device range table has four canonical int32 endpoints per global 128-row
// query tile and is traversed in ordered 64-key blocks. This pinned cooperative
// contract intentionally does not claim byte identity with shipped fused MMA.
// Inputs must satisfy the same finite BF16->FP16-V, score, denominator and PV
// accumulator domain documented by H3AttentionPlan.
void launch_deterministic_h3_attention(
    cudaStream_t stream, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* output, const int32_t* ranges, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, float scale,
    uint32_t query_row_offset = 0, uint32_t rows = 0,
    uint32_t output_row_offset = 0);

// Exact causal grouped-query reference paired with Vulkan's dedicated Qwen
// text plan. Q is [sequence,query_heads,128], K/V are
// [sequence,kv_heads,128], and query head h reads kv h/(H/Hkv). Every global
// query row sees keys [0,row] inclusive, including when rows are chunked.
// Exact mode requires finite Q/K/V, finite scaled scores, and finite PV
// accumulators/final numerators throughout the recurrence. The caller can use
// `causal_keys * max(abs(V)) <= max_finite_fp32` as a conservative proof.
void launch_deterministic_causal_gqa_attention(
    cudaStream_t stream, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* output, uint32_t sequence, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, float scale,
    uint32_t query_row_offset = 0, uint32_t rows = 0,
    uint32_t output_row_offset = 0);

}  // namespace vidfab::cuda
