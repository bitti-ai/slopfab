#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "vidfab/gemm.h"

namespace vidfab::cuda {

// Exact-tuple CUDA counterpart of the Vulkan cooperative BF16 plan. Shapes
// must be full M64/N16/K16 tiles; edge matrices use the scalar reference path.
void launch_deterministic_bf16_gemm_nt(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const void* bias, __nv_bfloat16* output, uint32_t rows,
    uint32_t out_features, uint32_t in_features, DenseGemmBias bias_type,
    uint32_t input_row_offset = 0, uint32_t output_row_offset = 0,
    cudaStream_t stream = nullptr);

// Exact-tuple CUDA counterpart of the Vulkan fp16 cooperative plan. Input is
// the once-per-chunk narrowed activation slot; output remains fp32.
void launch_deterministic_f16_gemm_nt(
    const __half* input, const __half* weight, float* output, uint32_t rows,
    uint32_t out_features, uint32_t in_features,
    uint32_t output_row_offset = 0, cudaStream_t stream = nullptr);

}  // namespace vidfab::cuda
