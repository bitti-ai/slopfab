#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace slopfab::cuda {

// Expands ComfyUI asymmetric W4A8 codes into the rotated INT8 weight consumed
// by the runtime GEMM. Channel scaling remains separate for the GEMM epilogue.
void launch_dequant_w4a8_weight(const int8_t* packed, const uint8_t* group_scale,
                                const float* codebook, int8_t* output,
                                int out_features, int in_features,
                                int group_size, cudaStream_t stream);

// Applies regular-Hadamard ConvRot-256 and dynamic row-wise INT8 activation
// quantization in one launch. Input is the FP16 operand used by the VAE GEMM.
void launch_quantize_w4a8_activation(const __half* input, int8_t* output,
                                     float* row_scale, int rows,
                                     int in_features, cudaStream_t stream);

// Converts an INT32 GEMM result in-place to FP32, applying the activation and
// per-output-channel weight scales.
void launch_dequant_w4a8_output(int32_t* input_output,
                                const float* activation_scale,
                                const float* weight_scale, int rows, int cols,
                                cudaStream_t stream);

}  // namespace slopfab::cuda
