#include "slopfab/cuda/w4a8.cuh"

#include <cuda_fp8.h>

#include <stdexcept>

#include "slopfab/cuda/device.h"

namespace slopfab::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kConvRotGroup = 256;

__device__ float h4(int row, float x0, float x1, float x2, float x3) {
  switch (row) {
  case 0:
    return x0 + x1 + x2 - x3;
  case 1:
    return x0 + x1 - x2 + x3;
  case 2:
    return x0 - x1 + x2 + x3;
  default:
    return -x0 + x1 + x2 + x3;
  }
}

__global__ void dequant_weight_kernel(const int8_t* __restrict__ packed,
                                      const uint8_t* __restrict__ group_scale,
                                      const float* __restrict__ codebook,
                                      int8_t* __restrict__ output, size_t packed_count,
                                      int packed_per_row, int in_features, int groups_per_row,
                                      int group_size) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= packed_count)
    return;
  const int row = static_cast<int>(index / packed_per_row);
  const int byte_col = static_cast<int>(index % packed_per_row);
  const int col = byte_col * 2;
  const uint8_t byte = static_cast<uint8_t>(packed[index]);
  const uint8_t scale_bits =
      group_scale[static_cast<size_t>(row) * groups_per_row + col / group_size];
  const __half_raw scale_half = __nv_cvt_fp8_to_halfraw(scale_bits, __NV_E4M3);
  const float scale = __half2float(*reinterpret_cast<const __half*>(&scale_half));
  const int c0 = byte & 15;
  const int c1 = byte >> 4;
  const int q0 = max(-127, min(127, __float2int_rn(codebook[c0] * scale)));
  const int q1 = max(-127, min(127, __float2int_rn(codebook[c1] * scale)));
  const size_t base = static_cast<size_t>(row) * in_features + col;
  output[base] = static_cast<int8_t>(q0);
  output[base + 1] = static_cast<int8_t>(q1);
}

__global__ void quantize_activation_kernel(const __half* __restrict__ input,
                                           int8_t* __restrict__ output,
                                           float* __restrict__ row_scale, int in_features) {
  extern __shared__ float shared[];
  float* row = shared;
  float* temporary = shared + in_features;
  const int tid = threadIdx.x;
  const int source_row = blockIdx.x;
  const size_t offset = static_cast<size_t>(source_row) * in_features;

  for (int col = tid; col < in_features; col += blockDim.x)
    row[col] = __half2float(input[offset + col]);
  __syncthreads();

  // H = kron(h4,h4,h4,h4)/16. Four radix-4 stages each contribute 1/2.
  for (int group = 0; group < in_features; group += kConvRotGroup) {
    float* src = row + group;
    float* dst = temporary;
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      const int stride = stage == 0 ? 1 : stage == 1 ? 4 : stage == 2 ? 16 : 64;
      const int digit = (tid / stride) & 3;
      const int base = tid - digit * stride;
      dst[tid] = 0.5f * h4(digit, src[base], src[base + stride], src[base + 2 * stride],
                           src[base + 3 * stride]);
      __syncthreads();
      float* swap = src;
      src = dst;
      dst = swap;
    }
  }

  float maximum = 0.0f;
  for (int col = tid; col < in_features; col += blockDim.x)
    maximum = fmaxf(maximum, fabsf(row[col]));
  temporary[tid] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (tid < stride)
      temporary[tid] = fmaxf(temporary[tid], temporary[tid + stride]);
    __syncthreads();
  }
  const float scale = fmaxf(fminf(temporary[0], 65504.0f) * (1.0f / 127.0f), 1.0e-30f);
  if (tid == 0)
    row_scale[source_row] = scale;
  // Comfy's FP16 path rounds the rotated value, scale, and quotient through
  // the activation dtype before the final nearest-even INT8 conversion.
  const float rounded_scale = __half2float(__float2half_rn(scale));
  for (int col = tid; col < in_features; col += blockDim.x) {
    const float rounded_value = __half2float(__float2half_rn(row[col]));
    const float divided = __half2float(__float2half_rn(rounded_value / rounded_scale));
    const int q = max(-128, min(127, __float2int_rn(divided)));
    output[offset + col] = static_cast<int8_t>(q);
  }
}

__global__ void dequant_output_kernel(int32_t* input_output, const float* activation_scale,
                                      const float* weight_scale, size_t count, int cols) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const int col = static_cast<int>(index % cols);
  const int row = static_cast<int>(index / cols);
  const float value =
      static_cast<float>(input_output[index]) * activation_scale[row] * weight_scale[col];
  reinterpret_cast<float*>(input_output)[index] = value;
}

} // namespace

void launch_dequant_w4a8_weight(const int8_t* packed, const uint8_t* group_scale,
                                const float* codebook, int8_t* output, int out_features,
                                int in_features, int group_size, cudaStream_t stream) {
  if (!packed || !group_scale || !codebook || !output)
    throw std::invalid_argument("W4A8 weight dequantization received a null buffer");
  if (out_features <= 0 || in_features <= 0 || group_size <= 0 || (in_features & 1) != 0 ||
      in_features % group_size != 0)
    throw std::invalid_argument("W4A8 weight dequantization received an invalid shape");
  const size_t count = static_cast<size_t>(out_features) * in_features / 2;
  dequant_weight_kernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads, 0,
                          stream>>>(packed, group_scale, codebook, output, count, in_features / 2,
                                    in_features, in_features / group_size, group_size);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_quantize_w4a8_activation(const __half* input, int8_t* output, float* row_scale,
                                     int rows, int in_features, cudaStream_t stream) {
  if (!input || !output || !row_scale)
    throw std::invalid_argument("W4A8 activation quantization received a null buffer");
  if (rows <= 0 || in_features <= 0 || in_features % kConvRotGroup != 0)
    throw std::invalid_argument("W4A8 activation requires K divisible by 256");
  const size_t shared = (static_cast<size_t>(in_features) + kConvRotGroup) * sizeof(float);
  quantize_activation_kernel<<<static_cast<unsigned>(rows), kThreads, shared, stream>>>(
      input, output, row_scale, in_features);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_dequant_w4a8_output(int32_t* input_output, const float* activation_scale,
                                const float* weight_scale, int rows, int cols,
                                cudaStream_t stream) {
  if (!input_output || !activation_scale || !weight_scale)
    throw std::invalid_argument("W4A8 output dequantization received a null buffer");
  if (rows <= 0 || cols <= 0)
    throw std::invalid_argument("W4A8 output dequantization received an invalid shape");
  const size_t count = static_cast<size_t>(rows) * cols;
  dequant_output_kernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads, 0,
                          stream>>>(input_output, activation_scale, weight_scale, count, cols);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

} // namespace slopfab::cuda
