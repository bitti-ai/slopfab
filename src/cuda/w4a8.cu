#include "slopfab/cuda/w4a8.cuh"

#include <cuda_fp8.h>

#include <stdexcept>

#include "slopfab/cuda/device.h"

namespace slopfab::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kConvRotGroup = 256;

__global__ void dequant_w6_kernel(const int8_t* packed, const uint8_t* group_scale, int8_t* output,
                                  size_t count, int width, int group_size) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const size_t row = index / width;
  const int col = int(index % width);
  const auto* bytes = reinterpret_cast<const uint8_t*>(packed) + row * (width * 3 / 4);
  const int low = (bytes[col / 2] >> ((col & 1) * 4)) & 15;
  const int high = (bytes[width / 2 + col / 4] >> ((col & 3) * 2)) & 3;
  const int level = (low | (high << 4)) - 32;
  const __half_raw half = __nv_cvt_fp8_to_halfraw(group_scale[index / group_size], __NV_E4M3);
  const float scale = __half2float(*reinterpret_cast<const __half*>(&half));
  output[index] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(level * scale))));
}

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

void launch_dequant_w6a8_weight(const int8_t* packed, const uint8_t* group_scale, int8_t* output,
                                int out_features, int in_features, int group_size,
                                cudaStream_t stream) {
  if (!packed || !group_scale || !output || out_features <= 0 || in_features <= 0 ||
      in_features % 32 || group_size < 16 || group_size % 16 || in_features % group_size)
    throw std::invalid_argument("W6A8 weight decode: invalid buffers or dimensions");
  const size_t count = size_t(out_features) * in_features;
  dequant_w6_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
      packed, group_scale, output, count, in_features, group_size);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

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

namespace {
__global__ void quantize_bf16(const __nv_bfloat16* input, int8_t* output, float* scales,
                              int width) {
  __shared__ float values[256];
  int t = threadIdx.x, row = blockIdx.x;
  size_t offset = size_t(row) * width;
  float maximum = 0;
  for (int i = t; i < width; i += 256)
    maximum = fmaxf(maximum, fabsf(__bfloat162float(input[offset + i])));
  values[t] = maximum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (t < d)
      values[t] = fmaxf(values[t], values[t + d]);
    __syncthreads();
  }
  const float scale = fmaxf(values[0] / 127.f, 1e-30f);
  if (t == 0)
    scales[row] = scale;
  const float rounded = __bfloat162float(__float2bfloat16(scale));
  for (int i = t; i < width; i += 256) {
    float q = __bfloat162float(__float2bfloat16(__bfloat162float(input[offset + i]) / rounded));
    output[offset + i] = static_cast<int8_t>(max(-128, min(127, __float2int_rn(q))));
  }
}

__global__ void output_bf16(const int32_t* input, const float* rows, const float* channels,
                            __nv_bfloat16* output, size_t n, int cols) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = __float2bfloat16(float(input[i]) * rows[i / cols] * channels[i % cols]);
}
}

void launch_quantize_w4a8_bf16(const __nv_bfloat16* input, int8_t* output, float* scales, int rows,
                               int width, cudaStream_t stream) {
  if (rows <= 0 || width <= 0)
    throw std::invalid_argument("W4A8 activation shape");
  quantize_bf16<<<rows, 256, 0, stream>>>(input, output, scales, width);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_w4a8_output_bf16(const int32_t* input, const float* rows, const float* channels,
                             __nv_bfloat16* output, int count, int cols, cudaStream_t stream) {
  size_t n = size_t(count) * cols;
  output_bf16<<<(n + 255) / 256, 256, 0, stream>>>(input, rows, channels, output, n, cols);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

} // namespace slopfab::cuda
