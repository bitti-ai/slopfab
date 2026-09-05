#include "slopfab/cuda/deterministic_gemm.cuh"

#include <mma.h>

#include <stdexcept>

#include "slopfab/cuda/device.h"

namespace slopfab::cuda {
namespace {

using namespace nvcuda;

__global__ void deterministic_bf16_gemm_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const void* bias, __nv_bfloat16* output, uint32_t rows,
    uint32_t out_features, uint32_t in_features, uint32_t bias_type,
    uint32_t input_row_offset, uint32_t output_row_offset) {
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t row = blockIdx.y * 64 + warp * 16;
  const uint32_t column = blockIdx.x * 16;
  using Bf16 = __nv_bfloat16;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum;
  wmma::fill_fragment(sum, 0.0f);
  for (uint32_t base = 0; base < in_features; base += 16) {
    wmma::load_matrix_sync(
        a, input + (input_row_offset + row) * in_features + base,
        in_features);
    wmma::load_matrix_sync(b, weight + column * in_features + base,
                           in_features);
    wmma::mma_sync(sum, a, b, sum);
  }
  __shared__ float accumulator[4][256];
  wmma::store_matrix_sync(accumulator[warp], sum, 16, wmma::mem_row_major);
  __syncwarp();
  auto load_bias = [&](uint32_t index) {
    if (bias_type == static_cast<uint32_t>(DenseGemmBias::kFloat32))
      return static_cast<const float*>(bias)[index];
    return __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[index]);
  };
  for (uint32_t pair = lane; pair < 128; pair += 32) {
    const uint32_t tile_row = pair >> 3;
    const uint32_t tile_column = (pair & 7) * 2;
    const uint32_t item = tile_row * 16 + tile_column;
    float lo = __bfloat162float(__float2bfloat16_rn(accumulator[warp][item]));
    float hi = __bfloat162float(__float2bfloat16_rn(accumulator[warp][item + 1]));
    if (bias_type != static_cast<uint32_t>(DenseGemmBias::kNone)) {
      lo += load_bias(column + tile_column);
      hi += load_bias(column + tile_column + 1);
    }
    const __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
    reinterpret_cast<__nv_bfloat162*>(output +
        (output_row_offset + row + tile_row) * out_features)[
            (column + tile_column) >> 1] = packed;
  }
}

__global__ void deterministic_f16_gemm_kernel(
    const __half* input, const __half* weight, float* output,
    uint32_t out_features, uint32_t in_features,
    uint32_t output_row_offset) {
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t row = blockIdx.y * 64 + warp * 16;
  const uint32_t column = blockIdx.x * 16;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum;
  wmma::fill_fragment(sum, 0.0f);
  for (uint32_t base = 0; base < in_features; base += 16) {
    wmma::load_matrix_sync(a, input + row * in_features + base, in_features);
    wmma::load_matrix_sync(b, weight + column * in_features + base, in_features);
    wmma::mma_sync(sum, a, b, sum);
  }
  __shared__ float accumulator[4][256];
  wmma::store_matrix_sync(accumulator[warp], sum, 16, wmma::mem_row_major);
  __syncwarp();
  for (uint32_t item = threadIdx.x & 31; item < 256; item += 32) {
    const uint32_t tile_row = item >> 4;
    const uint32_t tile_column = item & 15;
    output[(output_row_offset + row + tile_row) * out_features +
           column + tile_column] = accumulator[warp][item];
  }
}

__global__ void deterministic_scalar_gemm_kernel(
    const void* input, const void* weight, const void* bias, void* output,
    uint32_t rows, uint32_t out_features, uint32_t in_features,
    uint32_t mode, uint32_t bias_type, uint32_t input_row_offset,
    uint32_t output_row_offset) {
  const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || column >= out_features) return;
  float sum = 0.0f;
  for (uint32_t inner = 0; inner < in_features; ++inner) {
    float a = 0.0f, w = 0.0f;
    const size_t ai = size_t(input_row_offset + row) * in_features + inner;
    const size_t wi = size_t(column) * in_features + inner;
    if (mode == static_cast<uint32_t>(DenseGemmMode::kBFloat16)) {
      a = __bfloat162float(static_cast<const __nv_bfloat16*>(input)[ai]);
      w = __bfloat162float(static_cast<const __nv_bfloat16*>(weight)[wi]);
    } else if (mode == static_cast<uint32_t>(DenseGemmMode::kFloat16Vae)) {
      a = __half2float(static_cast<const __half*>(input)[ai]);
      w = __half2float(static_cast<const __half*>(weight)[wi]);
    } else {
      a = static_cast<const float*>(input)[ai];
      w = static_cast<const float*>(weight)[wi];
    }
    sum = fmaf(a, w, sum);
  }
  const size_t oi = size_t(output_row_offset + row) * out_features + column;
  if (mode == static_cast<uint32_t>(DenseGemmMode::kBFloat16)) {
    sum = __bfloat162float(__float2bfloat16_rn(sum));
    if (bias_type == static_cast<uint32_t>(DenseGemmBias::kFloat32))
      sum += static_cast<const float*>(bias)[column];
    else if (bias_type == static_cast<uint32_t>(DenseGemmBias::kBFloat16))
      sum += __bfloat162float(
          static_cast<const __nv_bfloat16*>(bias)[column]);
    static_cast<__nv_bfloat16*>(output)[oi] = __float2bfloat16_rn(sum);
  } else {
    if (bias_type == static_cast<uint32_t>(DenseGemmBias::kFloat32))
      sum += static_cast<const float*>(bias)[column];
    static_cast<float*>(output)[oi] = sum;
  }
}

}  // namespace

void launch_deterministic_bf16_gemm_nt(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const void* bias, __nv_bfloat16* output, uint32_t rows,
    uint32_t out_features, uint32_t in_features, DenseGemmBias bias_type,
    uint32_t input_row_offset, uint32_t output_row_offset,
    cudaStream_t stream) {
  if (!input || !weight || !output || rows == 0 ||
      rows % 64 != 0 || out_features == 0 || out_features % 16 != 0 ||
      in_features == 0 || in_features % 16 != 0 ||
      (bias_type == DenseGemmBias::kNone) != (bias == nullptr)) {
    throw std::invalid_argument("deterministic CUDA BF16 GEMM: invalid full-tile arguments");
  }
  if (bias_type != DenseGemmBias::kNone &&
      bias_type != DenseGemmBias::kFloat32 &&
      bias_type != DenseGemmBias::kBFloat16) {
    throw std::invalid_argument("deterministic CUDA BF16 GEMM: invalid bias type");
  }
  deterministic_bf16_gemm_kernel<<<
      dim3(out_features / 16, rows / 64), 128, 0, stream>>>(
          input, weight, bias, output, rows, out_features, in_features,
          static_cast<uint32_t>(bias_type), input_row_offset,
          output_row_offset);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_deterministic_f16_gemm_nt(
    const __half* input, const __half* weight, float* output, uint32_t rows,
    uint32_t out_features, uint32_t in_features,
    uint32_t output_row_offset, cudaStream_t stream) {
  if (!input || !weight || !output || rows == 0 || rows % 64 != 0 ||
      out_features == 0 || out_features % 16 != 0 ||
      in_features == 0 || in_features % 16 != 0) {
    throw std::invalid_argument(
        "deterministic CUDA F16 GEMM: invalid full-tile arguments");
  }
  deterministic_f16_gemm_kernel<<<
      dim3(out_features / 16, rows / 64), 128, 0, stream>>>(
          input, weight, output, out_features, in_features,
          output_row_offset);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_deterministic_scalar_gemm_nt(
    const void* input, const void* weight, const void* bias, void* output,
    uint32_t rows, uint32_t out_features, uint32_t in_features,
    DenseGemmMode mode, DenseGemmBias bias_type,
    uint32_t input_row_offset, uint32_t output_row_offset,
    cudaStream_t stream) {
  const bool mode_valid = mode == DenseGemmMode::kBFloat16 ||
      mode == DenseGemmMode::kFloat16Vae || mode == DenseGemmMode::kFloat32;
  const bool bias_valid = bias_type == DenseGemmBias::kNone ||
      (mode == DenseGemmMode::kBFloat16 &&
       (bias_type == DenseGemmBias::kFloat32 ||
        bias_type == DenseGemmBias::kBFloat16)) ||
      (mode == DenseGemmMode::kFloat32 && bias_type == DenseGemmBias::kFloat32);
  if (!input || !weight || !output || rows == 0 || out_features == 0 ||
      in_features == 0 || !mode_valid || !bias_valid ||
      ((bias_type == DenseGemmBias::kNone) != (bias == nullptr))) {
    throw std::invalid_argument("deterministic CUDA scalar GEMM: invalid arguments");
  }
  constexpr uint32_t tx = 16, ty = 8;
  deterministic_scalar_gemm_kernel<<<
      dim3((out_features + tx - 1) / tx, (rows + ty - 1) / ty),
      dim3(tx, ty), 0, stream>>>(
          input, weight, bias, output, rows, out_features, in_features,
          static_cast<uint32_t>(mode), static_cast<uint32_t>(bias_type),
          input_row_offset, output_row_offset);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace slopfab::cuda
