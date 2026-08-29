#include "vidfab/cuda/deterministic_gemm.cuh"

#include <mma.h>

#include <stdexcept>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {
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
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vidfab::cuda
