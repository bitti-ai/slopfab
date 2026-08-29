#include "vidfab/cuda/deterministic_attention.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/deterministic_math.cuh"

namespace vidfab::cuda {
namespace {

__device__ inline float bf16_as_f16(const __nv_bfloat16* values, size_t index) {
  return __half2float(__float2half_rn(__bfloat162float(values[index])));
}

__device__ inline float f16_round(float value) {
  return __half2float(__float2half_rn(value));
}

__device__ inline float negative_infinity() {
  return __uint_as_float(0xff800000u);
}

__global__ void blocked_attention_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ output, uint32_t sequence, uint32_t heads,
    uint32_t head_dim, float scale, uint32_t query_row_offset,
    uint32_t output_row_offset, uint32_t rows) {
  __shared__ float score_or_probability[128];
  __shared__ float reduction[128];
  __shared__ uint32_t output_bits[128];
  const uint32_t lane = threadIdx.x;
  const uint32_t local_row = blockIdx.x;
  const uint32_t head = blockIdx.y;
  if (local_row >= rows || head >= heads) return;
  const uint32_t query_row = query_row_offset + local_row;
  const size_t q_base = (static_cast<size_t>(query_row) * heads + head) * head_dim;
  float running_max = negative_infinity();
  float running_sum = 0.0f;
  float output_accumulator = 0.0f;

  for (uint32_t key_base = 0; key_base < sequence; key_base += 128) {
    const uint32_t key_row = key_base + lane;
    float score = 0.0f;
    if (key_row < sequence) {
      const size_t k_base = (static_cast<size_t>(key_row) * heads + head) * head_dim;
      for (uint32_t d = 0; d < head_dim; ++d) {
        score = fmaf(bf16_as_f16(query, q_base + d),
                     bf16_as_f16(key, k_base + d), score);
      }
      score = f16_round(score * scale);
    } else {
      score = negative_infinity();
    }
    score_or_probability[lane] = score;
    reduction[lane] = score;
    __syncthreads();
    for (uint32_t width = 64; width != 0; width >>= 1) {
      if (lane < width) reduction[lane] = fmaxf(reduction[lane], reduction[lane + width]);
      __syncthreads();
    }
    const float tile_max = reduction[0];
    const float next_max = fmaxf(running_max, tile_max);
    const float correction = running_max == negative_infinity()
        ? 0.0f : deterministic_exp_nonpositive(running_max - next_max);
    const float exponential = key_row < sequence
        ? deterministic_exp_nonpositive(score - next_max) : 0.0f;
    score_or_probability[lane] = f16_round(exponential);
    reduction[lane] = exponential;
    __syncthreads();
    for (uint32_t width = 64; width != 0; width >>= 1) {
      if (lane < width) reduction[lane] += reduction[lane + width];
      __syncthreads();
    }
    if (lane < head_dim) {
      output_accumulator *= correction;
      for (uint32_t j = 0; j < 128 && key_base + j < sequence; ++j) {
        const size_t v_index = ((static_cast<size_t>(key_base + j) * heads + head) *
                                head_dim) + lane;
        output_accumulator = fmaf(score_or_probability[j],
                                  bf16_as_f16(value, v_index), output_accumulator);
      }
    }
    running_sum = fmaf(running_sum, correction, reduction[0]);
    running_max = next_max;
    __syncthreads();
  }

  if (lane < head_dim) {
    output_bits[lane] = static_cast<uint32_t>(__bfloat16_as_ushort(
        __float2bfloat16_rn(deterministic_float_divide(output_accumulator,
                                                       running_sum))));
  }
  __syncthreads();
  if (lane < head_dim && (lane & 1u) == 0u) {
    const size_t output_index =
        ((static_cast<size_t>(output_row_offset + local_row) * heads + head) *
         head_dim) + lane;
    const uint32_t high = lane + 1 < head_dim ? output_bits[lane + 1] : 0u;
    reinterpret_cast<uint32_t*>(output)[output_index >> 1] =
        output_bits[lane] | (high << 16u);
  }
}

}  // namespace

void launch_deterministic_blocked_attention(
    cudaStream_t stream, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* output, uint32_t sequence, uint32_t heads,
    uint32_t head_dim, float scale, uint32_t query_row_offset,
    uint32_t rows, uint32_t output_row_offset) {
  if (!query || !key || !value || !output || query == key || query == value ||
      key == value || output == query || output == key || output == value ||
      sequence == 0 || heads == 0 ||
      (head_dim != 64 && head_dim != 72 && head_dim != 128) ||
      !std::isnormal(scale) || scale <= 0.0f || query_row_offset > sequence) {
    throw std::invalid_argument("deterministic attention: invalid tensor/configuration");
  }
  const uint32_t selected_rows = rows == 0 ? sequence - query_row_offset : rows;
  if (selected_rows == 0 || static_cast<uint64_t>(query_row_offset) + selected_rows > sequence ||
      static_cast<uint64_t>(output_row_offset) + selected_rows > sequence) {
    throw std::invalid_argument("deterministic attention: invalid row range");
  }
  const uint64_t elements = static_cast<uint64_t>(sequence) * heads * head_dim;
  if (elements > std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range("deterministic attention: uint32 indexing overflow");
  }
  blocked_attention_kernel<<<dim3(selected_rows, heads), 128, 0, stream>>>(
      query, key, value, output, sequence, heads, head_dim, scale,
      query_row_offset, output_row_offset, selected_rows);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vidfab::cuda
