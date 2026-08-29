#include "vidfab/cuda/deterministic_attention.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <limits>
#include <iterator>
#include <mutex>
#include <stdexcept>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/deterministic_math.cuh"
#include "vidfab/attention.h"

namespace vidfab::cuda {
namespace {

bool ranges_overlap(const void* a, uint64_t a_bytes,
                    const void* b, uint64_t b_bytes) noexcept {
  const uintptr_t begin_a = reinterpret_cast<uintptr_t>(a);
  const uintptr_t begin_b = reinterpret_cast<uintptr_t>(b);
  if (begin_a > std::numeric_limits<uintptr_t>::max() - a_bytes ||
      begin_b > std::numeric_limits<uintptr_t>::max() - b_bytes) return true;
  return begin_a < begin_b + b_bytes && begin_b < begin_a + a_bytes;
}

struct GridLimits { uint64_t x = 0; uint64_t y = 0; };

GridLimits cached_grid_limits() {
  constexpr int kCachedDevices = 32;
  static std::once_flag once[kCachedDevices];
  static GridLimits limits[kCachedDevices];
  int device = 0;
  VIDFAB_CUDA_CHECK(cudaGetDevice(&device));
  if (device < 0 || device >= kCachedDevices) {
    throw std::out_of_range("deterministic attention: CUDA device index is out of cache range");
  }
  std::call_once(once[device], [device] {
    cudaDeviceProp properties{};
    VIDFAB_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    limits[device] = {static_cast<uint64_t>(properties.maxGridSize[0]),
                      static_cast<uint64_t>(properties.maxGridSize[1])};
  });
  return limits[device];
}

template <typename T>
__device__ inline float prepared_value(const T* values, size_t index);

template <>
__device__ inline float prepared_value(const __nv_bfloat16* values, size_t index) {
  return __half2float(__float2half_rn(__bfloat162float(values[index])));
}

template <>
__device__ inline float prepared_value(const __half* values, size_t index) {
  return __half2float(values[index]);
}

__device__ inline float f16_round(float value) {
  return __half2float(__float2half_rn(value));
}

__device__ inline float negative_infinity() {
  return __uint_as_float(0xff800000u);
}

__device__ inline float canonicalize_subnormal(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x7fffffffu) < 0x00800000u
      ? __uint_as_float(bits & 0x80000000u) : value;
}

template <typename Input>
__global__ void blocked_attention_kernel(
    const Input* __restrict__ query,
    const Input* __restrict__ key,
    const Input* __restrict__ value,
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
        score = fmaf(prepared_value(query, q_base + d),
                     prepared_value(key, k_base + d), score);
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
    __syncthreads();
    score_or_probability[lane] = f16_round(exponential);
    reduction[lane] = exponential;
    __syncthreads();
    for (uint32_t width = 64; width != 0; width >>= 1) {
      if (lane < width) reduction[lane] += reduction[lane + width];
      __syncthreads();
    }
    if (lane < head_dim) {
      output_accumulator = canonicalize_subnormal(
          output_accumulator * correction);
      for (uint32_t j = 0; j < 128 && key_base + j < sequence; ++j) {
        const size_t v_index = ((static_cast<size_t>(key_base + j) * heads + head) *
                                head_dim) + lane;
        output_accumulator = canonicalize_subnormal(
            fmaf(score_or_probability[j], prepared_value(value, v_index),
                 output_accumulator));
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

__global__ void prepare_attention_inputs_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, __half* prepared_query,
    __half* prepared_key, __half* prepared_value, uint64_t elements) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  prepared_query[index] = __float2half_rn(__bfloat162float(query[index]));
  prepared_key[index] = __float2half_rn(__bfloat162float(key[index]));
  prepared_value[index] = __float2half_rn(__bfloat162float(value[index]));
}

template <typename Input>
void launch_attention_impl(
    cudaStream_t stream, const Input* query, const Input* key,
    const Input* value, __nv_bfloat16* output, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, float scale,
    uint32_t query_row_offset, uint32_t rows, uint32_t output_row_offset) {
  if (!query || !key || !value || !output || query == key || query == value ||
      key == value || sequence == 0 || heads == 0 ||
      (head_dim != 64 && head_dim != 72 && head_dim != 128) ||
      !is_exact_attention_scale(head_dim, scale) || query_row_offset > sequence) {
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
  const uint64_t tensor_bytes = elements * sizeof(Input);
  const uint64_t output_bytes = elements * sizeof(__nv_bfloat16);
  if (ranges_overlap(query, tensor_bytes, output, output_bytes) ||
      ranges_overlap(key, tensor_bytes, output, output_bytes) ||
      ranges_overlap(value, tensor_bytes, output, output_bytes)) {
    throw std::invalid_argument("deterministic attention: output must be distinct");
  }
  const GridLimits limits = cached_grid_limits();
  if (!deterministic_attention_grid_fits(
          selected_rows, heads, limits.x, limits.y)) {
    throw std::out_of_range("deterministic attention: CUDA grid limit exceeded");
  }
  blocked_attention_kernel<<<dim3(selected_rows, heads), 128, 0, stream>>>(
      query, key, value, output, sequence, heads, head_dim, scale,
      query_row_offset, output_row_offset, selected_rows);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

bool deterministic_attention_grid_fits(uint64_t rows, uint64_t heads,
                                       uint64_t max_grid_x,
                                       uint64_t max_grid_y) noexcept {
  return rows != 0 && heads != 0 && rows <= max_grid_x && heads <= max_grid_y;
}

void launch_prepare_deterministic_attention_inputs(
    cudaStream_t stream, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __half* prepared_query, __half* prepared_key, __half* prepared_value,
    uint64_t elements) {
  if (!query || !key || !value || !prepared_query || !prepared_key ||
      !prepared_value || elements == 0 || elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("deterministic attention: invalid preparation");
  }
  const uint64_t bytes = elements * sizeof(uint16_t);
  const void* buffers[] = {query, key, value, prepared_query, prepared_key,
                           prepared_value};
  for (size_t i = 0; i < std::size(buffers); ++i) {
    for (size_t j = i + 1; j < std::size(buffers); ++j) {
      if (ranges_overlap(buffers[i], bytes, buffers[j], bytes)) {
        throw std::invalid_argument(
            "deterministic attention: preparation buffers must be distinct");
      }
    }
  }
  const uint32_t blocks = static_cast<uint32_t>((elements + 255) / 256);
  prepare_attention_inputs_kernel<<<blocks, 256, 0, stream>>>(
      query, key, value, prepared_query, prepared_key, prepared_value, elements);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_deterministic_blocked_attention_f16(
    cudaStream_t stream, const __half* query, const __half* key,
    const __half* value, __nv_bfloat16* output, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, float scale,
    uint32_t query_row_offset, uint32_t rows, uint32_t output_row_offset) {
  launch_attention_impl(stream, query, key, value, output, sequence, heads,
                        head_dim, scale, query_row_offset, rows,
                        output_row_offset);
}

void launch_deterministic_blocked_attention(
    cudaStream_t stream, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* output, uint32_t sequence, uint32_t heads,
    uint32_t head_dim, float scale, uint32_t query_row_offset,
    uint32_t rows, uint32_t output_row_offset) {
  launch_attention_impl(stream, query, key, value, output, sequence, heads,
                        head_dim, scale, query_row_offset, rows,
                        output_row_offset);
}

}  // namespace vidfab::cuda
