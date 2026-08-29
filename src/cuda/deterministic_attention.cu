#include "vidfab/cuda/deterministic_attention.cuh"

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <iterator>
#include <mutex>
#include <stdexcept>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/deterministic_math.cuh"
#include "vidfab/attention.h"

namespace vidfab::cuda {
namespace {

using namespace nvcuda;

bool ranges_overlap(const void* a, uint64_t a_bytes,
                    const void* b, uint64_t b_bytes) noexcept {
  const uintptr_t begin_a = reinterpret_cast<uintptr_t>(a);
  const uintptr_t begin_b = reinterpret_cast<uintptr_t>(b);
  if (begin_a > std::numeric_limits<uintptr_t>::max() - a_bytes ||
      begin_b > std::numeric_limits<uintptr_t>::max() - b_bytes) return true;
  return begin_a < begin_b + b_bytes && begin_b < begin_a + a_bytes;
}

constexpr uint32_t kH3AttentionThreads = 1024;
constexpr uint32_t kH3AttentionSharedBytes = 99328;

struct GridLimits {
  uint64_t x = 0;
  uint64_t y = 0;
  uint32_t max_threads_per_block = 0;
  uint32_t max_shared_bytes_per_block = 0;
  bool exact_h3_tuple = false;
};

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
    int optin_shared = 0;
    VIDFAB_CUDA_CHECK(cudaDeviceGetAttribute(
        &optin_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    int driver_version = 0, runtime_version = 0;
    VIDFAB_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    VIDFAB_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    const bool exact_h3_tuple = deterministic_h3_cuda_tuple_fits(
        properties.major, properties.minor, properties.name, driver_version,
        runtime_version, static_cast<uint32_t>(properties.maxThreadsPerBlock),
        static_cast<uint32_t>(std::max<int>(
            static_cast<int>(properties.sharedMemPerBlock), optin_shared)),
        reinterpret_cast<const unsigned char*>(properties.uuid.bytes));
    limits[device] = {
        static_cast<uint64_t>(properties.maxGridSize[0]),
        static_cast<uint64_t>(properties.maxGridSize[1]),
        static_cast<uint32_t>(properties.maxThreadsPerBlock),
        static_cast<uint32_t>(std::max<int>(
            static_cast<int>(properties.sharedMemPerBlock), optin_shared)),
        exact_h3_tuple};
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

__device__ inline float causal_bf16_value(__nv_bfloat16 value) {
  const uint16_t bits = __bfloat16_as_ushort(value);
  if ((bits & 0x7f80u) == 0u)
    return __uint_as_float(static_cast<uint32_t>(bits & 0x8000u) << 16u);
  return __bfloat162float(value);
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


template <bool Banded>
__global__ __launch_bounds__(1024, 1) void h3_attention_coop64_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ ranges, uint32_t sequence, uint32_t heads,
    uint32_t head_dim, float scale, uint32_t query_row_offset,
    uint32_t output_row_offset, uint32_t rows) {
  __shared__ __align__(16) __nv_bfloat16 query_stage[64 * 128];
  __shared__ __align__(16) __nv_bfloat16 key_stage[64 * 128];
  __shared__ __align__(16) __half value_stage[16 * 128];
  __shared__ __align__(16) float scores[64 * 64];
  __shared__ __align__(16) __half probabilities[64 * 64];
  __shared__ __align__(16) float output_accumulators[64 * 128];
  __shared__ float soft_partial[1024], next_maximum[64];
  __shared__ float running_max[64], running_sum[64], correction[64];

  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5;
  const uint32_t aligned_first = query_row_offset & ~63u;
  const uint32_t query_base = aligned_first + blockIdx.x * 64u;
  const uint32_t request_end = query_row_offset + rows;
  const uint32_t head = blockIdx.y;
  for (uint32_t i = tid; i < 64u * head_dim; i += blockDim.x) {
    const uint32_t row = i / head_dim;
    const uint32_t d = i - row * head_dim;
    const uint32_t query_row = query_base + row;
    query_stage[i] = query_row < sequence
        ? __float2bfloat16_rn(causal_bf16_value(query[
              (static_cast<size_t>(query_row) * heads + head) * head_dim + d]))
        : __float2bfloat16_rn(0.0f);
    output_accumulators[i] = 0.0f;
  }
  if (tid < 64u) {
    running_max[tid] = negative_infinity();
    running_sum[tid] = 0.0f;
  }
  __syncthreads();

  int range_values[4] = {0, static_cast<int>(sequence), 0, 0};
  int range_count = 1;
  if constexpr (Banded) {
    const int32_t* selected = ranges + static_cast<size_t>(query_base / 128u) * 4u;
    for (int i = 0; i < 4; ++i) range_values[i] = selected[i];
    range_count = 2;
  }
  for (int range_index = 0; range_index < range_count; ++range_index) {
    const uint32_t range_start = static_cast<uint32_t>(range_values[range_index * 2]);
    const uint32_t range_stop = static_cast<uint32_t>(range_values[range_index * 2 + 1]);
    for (uint32_t key_base = range_start; key_base < range_stop; key_base += 64u) {
      for (uint32_t i = tid; i < 64u * head_dim; i += blockDim.x) {
        const uint32_t key_in_block = i / head_dim;
        const uint32_t d = i - key_in_block * head_dim;
        const uint32_t key_row = key_base + key_in_block;
        key_stage[i] = key_row < range_stop && key_row < sequence
            ? __float2bfloat16_rn(causal_bf16_value(key[
                  (static_cast<size_t>(key_row) * heads + head) * head_dim + d]))
            : __float2bfloat16_rn(0.0f);
      }
      __syncthreads();
      if (warp < 16u) {
        const uint32_t query_tile = warp >> 2;
        const uint32_t key_tile = warp & 3u;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_fragment;
        wmma::fill_fragment(score_fragment, 0.0f);
        for (uint32_t d_base = 0; d_base < head_dim; d_base += 16u) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                         wmma::row_major> a;
          wmma::load_matrix_sync(
              a, query_stage + query_tile * 16u * head_dim + d_base, head_dim);
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                         wmma::col_major> b;
          wmma::load_matrix_sync(
              b, key_stage + key_tile * 16u * head_dim + d_base, head_dim);
          wmma::mma_sync(score_fragment, a, b, score_fragment);
        }
        wmma::store_matrix_sync(
            scores + query_tile * 16u * 64u + key_tile * 16u,
            score_fragment, 64, wmma::mem_row_major);
      }
      __syncthreads();
      for (uint32_t i = tid; i < 4096u; i += blockDim.x) {
        const uint32_t key_row = key_base + (i & 63u);
        scores[i] = key_row < range_stop && key_row < sequence
            ? canonicalize_subnormal(__fmul_rn(scores[i], scale))
            : negative_infinity();
      }
      __syncthreads();

      const uint32_t row = tid >> 4;
      const uint32_t part = tid & 15u;
      float part_max = negative_infinity();
      for (uint32_t j = part * 4u; j < part * 4u + 4u; ++j)
        part_max = fmaxf(part_max, scores[row * 64u + j]);
      soft_partial[tid] = part_max;
      __syncthreads();
      if (part == 0u) {
        float tile_max = soft_partial[tid];
        for (uint32_t i = 1u; i < 16u; ++i)
          tile_max = fmaxf(tile_max, soft_partial[tid + i]);
        const float next = fmaxf(running_max[row], tile_max);
        correction[row] = running_max[row] == negative_infinity()
            ? 0.0f : deterministic_exp_nonpositive(running_max[row] - next);
        next_maximum[row] = next;
      }
      __syncthreads();
      for (uint32_t j = part * 4u; j < part * 4u + 4u; ++j) {
        const float score = scores[row * 64u + j];
        const float exponential = score == negative_infinity()
            ? 0.0f : deterministic_exp_nonpositive(score - next_maximum[row]);
        probabilities[row * 64u + j] = __float2half_rn(exponential);
        scores[row * 64u + j] = exponential;
      }
      __syncthreads();
      if (part < 8u) {
        float chunk_sum = 0.0f;
        for (uint32_t j = part * 8u; j < part * 8u + 8u; ++j)
          chunk_sum = __fadd_rn(chunk_sum, scores[row * 64u + j]);
        soft_partial[row * 8u + part] = chunk_sum;
      }
      __syncthreads();
      if (part == 0u) {
        float tile_sum = soft_partial[row * 8u];
        for (uint32_t i = 1u; i < 8u; ++i)
          tile_sum = __fadd_rn(tile_sum, soft_partial[row * 8u + i]);
        running_sum[row] = fmaf(running_sum[row], correction[row], tile_sum);
        running_max[row] = next_maximum[row];
      }
      __syncthreads();

      for (uint32_t i = tid; i < 64u * head_dim; i += blockDim.x) {
        const uint32_t output_row = i / head_dim;
        output_accumulators[i] = canonicalize_subnormal(__fmul_rn(
            output_accumulators[i], correction[output_row]));
      }
      __syncthreads();
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_fragment;
      const uint32_t d_tiles = head_dim / 16u;
      const uint32_t total_tiles = 4u * d_tiles;
      if (warp < total_tiles) {
        const uint32_t tile_query = warp / d_tiles;
        const uint32_t d_tile = warp - tile_query * d_tiles;
        wmma::load_matrix_sync(
            pv_fragment,
            output_accumulators + tile_query * 16u * head_dim + d_tile * 16u,
            head_dim, wmma::mem_row_major);
      }
      for (uint32_t key_part = 0; key_part < 64u; key_part += 16u) {
        for (uint32_t i = tid; i < 16u * head_dim; i += blockDim.x) {
          const uint32_t key_in_part = i / head_dim;
          const uint32_t d = i - key_in_part * head_dim;
          const uint32_t key_row = key_base + key_part + key_in_part;
          value_stage[i] = key_row < range_stop && key_row < sequence
              ? __float2half_rn(causal_bf16_value(value[
                    (static_cast<size_t>(key_row) * heads + head) * head_dim + d]))
              : __float2half_rn(0.0f);
        }
        __syncthreads();
        if (warp < total_tiles) {
            const uint32_t tile_query = warp / d_tiles;
            const uint32_t d_tile = warp - tile_query * d_tiles;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                           wmma::row_major> a;
            wmma::load_matrix_sync(
                a, probabilities + tile_query * 16u * 64u + key_part, 64);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                           wmma::row_major> b;
            wmma::load_matrix_sync(
                b, value_stage + d_tile * 16u, head_dim);
            wmma::mma_sync(pv_fragment, a, b, pv_fragment);
        }
        __syncthreads();
      }
      if (warp < total_tiles) {
          const uint32_t tile_query = warp / d_tiles;
          const uint32_t d_tile = warp - tile_query * d_tiles;
          wmma::store_matrix_sync(
              output_accumulators + tile_query * 16u * head_dim + d_tile * 16u,
              pv_fragment, head_dim, wmma::mem_row_major);
      }
      __syncthreads();
    }
  }
  const uint32_t pairs_per_row = head_dim >> 1u;
  for (uint32_t pair = tid; pair < 64u * pairs_per_row; pair += blockDim.x) {
    const uint32_t row = pair / pairs_per_row;
    const uint32_t d = (pair - row * pairs_per_row) * 2u;
    const uint32_t query_row = query_base + row;
    if (query_row >= query_row_offset && query_row < request_end &&
        query_row < sequence) {
      const uint32_t output_row = output_row_offset + query_row - query_row_offset;
      const size_t output_index =
          (static_cast<size_t>(output_row) * heads + head) * head_dim + d;
      const uint16_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(
          deterministic_float_divide(output_accumulators[row * head_dim + d],
                                     running_sum[row])));
      const uint16_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(
          deterministic_float_divide(output_accumulators[row * head_dim + d + 1u],
                                     running_sum[row])));
      reinterpret_cast<uint32_t*>(output)[output_index >> 1u] =
          static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16u);
    }
  }
}

__global__ void causal_gqa_attention_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ output, uint32_t sequence,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim, float scale,
    uint32_t query_row_offset, uint32_t output_row_offset, uint32_t rows) {
  __shared__ float score_or_probability[128];
  __shared__ float reduction[128];
  __shared__ uint32_t output_bits[128];
  const uint32_t lane = threadIdx.x;
  const uint32_t local_row = blockIdx.x;
  const uint32_t query_head = blockIdx.y;
  if (local_row >= rows || query_head >= query_heads) return;
  const uint32_t query_row = query_row_offset + local_row;
  const uint32_t kv_head = query_head / (query_heads / kv_heads);
  const size_t q_base =
      (static_cast<size_t>(query_row) * query_heads + query_head) * head_dim;
  const uint32_t causal_keys = query_row + 1;
  float running_max = negative_infinity();
  float running_sum = 0.0f;
  float output_accumulator = 0.0f;

  for (uint32_t key_base = 0; key_base < causal_keys; key_base += 128) {
    const uint32_t key_row = key_base + lane;
    float score = 0.0f;
    if (key_row < causal_keys) {
      const size_t k_base =
          (static_cast<size_t>(key_row) * kv_heads + kv_head) * head_dim;
      for (uint32_t d = 0; d < head_dim; ++d) {
        const float product = canonicalize_subnormal(
            __fmul_rn(causal_bf16_value(query[q_base + d]),
                      causal_bf16_value(key[k_base + d])));
        score = canonicalize_subnormal(__fadd_rn(score, product));
      }
      score = canonicalize_subnormal(__fmul_rn(score, scale));
    } else {
      score = negative_infinity();
    }
    score_or_probability[lane] = score;
    reduction[lane] = score;
    __syncthreads();
    for (uint32_t width = 64; width != 0; width >>= 1) {
      if (lane < width)
        reduction[lane] = fmaxf(reduction[lane], reduction[lane + width]);
      __syncthreads();
    }
    const float tile_max = reduction[0];
    const float next_max = fmaxf(running_max, tile_max);
    const float correction = running_max == negative_infinity()
        ? 0.0f : deterministic_exp_nonpositive(running_max - next_max);
    const float exponential = key_row < causal_keys
        ? deterministic_exp_nonpositive(score - next_max) : 0.0f;
    __syncthreads();
    score_or_probability[lane] =
        __bfloat162float(__float2bfloat16_rn(exponential));
    reduction[lane] = exponential;
    __syncthreads();
    for (uint32_t width = 64; width != 0; width >>= 1) {
      if (lane < width) reduction[lane] += reduction[lane + width];
      __syncthreads();
    }
    if (lane < head_dim) {
      output_accumulator = canonicalize_subnormal(output_accumulator * correction);
      for (uint32_t j = 0; j < 128 && key_base + j < causal_keys; ++j) {
        const size_t v_index =
            ((static_cast<size_t>(key_base + j) * kv_heads + kv_head) *
             head_dim) + lane;
        const float product = canonicalize_subnormal(
            __fmul_rn(score_or_probability[j],
                      causal_bf16_value(value[v_index])));
        output_accumulator = canonicalize_subnormal(
            __fadd_rn(output_accumulator, product));
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
        ((static_cast<size_t>(output_row_offset + local_row) * query_heads +
          query_head) * head_dim) + lane;
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

bool deterministic_h3_attention_available() {
  const GridLimits limits = cached_grid_limits();
  return limits.exact_h3_tuple &&
         limits.max_threads_per_block >= kH3AttentionThreads &&
         limits.max_shared_bytes_per_block >= kH3AttentionSharedBytes;
}

bool deterministic_h3_cuda_tuple_fits(
    int major, int minor, const char* model, int driver_api_version,
    int runtime_version, uint32_t max_threads_per_block,
    uint32_t max_shared_bytes_per_block,
    const unsigned char physical_uuid[16]) noexcept {
  (void)physical_uuid;  // Board identity must not exclude another RTX 5090.
  return major == 12 && minor == 0 && model != nullptr &&
      std::strcmp(model, "NVIDIA GeForce RTX 5090") == 0 &&
      driver_api_version >= 13000 && driver_api_version < 14000 &&
      runtime_version == 13000 &&
      max_threads_per_block >= kH3AttentionThreads &&
      max_shared_bytes_per_block >= kH3AttentionSharedBytes;
}

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

void launch_deterministic_h3_attention(
    cudaStream_t stream, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* output, const int32_t* ranges, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, float scale,
    uint32_t query_row_offset, uint32_t rows,
    uint32_t output_row_offset) {
  if (!query || !key || !value || !output || query == key || query == value ||
      key == value || sequence == 0 || heads == 0 ||
      (head_dim != 64 && head_dim != 128) ||
      !is_exact_attention_scale(head_dim, scale) ||
      query_row_offset > sequence) {
    throw std::invalid_argument(
        "deterministic H3 attention: invalid tensor/configuration");
  }
  const uint32_t selected_rows =
      rows == 0 ? sequence - query_row_offset : rows;
  if (selected_rows == 0 ||
      static_cast<uint64_t>(query_row_offset) + selected_rows > sequence ||
      static_cast<uint64_t>(output_row_offset) + selected_rows > sequence) {
    throw std::invalid_argument("deterministic H3 attention: invalid row range");
  }
  const uint64_t elements =
      static_cast<uint64_t>(sequence) * heads * head_dim;
  if (elements > std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range("deterministic H3 attention: uint32 indexing overflow");
  }
  const uint64_t bytes = elements * sizeof(__nv_bfloat16);
  if (ranges_overlap(query, bytes, key, bytes) ||
      ranges_overlap(query, bytes, value, bytes) ||
      ranges_overlap(key, bytes, value, bytes) ||
      ranges_overlap(query, bytes, output, bytes) ||
      ranges_overlap(key, bytes, output, bytes) ||
      ranges_overlap(value, bytes, output, bytes)) {
    throw std::invalid_argument("deterministic H3 attention: all tensors must be distinct");
  }
  const GridLimits limits = cached_grid_limits();
  if (!deterministic_attention_grid_fits(
          selected_rows, heads, limits.x, limits.y)) {
    throw std::out_of_range("deterministic H3 attention: CUDA grid limit exceeded");
  }
  if (limits.max_threads_per_block < kH3AttentionThreads ||
      limits.max_shared_bytes_per_block < kH3AttentionSharedBytes ||
      !limits.exact_h3_tuple) {
    throw std::runtime_error(
        "deterministic H3 attention: unqualified CUDA tuple or resources");
  }
  const uint32_t aligned_first = query_row_offset & ~63u;
  const uint32_t query_end = query_row_offset + selected_rows;
  const uint32_t query_groups = (query_end - aligned_first + 63u) / 64u;
  if (ranges) {
    h3_attention_coop64_kernel<true><<<dim3(query_groups, heads),
        kH3AttentionThreads, 0, stream>>>(
        query, key, value, output, ranges, sequence, heads, head_dim, scale,
        query_row_offset, output_row_offset, selected_rows);
  } else {
    h3_attention_coop64_kernel<false><<<dim3(query_groups, heads),
        kH3AttentionThreads, 0, stream>>>(
        query, key, value, output, nullptr, sequence, heads, head_dim, scale,
        query_row_offset, output_row_offset, selected_rows);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_deterministic_causal_gqa_attention(
    cudaStream_t stream, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* output, uint32_t sequence, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, float scale,
    uint32_t query_row_offset, uint32_t rows,
    uint32_t output_row_offset) {
  if (!query || !key || !value || !output || query == key || query == value ||
      key == value || sequence == 0 || query_heads == 0 || kv_heads == 0 ||
      query_heads % kv_heads != 0 || head_dim != 128 ||
      !is_exact_attention_scale(head_dim, scale) ||
      query_row_offset > sequence) {
    throw std::invalid_argument(
        "deterministic causal GQA attention: invalid tensor/configuration");
  }
  const uint32_t selected_rows =
      rows == 0 ? sequence - query_row_offset : rows;
  if (selected_rows == 0 ||
      static_cast<uint64_t>(query_row_offset) + selected_rows > sequence ||
      static_cast<uint64_t>(output_row_offset) + selected_rows > sequence) {
    throw std::invalid_argument(
        "deterministic causal GQA attention: invalid row range");
  }
  const uint64_t query_elements =
      static_cast<uint64_t>(sequence) * query_heads * head_dim;
  const uint64_t kv_elements =
      static_cast<uint64_t>(sequence) * kv_heads * head_dim;
  if (query_elements > std::numeric_limits<uint32_t>::max() ||
      kv_elements > std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range(
        "deterministic causal GQA attention: uint32 indexing overflow");
  }
  const uint64_t query_bytes = query_elements * sizeof(__nv_bfloat16);
  const uint64_t kv_bytes = kv_elements * sizeof(__nv_bfloat16);
  if (ranges_overlap(query, query_bytes, output, query_bytes) ||
      ranges_overlap(key, kv_bytes, output, query_bytes) ||
      ranges_overlap(value, kv_bytes, output, query_bytes)) {
    throw std::invalid_argument(
        "deterministic causal GQA attention: output must be distinct");
  }
  const GridLimits limits = cached_grid_limits();
  if (!deterministic_attention_grid_fits(
          selected_rows, query_heads, limits.x, limits.y)) {
    throw std::out_of_range(
        "deterministic causal GQA attention: CUDA grid limit exceeded");
  }
  causal_gqa_attention_kernel<<<dim3(selected_rows, query_heads), 128, 0,
                                stream>>>(
      query, key, value, output, sequence, query_heads, kv_heads, head_dim,
      scale, query_row_offset, output_row_offset, selected_rows);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vidfab::cuda
