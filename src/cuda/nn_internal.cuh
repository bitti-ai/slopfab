#pragma once
#include "slopfab/cuda/nn_kernels.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "slopfab/cuda/device.h"
#include "slopfab/cuda/deterministic_math.cuh"

namespace slopfab::cuda::nn_detail {
constexpr int kWarp = 32;
constexpr int kRowThreads = 256;

// Sums across the block and returns the total to every thread. `shared` needs
// blockDim.x / 32 floats. Callers that reduce twice must __syncthreads()
// between the calls: the second call's write to shared[warp] would otherwise
// race the first call's read of shared[0].
__device__ inline float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int warps = blockDim.x / kWarp;

  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();

  value = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    if (lane == 0)
      shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__device__ inline float block_norm_inverse(float sum, uint32_t dim, float eps, float* shared) {
  // block_reduce_sum returns shared[0] to every warp. Guard the read-to-reuse
  // phase boundary before lane 0 overwrites that word with the broadcast.
  __syncthreads();
  if (threadIdx.x == 0)
    shared[0] = deterministic_norm_rsqrt(sum, dim, eps);
  __syncthreads();
  return shared[0];
}

__device__ inline float block_mean(float sum, uint32_t dim, float* shared) {
  __syncthreads();
  if (threadIdx.x == 0)
    shared[0] = deterministic_divide(sum, dim);
  __syncthreads();
  return shared[0];
}

// --- packed load/store ------------------------------------------------------
//
// VEC is 8 (one 16-byte transaction) or 1 (scalar tail path). Everything
// arrives in the caller's register array as fp32.

template <int VEC> struct BfPack;

template <> struct BfPack<8> {
  static __device__ inline void load(const __nv_bfloat16* p, float* out) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(h[i]);
      out[2 * i] = f.x;
      out[2 * i + 1] = f.y;
    }
  }

  static __device__ inline void store(__nv_bfloat16* p, const float* in) {
    uint4 raw;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      h[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
    *reinterpret_cast<uint4*>(p) = raw;
  }
};

template <> struct BfPack<1> {
  static __device__ inline void load(const __nv_bfloat16* p, float* out) {
    out[0] = __bfloat162float(*p);
  }

  static __device__ inline void store(__nv_bfloat16* p, const float* in) {
    *p = __float2bfloat16(in[0]);
  }
};

template <int VEC> struct F32Pack;

template <> struct F32Pack<8> {
  static __device__ inline void load(const float* p, float* out) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    const float4 b = *reinterpret_cast<const float4*>(p + 4);
    out[0] = a.x;
    out[1] = a.y;
    out[2] = a.z;
    out[3] = a.w;
    out[4] = b.x;
    out[5] = b.y;
    out[6] = b.z;
    out[7] = b.w;
  }

  static __device__ inline void store(float* p, const float* in) {
    *reinterpret_cast<float4*>(p) = make_float4(in[0], in[1], in[2], in[3]);
    *reinterpret_cast<float4*>(p + 4) = make_float4(in[4], in[5], in[6], in[7]);
  }
};

template <> struct F32Pack<1> {
  static __device__ inline void load(const float* p, float* out) {
    out[0] = *p;
  }

  static __device__ inline void store(float* p, const float* in) {
    *p = in[0];
  }
};

// True when a row of `dim` elements can be walked in 8-wide packs. Row bases
// are always `row * dim`, so divisibility of `dim` also settles alignment
// provided the buffer itself is 16-byte aligned — cudaMalloc gives 256 and the
// workspace gives 256.
inline bool vectorisable(int dim) {
  return dim % 8 == 0;
}

inline int reduce_shared_bytes() {
  return (kRowThreads / kWarp) * static_cast<int>(sizeof(float));
}

inline int grid_1d(size_t n, int block) {
  return static_cast<int>((n + block - 1) / block);
}

inline void require_positive(int rows, int dim, const char* what) {
  if (rows <= 0 || dim <= 0) {
    throw std::runtime_error(std::string(what) + ": rows and dim must be positive");
  }
}

} // namespace slopfab::cuda::nn_detail
