// Transformer primitives. See the header for the contract and the operator
// order hazards; this file is the implementation and the reasons it is shaped
// the way it is.
//
// Two conventions run through everything here:
//
//   - Arithmetic is fp32 even where storage is bf16. bf16 has 8 mantissa bits,
//     so a 5376-term sum of squares accumulated in bf16 would stop growing
//     roughly a third of the way along the row.
//   - Row-wise kernels load 8 bf16 (one 16-byte transaction) per thread where
//     the row length allows it. Every production width here — 128, 5376, 7168,
//     14336 — is a multiple of 8; the scalar path exists for tests and for the
//     odd width a future component might bring.

#include "vidfab/cuda/nn_kernels.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/deterministic_math.cuh"

namespace vidfab::cuda {
namespace {

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
  if (lane == 0) shared[warp] = value;
  __syncthreads();

  value = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__device__ inline float block_norm_inverse(float sum, uint32_t dim, float eps,
                                           float* shared) {
  if (threadIdx.x == 0) shared[0] = deterministic_norm_rsqrt(sum, dim, eps);
  __syncthreads();
  return shared[0];
}

__device__ inline float block_mean(float sum, uint32_t dim, float* shared) {
  if (threadIdx.x == 0) shared[0] = deterministic_divide(sum, dim);
  __syncthreads();
  return shared[0];
}

// --- packed load/store ------------------------------------------------------
//
// VEC is 8 (one 16-byte transaction) or 1 (scalar tail path). Everything
// arrives in the caller's register array as fp32.

template <int VEC>
struct BfPack;

template <>
struct BfPack<8> {
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
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
    *reinterpret_cast<uint4*>(p) = raw;
  }
};

template <>
struct BfPack<1> {
  static __device__ inline void load(const __nv_bfloat16* p, float* out) {
    out[0] = __bfloat162float(*p);
  }
  static __device__ inline void store(__nv_bfloat16* p, const float* in) {
    *p = __float2bfloat16(in[0]);
  }
};

template <int VEC>
struct F32Pack;

template <>
struct F32Pack<8> {
  static __device__ inline void load(const float* p, float* out) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    const float4 b = *reinterpret_cast<const float4*>(p + 4);
    out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
    out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
  }
  static __device__ inline void store(float* p, const float* in) {
    *reinterpret_cast<float4*>(p) = make_float4(in[0], in[1], in[2], in[3]);
    *reinterpret_cast<float4*>(p + 4) = make_float4(in[4], in[5], in[6], in[7]);
  }
};

template <>
struct F32Pack<1> {
  static __device__ inline void load(const float* p, float* out) { out[0] = *p; }
  static __device__ inline void store(float* p, const float* in) { *p = in[0]; }
};

// True when a row of `dim` elements can be walked in 8-wide packs. Row bases
// are always `row * dim`, so divisibility of `dim` also settles alignment
// provided the buffer itself is 16-byte aligned — cudaMalloc gives 256 and the
// workspace gives 256.
inline bool vectorisable(int dim) { return dim % 8 == 0; }

inline int reduce_shared_bytes() { return (kRowThreads / kWarp) * static_cast<int>(sizeof(float)); }

// --- rmsnorm ----------------------------------------------------------------
//
// out = x * rsqrt(mean(x^2) + eps) * w. eps is inside the sqrt and added to the
// mean of squares (spec 9.3). x is read twice: the second pass hits L2, which
// is cheaper than the shared memory it would take to cache a 5376-wide row.

template <int VEC, class XPack, class WPack, class OPack, class XT, class WT, class OT>
__device__ inline void rmsnorm_body(const XT* xr, const WT* w, OT* outr, int dim, float eps,
                                    float* shared) {
  const int packs = dim / VEC;
  float buf[VEC];

  float sum_sq = 0.0f;
  for (int p = threadIdx.x; p < packs; p += blockDim.x) {
    XPack::load(xr + p * VEC, buf);
#pragma unroll
    for (int i = 0; i < VEC; ++i) sum_sq += buf[i] * buf[i];
  }
  const float inv = block_norm_inverse(block_reduce_sum(sum_sq, shared),
                                       static_cast<uint32_t>(dim), eps, shared);

  float wbuf[VEC];
  for (int p = threadIdx.x; p < packs; p += blockDim.x) {
    XPack::load(xr + p * VEC, buf);
    WPack::load(w + p * VEC, wbuf);
#pragma unroll
    for (int i = 0; i < VEC; ++i) buf[i] = buf[i] * inv * wbuf[i];
    OPack::store(outr + p * VEC, buf);
  }
}

template <int VEC>
__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ w,
                                    __nv_bfloat16* __restrict__ out, int dim, float eps) {
  extern __shared__ float shared[];
  const size_t row = blockIdx.x;
  rmsnorm_body<VEC, BfPack<VEC>, BfPack<VEC>, BfPack<VEC>>(x + row * dim, w, out + row * dim, dim,
                                                           eps, shared);
}

template <int VEC>
__global__ void rmsnorm_f32_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                   float* __restrict__ out, int dim, float eps) {
  extern __shared__ float shared[];
  const size_t row = blockIdx.x;
  rmsnorm_body<VEC, F32Pack<VEC>, F32Pack<VEC>, F32Pack<VEC>>(x + row * dim, w, out + row * dim,
                                                              dim, eps, shared);
}

__global__ void layernorm_affine_kernel(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ w,
                                        const __nv_bfloat16* __restrict__ bias,
                                        __nv_bfloat16* __restrict__ out, int dim, float eps) {
  extern __shared__ float shared[];
  const size_t base = static_cast<size_t>(blockIdx.x) * dim;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) sum += __bfloat162float(x[base + i]);
  const float mean = block_mean(block_reduce_sum(sum, shared),
                                static_cast<uint32_t>(dim), shared);
  __syncthreads();
  float sq = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = __bfloat162float(x[base + i]) - mean; sq += d * d;
  }
  const float inv = block_norm_inverse(block_reduce_sum(sq, shared),
                                       static_cast<uint32_t>(dim), eps, shared);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float y = (__bfloat162float(x[base + i]) - mean) * inv * __bfloat162float(w[i]) +
                    __bfloat162float(bias[i]);
    out[base + i] = __float2bfloat16(y);
  }
}

// --- narrow-row rmsnorm -----------------------------------------------------
//
// q_norm/k_norm normalise over head_dim = 128, which is 16 packs. The block
// kernel above would put 256 threads on those 16 packs — 6 % of the lanes doing
// work while 7.5 warps idle through a shared-memory reduction over zeros and
// two barriers. Measured at rows*heads = 2.11M: 2.02 ms against 0.71 ms here.
//
// One warp owns a row, so the reduction is a pure shuffle butterfly with no
// shared memory and no __syncthreads(). The row also fits in registers, which
// removes the second load of x that the block kernel pays (it re-reads to apply
// the weight); that is a third of the DRAM traffic gone on top of the
// utilisation fix.
//
// blockDim.x is exactly 32 so a warp never straddles two rows: threadIdx.y is
// warp-uniform, which is what makes the early return and the full-mask shuffle
// legal.
template <int VEC, class XPack, class WPack, class OPack, class XT, class WT, class OT>
__global__ void rmsnorm_warp_kernel(const XT* __restrict__ x, const WT* __restrict__ w,
                                    OT* __restrict__ out, int rows, int dim, float eps) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (row >= static_cast<size_t>(rows)) return;

  const int packs = dim / VEC;  // <= 32, enforced by the launcher
  const int lane = static_cast<int>(threadIdx.x);
  const bool active = lane < packs;

  float buf[VEC];
  float sum_sq = 0.0f;
  if (active) {
    XPack::load(x + row * dim + lane * VEC, buf);
#pragma unroll
    for (int i = 0; i < VEC; ++i) sum_sq += buf[i] * buf[i];
  }
  // Butterfly, not shfl_down: every lane needs the total, and xor leaves it in
  // all 32 without a broadcast. Idle lanes contribute their zero.
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor_sync(0xFFFFFFFFu, sum_sq, offset);
  }
  float inv = lane == 0
                  ? deterministic_norm_rsqrt(sum_sq, static_cast<uint32_t>(dim), eps)
                  : 0.0f;
  inv = __shfl_sync(0xFFFFFFFFu, inv, 0);

  if (active) {
    float wbuf[VEC];
    WPack::load(w + lane * VEC, wbuf);
#pragma unroll
    for (int i = 0; i < VEC; ++i) buf[i] = buf[i] * inv * wbuf[i];
    OPack::store(out + row * dim + lane * VEC, buf);
  }
}

// Rows per block for the warp kernel: 8 warps is the usual sweet spot and keeps
// the grid under 2^31 for every sequence length this project sees.
constexpr int kWarpRowsPerBlock = 8;

// The warp kernel needs the whole row to land on one warp.
inline bool narrow_row(int dim, int vec) { return dim / vec <= kWarp; }

// --- rmsnorm + AdaLN modulation ---------------------------------------------
//
// n * (1 + scale) + shift, in that order (spec 9.4.1). Written as a single
// fmaf chain per element so the compiler cannot re-associate it into
// n*scale + (n + shift), which rounds differently.

template <int VEC, class XPack, class WPack, class OPack, class XT, class WT, class OT>
__device__ inline void modulate_body(const XT* xr, const WT* w, const float* scale_row,
                                     const float* shift_row, OT* outr, int dim, float eps,
                                     float* shared) {
  const int packs = dim / VEC;
  float buf[VEC];

  float sum_sq = 0.0f;
  for (int p = threadIdx.x; p < packs; p += blockDim.x) {
    XPack::load(xr + p * VEC, buf);
#pragma unroll
    for (int i = 0; i < VEC; ++i) sum_sq += buf[i] * buf[i];
  }
  const float inv = block_norm_inverse(block_reduce_sum(sum_sq, shared),
                                       static_cast<uint32_t>(dim), eps, shared);

  float wbuf[VEC];
  float sc[VEC];
  float sh[VEC];
  for (int p = threadIdx.x; p < packs; p += blockDim.x) {
    XPack::load(xr + p * VEC, buf);
    WPack::load(w + p * VEC, wbuf);
    F32Pack<VEC>::load(scale_row + p * VEC, sc);
    F32Pack<VEC>::load(shift_row + p * VEC, sh);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float n = buf[i] * inv * wbuf[i];
      buf[i] = fmaf(n, 1.0f + sc[i], sh[i]);
    }
    OPack::store(outr + p * VEC, buf);
  }
}

template <int VEC>
__global__ void rmsnorm_modulate_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ w,
                                             const float* __restrict__ scale,
                                             const float* __restrict__ shift,
                                             const int32_t* __restrict__ a,
                                             __nv_bfloat16* __restrict__ out, int dim, float eps) {
  extern __shared__ float shared[];
  const size_t row = blockIdx.x;
  const size_t mod = static_cast<size_t>(a[row]) * dim;
  modulate_body<VEC, BfPack<VEC>, BfPack<VEC>, BfPack<VEC>>(x + row * dim, w, scale + mod,
                                                            shift + mod, out + row * dim, dim, eps,
                                                            shared);
}

template <int VEC>
__global__ void rmsnorm_modulate_f32_kernel(const float* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ w,
                                            const float* __restrict__ scale,
                                            const float* __restrict__ shift,
                                            const int32_t* __restrict__ a, float* __restrict__ out,
                                            int dim, float eps) {
  extern __shared__ float shared[];
  const size_t row = blockIdx.x;
  const size_t mod = static_cast<size_t>(a[row]) * dim;
  modulate_body<VEC, F32Pack<VEC>, BfPack<VEC>, F32Pack<VEC>>(x + row * dim, w, scale + mod,
                                                              shift + mod, out + row * dim, dim,
                                                              eps, shared);
}

// --- gated residual ---------------------------------------------------------
//
// x += gate[a[row]] * branch. The gate multiplies the branch only; the
// residual passes through ungated (spec 9.4.3).

// Rows go on grid.x throughout this file: a 768p/10s request packs ~73.4k
// rows, and gridDim.y is capped at 65535.
template <int VEC>
__global__ void add_gated_kernel(__nv_bfloat16* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ branch,
                                 const float* __restrict__ gate, const int32_t* __restrict__ a,
                                 int dim) {
  const size_t row = blockIdx.x;
  const size_t base = row * dim;
  const float* g = gate + static_cast<size_t>(a[row]) * dim;

  const int packs = dim / VEC;
  const int p = blockIdx.y * blockDim.x + threadIdx.x;
  if (p >= packs) return;

  float xb[VEC];
  float bb[VEC];
  float gb[VEC];
  BfPack<VEC>::load(x + base + p * VEC, xb);
  BfPack<VEC>::load(branch + base + p * VEC, bb);
  F32Pack<VEC>::load(g + p * VEC, gb);
#pragma unroll
  for (int i = 0; i < VEC; ++i) xb[i] = fmaf(gb[i], bb[i], xb[i]);
  BfPack<VEC>::store(x + base + p * VEC, xb);
}

// --- SwiGLU -----------------------------------------------------------------
//
// out[i] = silu(fused[i]) * fused[inner + i]. First half is the gate: our
// checkpoints use the original `mlp.fc1` naming (spec 4.4). Swapping the
// halves produces a well-scaled wrong answer.

template <int VEC>
__global__ void swiglu_kernel(const __nv_bfloat16* __restrict__ fused,
                              __nv_bfloat16* __restrict__ out, int inner) {
  const size_t row = blockIdx.x;
  const int packs = inner / VEC;
  const int p = blockIdx.y * blockDim.x + threadIdx.x;
  if (p >= packs) return;

  const __nv_bfloat16* src = fused + row * (2 * static_cast<size_t>(inner)) + p * VEC;
  float gate[VEC];
  float value[VEC];
  BfPack<VEC>::load(src, gate);
  BfPack<VEC>::load(src + inner, value);
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    gate[i] = (gate[i] / (1.0f + __expf(-gate[i]))) * value[i];
  }
  BfPack<VEC>::store(out + row * static_cast<size_t>(inner) + p * VEC, gate);
}

__global__ void silu_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  out[i] = v / (1.0f + __expf(-v));
}

__global__ void gelu_tanh_kernel(__nv_bfloat16* x, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = __bfloat162float(x[i]);
    constexpr float k = 0.7978845608028654f; // sqrt(2/pi)
    x[i] = __float2bfloat16(0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v))));
  }
}

__global__ void add_bf16_kernel(__nv_bfloat16* x, const __nv_bfloat16* branch, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(branch[i]));
}

// out = a - b, VEC elements per thread. The block cache captures a delta the
// size of the whole residual stream (844 MB at the production geometry), so
// this reads 1.7 GB and writes 844 MB every time it runs; it is purely
// bandwidth-bound and the vectorised load is the only thing that matters.
//
// Only `a` is `__restrict__`: `out` aliases `b` on the block cache's own call,
// which is deliberate (nn_kernels.cuh) and would be undefined behaviour if `b`
// claimed not to alias.
template <int VEC>
__global__ void sub_bf16_kernel(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* b,
                                __nv_bfloat16* out, size_t n) {
  const size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC;
  // `n` is a multiple of VEC by construction in `launch_sub_bf16`, which splits
  // the tail into its own scalar launch. This *skips* a partial pack, it does
  // not clamp one — an n that broke that invariant would leave the last 1-7
  // elements of `out` holding the previous step's delta, which is finite,
  // plausible, and invisible.
  if (i + VEC > n) return;
  float av[VEC];
  float bv[VEC];
  BfPack<VEC>::load(a + i, av);
  BfPack<VEC>::load(b + i, bv);
#pragma unroll
  for (int j = 0; j < VEC; ++j) av[j] -= bv[j];
  BfPack<VEC>::store(out + i, av);
}

__global__ void scatter_add_rows_kernel(const __nv_bfloat16* src, const int32_t* index,
                                        __nv_bfloat16* dst, int dim) {
  const int r = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d < dim) {
    const size_t to = static_cast<size_t>(index[r]) * dim + d;
    dst[to] = __float2bfloat16(__bfloat162float(dst[to]) +
                              __bfloat162float(src[static_cast<size_t>(r) * dim + d]));
  }
}

__global__ void merge_four_rows_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                       int dim) {
  const int g = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d < 4 * dim) dst[static_cast<size_t>(g) * 4 * dim + d] =
      src[static_cast<size_t>(g) * 4 * dim + d];
}

// --- rotary -----------------------------------------------------------------
//
// MM-RoPE rotates the first 96 of 128 head channels, pairing j with j+48
// (spec 5.3). Channels 96..127 are copied untouched, which the kernel achieves
// by not writing them at all — the transform is in place.
//
// One warp per (row, head): 48 pairs, so each of 32 lanes handles one or two.

constexpr int kRopeHalf = 48;
constexpr int kRopeDim = 2 * kRopeHalf;  // 96 rotary channels of 128

__device__ float rope_bf16_value(__nv_bfloat16 value) {
  const unsigned short bits = __bfloat16_as_ushort(value);
  return (bits & 0x7fffu) < 0x0080u
      ? __uint_as_float(static_cast<unsigned>(bits & 0x8000u) << 16u)
      : __bfloat162float(value);
}

__device__ float rope_float_value(float value) {
  const unsigned bits = __float_as_uint(value);
  return (bits & 0x7fffffffu) < 0x00800000u
      ? __uint_as_float(bits & 0x80000000u) : value;
}

__device__ __nv_bfloat16 rope_bf16_result(float value) {
  const __nv_bfloat16 rounded = __float2bfloat16(value);
  const unsigned short bits = __bfloat16_as_ushort(rounded);
  return (bits & 0x7fffu) < 0x0080u
      ? __ushort_as_bfloat16(bits & 0x8000u) : rounded;
}

__global__ void rope_h3_kernel(__nv_bfloat16* __restrict__ x, const float* __restrict__ cos_tab,
                               const float* __restrict__ sin_tab, int rows, int heads,
                               int head_dim) {
  const int lane = threadIdx.x;
  const int head = blockIdx.y * blockDim.y + threadIdx.y;
  const int row = blockIdx.x;
  if (head >= heads || row >= rows) return;

  __nv_bfloat16* v = x + (static_cast<size_t>(row) * heads + head) * head_dim;
  const float* cos_row = cos_tab + static_cast<size_t>(row) * kRopeDim;
  const float* sin_row = sin_tab + static_cast<size_t>(row) * kRopeDim;

  for (int j = lane; j < kRopeHalf; j += blockDim.x) {
    const float c = rope_float_value(cos_row[j]);
    const float s = rope_float_value(sin_row[j]);
    const float lo = rope_bf16_value(v[j]);
    const float hi = rope_bf16_value(v[j + kRopeHalf]);
    const float left = rope_float_value(lo * c);
    const float right = rope_float_value(hi * s);
    const float base = rope_float_value(hi * c);
    v[j] = rope_bf16_result(left - right);
    v[j + kRopeHalf] = rope_bf16_result(fmaf(lo, s, base));
  }
}

// GPT-NeoX rotary over the whole head dimension, pairing j with j + dim/2.
// cos/sin are `[rows, head_dim]`, i.e. the half-period duplicated, matching
// what the reference builds for the encoder.
__global__ void rope_neox_kernel(__nv_bfloat16* __restrict__ x, const float* __restrict__ cos_tab,
                                 const float* __restrict__ sin_tab, int rows, int heads,
                                 int head_dim) {
  const int lane = threadIdx.x;
  const int head = blockIdx.y * blockDim.y + threadIdx.y;
  const int row = blockIdx.x;
  if (head >= heads || row >= rows) return;

  const int half = head_dim / 2;
  __nv_bfloat16* v = x + (static_cast<size_t>(row) * heads + head) * head_dim;
  const float* cos_row = cos_tab + static_cast<size_t>(row) * head_dim;
  const float* sin_row = sin_tab + static_cast<size_t>(row) * head_dim;

  for (int j = lane; j < half; j += blockDim.x) {
    const float lo = rope_bf16_value(v[j]);
    const float hi = rope_bf16_value(v[j + half]);
    const float cos_lo = rope_float_value(cos_row[j]);
    const float sin_lo = rope_float_value(sin_row[j]);
    const float cos_hi = rope_float_value(cos_row[j + half]);
    const float sin_hi = rope_float_value(sin_row[j + half]);
    const float left = rope_float_value(lo * cos_lo);
    const float right = rope_float_value(hi * sin_lo);
    const float base = rope_float_value(hi * cos_hi);
    v[j] = rope_bf16_result(left - right);
    v[j + half] = rope_bf16_result(fmaf(lo, sin_hi, base));
  }
}

// --- row permutation --------------------------------------------------------
//
// A permutation is a pure move, so the packed path copies raw uint4 rather than
// converting anything: 8 bf16 or 4 fp32 per thread instead of one element.
// Measured on the bf16 gather at n*dim = 270M: 0.618 ms / 1312 GB/s scalar
// against 0.543 / 1494 packed.

// Elements of T in one 16-byte transaction.
template <typename T>
struct PackWidth {
  static constexpr int value = static_cast<int>(16 / sizeof(T));
};

template <typename T>
__global__ void gather_rows_kernel(const T* __restrict__ src, const int32_t* __restrict__ index,
                                   T* __restrict__ dst, int n, int dim) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= n || col >= dim) return;
  dst[static_cast<size_t>(row) * dim + col] =
      src[static_cast<size_t>(index[row]) * dim + col];
}

template <typename T>
__global__ void gather_rows_packed_kernel(const T* __restrict__ src,
                                          const int32_t* __restrict__ index, T* __restrict__ dst,
                                          int n, int packs) {
  const int row = blockIdx.x;
  const int p = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= n || p >= packs) return;
  const size_t dim = static_cast<size_t>(packs) * PackWidth<T>::value;
  reinterpret_cast<uint4*>(dst + static_cast<size_t>(row) * dim)[p] =
      reinterpret_cast<const uint4*>(src + static_cast<size_t>(index[row]) * dim)[p];
}

template <typename T>
__global__ void scatter_rows_kernel(const T* __restrict__ src, const int32_t* __restrict__ index,
                                    T* __restrict__ dst, int n, int dim) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= n || col >= dim) return;
  dst[static_cast<size_t>(index[row]) * dim + col] =
      src[static_cast<size_t>(row) * dim + col];
}

template <typename T>
__global__ void scatter_rows_packed_kernel(const T* __restrict__ src,
                                           const int32_t* __restrict__ index, T* __restrict__ dst,
                                           int n, int packs) {
  const int row = blockIdx.x;
  const int p = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= n || p >= packs) return;
  const size_t dim = static_cast<size_t>(packs) * PackWidth<T>::value;
  reinterpret_cast<uint4*>(dst + static_cast<size_t>(index[row]) * dim)[p] =
      reinterpret_cast<const uint4*>(src + static_cast<size_t>(row) * dim)[p];
}

// Row bases are `row * dim`, so a `dim` divisible by the pack width settles
// alignment given a 16-byte-aligned buffer — cudaMalloc gives 256 and the
// workspace gives 256.
template <typename T>
bool row_packable(const T* a, const T* b, int dim) {
  return dim % PackWidth<T>::value == 0 && reinterpret_cast<uintptr_t>(a) % 16 == 0 &&
         reinterpret_cast<uintptr_t>(b) % 16 == 0;
}

// --- elementwise ------------------------------------------------------------

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void axpby_kernel(const float* __restrict__ x, float a, const float* __restrict__ y,
                             float b, float* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmaf(a, x[i], b * y[i]);
}

inline int grid_1d(size_t n, int block) {
  return static_cast<int>((n + block - 1) / block);
}

void require_positive(int rows, int dim, const char* what) {
  if (rows <= 0 || dim <= 0) {
    throw std::runtime_error(std::string(what) + ": rows and dim must be positive");
  }
}

}  // namespace

// --- launchers --------------------------------------------------------------

void launch_rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out, int rows,
                    int dim, float eps, cudaStream_t stream) {
  require_positive(rows, dim, "launch_rmsnorm");
  const int vec = vectorisable(dim) ? 8 : 1;
  const dim3 warp_block(kWarp, kWarpRowsPerBlock);
  const int warp_grid = (rows + kWarpRowsPerBlock - 1) / kWarpRowsPerBlock;
  const int shared = reduce_shared_bytes();
  if (vec == 8 && narrow_row(dim, 8)) {
    rmsnorm_warp_kernel<8, BfPack<8>, BfPack<8>, BfPack<8>>
        <<<warp_grid, warp_block, 0, stream>>>(x, w, out, rows, dim, eps);
  } else if (vec == 1 && narrow_row(dim, 1)) {
    rmsnorm_warp_kernel<1, BfPack<1>, BfPack<1>, BfPack<1>>
        <<<warp_grid, warp_block, 0, stream>>>(x, w, out, rows, dim, eps);
  } else if (vec == 8) {
    rmsnorm_bf16_kernel<8><<<rows, kRowThreads, shared, stream>>>(x, w, out, dim, eps);
  } else {
    rmsnorm_bf16_kernel<1><<<rows, kRowThreads, shared, stream>>>(x, w, out, dim, eps);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_rmsnorm_f32(const float* x, const float* w, float* out, int rows, int dim, float eps,
                        cudaStream_t stream) {
  require_positive(rows, dim, "launch_rmsnorm_f32");
  const int vec = vectorisable(dim) ? 8 : 1;
  const dim3 warp_block(kWarp, kWarpRowsPerBlock);
  const int warp_grid = (rows + kWarpRowsPerBlock - 1) / kWarpRowsPerBlock;
  const int shared = reduce_shared_bytes();
  if (vec == 8 && narrow_row(dim, 8)) {
    rmsnorm_warp_kernel<8, F32Pack<8>, F32Pack<8>, F32Pack<8>>
        <<<warp_grid, warp_block, 0, stream>>>(x, w, out, rows, dim, eps);
  } else if (vec == 1 && narrow_row(dim, 1)) {
    rmsnorm_warp_kernel<1, F32Pack<1>, F32Pack<1>, F32Pack<1>>
        <<<warp_grid, warp_block, 0, stream>>>(x, w, out, rows, dim, eps);
  } else if (vec == 8) {
    rmsnorm_f32_kernel<8><<<rows, kRowThreads, shared, stream>>>(x, w, out, dim, eps);
  } else {
    rmsnorm_f32_kernel<1><<<rows, kRowThreads, shared, stream>>>(x, w, out, dim, eps);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_layernorm_affine(const __nv_bfloat16* x, const __nv_bfloat16* w,
                             const __nv_bfloat16* bias, __nv_bfloat16* out,
                             int rows, int dim, float eps, cudaStream_t stream) {
  require_positive(rows, dim, "launch_layernorm_affine");
  layernorm_affine_kernel<<<rows, kRowThreads, reduce_shared_bytes(), stream>>>(x, w, bias, out,
                                                                                dim, eps);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_rmsnorm_modulate(const __nv_bfloat16* x, const __nv_bfloat16* w, const float* scale,
                             const float* shift, const int32_t* a, __nv_bfloat16* out, int rows,
                             int dim, float eps, cudaStream_t stream) {
  require_positive(rows, dim, "launch_rmsnorm_modulate");
  const int shared = reduce_shared_bytes();
  if (vectorisable(dim)) {
    rmsnorm_modulate_bf16_kernel<8>
        <<<rows, kRowThreads, shared, stream>>>(x, w, scale, shift, a, out, dim, eps);
  } else {
    rmsnorm_modulate_bf16_kernel<1>
        <<<rows, kRowThreads, shared, stream>>>(x, w, scale, shift, a, out, dim, eps);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_rmsnorm_modulate_f32(const float* x, const __nv_bfloat16* w, const float* scale,
                                 const float* shift, const int32_t* a, float* out, int rows,
                                 int dim, float eps, cudaStream_t stream) {
  require_positive(rows, dim, "launch_rmsnorm_modulate_f32");
  const int shared = reduce_shared_bytes();
  if (vectorisable(dim)) {
    rmsnorm_modulate_f32_kernel<8>
        <<<rows, kRowThreads, shared, stream>>>(x, w, scale, shift, a, out, dim, eps);
  } else {
    rmsnorm_modulate_f32_kernel<1>
        <<<rows, kRowThreads, shared, stream>>>(x, w, scale, shift, a, out, dim, eps);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_add_gated(__nv_bfloat16* x, const __nv_bfloat16* branch, const float* gate,
                      const int32_t* a, int rows, int dim, cudaStream_t stream) {
  require_positive(rows, dim, "launch_add_gated");
  const int vec = vectorisable(dim) ? 8 : 1;
  const dim3 grid(rows, grid_1d(static_cast<size_t>(dim / vec), kRowThreads));
  if (vec == 8) {
    add_gated_kernel<8><<<grid, kRowThreads, 0, stream>>>(x, branch, gate, a, dim);
  } else {
    add_gated_kernel<1><<<grid, kRowThreads, 0, stream>>>(x, branch, gate, a, dim);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_swiglu(const __nv_bfloat16* fused, __nv_bfloat16* out, int rows, int inner,
                   cudaStream_t stream) {
  require_positive(rows, inner, "launch_swiglu");
  const int vec = vectorisable(inner) ? 8 : 1;
  const dim3 grid(rows, grid_1d(static_cast<size_t>(inner / vec), kRowThreads));
  if (vec == 8) {
    swiglu_kernel<8><<<grid, kRowThreads, 0, stream>>>(fused, out, inner);
  } else {
    swiglu_kernel<1><<<grid, kRowThreads, 0, stream>>>(fused, out, inner);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_silu(const float* x, float* out, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  silu_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(x, out, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_gelu_tanh(__nv_bfloat16* x, size_t n, cudaStream_t stream) {
  if (!n) return;
  gelu_tanh_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(x, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_add_bf16(__nv_bfloat16* x, const __nv_bfloat16* branch, size_t n,
                     cudaStream_t stream) {
  if (!n) return;
  add_bf16_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(x, branch, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_sub_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, size_t n,
                     cudaStream_t stream) {
  if (!n) return;
  // `a` is `__restrict__`, so writing through `out` over it is undefined
  // behaviour rather than a wrong answer — the compiler is entitled to hoist
  // the `a` load above the `out` store. Checked rather than documented, because
  // the alignment precondition below is already checked and the asymmetry
  // between a checked minor hazard and a documented major one is exactly how
  // the major one gets violated.
  if (out == a) {
    throw std::runtime_error("launch_sub_bf16: out must not alias a (out may alias b)");
  }
  // Alignment is a property of the pointers, not of `n`. Every caller today
  // passes `DeviceBuffer::get()` results, which are 256-byte aligned, but this
  // is a public declaration and the first caller to pass `x + row * dim` with
  // an unlucky `dim` would take a misaligned `uint4` load — an async fault that
  // kills the context rather than a slow path. Checked here so that never
  // becomes a crash somebody has to bisect.
  const bool aligned = ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
                         reinterpret_cast<uintptr_t>(out)) &
                        0xF) == 0;
  // The residual stream's width is 5376, a multiple of 8, so on an aligned call
  // the vector path covers every element and the tail below is dead. It is kept
  // because the alternative — asserting the divisibility — turns a future
  // geometry into a crash instead of a slower kernel. On the misaligned
  // fallback `vec_n` is 0 and the "tail" is the entire array.
  const size_t vec_n = aligned ? n / 8 * 8 : 0;
  if (vec_n) {
    sub_bf16_kernel<8><<<grid_1d(vec_n / 8, kRowThreads), kRowThreads, 0, stream>>>(a, b, out,
                                                                                    vec_n);
    VIDFAB_CUDA_CHECK(cudaGetLastError());
  }
  if (vec_n < n) {
    const size_t rest = n - vec_n;
    sub_bf16_kernel<1><<<grid_1d(rest, kRowThreads), kRowThreads, 0, stream>>>(a + vec_n, b + vec_n,
                                                                              out + vec_n, rest);
    VIDFAB_CUDA_CHECK(cudaGetLastError());
  }
}

void launch_rope_h3(__nv_bfloat16* x, const float* cos, const float* sin, int rows, int heads,
                    int head_dim, cudaStream_t stream) {
  require_positive(rows, heads, "launch_rope_h3");
  if (head_dim < kRopeDim) {
    throw std::runtime_error("launch_rope_h3: head_dim must be at least 96 rotary channels, got " +
                             std::to_string(head_dim));
  }
  // One warp per head, four heads per block.
  const dim3 block(kWarp, 4);
  const dim3 grid(rows, (heads + 3) / 4);
  rope_h3_kernel<<<grid, block, 0, stream>>>(x, cos, sin, rows, heads, head_dim);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_rope_neox(__nv_bfloat16* x, const float* cos, const float* sin, int rows, int heads,
                      int head_dim, cudaStream_t stream) {
  require_positive(rows, heads, "launch_rope_neox");
  if (head_dim <= 0 || head_dim % 2 != 0) {
    throw std::runtime_error("launch_rope_neox: head_dim must be even");
  }
  const dim3 block(kWarp, 4);
  const dim3 grid(rows, (heads + 3) / 4);
  rope_neox_kernel<<<grid, block, 0, stream>>>(x, cos, sin, rows, heads, head_dim);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_head_rmsnorm(__nv_bfloat16* x, const __nv_bfloat16* w, int rows, int heads, int dim,
                         float eps, cudaStream_t stream) {
  require_positive(rows, dim, "launch_head_rmsnorm");
  // `[rows, heads, dim]` is contiguous, so normalising over the head dimension
  // is the plain row kernel with rows*heads rows. The distinction that matters
  // is that `dim` is 128 here, not 7168 (spec 9.3) — and that 128 is what puts
  // this on `launch_rmsnorm`'s warp-per-row path, which exists for exactly this
  // caller.
  launch_rmsnorm(x, w, x, rows * heads, dim, eps, stream);
}

namespace {

template <typename T>
void launch_gather_impl(const T* src, const int32_t* index, T* dst, int n, int dim,
                        cudaStream_t stream) {
  if (row_packable(src, dst, dim)) {
    const int packs = dim / PackWidth<T>::value;
    const dim3 grid(n, grid_1d(static_cast<size_t>(packs), kRowThreads));
    gather_rows_packed_kernel<<<grid, kRowThreads, 0, stream>>>(src, index, dst, n, packs);
  } else {
    const dim3 grid(n, grid_1d(static_cast<size_t>(dim), kRowThreads));
    gather_rows_kernel<<<grid, kRowThreads, 0, stream>>>(src, index, dst, n, dim);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_scatter_impl(const T* src, const int32_t* index, T* dst, int n, int dim,
                         cudaStream_t stream) {
  if (row_packable(src, dst, dim)) {
    const int packs = dim / PackWidth<T>::value;
    const dim3 grid(n, grid_1d(static_cast<size_t>(packs), kRowThreads));
    scatter_rows_packed_kernel<<<grid, kRowThreads, 0, stream>>>(src, index, dst, n, packs);
  } else {
    const dim3 grid(n, grid_1d(static_cast<size_t>(dim), kRowThreads));
    scatter_rows_kernel<<<grid, kRowThreads, 0, stream>>>(src, index, dst, n, dim);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launch_gather_rows(const __nv_bfloat16* src, const int32_t* index, __nv_bfloat16* dst, int n,
                        int dim, cudaStream_t stream) {
  require_positive(n, dim, "launch_gather_rows");
  launch_gather_impl(src, index, dst, n, dim, stream);
}

void launch_gather_rows_f32(const float* src, const int32_t* index, float* dst, int n, int dim,
                            cudaStream_t stream) {
  require_positive(n, dim, "launch_gather_rows_f32");
  launch_gather_impl(src, index, dst, n, dim, stream);
}

void launch_scatter_rows(const __nv_bfloat16* src, const int32_t* index, __nv_bfloat16* dst, int n,
                         int dim, cudaStream_t stream) {
  require_positive(n, dim, "launch_scatter_rows");
  launch_scatter_impl(src, index, dst, n, dim, stream);
}

void launch_scatter_rows_f32(const float* src, const int32_t* index, float* dst, int n, int dim,
                             cudaStream_t stream) {
  require_positive(n, dim, "launch_scatter_rows_f32");
  launch_scatter_impl(src, index, dst, n, dim, stream);
}

void launch_scatter_add_rows(const __nv_bfloat16* src, const int32_t* index,
                             __nv_bfloat16* dst, int n, int dim, cudaStream_t stream) {
  require_positive(n, dim, "launch_scatter_add_rows");
  scatter_add_rows_kernel<<<dim3(n, grid_1d(dim, kRowThreads)), kRowThreads, 0, stream>>>(
      src, index, dst, dim);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_merge_four_rows(const __nv_bfloat16* src, __nv_bfloat16* dst,
                            int groups, int dim, cudaStream_t stream) {
  require_positive(groups, dim, "launch_merge_four_rows");
  merge_four_rows_kernel<<<dim3(groups, grid_1d(4 * dim, kRowThreads)), kRowThreads, 0, stream>>>(
      src, dst, dim);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_add(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  add_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(a, b, out, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_axpby(const float* x, float a, const float* y, float b, float* out, size_t n,
                  cudaStream_t stream) {
  if (n == 0) return;
  axpby_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(x, a, y, b, out, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vidfab::cuda
