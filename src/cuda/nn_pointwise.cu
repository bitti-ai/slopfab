#include "nn_internal.cuh"

namespace slopfab::cuda {
using namespace nn_detail;
namespace {
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

__device__ inline __nv_bfloat16 exact_vision_bf16_result(float value) {
  value = canonicalize_pointwise_float(value);
  const uint32_t magnitude = __float_as_uint(value) & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return __ushort_as_bfloat16(0x7fffu);
  const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
  const uint16_t rounded_bits = __bfloat16_as_ushort(rounded);
  return (rounded_bits & 0x7fffu) < 0x0080u
      ? __ushort_as_bfloat16(rounded_bits & 0x8000u) : rounded;
}

__global__ void gelu_tanh_exact_kernel(__nv_bfloat16* x, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint16_t input_bits = __bfloat16_as_ushort(x[i]);
  float value = __bfloat162float(x[i]);
  if ((input_bits & 0x7fffu) < 0x0080u)
    value = __uint_as_float(static_cast<uint32_t>(input_bits & 0x8000u) << 16u);
  if ((input_bits & 0x7fffu) > 0x7f80u) value = __uint_as_float(0x7fc00000u);
  x[i] = exact_vision_bf16_result(deterministic_pointwise_gelu_tanh(value));
}

template <int VEC>
__global__ void swiglu_exact_kernel(const __nv_bfloat16* __restrict__ fused,
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
    const float denominator = 1.0f + deterministic_exp(-gate[i]);
    gate[i] = deterministic_float_divide(gate[i], denominator) * value[i];
  }
  BfPack<VEC>::store(out + row * static_cast<size_t>(inner) + p * VEC, gate);
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


}  // namespace

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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_swiglu_exact(const __nv_bfloat16* fused, __nv_bfloat16* out,
                         int rows, int inner, cudaStream_t stream) {
  require_positive(rows, inner, "launch_swiglu_exact");
  const int vec = vectorisable(inner) ? 8 : 1;
  const dim3 grid(rows, grid_1d(static_cast<size_t>(inner / vec), kRowThreads));
  if (vec == 8) {
    swiglu_exact_kernel<8><<<grid, kRowThreads, 0, stream>>>(fused, out, inner);
  } else {
    swiglu_exact_kernel<1><<<grid, kRowThreads, 0, stream>>>(fused, out, inner);
  }
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_silu(const float* x, float* out, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  silu_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(x, out, n);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_gelu_tanh(__nv_bfloat16* x, size_t n, cudaStream_t stream) {
  if (!n) return;
  gelu_tanh_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(x, n);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_gelu_tanh_exact(__nv_bfloat16* x, size_t n,
                            cudaStream_t stream) {
  if (!n) return;
  gelu_tanh_exact_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(x, n);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_add_bf16(__nv_bfloat16* x, const __nv_bfloat16* branch, size_t n,
                     cudaStream_t stream) {
  if (!n) return;
  add_bf16_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(x, branch, n);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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
    SLOPFAB_CUDA_CHECK(cudaGetLastError());
  }
  if (vec_n < n) {
    const size_t rest = n - vec_n;
    sub_bf16_kernel<1><<<grid_1d(rest, kRowThreads), kRowThreads, 0, stream>>>(a + vec_n, b + vec_n,
                                                                              out + vec_n, rest);
    SLOPFAB_CUDA_CHECK(cudaGetLastError());
  }
}

void launch_add(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  add_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(a, b, out, n);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_axpby(const float* x, float a, const float* y, float b, float* out, size_t n,
                  cudaStream_t stream) {
  if (n == 0) return;
  axpby_kernel<<<grid_1d(n, kRowThreads), kRowThreads, 0, stream>>>(x, a, y, b, out, n);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}


}  // namespace slopfab::cuda
