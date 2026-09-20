#include "nn_internal.cuh"

namespace slopfab::cuda {
using namespace nn_detail;
namespace {
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


}  // namespace

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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_layernorm_affine(const __nv_bfloat16* x, const __nv_bfloat16* w,
                             const __nv_bfloat16* bias, __nv_bfloat16* out,
                             int rows, int dim, float eps, cudaStream_t stream) {
  require_positive(rows, dim, "launch_layernorm_affine");
  layernorm_affine_kernel<<<rows, kRowThreads, reduce_shared_bytes(), stream>>>(x, w, bias, out,
                                                                                dim, eps);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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


}  // namespace slopfab::cuda
