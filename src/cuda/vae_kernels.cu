// Kernels for the H3 video VAE decoder.
//
// Everything computes in fp32. The reference loader casts the checkpoint's
// fp16 weights up to fp32 parameters, so fp32 is the arithmetic we are trying
// to match, not a conservative choice. Lower-precision paths come later and
// will be measured against this one.

#include "vidfab/cuda/vae_kernels.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace vidfab::cuda {
namespace {

constexpr int kWarp = 32;

// Sums `value` across the block using shared memory, returning the total to
// every thread. Block size must be a multiple of the warp size.
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

__device__ inline float block_reduce_max(float value, float* shared) {
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int warps = blockDim.x / kWarp;

  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
  }
  if (lane == 0) shared[warp] = value;
  __syncthreads();

  value = (threadIdx.x < warps) ? shared[threadIdx.x] : -INFINITY;
  if (warp == 0) {
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

// y = x / sqrt(mean(x^2) + eps) * weight
// PyTorch RMSNorm: no mean subtraction, eps added to the mean square.
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                               float* __restrict__ out, int dim, float eps) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const float* xr = x + static_cast<size_t>(row) * dim;
  float* outr = out + static_cast<size_t>(row) * dim;

  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = xr[i];
    sum_sq += v * v;
  }
  const float total = block_reduce_sum(sum_sq, shared);
  const float inv = rsqrtf(total / static_cast<float>(dim) + eps);

  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    outr[i] = xr[i] * inv * weight[i];
  }
}

// y = (x - mean) / sqrt(var + eps) * weight + bias, biased variance.
__global__ void layernorm_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                                 const float* __restrict__ bias, float* __restrict__ out,
                                 int dim, float eps) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const float* xr = x + static_cast<size_t>(row) * dim;
  float* outr = out + static_cast<size_t>(row) * dim;

  float sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) sum += xr[i];
  const float mean = block_reduce_sum(sum, shared) / static_cast<float>(dim);

  __syncthreads();
  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = xr[i] - mean;
    sum_sq += d * d;
  }
  const float var = block_reduce_sum(sum_sq, shared) / static_cast<float>(dim);
  const float inv = rsqrtf(var + eps);

  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    outr[i] = (xr[i] - mean) * inv * weight[i] + bias[i];
  }
}

__global__ void add_bias_kernel(float* __restrict__ y, const float* __restrict__ bias, int rows,
                                int cols) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(rows) * cols;
  if (idx >= total) return;
  y[idx] += bias[idx % cols];
}

// Splits the fused QKV projection, applies non-affine RMSNorm over head_dim,
// then rotary embedding over the leading rope_dim dimensions.
//
// Row r of to_qkv.weight is laid out r = h*(3*D) + s, with s in [0,D) -> Q,
// [D,2D) -> K, [2D,3D) -> V. Q/K/V are interleaved per head, NOT blocked.
// Reading this as [Q|K|V] blocks produces plausible-looking garbage.
//
// Output layout is head-major [H][S][D] so attention can use strided batched
// GEMM directly.
// One warp per (token, head). head_dim is 64, so each lane holds exactly two
// elements in registers: indices `lane` and `lane + 32`. The RMS reduction is
// a pure warp shuffle — no shared memory, no __syncthreads — and the rotary
// partner is fetched by shuffle rather than re-read from global.
//
// Requires head_dim == 2 * warpSize (64), which the launcher asserts.
__global__ void split_qkv_norm_rope_kernel(const float* __restrict__ qkv,
                                           const float* __restrict__ bias,
                                           const float* __restrict__ cos_tab,
                                           const float* __restrict__ sin_tab,
                                           float* __restrict__ q_out, float* __restrict__ k_out,
                                           float* __restrict__ v_out, int seq, int heads,
                                           int head_dim, int rope_dim, int num_patches,
                                           float eps) {
  const int warp_in_block = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int pair = blockIdx.x * (blockDim.x / kWarp) + warp_in_block;
  const int total_pairs = seq * heads;
  if (pair >= total_pairs) return;

  const int head = pair % heads;
  const int token = pair / heads;

  const int triple = 3 * head_dim;
  const float* row = qkv + static_cast<size_t>(token) * heads * triple + head * triple;
  // The to_qkv bias shares the projection's interleaved layout, so this head's
  // slice sits at the same offset. Folding it in here removes the last
  // standalone bias pass over the 88 MB qkv buffer.
  const float* bias_row = (bias != nullptr) ? (bias + head * triple) : nullptr;
  const size_t out_base = (static_cast<size_t>(head) * seq + token) * head_dim;

  // V needs no normalisation or rotation.
  v_out[out_base + lane] =
      row[2 * head_dim + lane] + (bias != nullptr ? bias_row[2 * head_dim + lane] : 0.0f);
  v_out[out_base + lane + kWarp] =
      row[2 * head_dim + lane + kWarp] +
      (bias != nullptr ? bias_row[2 * head_dim + lane + kWarp] : 0.0f);

  // Suffix tokens (register + zero cls) carry position id 0, so their rotation
  // is the identity. They still take part in attention.
  const bool rotate = token < num_patches;
  const int half = rope_dim / 2;

  for (int which = 0; which < 2; ++which) {
    const float* src = row + which * head_dim;
    float* dst = (which == 0 ? q_out : k_out) + out_base;

    const float v0 = src[lane] + (bias != nullptr ? bias_row[which * head_dim + lane] : 0.0f);
    const float v1 = src[lane + kWarp] +
                     (bias != nullptr ? bias_row[which * head_dim + lane + kWarp] : 0.0f);

    // The two halves are reduced separately and summed at the end. Folding
    // them into one accumulator first would reassociate the 64-element sum and
    // shift every Q/K element by about an ulp; keeping them apart reproduces
    // the block-reduction order this kernel originally used.
    float s0 = v0 * v0;
    float s1 = v1 * v1;
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
      s0 += __shfl_xor_sync(0xFFFFFFFFu, s0, offset);
      s1 += __shfl_xor_sync(0xFFFFFFFFu, s1, offset);
    }
    const float sum_sq = s0 + s1;
    const float inv = rsqrtf(sum_sq / static_cast<float>(head_dim) + eps);

    // Normalise first, then rotate — the reference order. Swapping them
    // changes the result because RMSNorm is not rotation-invariant per pair.
    const float n0 = v0 * inv;
    const float n1 = v1 * inv;

    // Each output index d in [0, rope_dim) pairs with d+half (negated) when
    // d < half, or d-half when d >= half.
    //
    // The shuffles must be executed by every lane of the warp, including lanes
    // whose own d falls outside rope_dim (for head_dim 64 / rope_dim 48 that is
    // lanes 16..31 of the upper register). So the partner is always fetched and
    // the rotation is selected afterwards, never branched around.
    auto rotated = [&](int d, float normalised) -> float {
      const bool active = rotate && d < rope_dim;
      const int partner_index = (d < half) ? (d + half) : (d - half);
      const int src_lane = partner_index & (kWarp - 1);
      const int src_reg = partner_index >> 5;
      const float p0 = __shfl_sync(0xFFFFFFFFu, n0, src_lane);
      const float p1 = __shfl_sync(0xFFFFFFFFu, n1, src_lane);
      if (!active) return normalised;
      const float sign = (d < half) ? -1.0f : 1.0f;
      const float partner = sign * ((src_reg == 0) ? p0 : p1);
      const float c = cos_tab[static_cast<size_t>(token) * rope_dim + d];
      const float s = sin_tab[static_cast<size_t>(token) * rope_dim + d];
      return normalised * c + partner * s;
    };

    const float out0 = rotated(lane, n0);
    const float out1 = rotated(lane + kWarp, n1);
    dst[lane] = out0;
    dst[lane + kWarp] = out1;
  }
}

// Scales by 1/sqrt(head_dim) and softmaxes each row in place.
__global__ void softmax_rows_kernel(float* __restrict__ scores, int cols, float scale) {
  extern __shared__ float shared[];
  const size_t row = blockIdx.x;
  float* r = scores + row * cols;

  // The scale is applied on the fly rather than written back first, which
  // removes a full read+write pass over a 394 MiB buffer.
  //
  // __fmul_rn is required, not stylistic: nvcc contracts by default, and a
  // plain `r[i] * scale - row_max` fuses into an FMA that keeps the product at
  // internal precision. Writing the scaled value to global first, as this
  // kernel originally did, forces a round to fp32. The intrinsic blocks
  // contraction so the arithmetic stays identical to that version.
  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    local_max = fmaxf(local_max, __fmul_rn(r[i], scale));
  }
  const float row_max = block_reduce_max(local_max, shared);

  __syncthreads();
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float e = __expf(__fmul_rn(r[i], scale) - row_max);
    r[i] = e;
    local_sum += e;
  }
  const float total = block_reduce_sum(local_sum, shared);
  const float inv = 1.0f / total;

  for (int i = threadIdx.x; i < cols; i += blockDim.x) r[i] *= inv;
}

// x += (y + bias) * scale, with bias and scale broadcast over columns.
//
// The bias add is folded in because this kernel already reads y: applying it
// as a separate pass costs a full read+write of the buffer for one FMA.
// `bias` may be null when the caller has already applied it.
// Row index comes from blockIdx.y, avoiding a 64-bit modulo per element.
__global__ void layerscale_residual_kernel(float* __restrict__ x, const float* __restrict__ y,
                                           const float* __restrict__ bias,
                                           const float* __restrict__ scale, int cols) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  const size_t idx = static_cast<size_t>(blockIdx.y) * cols + c;
  const float v = (bias != nullptr) ? (y[idx] + bias[c]) : y[idx];
  x[idx] += v * scale[c];
}

// SwiGLU: gate is the FIRST half of w1's output, value the second.
// out = silu(gate) * value. Reversing these is a silent correctness bug.
//
// The w1 bias is folded in for the same reason as above: this kernel already
// streams the whole 2*inner-wide row.
__global__ void swiglu_kernel(const float* __restrict__ in, const float* __restrict__ bias,
                              float* __restrict__ out, int inner) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= inner) return;
  const size_t row = blockIdx.y;
  const float* r = in + row * 2 * inner;
  float gate = r[c];
  float value = r[inner + c];
  if (bias != nullptr) {
    gate += bias[c];
    value += bias[inner + c];
  }
  out[row * inner + c] = (gate / (1.0f + __expf(-gate))) * value;
}

// Widens fp16 checkpoint bytes to fp32 on the device, so the host never has to
// run a scalar conversion loop and only half as many bytes cross PCIe.
__global__ void widen_f16_kernel(const __half* __restrict__ src, float* __restrict__ dst,
                                 size_t count) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  dst[idx] = __half2float(src[idx]);
}

// [channels, voxels] -> [voxels, channels]. The ViT consumes one channel-last
// token per latent voxel.
__global__ void transpose_cn_to_nc_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                          int channels, int voxels) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= voxels) return;
  const int c = blockIdx.y;
  dst[static_cast<size_t>(n) * channels + c] = src[static_cast<size_t>(c) * voxels + n];
}

// Depth-to-space with channel-major ordering.
//
// Flat index inside the 3072-wide projection is
//   c*1024 + pt*256 + ph*16 + pw
// with c outermost. This is NOT nn.PixelShuffle ordering and not the common
// (pt, ph, pw, c) ordering; getting it wrong yields a scrambled image that
// still looks structured.
__global__ void depth_to_space_kernel(const float* __restrict__ tokens, float* __restrict__ out,
                                      int T, int H, int W, int channels, int patch_t, int patch) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int out_T = T * patch_t;
  const int out_H = H * patch;
  const int out_W = W * patch;
  const size_t total = static_cast<size_t>(channels) * out_T * out_H * out_W;
  if (idx >= total) return;

  // Decompose the destination index.
  const int ow = static_cast<int>(idx % out_W);
  size_t rest = idx / out_W;
  const int oh = static_cast<int>(rest % out_H);
  rest /= out_H;
  const int ot = static_cast<int>(rest % out_T);
  const int c = static_cast<int>(rest / out_T);

  const int w = ow / patch;
  const int pw = ow % patch;
  const int h = oh / patch;
  const int ph = oh % patch;
  const int t = ot / patch_t;
  const int pt = ot % patch_t;

  const int patch_dim = channels * patch_t * patch * patch;
  const size_t token = (static_cast<size_t>(t) * H + h) * W + w;
  const int offset = ((c * patch_t + pt) * patch + ph) * patch + pw;
  out[idx] = tokens[token * patch_dim + offset];
}

// z = z_norm * std + mean, per latent channel.
__global__ void latent_denorm_kernel(const float* __restrict__ z_norm,
                                     const float* __restrict__ mean,
                                     const float* __restrict__ std_dev, float* __restrict__ out,
                                     int channels, int voxels) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(channels) * voxels;
  if (idx >= total) return;
  const int c = static_cast<int>(idx / voxels);
  out[idx] = z_norm[idx] * std_dev[c] + mean[c];
}

}  // namespace

void launch_rmsnorm(const float* x, const float* weight, float* out, int rows, int dim, float eps,
                    cudaStream_t stream) {
  const int threads = 256;
  const size_t shared = (threads / kWarp) * sizeof(float);
  rmsnorm_kernel<<<rows, threads, shared, stream>>>(x, weight, out, dim, eps);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_layernorm(const float* x, const float* weight, const float* bias, float* out, int rows,
                      int dim, float eps, cudaStream_t stream) {
  const int threads = 256;
  const size_t shared = (threads / kWarp) * sizeof(float);
  layernorm_kernel<<<rows, threads, shared, stream>>>(x, weight, bias, out, dim, eps);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_add_bias(float* y, const float* bias, int rows, int cols, cudaStream_t stream) {
  const size_t total = static_cast<size_t>(rows) * cols;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  add_bias_kernel<<<blocks, threads, 0, stream>>>(y, bias, rows, cols);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_split_qkv_norm_rope(const float* qkv, const float* bias, const float* cos_tab,
                                const float* sin_tab, float* q, float* k, float* v, int seq,
                                int heads, int head_dim, int rope_dim, int num_patches, float eps,
                                cudaStream_t stream) {
  const int threads = 256;
  const int warps_per_block = threads / kWarp;
  const int pairs = seq * heads;
  const int blocks = (pairs + warps_per_block - 1) / warps_per_block;
  split_qkv_norm_rope_kernel<<<blocks, threads, 0, stream>>>(
      qkv, bias, cos_tab, sin_tab, q, k, v, seq, heads, head_dim, rope_dim, num_patches, eps);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_softmax_rows(float* scores, int rows, int cols, float scale, cudaStream_t stream) {
  const int threads = 256;
  const size_t shared = (threads / kWarp) * sizeof(float);
  softmax_rows_kernel<<<rows, threads, shared, stream>>>(scores, cols, scale);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_layerscale_residual(float* x, const float* y, const float* bias, const float* scale,
                                int rows, int cols, cudaStream_t stream) {
  const int threads = 256;
  const dim3 grid((cols + threads - 1) / threads, rows);
  layerscale_residual_kernel<<<grid, threads, 0, stream>>>(x, y, bias, scale, cols);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_swiglu(const float* in, const float* bias, float* out, int rows, int inner,
                   cudaStream_t stream) {
  const int threads = 256;
  const dim3 grid((inner + threads - 1) / threads, rows);
  swiglu_kernel<<<grid, threads, 0, stream>>>(in, bias, out, inner);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_widen_f16(const void* src, float* dst, size_t count, cudaStream_t stream) {
  const int threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
  widen_f16_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __half*>(src), dst, count);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_transpose_cn_to_nc(const float* src, float* dst, int channels, int voxels,
                               cudaStream_t stream) {
  const int threads = 256;
  const dim3 grid((voxels + threads - 1) / threads, channels);
  transpose_cn_to_nc_kernel<<<grid, threads, 0, stream>>>(src, dst, channels, voxels);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_depth_to_space(const float* tokens, float* out, int T, int H, int W, int channels,
                           int patch_t, int patch, cudaStream_t stream) {
  const size_t total = static_cast<size_t>(channels) * (T * patch_t) * (H * patch) * (W * patch);
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  depth_to_space_kernel<<<blocks, threads, 0, stream>>>(tokens, out, T, H, W, channels, patch_t,
                                                        patch);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_latent_denorm(const float* z_norm, const float* mean, const float* std_dev, float* out,
                          int channels, int voxels, cudaStream_t stream) {
  const size_t total = static_cast<size_t>(channels) * voxels;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  latent_denorm_kernel<<<blocks, threads, 0, stream>>>(z_norm, mean, std_dev, out, channels,
                                                       voxels);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vidfab::cuda
