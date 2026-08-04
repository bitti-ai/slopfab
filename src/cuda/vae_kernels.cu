// Kernels for the H3 video VAE decoder.
//
// Everything computes in fp32. The reference loader casts the checkpoint's
// fp16 weights up to fp32 parameters, so fp32 is the arithmetic we are trying
// to match, not a conservative choice. Lower-precision paths come later and
// will be measured against this one.

#include "vidfab/cuda/vae_kernels.cuh"

#include <cmath>

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
__global__ void split_qkv_norm_rope_kernel(const float* __restrict__ qkv,
                                           const float* __restrict__ cos_tab,
                                           const float* __restrict__ sin_tab,
                                           float* __restrict__ q_out, float* __restrict__ k_out,
                                           float* __restrict__ v_out, int seq, int heads,
                                           int head_dim, int rope_dim, int num_patches,
                                           float eps) {
  extern __shared__ float shared[];
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  if (token >= seq || head >= heads) return;

  const int triple = 3 * head_dim;
  const float* row = qkv + static_cast<size_t>(token) * heads * triple + head * triple;
  const size_t out_base = (static_cast<size_t>(head) * seq + token) * head_dim;

  // V needs no normalisation or rotation.
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    v_out[out_base + d] = row[2 * head_dim + d];
  }

  // Suffix tokens (register + zero cls) carry position id 0, so their rotation
  // is the identity. They still take part in attention.
  const bool rotate = token < num_patches;

  for (int which = 0; which < 2; ++which) {
    const float* src = row + which * head_dim;
    float* dst = (which == 0 ? q_out : k_out) + out_base;

    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      const float v = src[d];
      sum_sq += v * v;
    }
    __syncthreads();
    const float total = block_reduce_sum(sum_sq, shared);
    const float inv = rsqrtf(total / static_cast<float>(head_dim) + eps);

    // Normalise first, then rotate — the reference order. Swapping them
    // changes the result because RMSNorm is not rotation-invariant per pair.
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      const float normalised = src[d] * inv;
      if (!rotate || d >= rope_dim) {
        dst[d] = normalised;
        continue;
      }
      // GPT-NeoX half-split: dimension d pairs with d + rope_dim/2.
      const int half = rope_dim / 2;
      const float c = cos_tab[static_cast<size_t>(token) * rope_dim + d];
      const float s = sin_tab[static_cast<size_t>(token) * rope_dim + d];
      float partner;
      if (d < half) {
        partner = -(src[d + half] * inv);
      } else {
        partner = src[d - half] * inv;
      }
      dst[d] = normalised * c + partner * s;
    }
    __syncthreads();
  }
}

// Scales by 1/sqrt(head_dim) and softmaxes each row in place.
__global__ void softmax_rows_kernel(float* __restrict__ scores, int cols, float scale) {
  extern __shared__ float shared[];
  const size_t row = blockIdx.x;
  float* r = scores + row * cols;

  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    r[i] *= scale;
    local_max = fmaxf(local_max, r[i]);
  }
  const float row_max = block_reduce_max(local_max, shared);

  __syncthreads();
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float e = __expf(r[i] - row_max);
    r[i] = e;
    local_sum += e;
  }
  const float total = block_reduce_sum(local_sum, shared);
  const float inv = 1.0f / total;

  for (int i = threadIdx.x; i < cols; i += blockDim.x) r[i] *= inv;
}

// Repacks attention output from head-major [H][S][D] to token-major [S][H*D].
__global__ void merge_heads_kernel(const float* __restrict__ in, float* __restrict__ out, int seq,
                                   int heads, int head_dim) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(seq) * heads * head_dim;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % head_dim);
  const size_t rest = idx / head_dim;
  const int h = static_cast<int>(rest % heads);
  const int s = static_cast<int>(rest / heads);
  out[idx] = in[(static_cast<size_t>(h) * seq + s) * head_dim + d];
}

// x += y * scale, with scale broadcast over the channel dimension (LayerScale).
__global__ void layerscale_residual_kernel(float* __restrict__ x, const float* __restrict__ y,
                                           const float* __restrict__ scale, int rows, int cols) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(rows) * cols;
  if (idx >= total) return;
  const int c = static_cast<int>(idx % cols);
  x[idx] += y[idx] * scale[c];
}

// SwiGLU: gate is the FIRST half of w1's output, value the second.
// out = silu(gate) * value. Reversing these is a silent correctness bug.
__global__ void swiglu_kernel(const float* __restrict__ in, float* __restrict__ out, int rows,
                              int inner) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(rows) * inner;
  if (idx >= total) return;
  const int c = static_cast<int>(idx % inner);
  const size_t row = idx / inner;
  const float* r = in + row * 2 * inner;
  const float gate = r[c];
  const float value = r[inner + c];
  out[idx] = (gate / (1.0f + __expf(-gate))) * value;
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

void launch_split_qkv_norm_rope(const float* qkv, const float* cos_tab, const float* sin_tab,
                                float* q, float* k, float* v, int seq, int heads, int head_dim,
                                int rope_dim, int num_patches, float eps, cudaStream_t stream) {
  const int threads = 64;
  const size_t shared = ((threads + kWarp - 1) / kWarp) * sizeof(float);
  const dim3 grid(seq, heads);
  split_qkv_norm_rope_kernel<<<grid, threads, shared, stream>>>(
      qkv, cos_tab, sin_tab, q, k, v, seq, heads, head_dim, rope_dim, num_patches, eps);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_softmax_rows(float* scores, int rows, int cols, float scale, cudaStream_t stream) {
  const int threads = 256;
  const size_t shared = (threads / kWarp) * sizeof(float);
  softmax_rows_kernel<<<rows, threads, shared, stream>>>(scores, cols, scale);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_merge_heads(const float* in, float* out, int seq, int heads, int head_dim,
                        cudaStream_t stream) {
  const size_t total = static_cast<size_t>(seq) * heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  merge_heads_kernel<<<blocks, threads, 0, stream>>>(in, out, seq, heads, head_dim);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_layerscale_residual(float* x, const float* y, const float* scale, int rows, int cols,
                                cudaStream_t stream) {
  const size_t total = static_cast<size_t>(rows) * cols;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  layerscale_residual_kernel<<<blocks, threads, 0, stream>>>(x, y, scale, rows, cols);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_swiglu(const float* in, float* out, int rows, int inner, cudaStream_t stream) {
  const size_t total = static_cast<size_t>(rows) * inner;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  swiglu_kernel<<<blocks, threads, 0, stream>>>(in, out, rows, inner);
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
