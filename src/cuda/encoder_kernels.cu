// The Qwen3-VL conditioner's forward pass.
//
// Three things live here that could not be reused from elsewhere:
//
//   1. **Causal GQA attention.** `vidfab/cuda/attention.cuh` is unmasked by
//      design — the H3 DiT has no mask anywhere (transformer spec 2.2) — but
//      `Qwen3VLTextAttention` sets `is_causal = True` unconditionally and the
//      model calls `create_causal_mask`. The `attention_mask=ones_like(input_ids)`
//      in the reference pipeline is a *padding* mask meaning "nothing is
//      padded"; it does not disable causality. Bidirectional attention here
//      would produce a plausible, entirely wrong conditioning
//      (docs/text_encoder_spec.md section 3).
//
//      The problem is much smaller than the DiT's: L is a prompt length, a few
//      hundred to a few thousand and capped at 8192, not 37710. So this is the
//      same blocked online-softmax algorithm as attention.cu with two additions
//      — key blocks entirely past the diagonal are skipped, and the diagonal
//      block is masked per row — rather than a fused flash kernel.
//
//   2. **SwiGLU over two separate tensors.** Qwen3-VL ships `gate_proj` and
//      `up_proj` as independent weights, so `launch_swiglu`'s fused-halves
//      layout does not apply.
//
//   3. **The decoder layer and the encoder itself**, including both residency
//      modes. `Encoder`'s methods are defined here rather than in
//      src/text/encoder.cpp because that file is compiled by the host compiler
//      and everything below needs nvcc.

#include "vidfab/text/encoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nn_kernels.cuh"
#include "vidfab/cuda/workspace.cuh"

namespace vidfab::text {
namespace {

using vidfab::cuda::ComputeType;
using vidfab::cuda::DeviceBuffer;
using vidfab::cuda::PinnedBuffer;
using vidfab::cuda::QuantFormat;
using vidfab::cuda::QuantWeight;
using vidfab::cuda::Workspace;

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kSoftmaxThreads = 256;
constexpr int kDefaultQueryBlock = 256;

// Score tiles dominate the attention footprint: `heads * bq * bk` elements at
// fp32 plus bf16. A quarter of what attention.cu allows itself, because this
// module may be sharing the card with 24.4 GB of resident weights and its
// sequences are two orders of magnitude shorter.
constexpr size_t kScoreTileBudget = 256ull << 20;

// MSVC's INFINITY macro is a double expression, which nvcc warns about on every
// use in float context. Build the bit pattern instead.
__device__ inline float neg_inf() { return __int_as_float(0xFF800000); }
constexpr float kHostNegInf = -std::numeric_limits<float>::infinity();

inline size_t align_up(size_t n) { return (n + 255) / 256 * 256; }

inline int grid_1d(size_t n, int block) { return static_cast<int>((n + block - 1) / block); }

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("text encoder: " + message);
}

// --- elementwise ------------------------------------------------------------

// SiLU evaluates in fp32 and the product rounds once. Free on a kernel that is
// entirely bandwidth bound.
__global__ void swiglu_split_kernel(const __nv_bfloat16* __restrict__ gate,
                                    const __nv_bfloat16* __restrict__ up,
                                    __nv_bfloat16* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = __bfloat162float(gate[i]);
  const float u = __bfloat162float(up[i]);
  out[i] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
}

__global__ void residual_add_kernel(__nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ branch, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(branch[i]));
}

__global__ void fill_kernel(float* __restrict__ dst, float value, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = value;
}

// --- causal attention -------------------------------------------------------

__device__ inline float block_reduce_max(float value, float* shared) {
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int warps = blockDim.x / kWarp;
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
  }
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (threadIdx.x < warps) ? shared[threadIdx.x] : neg_inf();
  if (warp == 0) {
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

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

// One block per (head, query row) of the current tile. Identical to
// attention.cu's online softmax except for `limit`: token `q0 + row` may only
// see keys `0 .. q0 + row`, so within a tile starting at `k0` the first `limit`
// columns are live and the rest are masked to zero probability.
//
// Masked columns still have their probability written, because the PV GEMM
// consumes the whole `[bq, bk]` tile and would otherwise multiply stale memory.
__global__ void causal_softmax_kernel(const float* __restrict__ scores,
                                      __nv_bfloat16* __restrict__ probs, float* __restrict__ acc,
                                      float* __restrict__ m_run, float* __restrict__ l_run,
                                      int rows_in_block, int key_block, int head_dim, int q0,
                                      int k0) {
  __shared__ float shared[kSoftmaxThreads / kWarp];
  const int idx = blockIdx.x;  // head * rows_in_block + row
  const int row = idx % rows_in_block;
  const size_t base = static_cast<size_t>(idx) * key_block;
  const float* s = scores + base;
  __nv_bfloat16* p = probs + base;
  float* a = acc + static_cast<size_t>(idx) * head_dim;

  int limit = q0 + row - k0 + 1;
  if (limit < 0) limit = 0;
  if (limit > key_block) limit = key_block;

  float local = neg_inf();
  for (int j = threadIdx.x; j < limit; j += blockDim.x) local = fmaxf(local, s[j]);
  const float tile_max = block_reduce_max(local, shared);
  __syncthreads();  // shared[] is reused by the sum reduction below

  const float m_old = m_run[idx];
  const float m_new = fmaxf(m_old, tile_max);
  // Three cases, and the guards matter because `limit` can be zero for a row
  // whose whole tile is above the diagonal:
  //   m_old = -inf  -> nothing accumulated yet, correction is 0
  //   tile empty    -> m unchanged, correction is 1, tile contributes nothing
  //   otherwise     -> the usual exp(m_old - m_new)
  const bool empty = (m_new == neg_inf());
  const float corr = (m_old == neg_inf()) ? 0.0f : (empty ? 1.0f : __expf(m_old - m_new));

  float sum = 0.0f;
  for (int j = threadIdx.x; j < key_block; j += blockDim.x) {
    const float e = (j < limit && !empty) ? __expf(s[j] - m_new) : 0.0f;
    p[j] = __float2bfloat16(e);
    sum += e;
  }
  const float tile_sum = block_reduce_sum(sum, shared);

  if (threadIdx.x == 0) {
    m_run[idx] = m_new;
    l_run[idx] = l_run[idx] * corr + tile_sum;
  }
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) a[d] *= corr;
}

__global__ void finalise_kernel(const float* __restrict__ acc, const float* __restrict__ l_run,
                                __nv_bfloat16* __restrict__ out, int rows_in_block, int heads,
                                int head_dim, int q0) {
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= head_dim) return;
  const int row = blockIdx.x % rows_in_block;
  const int head = blockIdx.x / rows_in_block;
  const size_t src = (static_cast<size_t>(head) * rows_in_block + row) * head_dim + d;
  const size_t dst = static_cast<size_t>(q0 + row) * heads * head_dim +
                     static_cast<size_t>(head) * head_dim + d;
  // Row 0 sees exactly one key, so l is never zero: every query row has at
  // least key 0 inside the first tile.
  const float l = l_run[static_cast<size_t>(head) * rows_in_block + row];
  out[dst] = __float2bfloat16(acc[src] / l);
}

void check_causal_config(const CausalAttentionConfig& cfg) {
  require(cfg.seq_len > 0 && cfg.num_heads > 0 && cfg.head_dim > 0,
          "causal attention: seq_len, num_heads and head_dim must be positive");
  require(cfg.num_kv_heads > 0 && cfg.num_heads % cfg.num_kv_heads == 0,
          "causal attention: num_heads must be a multiple of num_kv_heads");
}

int effective_query_block(const CausalAttentionConfig& cfg) {
  const int bq = cfg.query_block > 0 ? cfg.query_block : kDefaultQueryBlock;
  return std::min(bq, cfg.seq_len);
}

int choose_key_block(const CausalAttentionConfig& cfg) {
  const int bq = effective_query_block(cfg);
  const size_t per_key = static_cast<size_t>(cfg.num_heads) * bq * 6;
  size_t bk = per_key > 0 ? kScoreTileBudget / per_key : static_cast<size_t>(cfg.seq_len);
  bk = bk / 256 * 256;
  if (bk < 256) bk = 256;
  if (bk > static_cast<size_t>(cfg.seq_len)) bk = static_cast<size_t>(cfg.seq_len);
  return static_cast<int>(bk);
}

}  // namespace

// --- causal attention, public ------------------------------------------------

float causal_attention_scale(const CausalAttentionConfig& cfg) {
  // 1/sqrt(128); nothing in Qwen3-VL overrides `self.scaling`.
  return cfg.scale > 0.0f ? cfg.scale : 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
}

size_t causal_attention_workspace_bytes(const CausalAttentionConfig& cfg) {
  if (cfg.seq_len <= 0 || cfg.num_heads <= 0 || cfg.head_dim <= 0) return 0;
  const int bq = effective_query_block(cfg);
  const int bk = choose_key_block(cfg);
  const size_t tile = static_cast<size_t>(cfg.num_heads) * bq * bk;
  const size_t stats = static_cast<size_t>(cfg.num_heads) * bq;

  size_t total = 0;
  total += align_up(tile * sizeof(float));                   // scores
  total += align_up(tile * sizeof(__nv_bfloat16));           // probabilities
  total += align_up(stats * cfg.head_dim * sizeof(float));   // accumulator
  total += align_up(stats * sizeof(float));                  // running max
  total += align_up(stats * sizeof(float));                  // running sum
  return total;
}

void causal_attention_forward(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                              const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                              const CausalAttentionConfig& cfg, Workspace& ws) {
  check_causal_config(cfg);

  const int S = cfg.seq_len;
  const int H = cfg.num_heads;
  const int D = cfg.head_dim;
  const int G = H / cfg.num_kv_heads;  // query heads per kv head
  const int qld = H * D;
  const int kvld = cfg.num_kv_heads * D;
  const int bq_max = effective_query_block(cfg);
  const int bk_max = choose_key_block(cfg);
  const float scale = causal_attention_scale(cfg);

  Workspace::Scope scope(ws);
  const size_t tile = static_cast<size_t>(H) * bq_max * bk_max;
  float* scores = ws.alloc_n<float>(tile);
  __nv_bfloat16* probs = ws.alloc_n<__nv_bfloat16>(tile);
  float* acc = ws.alloc_n<float>(static_cast<size_t>(H) * bq_max * D);
  float* m_run = ws.alloc_n<float>(static_cast<size_t>(H) * bq_max);
  float* l_run = ws.alloc_n<float>(static_cast<size_t>(H) * bq_max);

  VIDFAB_CUBLAS_CHECK(cublasSetStream(handle, stream));
  const float one = 1.0f;
  const float zero = 0.0f;

  for (int q0 = 0; q0 < S; q0 += bq_max) {
    const int bq = std::min(bq_max, S - q0);
    const size_t stat_n = static_cast<size_t>(H) * bq;

    VIDFAB_CUDA_CHECK(cudaMemsetAsync(acc, 0, stat_n * D * sizeof(float), stream));
    VIDFAB_CUDA_CHECK(cudaMemsetAsync(l_run, 0, stat_n * sizeof(float), stream));
    fill_kernel<<<grid_1d(stat_n, 256), 256, 0, stream>>>(m_run, kHostNegInf, stat_n);
    VIDFAB_CUDA_CHECK(cudaGetLastError());

    // The whole point: the last query in this block is `q0 + bq - 1`, so no key
    // beyond it can ever be attended to and those tiles are never computed.
    // Roughly half the score matrix disappears.
    const int k_last = q0 + bq;

    for (int k0 = 0; k0 < k_last; k0 += bk_max) {
      const int bk = std::min(bk_max, k_last - k0);

      // S_tile[h] (row-major [bq, bk]) = Q[h] K[h]^T * scale.
      // Column-major: C[bk, bq] = op_T(K[D, bk]) * op_N(Q[D, bq]).
      if (G == 1) {
        VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, bk, bq, D, &scale,
            k + static_cast<size_t>(k0) * kvld, CUDA_R_16BF, kvld, D,
            q + static_cast<size_t>(q0) * qld, CUDA_R_16BF, qld, D, &zero, scores, CUDA_R_32F, bk,
            static_cast<long long>(bq) * bk, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else {
        // Query head h reads kv head h/G — contiguous blocks, not interleaved
        // (spec section 4.2), which is exactly what makes each kv head one
        // batched call with a zero stride on K.
        for (int kv = 0; kv < cfg.num_kv_heads; ++kv) {
          VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
              handle, CUBLAS_OP_T, CUBLAS_OP_N, bk, bq, D, &scale,
              k + static_cast<size_t>(k0) * kvld + static_cast<size_t>(kv) * D, CUDA_R_16BF, kvld,
              0, q + static_cast<size_t>(q0) * qld + static_cast<size_t>(kv) * G * D, CUDA_R_16BF,
              qld, D, &zero, scores + static_cast<size_t>(kv) * G * bq * bk, CUDA_R_32F, bk,
              static_cast<long long>(bq) * bk, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
      }

      causal_softmax_kernel<<<static_cast<int>(stat_n), kSoftmaxThreads, 0, stream>>>(
          scores, probs, acc, m_run, l_run, bq, bk, D, q0, k0);
      VIDFAB_CUDA_CHECK(cudaGetLastError());

      // acc[h] (row-major [bq, D]) += P[h] V[h], beta = 1.
      if (G == 1) {
        VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, D, bq, bk, &one,
            v + static_cast<size_t>(k0) * kvld, CUDA_R_16BF, kvld, D, probs, CUDA_R_16BF, bk,
            static_cast<long long>(bq) * bk, &one, acc, CUDA_R_32F, D,
            static_cast<long long>(bq) * D, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else {
        for (int kv = 0; kv < cfg.num_kv_heads; ++kv) {
          VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
              handle, CUBLAS_OP_N, CUBLAS_OP_N, D, bq, bk, &one,
              v + static_cast<size_t>(k0) * kvld + static_cast<size_t>(kv) * D, CUDA_R_16BF, kvld,
              0, probs + static_cast<size_t>(kv) * G * bq * bk, CUDA_R_16BF, bk,
              static_cast<long long>(bq) * bk, &one, acc + static_cast<size_t>(kv) * G * bq * D,
              CUDA_R_32F, D, static_cast<long long>(bq) * D, G, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT));
        }
      }
    }

    const dim3 grid(static_cast<unsigned>(stat_n), static_cast<unsigned>((D + 127) / 128));
    finalise_kernel<<<grid, 128, 0, stream>>>(acc, l_run, out, bq, H, D, q0);
    VIDFAB_CUDA_CHECK(cudaGetLastError());
  }
}

// --- elementwise, public ------------------------------------------------------

void launch_swiglu_split(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                         size_t n, cudaStream_t stream) {
  if (n == 0) return;
  swiglu_split_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(gate, up, out, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_residual_add(__nv_bfloat16* x, const __nv_bfloat16* branch, size_t n,
                         cudaStream_t stream) {
  if (n == 0) return;
  residual_add_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(x, branch, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vidfab::text
