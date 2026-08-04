// Blocked full attention with an online softmax.
//
// The shape of the problem dictates the shape of the code. At seq 37710 and 56
// heads the score matrix is 79.6 billion elements — 318 GB at fp32 — so it can
// only ever exist a tile at a time. The algorithm is FlashAttention's, with
// cuBLAS doing the two GEMMs instead of a fused kernel:
//
//   for each block of queries:
//     m = -inf, l = 0, acc = 0
//     for each block of keys:
//       S = Q K^T * scale                          (strided-batched GEMM)
//       m' = max(m, rowmax(S))
//       c  = exp(m - m')                           correction for what is
//       l  = l*c + rowsum(exp(S - m'))             already in acc
//       acc = acc*c                                (fused into the same kernel)
//       acc += exp(S - m') V                       (strided-batched GEMM, beta=1)
//     out = acc / l
//
// The correction factor is the whole trick and the whole risk: get it wrong and
// the result is still finite, still plausibly scaled, and depends on the block
// size. That last property is what the unit test pins.
//
// **Layout.** q/k/v are `[seq, heads*head_dim]` row-major. cuBLAS is
// column-major, and a row-major `[S, H*D]` buffer viewed column-major with
// ld = H*D and base offset h*D is exactly the `[D, S]` matrix of head h. No
// transpose, no repacking — the per-head "batch" is just a stride of D.
//
// The probabilities are written as bf16 for the second GEMM. That is what
// FlashAttention does too: the score tile is fp32 in the softmax and bf16 into
// the PV product, and the error it introduces is well inside the per-tensor
// tolerance this project works to.

#include "vidfab/cuda/attention.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"

namespace vidfab::cuda {
namespace {

constexpr int kWarp = 32;

// MSVC's INFINITY macro is a double expression, which nvcc warns about on every
// use in float context. Build the bit pattern instead.
__device__ inline float neg_inf() { return __int_as_float(0xFF800000); }
constexpr float kHostNegInf = -std::numeric_limits<float>::infinity();

constexpr int kSoftmaxThreads = 256;

// Score tiles dominate the footprint: `heads * query_block * key_block`
// elements at 4 bytes fp32 plus 2 bytes bf16. Capping the tile rather than the
// key block means a longer sequence costs more iterations, not more memory.
//
// Measured on an RTX 5090 at seq 37710, 56 heads, query_block 1024:
// 192 MiB -> 1010 ms, 640 MiB -> 747 ms, 1536 MiB -> 711 ms. The knee is at
// 640 MiB, and with 19 GB of weights resident there is no case for spending
// another gigabyte to buy 5 %.
constexpr size_t kScoreTileBudget = 640ull << 20;

inline size_t align_up(size_t n) { return (n + 255) / 256 * 256; }

int effective_query_block(const AttentionConfig& cfg) {
  int bq = cfg.query_block > 0 ? cfg.query_block : 1024;
  return std::min(bq, cfg.seq_len);
}

int choose_key_block(const AttentionConfig& cfg) {
  const int bq = effective_query_block(cfg);
  const size_t per_key = static_cast<size_t>(cfg.num_heads) * bq * 6;
  size_t bk = per_key > 0 ? kScoreTileBudget / per_key : static_cast<size_t>(cfg.seq_len);
  bk = bk / 256 * 256;
  if (bk < 256) bk = 256;
  if (bk > static_cast<size_t>(cfg.seq_len)) bk = static_cast<size_t>(cfg.seq_len);
  return static_cast<int>(bk);
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

// One block per (head, query row). Consumes one score tile row, updates the
// running max and sum, writes the bf16 probabilities the PV GEMM reads, and
// rescales that row of the accumulator so the GEMM can accumulate with beta=1.
//
// Rescaling here rather than in a separate kernel keeps `acc` out of HBM one
// extra time and — more importantly — makes it impossible to accumulate into
// an unrescaled accumulator by launching the two out of order.
__global__ void online_softmax_kernel(const float* __restrict__ scores,
                                      __nv_bfloat16* __restrict__ probs, float* __restrict__ acc,
                                      float* __restrict__ m_run, float* __restrict__ l_run,
                                      int key_block, int head_dim) {
  __shared__ float shared[kSoftmaxThreads / kWarp];
  const size_t idx = blockIdx.x;  // head * rows_in_block + query row
  const float* s = scores + idx * key_block;
  __nv_bfloat16* p = probs + idx * key_block;
  float* a = acc + idx * head_dim;

  float local = neg_inf();
  for (int j = threadIdx.x; j < key_block; j += blockDim.x) local = fmaxf(local, s[j]);
  const float tile_max = block_reduce_max(local, shared);
  __syncthreads();  // shared[] is reused by the sum reduction below

  const float m_old = m_run[idx];
  const float m_new = fmaxf(m_old, tile_max);
  // The first tile has m_old = -inf; exp(-inf - m_new) is 0, which is exactly
  // the right correction for an accumulator that is still zero.
  const float corr = (m_old == neg_inf()) ? 0.0f : __expf(m_old - m_new);

  float sum = 0.0f;
  for (int j = threadIdx.x; j < key_block; j += blockDim.x) {
    const float e = __expf(s[j] - m_new);
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

// acc[head][row][d] / l[head][row] -> out[q0 + row][head*head_dim + d]
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
  const float l = l_run[static_cast<size_t>(head) * rows_in_block + row];
  out[dst] = __float2bfloat16(acc[src] / l);
}

__global__ void fill_kernel(float* __restrict__ dst, float value, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = value;
}

void check_config(const AttentionConfig& cfg, int num_kv_heads) {
  if (cfg.seq_len <= 0 || cfg.num_heads <= 0 || cfg.head_dim <= 0) {
    throw std::runtime_error("attention: seq_len, num_heads and head_dim must be positive");
  }
  if (num_kv_heads <= 0 || cfg.num_heads % num_kv_heads != 0) {
    throw std::runtime_error("attention: num_heads must be a multiple of num_kv_heads");
  }
}

void run_blocked(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                 const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                 const AttentionConfig& cfg, int num_kv_heads, Workspace& ws) {
  check_config(cfg, num_kv_heads);

  const int S = cfg.seq_len;
  const int H = cfg.num_heads;
  const int D = cfg.head_dim;
  const int G = H / num_kv_heads;  // query heads per kv head
  const int qld = H * D;
  const int kvld = num_kv_heads * D;
  const int bq_max = effective_query_block(cfg);
  const int bk_max = choose_key_block(cfg);
  const float scale = cfg.effective_scale();

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

    VIDFAB_CUDA_CHECK(
        cudaMemsetAsync(acc, 0, stat_n * D * sizeof(float), stream));
    VIDFAB_CUDA_CHECK(cudaMemsetAsync(l_run, 0, stat_n * sizeof(float), stream));
    fill_kernel<<<static_cast<int>((stat_n + 255) / 256), 256, 0, stream>>>(m_run, kHostNegInf,
                                                                           stat_n);
    VIDFAB_CUDA_CHECK(cudaGetLastError());

    for (int k0 = 0; k0 < S; k0 += bk_max) {
      const int bk = std::min(bk_max, S - k0);

      // S_tile[h] (row-major [bq, bk]) = Q[h] K[h]^T * scale.
      // Column-major: C[bk, bq] = op_T(K[D, bk]) * op_N(Q[D, bq]).
      if (G == 1) {
        VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, bk, bq, D, &scale,
            k + static_cast<size_t>(k0) * kvld, CUDA_R_16BF, kvld, D,
            q + static_cast<size_t>(q0) * qld, CUDA_R_16BF, qld, D, &zero, scores, CUDA_R_32F, bk,
            static_cast<long long>(bq) * bk, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else {
        // Grouped-query: the G query heads sharing a kv head are contiguous, so
        // each kv head is one batched call with a zero stride on K.
        for (int kv = 0; kv < num_kv_heads; ++kv) {
          VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
              handle, CUBLAS_OP_T, CUBLAS_OP_N, bk, bq, D, &scale,
              k + static_cast<size_t>(k0) * kvld + static_cast<size_t>(kv) * D, CUDA_R_16BF, kvld,
              0, q + static_cast<size_t>(q0) * qld + static_cast<size_t>(kv) * G * D, CUDA_R_16BF,
              qld, D, &zero, scores + static_cast<size_t>(kv) * G * bq * bk, CUDA_R_32F, bk,
              static_cast<long long>(bq) * bk, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
      }

      online_softmax_kernel<<<static_cast<int>(stat_n), kSoftmaxThreads, 0, stream>>>(
          scores, probs, acc, m_run, l_run, bk, D);
      VIDFAB_CUDA_CHECK(cudaGetLastError());

      // acc[h] (row-major [bq, D]) += P[h] V[h].
      // Column-major: C[D, bq] = op_N(V[D, bk]) * op_N(P[bk, bq]), beta = 1.
      if (G == 1) {
        VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, D, bq, bk, &one,
            v + static_cast<size_t>(k0) * kvld, CUDA_R_16BF, kvld, D, probs, CUDA_R_16BF, bk,
            static_cast<long long>(bq) * bk, &one, acc, CUDA_R_32F, D,
            static_cast<long long>(bq) * D, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else {
        for (int kv = 0; kv < num_kv_heads; ++kv) {
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

}  // namespace

float AttentionConfig::effective_scale() const {
  return scale > 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
}

size_t attention_workspace_bytes(const AttentionConfig& cfg, AttentionBackend backend) {
  if (backend != AttentionBackend::kBlocked) {
    throw std::runtime_error("attention: kFused is not implemented yet");
  }
  if (cfg.seq_len <= 0 || cfg.num_heads <= 0 || cfg.head_dim <= 0) return 0;

  const int bq = effective_query_block(cfg);
  const int bk = choose_key_block(cfg);
  const size_t tile = static_cast<size_t>(cfg.num_heads) * bq * bk;
  const size_t stats = static_cast<size_t>(cfg.num_heads) * bq;

  size_t total = 0;
  total += align_up(tile * sizeof(float));           // scores
  total += align_up(tile * sizeof(__nv_bfloat16));   // probabilities
  total += align_up(stats * cfg.head_dim * sizeof(float));  // accumulator
  total += align_up(stats * sizeof(float));          // running max
  total += align_up(stats * sizeof(float));          // running sum
  return total;
}

void attention_forward(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                       const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                       const AttentionConfig& cfg, AttentionBackend backend, Workspace& ws) {
  if (backend != AttentionBackend::kBlocked) {
    throw std::runtime_error("attention: kFused is not implemented yet");
  }
  run_blocked(handle, stream, q, k, v, out, cfg, cfg.num_heads, ws);
}

void attention_forward_gqa(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                           const AttentionConfig& cfg, int num_kv_heads, AttentionBackend backend,
                           Workspace& ws) {
  if (backend != AttentionBackend::kBlocked) {
    throw std::runtime_error("attention: kFused is not implemented yet");
  }
  run_blocked(handle, stream, q, k, v, out, cfg, num_kv_heads, ws);
}

}  // namespace vidfab::cuda
