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
// **Why the score tile is fp16 and not fp32 or bf16.** Both the scores and the
// probabilities live in one fp16 buffer, written by the QK GEMM epilogue and
// read back by the softmax, which halves the traffic through the single largest
// buffer in the pipeline and lets the key block grow threefold inside the same
// budget. Measured at seq 37710, 56 heads: 744 ms with fp32 scores and bf16
// probabilities, 561 ms this way. (Measure it twice — another process on this
// box intermittently takes half the card, and a contended run reads ~1180 ms.)
//
// It is not a precision regression. fp16 carries 11 mantissa bits to bf16's 8
// and the accumulation is fp32 either way, so the probabilities are strictly
// better resolved than before; `attention_fp16_score_tile` measures the whole
// kernel at rms_rel 1.01x the bf16 *output* quantisation, i.e. the score path is
// no longer distinguishable from exact arithmetic at the precision `out` is
// stored in.
//
// The trade is range, not precision. fp16 saturates at 65504 and an overflow
// there becomes a NaN in the softmax, not a large number. H3 scores land at
// ±10..30 because q_norm/k_norm pin the per-head RMS and RoPE preserves norm;
// `attention_forward` documents the limit for callers who do neither.
//
// bf16 in, fp16 out is not a combination cuBLAS accepts, so q, k and v are
// converted to fp16 first. k and v are converted once per call; q only a query
// block at a time, which is the same arithmetic for 1/37th of the buffer.

#include "vidfab/cuda/attention.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

// Score tiles dominate the footprint: `heads * query_block * key_block`
// elements. Scores and probabilities share one fp16 buffer, so that is 2 bytes
// per element, not the 6 the fp32-scores-plus-bf16-probabilities layout cost.
// Capping the tile rather than the key block means a longer sequence costs more
// iterations, not more memory.
//
// Measured on an RTX 5090 at seq 37710, 56 heads, query_block 1024:
// 192 MiB -> 1010 ms, 640 MiB -> 747 ms, 1536 MiB -> 711 ms, all at the old
// 6 bytes per element. The knee is at 640 MiB, and with 19 GB of weights
// resident there is no case for spending another gigabyte to buy 5 %.
constexpr size_t kScoreTileBudget = 640ull << 20;
constexpr size_t kScoreTileBytesPerKey = 2;

inline size_t align_up(size_t n) { return (n + 255) / 256 * 256; }

int effective_query_block(const AttentionConfig& cfg) {
  int bq = cfg.query_block > 0 ? cfg.query_block : 1024;
  return std::min(bq, cfg.seq_len);
}

// Largest key block the tile budget allows, then *balanced* rather than
// truncated. Taking the budget maximum leaves a ragged tail — at bk 1792 the
// last of 22 blocks was 78 rows, so both GEMMs and the softmax launch ran at a
// few percent utilisation once per query block, 37 times per layer. Spreading
// the same number of blocks evenly costs nothing and removes it.
//
// The grain is 64, not 256: 37710 over 22 blocks wants 1728, which is a
// multiple of 64 but not of 256. 64 elements is 128 bytes of fp16, which keeps
// the tile's leading dimension comfortably aligned for cuBLAS.
int choose_key_block(const AttentionConfig& cfg) {
  if (cfg.key_block > 0) return std::min(cfg.key_block, cfg.seq_len);
  const int bq = effective_query_block(cfg);
  const size_t per_key = static_cast<size_t>(cfg.num_heads) * bq * kScoreTileBytesPerKey;
  size_t bk_max = per_key > 0 ? kScoreTileBudget / per_key : static_cast<size_t>(cfg.seq_len);
  bk_max = bk_max / 256 * 256;
  if (bk_max < 256) bk_max = 256;
  if (bk_max > static_cast<size_t>(cfg.seq_len)) bk_max = static_cast<size_t>(cfg.seq_len);

  const size_t S = static_cast<size_t>(cfg.seq_len);
  const size_t nblocks = (S + bk_max - 1) / bk_max;
  size_t bk = (S + nblocks - 1) / nblocks;
  bk = (bk + 63) / 64 * 64;
  // bk_max is a multiple of 64, so rounding a value already <= bk_max up to the
  // next multiple of 64 cannot overshoot it.
  if (bk > bk_max) bk = bk_max;
  if (bk < 1) bk = 1;
  return static_cast<int>(bk);
}

// --- softmax launch shape ---------------------------------------------------
//
// The softmax holds its row of the score tile in registers so `s[j]` is loaded
// once instead of twice, which needs the chunk count as a compile-time
// constant. Chunks are rounded up to the next instantiated size; the surplus
// iterations fall out on the `j >= key_block` guard.
//
// **The block stays at 256 threads.** The obvious way to keep the chunk count
// small is to widen the block, and it is a trap: measured at seq 37710,
// key_block 5440, 256 threads with 22 chunks is 562 ms, 512 threads with 11 is
// 582 ms and 1024 with 6 is 593 ms. A 1024-thread block is one block per SM.
// 22 registers of row buys more than it costs.
constexpr int kSoftmaxThreads = 256;
constexpr int kSoftmaxChunkSizes[] = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32};

// 0 selects a strided two-pass fallback, for key blocks past the largest
// instantiation. Nothing in this project reaches it — the tile budget caps
// key_block near 5.5k against this ceiling of 8192 — but a raised budget or a
// smaller head count could.
int choose_softmax_chunks(int key_block) {
  const int want = (key_block + kSoftmaxThreads - 1) / kSoftmaxThreads;
  for (int c : kSoftmaxChunkSizes) {
    if (want <= c) return c;
  }
  return 0;
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
// running max and sum, overwrites the scores in place with the fp16
// probabilities the PV GEMM reads, and rescales that row of the accumulator so
// the GEMM can accumulate with beta = 1.
//
// Rescaling here rather than in a separate kernel keeps `acc` out of HBM one
// extra time and — more importantly — makes it impossible to accumulate into
// an unrescaled accumulator by launching the two out of order.
//
// **`tile` is read and written in place.** Every block owns a disjoint row, and
// within a thread the read of element j always precedes the write of element j:
// with CHUNKS > 0 the whole row is in registers before anything is stored, and
// the fallback reads and writes the same j in the same iteration. `__restrict__`
// is still honest — nothing *else* aliases the tile.
template <int CHUNKS>
__global__ void online_softmax_kernel(__half* __restrict__ tile, float* __restrict__ acc,
                                      float* __restrict__ m_run, float* __restrict__ l_run,
                                      int key_block, int head_dim) {
  extern __shared__ float shared[];  // blockDim.x / 32 floats
  const size_t idx = blockIdx.x;     // head * rows_in_block + query row
  __half* s = tile + idx * key_block;
  float* a = acc + idx * head_dim;

  float row[CHUNKS > 0 ? CHUNKS : 1];
  float local = neg_inf();
  if (CHUNKS > 0) {
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int j = static_cast<int>(threadIdx.x) + c * static_cast<int>(blockDim.x);
      row[c] = (j < key_block) ? __half2float(s[j]) : neg_inf();
      local = fmaxf(local, row[c]);
    }
  } else {
    for (int j = threadIdx.x; j < key_block; j += blockDim.x) {
      local = fmaxf(local, __half2float(s[j]));
    }
  }
  const float tile_max = block_reduce_max(local, shared);
  __syncthreads();  // shared[] is reused by the sum reduction below

  const float m_old = m_run[idx];
  const float m_new = fmaxf(m_old, tile_max);
  // The first tile has m_old = -inf; exp(-inf - m_new) is 0, which is exactly
  // the right correction for an accumulator that is still zero.
  const float corr = (m_old == neg_inf()) ? 0.0f : __expf(m_old - m_new);

  float sum = 0.0f;
  if (CHUNKS > 0) {
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int j = static_cast<int>(threadIdx.x) + c * static_cast<int>(blockDim.x);
      if (j >= key_block) continue;
      const float e = __expf(row[c] - m_new);
      s[j] = __float2half(e);
      sum += e;
    }
  } else {
    for (int j = threadIdx.x; j < key_block; j += blockDim.x) {
      const float e = __expf(__half2float(s[j]) - m_new);
      s[j] = __float2half(e);
      sum += e;
    }
  }
  const float tile_sum = block_reduce_sum(sum, shared);

  if (threadIdx.x == 0) {
    m_run[idx] = m_new;
    l_run[idx] = l_run[idx] * corr + tile_sum;
  }

  // `corr` is block-uniform, so both of these are uniform branches.
  //
  // corr == 1 is the common case by a wide margin: the running max stops moving
  // after the first few tiles, and over 22 tiles the expected number of record
  // updates is H(22) ~ 3.6. Skipping the no-op multiply saves ~84 % of a 28 MB
  // read-modify-write per tile.
  //
  // corr == 0 marks the *first* tile, where `acc` has not been written yet —
  // the PV GEMM runs with beta = 0 instead of being memset. Rescaling it here
  // would multiply uninitialised workspace by zero, and a stray NaN bit pattern
  // would survive that. Both exclusions are load-bearing.
  if (corr != 1.0f && corr != 0.0f) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) a[d] *= corr;
  }
}

// One instantiation per entry in kSoftmaxChunkSizes. A tail key block is
// shorter than the one `chunks` was chosen for, which the `j >= key_block`
// guard inside the kernel already covers.
void launch_online_softmax(int chunks, int blocks, __half* tile, float* acc, float* m_run,
                           float* l_run, int key_block, int head_dim, cudaStream_t stream) {
  constexpr int shared = kSoftmaxThreads / kWarp * static_cast<int>(sizeof(float));
#define VIDFAB_SOFTMAX_CASE(n)                                                                  \
  case n:                                                                                       \
    online_softmax_kernel<n><<<blocks, kSoftmaxThreads, shared, stream>>>(tile, acc, m_run,      \
                                                                         l_run, key_block,      \
                                                                         head_dim);             \
    break;
  switch (chunks) {
    VIDFAB_SOFTMAX_CASE(1)
    VIDFAB_SOFTMAX_CASE(2)
    VIDFAB_SOFTMAX_CASE(4)
    VIDFAB_SOFTMAX_CASE(8)
    VIDFAB_SOFTMAX_CASE(12)
    VIDFAB_SOFTMAX_CASE(16)
    VIDFAB_SOFTMAX_CASE(20)
    VIDFAB_SOFTMAX_CASE(24)
    VIDFAB_SOFTMAX_CASE(28)
    VIDFAB_SOFTMAX_CASE(32)
    default:
      online_softmax_kernel<0><<<blocks, kSoftmaxThreads, shared, stream>>>(
          tile, acc, m_run, l_run, key_block, head_dim);
      break;
  }
#undef VIDFAB_SOFTMAX_CASE
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

// --- bf16 -> fp16 -----------------------------------------------------------
//
// cuBLAS will not take bf16 inputs with an fp16 output, so q, k and v are
// widened-then-narrowed once. Both formats round-trip exactly through fp32
// here: every bf16 value is representable in fp16 unless its magnitude is
// outside [6e-8, 65504], which post-norm activations never are.

__global__ void bf16_to_f16_packed_kernel(const __nv_bfloat16* __restrict__ src,
                                          __half* __restrict__ dst, size_t packs) {
  const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= packs) return;
  const size_t base = p * 8;
  const uint4 raw = *reinterpret_cast<const uint4*>(src + base);
  const __nv_bfloat162* in = reinterpret_cast<const __nv_bfloat162*>(&raw);
  uint4 out;
  __half2* h = reinterpret_cast<__half2*>(&out);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __float22half2_rn(__bfloat1622float2(in[i]));
  *reinterpret_cast<uint4*>(dst + base) = out;
}

__global__ void bf16_to_f16_scalar_kernel(const __nv_bfloat16* __restrict__ src,
                                          __half* __restrict__ dst, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(__bfloat162float(src[i]));
}

void convert_bf16_to_f16(const __nv_bfloat16* src, __half* dst, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  // The packed path needs 16-byte alignment at both ends. Query-block bases are
  // `q0 * heads * head_dim`, which is 16-byte aligned for every production
  // width but not for an arbitrary one, so check rather than assume.
  const bool aligned = n % 8 == 0 && reinterpret_cast<uintptr_t>(src) % 16 == 0 &&
                       reinterpret_cast<uintptr_t>(dst) % 16 == 0;
  if (aligned) {
    const size_t packs = n / 8;
    bf16_to_f16_packed_kernel<<<static_cast<int>((packs + 255) / 256), 256, 0, stream>>>(src, dst,
                                                                                         packs);
  } else {
    bf16_to_f16_scalar_kernel<<<static_cast<int>((n + 255) / 256), 256, 0, stream>>>(src, dst, n);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
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
  // The blocked path has no banding. Ignoring the request would hand back full
  // attention while the caller believed it was banded -- correct output, a
  // plausible time, and a silently different model. Refuse instead.
  if (cfg.band_ranges != nullptr) {
    throw std::runtime_error("attention: band_ranges requires the fused backend");
  }

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
  // Scores in, probabilities out, same buffer. See online_softmax_kernel for
  // why that is safe.
  __half* tile_buf = ws.alloc_n<__half>(tile);
  float* acc = ws.alloc_n<float>(static_cast<size_t>(H) * bq_max * D);
  float* m_run = ws.alloc_n<float>(static_cast<size_t>(H) * bq_max);
  float* l_run = ws.alloc_n<float>(static_cast<size_t>(H) * bq_max);
  __half* k16 = ws.alloc_n<__half>(static_cast<size_t>(S) * kvld);
  __half* v16 = ws.alloc_n<__half>(static_cast<size_t>(S) * kvld);
  __half* q16 = ws.alloc_n<__half>(static_cast<size_t>(bq_max) * qld);

  // The handle is retargeted on every call. Harmless while everything runs on
  // one stream, which is the only configuration this project has; if attention
  // and a linear ever share a handle across two streams, this line silently
  // moves the linear's work as well. Give each stream its own handle then.
  VIDFAB_CUBLAS_CHECK(cublasSetStream(handle, stream));

  convert_bf16_to_f16(k, k16, static_cast<size_t>(S) * kvld, stream);
  convert_bf16_to_f16(v, v16, static_cast<size_t>(S) * kvld, stream);

  const float one = 1.0f;
  const float zero = 0.0f;

  const int sm_chunks = choose_softmax_chunks(bk_max);

  for (int q0 = 0; q0 < S; q0 += bq_max) {
    const int bq = std::min(bq_max, S - q0);
    const size_t stat_n = static_cast<size_t>(H) * bq;

    // `acc` is deliberately *not* zeroed: the k0 == 0 PV GEMM runs with
    // beta = 0, which writes every element it would have read. That is 28 MB of
    // memset saved per query block.
    VIDFAB_CUDA_CHECK(cudaMemsetAsync(l_run, 0, stat_n * sizeof(float), stream));
    fill_kernel<<<static_cast<int>((stat_n + 255) / 256), 256, 0, stream>>>(m_run, kHostNegInf,
                                                                           stat_n);
    VIDFAB_CUDA_CHECK(cudaGetLastError());

    convert_bf16_to_f16(q + static_cast<size_t>(q0) * qld, q16, static_cast<size_t>(bq) * qld,
                        stream);

    for (int k0 = 0; k0 < S; k0 += bk_max) {
      const int bk = std::min(bk_max, S - k0);
      // beta for the PV GEMM: the first key block initialises the accumulator
      // rather than adding to it, which is what makes the memset unnecessary.
      const float* pv_beta = (k0 == 0) ? &zero : &one;

      // S_tile[h] (row-major [bq, bk]) = Q[h] K[h]^T * scale.
      // Column-major: C[bk, bq] = op_T(K[D, bk]) * op_N(Q[D, bq]).
      if (G == 1) {
        VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, bk, bq, D, &scale,
            k16 + static_cast<size_t>(k0) * kvld, CUDA_R_16F, kvld, D, q16, CUDA_R_16F, qld, D,
            &zero, tile_buf, CUDA_R_16F, bk, static_cast<long long>(bq) * bk, H,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else {
        // Grouped-query: the G query heads sharing a kv head are contiguous, so
        // each kv head is one batched call with a zero stride on K.
        for (int kv = 0; kv < num_kv_heads; ++kv) {
          VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
              handle, CUBLAS_OP_T, CUBLAS_OP_N, bk, bq, D, &scale,
              k16 + static_cast<size_t>(k0) * kvld + static_cast<size_t>(kv) * D, CUDA_R_16F, kvld,
              0, q16 + static_cast<size_t>(kv) * G * D, CUDA_R_16F, qld, D, &zero,
              tile_buf + static_cast<size_t>(kv) * G * bq * bk, CUDA_R_16F, bk,
              static_cast<long long>(bq) * bk, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
      }

      launch_online_softmax(sm_chunks, static_cast<int>(stat_n), tile_buf, acc, m_run, l_run, bk, D,
                            stream);

      // acc[h] (row-major [bq, D]) += P[h] V[h].
      // Column-major: C[D, bq] = op_N(V[D, bk]) * op_N(P[bk, bq]).
      if (G == 1) {
        VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, D, bq, bk, &one,
            v16 + static_cast<size_t>(k0) * kvld, CUDA_R_16F, kvld, D, tile_buf, CUDA_R_16F, bk,
            static_cast<long long>(bq) * bk, pv_beta, acc, CUDA_R_32F, D,
            static_cast<long long>(bq) * D, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else {
        for (int kv = 0; kv < num_kv_heads; ++kv) {
          VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
              handle, CUBLAS_OP_N, CUBLAS_OP_N, D, bq, bk, &one,
              v16 + static_cast<size_t>(k0) * kvld + static_cast<size_t>(kv) * D, CUDA_R_16F, kvld,
              0, tile_buf + static_cast<size_t>(kv) * G * bq * bk, CUDA_R_16F, bk,
              static_cast<long long>(bq) * bk, pv_beta, acc + static_cast<size_t>(kv) * G * bq * D,
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

// --- fused backend ----------------------------------------------------------
//
// The blocked path above is FlashAttention's algorithm with cuBLAS doing the two
// GEMMs, which means the score tile is written to HBM by the QK GEMM, read back
// by the softmax, and read a third time by the PV GEMM. At seq 37710 that round
// trip is ~12 bytes per score element and it is 86% of a denoising step. This
// kernel exists to delete it: S and P never leave shared memory.
//
// **Shape of the decomposition.** One block owns one query tile of `kBr` rows
// for one head; `kWarps` warps split it so that **each warp owns 16 whole rows
// and every column of them**. That is the load-bearing choice: a row's softmax
// never spans two warps, so the running max, the correction factor and the sum
// are warp-local and need no `__syncthreads` and no shared reduction. Only the
// K/V staging and the O rescale are block-wide.
//
// **Why bf16 fragments and not the fp16 the blocked path uses.** The fp16
// conversion of k and v exists only because cuBLAS will not take bf16 in and
// fp16 out; `mma.sync` takes bf16 directly with an fp32 accumulator, so the two
// full-sequence conversion buffers and their HBM traffic disappear. The
// probabilities stay fp16 -- P feeds a second mma whose operands must share a
// type, so V is converted to fp16 during SMEM staging, which costs no HBM
// traffic and keeps the 11 mantissa bits the blocked path deliberately chose
// over bf16's 8.
//
// **Accumulator in registers, via `mma.sync` rather than `wmma`.** O is
// rescaled by a *per-row* factor every key block, and `nvcuda::wmma` does not
// say which row an accumulator element belongs to -- the mapping is
// unspecified, so a register-resident O could not be scaled correctly through
// that API. Raw `mma.sync.aligned.m16n8k16` does specify it (PTX ISA 9.7.14),
// and the specification is what makes this kernel possible:
//
//     groupID = lane >> 2      tig = lane & 3
//     c0,c1 -> row groupID      cols tig*2, tig*2+1
//     c2,c3 -> row groupID + 8  cols tig*2, tig*2+1
//
// Three consequences, and each one deletes a shared-memory buffer:
//
//   * A thread's four accumulator registers span exactly **two** rows, so the
//     per-row rescale is four multiplies on registers -- not an 8192-element
//     read-modify-write over a shared tile.
//   * The four threads of a group hold all eight columns of one row, and they
//     are consecutive lanes, so a row reduction is `shfl_xor` by 1 then 2.
//     Rows `groupID` and `groupID+8` reduce simultaneously in separate
//     registers, and all 32 lanes work rather than 16.
//   * The QK accumulator tiles at n-offset 0 and 8 supply **exactly** the four
//     A-fragment registers the PV `mma` wants. S becomes P by a register
//     permute and an `f32 -> f16x2` convert, with no transpose and no store.
//
// So `st`, `ps`, `os` and `cs` are all gone, along with the fp32 shared tiles
// whose 2-way bank conflict was unfixable: `wmma` requires an fp32 `ldm` that
// is a multiple of 4 floats, and for any stride divisible by 4, rows 8 apart
// land in the same bank. Only K and V staging remain in shared memory.
//
// **V is staged transposed.** The PV `mma` wants B in column-major, and V
// arrives row-major. Writing V^T during staging turns each B fragment load
// into one aligned 32-bit read instead of two scattered 16-bit reads. Its row
// stride is `kBc + 2` halves = 17 four-byte words, and 17 is coprime with 32,
// so the transposing writes are conflict-free.
namespace fused {

// Eight warps, not four. Four warps is exactly one warp per sub-partition, so
// every shared-load and MUFU latency in the inner loop is fully exposed -- the
// kernel was ~14% of the tensor-pipe ceiling with nothing co-resident to cover
// it. Occupancy *percentage* is the wrong metric here (a well-tuned FA2 kernel
// runs at 12.5% and still reaches 70% of peak); what matters is independent
// instruction streams per scheduler, and eight warps gives two.
//
// kBc doubles with it. Everything proportional to kBr*D -- the rescale, the
// epilogue -- is paid once per key block, so its cost per FLOP falls as 1/kBc,
// and kBr=128 halves how many times K and V are re-read per head.
//
// --- REGISTER BUDGET: the two instantiations are in different regimes -------
//
// Measured on this file with `-Xptxas -v`, sm_120a, at the commit that added
// this comment. Re-measure rather than trusting the numbers; the *ceiling* is
// what does not move.
//
//   D = 128 : 173 registers, 0 spills, 34304 B smem -> 1 block/SM.
//             No cliff to fall off, but note *why* has changed: deleting the
//             shared Q tile took smem from 69120 to 34304 B, so two blocks now
//             fit in shared memory (68608 <= 102400) and are stopped by
//             registers alone (173 * 256 * 2 = 88576 > 65536). It reads like
//             an occupancy win and is not one. `o[kOTiles][4]` alone is 64
//             registers of irreducible accumulator, so 2 blocks/SM here would
//             need <=128 and is unreachable. Instruction count is still the
//             only currency; spending a register costs nothing.
//
//   D = 64  : 125 registers, 0 spills -> 2 blocks/SM, with **three registers
//             of margin**. The ceiling is 128 and it is exact: 128 * 256
//             threads is precisely half the 65536-register file. At 129 this
//             path drops to 1 block/SM -- a 2x occupancy loss.
//
// The trap: **no test in this suite can see that happen.** The tests check
// numbers, and this failure only moves the clock. A change that is bit-exact,
// digest-identical and obviously correct can still halve D=64's occupancy, and
// the only thing that reports it is `-Xptxas -v`. So if you touch this kernel,
// read the register count for **both** instantiations, not just D=128 -- the
// one with headroom is the one people quote.
//
// Deliberately not a `__launch_bounds__` / `minBlocksPerMultiprocessor` cap --
// and this has now been measured rather than assumed.
// `__launch_bounds__(kThreads, D == 64 ? 2 : 1)` *does* hold D=64 to 128
// registers and keep 2 blocks/SM. It pays 8 B stack frame, 12 B spill stores
// and 8 B spill loads **in the inner loop**, on the very path it rescues, and
// pushes D=128 to 202 registers (harmless -- no cliff there). So the trade is
// 2x occupancy against spills in the hot loop, which settles on a clock and
// not on an argument. Measured, understood, not applied: nothing in this port
// takes D=64, so the clock that would decide it has never been worth running.
//
// **These counts are branch-local, and this comment merges silently.** They
// were measured on the commit that wrote them. A merge will not conflict on
// them and will not update them, so this block can arrive in a tree where it
// is false -- which has already happened once, cleanly, with no diff to
// review. Therefore: **whoever merges a change to this file re-measures both
// instantiations and edits these numbers in the same commit.** Nothing checks
// this. It is not optional and it is not the author's job, it is the merger's.
//
// Two measurements from the campaign that established the rule, both of which
// would have fooled a careful person:
//
//   * Two branches each started from D=64 = 125. One spent 3 registers, the
//     other 2, each correctly under the ceiling on its own branch. This was
//     first written as a prediction that the merge would land at 129-130 by
//     adding the two spends. **That prediction was wrong and understated its
//     own case: measured after the merge, D=64 came out at 132.** ptxas
//     reallocates across the whole function body rather than summing
//     per-branch costs, so a merge can overshoot what its parents spent.
//     Per-branch numbers are not additive and therefore not predictive:
//     **only a measurement on the integration branch, after the second change
//     lands, tells you anything.**
//   * A variant that derived `vt` from `ks` instead of tracking it **saved a
//     register at D=128 and cost one at D=64** -- ptxas rescheduled and the
//     trade inverted. D=128 is 1 block/SM at any count in this range, so the
//     saving bought nothing while the cost was the entire margin. Optimising
//     on the D=128 number alone would have shipped at the ceiling and called
//     it an improvement.
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * kWarp;
constexpr int kBr = 16 * kWarps;  // query rows per block, 16 per warp
constexpr int kBc = 64;           // key rows per step
constexpr int kMmaN = 8;          // n extent of one mma tile

// 16-bit tiles only, so the bank argument is the honest one: at stride 136
// halves the QK operand addresses reduce to `groupID*4 + tig`, which is a
// permutation of 0..31. kPadV = 2 makes V^T's stride 17 words, coprime with 32.
constexpr int kPadH = 8;
constexpr int kPadV = 2;

// C = A*B + C on one m16n8k16 tile. Separate bf16 and f16 forms because the QK
// product consumes bf16 straight from the checkpoint while the PV product
// consumes fp16 probabilities -- the 11 mantissa bits the blocked path
// deliberately chose over bf16's 8.
__device__ inline void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ inline void mma_f16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ inline uint32_t ld32(const void* p) { return *reinterpret_cast<const uint32_t*>(p); }

// --- K/V staging ------------------------------------------------------------
//
// Staging one key block moves kBc*D halves of K and the same of V from global
// into shared. It sits between two `__syncthreads`, so no `mma` can issue while
// any of it is in flight, and that made it the kernel's largest single cost.
//
// **Nothing about a thread's share of it depends on the key block.** Its row,
// its column, both shared destinations and both global sources are fixed by the
// thread index; from one key block to the next the sources advance by exactly
// `kBc * kvld` and the destinations do not move at all. The previous version
// derived all of it from a flat index on every element -- `r = i / D`,
// `c = i % D`, then a 64-bit `row*kvld + kv_head*D + c`, reloading the kernel
// parameters from the constant bank as it went -- and cost **172 SASS
// instructions per trip over 8 trips, 1376 per warp per key block**, of which
// 128 were the loads and 128 the stores. `KvStage` hoists the lot: the staging
// call inside the `k0` loop does not reference `k0`.
//
// **Each thread takes eight contiguous columns**, one 16-byte access. K is
// staged row-major, so that is one `LDG.128` and one `STS.128`. V is staged
// transposed on purpose (see above), so its eight halves land in eight
// different rows of `vt` and its stores stay scalar `STS.U16`; only its load
// widens.
//
// **The lane -> row map is permuted, and that is load-bearing.** A warp covers
// 8 rows x 32 columns as (row rr, 16-byte column cc). K's 16-byte store starts
// in bank `4*rr + 4*cc + 16*cg (mod 32)`, and a 128-bit store is issued in
// phases of eight lanes whose starting banks must be eight distinct multiples
// of four. The natural `rr = lane/4` gives a phase `rr + cc` of
// 0,1,2,3,1,2,3,4 -- three collisions, a 2-way conflict on every K store.
// `rr = lane/8 + 4*((lane/4) & 1)` gives it {0,4} x {0,1,2,3}, which is
// 0,4,8,...,28. V's scatter is indifferent to the permutation: `kVStride` is 66
// halves, so 33 words, so its bank reduces to `col + e + row/2 (mod 32)` and a
// warp covers 16 distinct words with the two lanes of each pair writing the two
// halves of one word.
constexpr int kVec = 8;                          // halves in a 16-byte access
constexpr int kStageCols = 4;                    // 16-byte columns one warp covers
constexpr int kStageRows = kWarp / kStageCols;   // rows one warp covers

// Hoisted staging state for one thread. Constructed once; `run` stages the
// current key block and advances to the next, so the loop body carries no
// addressing of its own. Keeping it a value rather than inline code is what
// lets a second buffer be staged ahead of the one being consumed without
// touching any of the arithmetic.
template <int D>
struct KvStage {
  static constexpr int kColGroups = D / (kVec * kStageCols);          // 4 at D=128, 2 at D=64
  static constexpr int kRowsPerPass = kWarps / kColGroups * kStageRows;
  static constexpr int kPasses = kBc / kRowsPerPass;
  static constexpr int kKStride = D + kPadH;
  static constexpr int kVStride = kBc + kPadV;

  static_assert(kStageCols == 4 && kStageRows == 8, "the bank argument above assumes 4x8 warps");
  static_assert(D % (kVec * kStageCols) == 0, "D must tile into whole 16-byte column groups");
  static_assert(kWarps % kColGroups == 0, "warps must split evenly over the column groups");
  static_assert(kPasses * kRowsPerPass == kBc, "the passes must tile the key block exactly");
  // 16-byte shared stores need 16-byte-aligned rows. `ks` itself is at offset 0
  // of the dynamic allocation, which `__align__(16)` pins, so only the row
  // stride is left to check.
  static_assert((kKStride * 2) % 16 == 0, "K rows must start on a 16-byte boundary");

  const __nv_bfloat16* kp;  // this thread's K source for pass 0 of the current key block
  const __nv_bfloat16* vp;
  size_t pass_stride;       // halves between one pass and the next
  int ks_off;               // halves into ks
  int vt_off;               // halves into vt
  int row;                  // absolute key row of pass 0

  __device__ KvStage(const __nv_bfloat16* k, const __nv_bfloat16* v, int tid, int kv_head,
                     size_t kvld) {
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int cc = lane & (kStageCols - 1);
    const int rr = (lane >> 3) + 4 * ((lane >> 2) & 1);
    const int cg = warp % kColGroups;
    const int rg = warp / kColGroups;
    const int col = (cg * kStageCols + cc) * kVec;
    row = rg * kStageRows + rr;
    ks_off = row * kKStride + col;
    vt_off = col * kVStride + row;
    const size_t off = static_cast<size_t>(row) * kvld + static_cast<size_t>(kv_head) * D + col;
    kp = k + off;
    vp = v + off;
    pass_stride = static_cast<size_t>(kRowsPerPass) * kvld;
  }

  // Skip forward by `delta` key rows. Frame banding gives a query tile two
  // disjoint key ranges -- the text/audio prefix and its own band -- so the
  // staging has to move between them without re-deriving the per-thread
  // addressing that F1 exists to hoist.
  //
  // The caller passes the delta rather than a destination, deliberately: it
  // already knows both ends, and remembering this thread's own row offset here
  // would cost a live register on the unbanded path too, which never calls this
  // at all.
  __device__ void advance(int delta, size_t kvld) {
    const ptrdiff_t step = static_cast<ptrdiff_t>(delta) * static_cast<ptrdiff_t>(kvld);
    kp += step;
    vp += step;
    row += delta;
  }

  // Rows past the end of the sequence are zeroed exactly as before -- the
  // softmax excludes dead *columns* itself and must keep seeing zeros here, not
  // stale shared memory. Barriers are the caller's.
  __device__ void run(__nv_bfloat16* ks, __half* vt, int seq) {
#pragma unroll
    for (int p = 0; p < kPasses; ++p) {
      uint4 kw = {0u, 0u, 0u, 0u};
      uint4 vw = {0u, 0u, 0u, 0u};
      if (row < seq) {
        kw = *reinterpret_cast<const uint4*>(kp);
        vw = *reinterpret_cast<const uint4*>(vp);
      }
      *reinterpret_cast<uint4*>(ks + ks_off + p * kRowsPerPass * kKStride) = kw;
      const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(&vw);
      __half* dst = vt + vt_off + p * kRowsPerPass;
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        dst[e * kVStride] = __float2half(__bfloat162float(src[e]));
      }
      kp += pass_stride;
      vp += pass_stride;
      row += kRowsPerPass;
    }
  }
};

__device__ inline uint32_t pack_h2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// Q is not here. It is read straight from global into the A-fragments at block
// entry and never staged -- see the load in `fused_kernel`.
template <int D>
__host__ __device__ inline size_t smem_bytes() {
  size_t n = 0;
  n += sizeof(__nv_bfloat16) * kBc * (D + kPadH);  // K
  n += sizeof(__half) * D * (kBc + kPadV);         // V transposed
  return n;
}

template <int D, bool kBanded>
__global__ __launch_bounds__(kThreads) void fused_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out, int seq, int heads,
    int num_kv_heads, float scale, const int4* __restrict__ band) {
  constexpr int kKStride = D + kPadH;
  constexpr int kVStride = kBc + kPadV;
  constexpr int kDSteps = D / 16;      // k-steps of the QK product
  constexpr int kSTiles = kBc / kMmaN; // n-tiles of S
  constexpr int kOTiles = D / kMmaN;   // n-tiles of O
  constexpr int kPSteps = kBc / 16;    // k-steps of the PV product

  // 16-byte aligned because the K stage stores `uint4`. The dynamic allocation
  // is suitably aligned already; saying so keeps it true if the declaration
  // ever moves.
  extern __shared__ __align__(16) char raw_smem[];
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(raw_smem);
  __half* vt = reinterpret_cast<__half*>(ks + kBc * kKStride);

  const int q0 = blockIdx.x * kBr;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;
  const int gid = lane >> 2;   // 0..7, selects the row pair
  const int tig = lane & 3;    // 0..3, selects the column pair

  const int group = heads / num_kv_heads;
  const int kv_head = head / group;
  const size_t qld = static_cast<size_t>(heads) * D;
  const size_t kvld = static_cast<size_t>(num_kv_heads) * D;

  // The two output rows this thread owns, block-relative.
  const int row_a = warp * 16 + gid;
  const int row_b = row_a + 8;

  // Q A-fragments, straight from global, loaded once and reused across every
  // key block.
  //
  // Q used to be staged through a kBr x D shared tile that existed for exactly
  // this one read: 34816 B alive for the kernel's whole lifetime, written once,
  // read once, dead from here on. The `mma` A-fragment layout says which two
  // elements of which row each lane wants, so the tile was only ever a
  // transpose-free reshuffle of bytes the thread could address itself --
  // 32 four-byte loads per thread, once per block, amortised over 590 key
  // blocks against a barrier and a kBr*D staging loop it also removes.
  //
  // Deleting it is the precondition for double-buffering K/V rather than a
  // saving in its own right: two K/V buffers are 68608 B, and
  // 68608 + 34816 = 103424 exceeds the 102400 B a block can hold. With Q gone
  // the pair fits with room to spare.
  //
  // Rows past the end of the sequence read zero, exactly as the zero-filled
  // shared tile gave them. They are discarded by the guard in the epilogue,
  // never by the arithmetic.
  const bool q_live_a = q0 + row_a < seq;
  const bool q_live_b = q0 + row_b < seq;
  const __nv_bfloat16* qsrc_a =
      q + static_cast<size_t>(q0 + row_a) * qld + head * D + tig * 2;
  const __nv_bfloat16* qsrc_b =
      q + static_cast<size_t>(q0 + row_b) * qld + head * D + tig * 2;
  uint32_t qa[kDSteps][4];
#pragma unroll
  for (int t = 0; t < kDSteps; ++t) {
    qa[t][0] = q_live_a ? ld32(qsrc_a + t * 16) : 0u;
    qa[t][1] = q_live_b ? ld32(qsrc_b + t * 16) : 0u;
    qa[t][2] = q_live_a ? ld32(qsrc_a + t * 16 + 8) : 0u;
    qa[t][3] = q_live_b ? ld32(qsrc_b + t * 16 + 8) : 0u;
  }

  float o[kOTiles][4];
#pragma unroll
  for (int j = 0; j < kOTiles; ++j) {
    o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
  }
  float m_a = kHostNegInf, m_b = kHostNegInf;
  float l_a = 0.0f, l_b = 0.0f;

  // Every address the staging needs, computed once. See KvStage.
  KvStage<D> stage(k, v, tid, kv_head, kvld);

  // Which keys this query tile may see. Without a band that is the whole
  // sequence in one range; with one it is the text/audio prefix plus this
  // tile's frame band, which are disjoint and in order. Both bounds are already
  // key-block aligned by the host, so the loop below is unchanged in shape --
  // banding moves the *bound*, not the body.
  //
  // Nothing else needs to know. The online softmax is a running max and sum
  // over whatever columns it is shown, so a subset is not a special case, and
  // it already excludes columns past `seq` rather than zeroing keys.
  // Flattened to one loop over key *blocks* rather than nested range-then-block
  // loops. The nested form cost 14 registers at D=64 -- 125 to 139, straight
  // through the 128 ceiling and down to 1 block/SM -- because the outer loop
  // kept the inner one's state live across it.
  //
  // `kBanded` is a template parameter and not a runtime test, because banding is
  // off by default and the default path must stay the kernel it already was.
  // With it false every line below compiles away and the loop is the original
  // walk from 0 to seq; the register counts say so -- 173/125 unbanded against
  // 199/128 banded, i.e. the cost lands only on the path that opted in.
  int k0 = 0;
  int k_stop = seq;
  int seam_lo = 0, seam_stop = 0;
  if constexpr (kBanded) {
    const int4 r = band[blockIdx.x];
    k0 = r.x;
    k_stop = r.y;
    seam_lo = r.z;
    seam_stop = r.w;
    stage.advance(k0, kvld);  // from row 0 to the first range
  }

  while (true) {
    if (k0 >= k_stop) {
      if constexpr (kBanded) {
        if (seam_stop <= seam_lo) break;
        stage.advance(seam_lo - k0, kvld);
        k0 = seam_lo;
        k_stop = seam_stop;
        seam_lo = seam_stop = 0;
      } else {
        break;
      }
    }
    __syncthreads();  // last iteration's mma has finished reading ks/vt
    stage.run(ks, vt, seq);
    __syncthreads();

    // S = Q K^T. K is row-major in shared memory and the B operand is
    // column-major, which is exactly K^T -- the same free transpose the
    // blocked path gets from cuBLAS.
    float s[kSTiles][4];
#pragma unroll
    for (int j = 0; j < kSTiles; ++j) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
      for (int t = 0; t < kDSteps; ++t) {
        const __nv_bfloat16* kb = ks + (j * kMmaN + gid) * kKStride + t * 16 + tig * 2;
        uint32_t b[2] = {ld32(kb), ld32(kb + 8)};
        mma_bf16(s[j], qa[t], b);
      }
    }

    // Online softmax, entirely in registers. Columns past the end of the
    // sequence are excluded here rather than by zeroing k: a zero key scores 0,
    // whose exp is 1, which would inflate the denominator.
    const int col0 = k0 + tig * 2;
    float ma = kHostNegInf, mb = kHostNegInf;
#pragma unroll
    for (int j = 0; j < kSTiles; ++j) {
      const int c = col0 + j * kMmaN;
      if (c < seq) {
        ma = fmaxf(ma, s[j][0] * scale);
        mb = fmaxf(mb, s[j][2] * scale);
      }
      if (c + 1 < seq) {
        ma = fmaxf(ma, s[j][1] * scale);
        mb = fmaxf(mb, s[j][3] * scale);
      }
    }
    // The four lanes of a group hold all eight columns of a row and are
    // consecutive, so xor by 1 then 2 reduces both rows at once.
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, off));
      mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, off));
    }

    const float new_a = fmaxf(m_a, ma);
    const float new_b = fmaxf(m_b, mb);
    const float c_a = (m_a == kHostNegInf || new_a == kHostNegInf) ? 0.0f : __expf(m_a - new_a);
    const float c_b = (m_b == kHostNegInf || new_b == kHostNegInf) ? 0.0f : __expf(m_b - new_b);

    float sum_a = 0.0f, sum_b = 0.0f;
#pragma unroll
    for (int j = 0; j < kSTiles; ++j) {
      const int c = col0 + j * kMmaN;
      const bool l0 = c < seq && new_a != kHostNegInf;
      const bool l1 = c + 1 < seq && new_a != kHostNegInf;
      const bool r0 = c < seq && new_b != kHostNegInf;
      const bool r1 = c + 1 < seq && new_b != kHostNegInf;
      s[j][0] = l0 ? __expf(s[j][0] * scale - new_a) : 0.0f;
      s[j][1] = l1 ? __expf(s[j][1] * scale - new_a) : 0.0f;
      s[j][2] = r0 ? __expf(s[j][2] * scale - new_b) : 0.0f;
      s[j][3] = r1 ? __expf(s[j][3] * scale - new_b) : 0.0f;
      sum_a += s[j][0] + s[j][1];
      sum_b += s[j][2] + s[j][3];
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      sum_a += __shfl_xor_sync(0xffffffffu, sum_a, off);
      sum_b += __shfl_xor_sync(0xffffffffu, sum_b, off);
    }
    l_a = l_a * c_a + sum_a;
    l_b = l_b * c_b + sum_b;
    m_a = new_a;
    m_b = new_b;

    // Four multiplies, because c0/c1 and c2/c3 are one row each.
#pragma unroll
    for (int j = 0; j < kOTiles; ++j) {
      o[j][0] *= c_a;
      o[j][1] *= c_a;
      o[j][2] *= c_b;
      o[j][3] *= c_b;
    }

    // S -> P with no data movement: the n-offset 0 and 8 tiles are precisely
    // the A-fragment's four registers.
    uint32_t pa[kPSteps][4];
#pragma unroll
    for (int t = 0; t < kPSteps; ++t) {
      const int j0 = t * 2;
      pa[t][0] = pack_h2(s[j0][0], s[j0][1]);
      pa[t][1] = pack_h2(s[j0][2], s[j0][3]);
      pa[t][2] = pack_h2(s[j0 + 1][0], s[j0 + 1][1]);
      pa[t][3] = pack_h2(s[j0 + 1][2], s[j0 + 1][3]);
    }

#pragma unroll
    for (int j = 0; j < kOTiles; ++j) {
#pragma unroll
      for (int t = 0; t < kPSteps; ++t) {
        const __half* vb = vt + (j * kMmaN + gid) * kVStride + t * 16 + tig * 2;
        uint32_t b[2] = {ld32(vb), ld32(vb + 8)};
        mma_f16(o[j], pa[t], b);
      }
    }
    k0 += kBc;
  }

  const float inv_a = l_a > 0.0f ? 1.0f / l_a : 0.0f;
  const float inv_b = l_b > 0.0f ? 1.0f / l_b : 0.0f;
  const int out_a = q0 + row_a;
  const int out_b = q0 + row_b;
  __nv_bfloat16* base = out + head * D;
#pragma unroll
  for (int j = 0; j < kOTiles; ++j) {
    const int c = j * kMmaN + tig * 2;
    if (out_a < seq) {
      base[static_cast<size_t>(out_a) * qld + c] = __float2bfloat16(o[j][0] * inv_a);
      base[static_cast<size_t>(out_a) * qld + c + 1] = __float2bfloat16(o[j][1] * inv_a);
    }
    if (out_b < seq) {
      base[static_cast<size_t>(out_b) * qld + c] = __float2bfloat16(o[j][2] * inv_b);
      base[static_cast<size_t>(out_b) * qld + c + 1] = __float2bfloat16(o[j][3] * inv_b);
    }
  }
}

// >48 KB of shared memory per block is opt-in and the opt-in is per function.
// Doing it once per (function, thread) rather than per launch keeps it off the
// hot path; the call is idempotent so a race is harmless.
template <int D>
void ensure_smem_optin() {
  static thread_local bool done = false;
  if (done) return;
  VIDFAB_CUDA_CHECK(cudaFuncSetAttribute(fused_kernel<D, false>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem_bytes<D>())));
  VIDFAB_CUDA_CHECK(cudaFuncSetAttribute(fused_kernel<D, true>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem_bytes<D>())));
  done = true;
}

template <int D>
void launch(cudaStream_t stream, const __nv_bfloat16* q, const __nv_bfloat16* k,
            const __nv_bfloat16* v, __nv_bfloat16* out, const AttentionConfig& cfg,
            int num_kv_heads) {
  ensure_smem_optin<D>();
  const dim3 grid(static_cast<unsigned>((cfg.seq_len + kBr - 1) / kBr),
                  static_cast<unsigned>(cfg.num_heads));
  // Grid order matters and is load-bearing: x is the query tile and y is the
  // head, and CUDA dispatches x fastest, so the resident blocks share a head
  // and walk one K/V stream together. That stream is 19.3 MB at production
  // shape, which lives in L2. Swapping the dimensions would turn L2 hits into
  // DRAM traffic.
  const int4* band = reinterpret_cast<const int4*>(cfg.band_ranges);
  if (band != nullptr) {
    fused_kernel<D, true><<<grid, kThreads, smem_bytes<D>(), stream>>>(
        q, k, v, out, cfg.seq_len, cfg.num_heads, num_kv_heads, cfg.effective_scale(), band);
  } else {
    fused_kernel<D, false><<<grid, kThreads, smem_bytes<D>(), stream>>>(
        q, k, v, out, cfg.seq_len, cfg.num_heads, num_kv_heads, cfg.effective_scale(), nullptr);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

bool supported(const AttentionConfig& cfg) { return cfg.head_dim == 64 || cfg.head_dim == 128; }

}  // namespace fused

void run_fused(cudaStream_t stream, const __nv_bfloat16* q, const __nv_bfloat16* k,
               const __nv_bfloat16* v, __nv_bfloat16* out, const AttentionConfig& cfg,
               int num_kv_heads) {
  check_config(cfg, num_kv_heads);
  if (!fused::supported(cfg)) {
    throw std::runtime_error("attention: kFused supports head_dim 64 or 128, got " +
                             std::to_string(cfg.head_dim));
  }
  // K and V are staged with 16-byte vector loads, and Q's A-fragments are read
  // from global as 4-byte pairs. Every element offset the kernel forms is a
  // multiple of eight halves for K/V and of two for Q -- head_dim is 64 or 128
  // and each thread starts on an eight-column boundary -- so only the base
  // pointers can break it, and every allocator here returns at least 256 bytes
  // of alignment. Check rather than fault: a misaligned address is a kernel
  // abort with no indication of which pointer was wrong.
  const uintptr_t bases = reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) |
                          reinterpret_cast<uintptr_t>(v);
  if ((bases & 15u) != 0) {
    throw std::runtime_error("attention: kFused needs 16-byte aligned q, k and v");
  }
  if (cfg.head_dim == 64) {
    fused::launch<64>(stream, q, k, v, out, cfg, num_kv_heads);
  } else {
    fused::launch<128>(stream, q, k, v, out, cfg, num_kv_heads);
  }
}

}  // namespace

int attention_fused_query_tile() { return fused::kBr; }
int attention_fused_key_align() { return fused::kBc; }

AttentionBackend attention_preferred_backend(const AttentionConfig& cfg) {
  return fused::supported(cfg) ? AttentionBackend::kFused : AttentionBackend::kBlocked;
}

float AttentionConfig::effective_scale() const {
  return scale > 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
}

size_t attention_workspace_bytes(const AttentionConfig& cfg, AttentionBackend backend) {
  // The fused backend keeps the score tile, the probabilities and the
  // accumulator in shared memory, and reads q, k and v as bf16 without the
  // fp16 conversion copies the blocked path needs. It therefore wants no
  // workspace at all -- not a smaller one.
  if (backend == AttentionBackend::kFused) return 0;
  if (cfg.seq_len <= 0 || cfg.num_heads <= 0 || cfg.head_dim <= 0) return 0;

  const int bq = effective_query_block(cfg);
  const int bk = choose_key_block(cfg);
  const size_t tile = static_cast<size_t>(cfg.num_heads) * bq * bk;
  const size_t stats = static_cast<size_t>(cfg.num_heads) * bq;
  const size_t width = static_cast<size_t>(cfg.num_heads) * cfg.head_dim;

  size_t total = 0;
  total += align_up(tile * sizeof(__half));                 // scores, then probabilities
  total += align_up(stats * cfg.head_dim * sizeof(float));  // accumulator
  total += align_up(stats * sizeof(float));                 // running max
  total += align_up(stats * sizeof(float));                 // running sum
  // fp16 copies of k and v, and of one query block. The config carries no kv
  // head count, so k and v are sized for the full-width case; a grouped-query
  // caller over-reserves by the group factor on a sequence short enough that it
  // does not matter.
  total += 2 * align_up(static_cast<size_t>(cfg.seq_len) * width * sizeof(__half));
  total += align_up(static_cast<size_t>(bq) * width * sizeof(__half));
  return total;
}

void attention_forward(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                       const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                       const AttentionConfig& cfg, AttentionBackend backend, Workspace& ws) {
  if (backend == AttentionBackend::kFused) {
    run_fused(stream, q, k, v, out, cfg, cfg.num_heads);
    return;
  }
  run_blocked(handle, stream, q, k, v, out, cfg, cfg.num_heads, ws);
}

void attention_forward_gqa(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                           const AttentionConfig& cfg, int num_kv_heads, AttentionBackend backend,
                           Workspace& ws) {
  if (backend == AttentionBackend::kFused) {
    run_fused(stream, q, k, v, out, cfg, num_kv_heads);
    return;
  }
  run_blocked(handle, stream, q, k, v, out, cfg, num_kv_heads, ws);
}

}  // namespace vidfab::cuda
