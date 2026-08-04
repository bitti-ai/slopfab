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
#include <mma.h>

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
// fp16 out; `wmma` takes bf16 directly with an fp32 accumulator, so the two
// full-sequence conversion buffers and their HBM traffic disappear. The
// probabilities stay fp16 -- P feeds a second mma whose operands must share a
// type, so V is converted to fp16 during SMEM staging, which costs no HBM
// traffic and keeps the 11 mantissa bits the blocked path deliberately chose
// over bf16's 8.
//
// **Accumulator in shared memory, not registers.** O is rescaled by a
// *per-row* factor every key block. `wmma`'s accumulator fragment does not
// expose which row an element belongs to -- the mapping is unspecified -- so a
// register-resident O cannot be scaled correctly without assuming a layout the
// API does not promise. Keeping O in SMEM makes the rescale an ordinary
// indexed loop. It costs SMEM bandwidth per key block and no HBM traffic, which
// is the trade this kernel is for.
namespace fused {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * kWarp;
constexpr int kBr = 16 * kWarps;  // query rows per block, 16 per warp
constexpr int kBc = 32;           // key rows per step
constexpr int kM = 16;            // wmma tile extent

// Padding keeps successive rows out of the same SMEM bank. 8 elements of a
// 2-byte type and 4 of a 4-byte type are both 16 bytes, which is also the
// alignment `load_matrix_sync` wants.
constexpr int kPadH = 8;
constexpr int kPadF = 4;

struct Smem {
  __nv_bfloat16* qs;  // [kBr][D + kPadH]
  __nv_bfloat16* ks;  // [kBc][D + kPadH]
  __half* vs;         // [kBc][D + kPadH]
  float* st;          // [kBr][kBc + kPadF]
  __half* ps;         // [kBr][kBc + kPadH]
  float* os;          // [kBr][D + kPadF]
  float* cs;          // [kBr]
};

__host__ __device__ inline size_t smem_bytes(int D) {
  size_t n = 0;
  n += sizeof(__nv_bfloat16) * kBr * (D + kPadH);
  n += sizeof(__nv_bfloat16) * kBc * (D + kPadH);
  n += sizeof(__half) * kBc * (D + kPadH);
  n += sizeof(float) * kBr * (kBc + kPadF);
  n += sizeof(__half) * kBr * (kBc + kPadH);
  n += sizeof(float) * kBr * (D + kPadF);
  n += sizeof(float) * kBr;
  return n;
}

__device__ inline Smem carve(char* raw, int D) {
  Smem s;
  s.qs = reinterpret_cast<__nv_bfloat16*>(raw);
  raw += sizeof(__nv_bfloat16) * kBr * (D + kPadH);
  s.ks = reinterpret_cast<__nv_bfloat16*>(raw);
  raw += sizeof(__nv_bfloat16) * kBc * (D + kPadH);
  s.vs = reinterpret_cast<__half*>(raw);
  raw += sizeof(__half) * kBc * (D + kPadH);
  s.st = reinterpret_cast<float*>(raw);
  raw += sizeof(float) * kBr * (kBc + kPadF);
  s.ps = reinterpret_cast<__half*>(raw);
  raw += sizeof(__half) * kBr * (kBc + kPadH);
  s.os = reinterpret_cast<float*>(raw);
  raw += sizeof(float) * kBr * (D + kPadF);
  s.cs = reinterpret_cast<float*>(raw);
  return s;
}

// D is a template parameter so the fragment loops unroll to a constant count and
// the SMEM strides fold into the addressing.
template <int D>
__global__ __launch_bounds__(kThreads) void fused_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out, int seq, int heads,
    int num_kv_heads, float scale) {
  using namespace nvcuda;

  extern __shared__ char raw_smem[];
  Smem sm = carve(raw_smem, D);

  const int q0 = blockIdx.x * kBr;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;

  const int group = heads / num_kv_heads;
  const int kv_head = head / group;
  const size_t qld = static_cast<size_t>(heads) * D;
  const size_t kvld = static_cast<size_t>(num_kv_heads) * D;

  constexpr int kDTiles = D / kM;
  constexpr int kCTiles = kBc / kM;

  // Q tile -> SMEM, zero-filled past the end of the sequence so the tail block's
  // fragments read defined data. Its scores are discarded by the row guard at
  // the end, never by the arithmetic.
  for (int i = tid; i < kBr * D; i += kThreads) {
    const int r = i / D;
    const int c = i % D;
    const int row = q0 + r;
    sm.qs[r * (D + kPadH) + c] =
        row < seq ? q[static_cast<size_t>(row) * qld + head * D + c] : __float2bfloat16(0.0f);
  }
  for (int i = tid; i < kBr * D; i += kThreads) sm.os[(i / D) * (D + kPadF) + (i % D)] = 0.0f;
  __syncthreads();

  // Per-row softmax state. Each warp owns rows [warp*16, warp*16+16), so lane r
  // (r < 16) is the sole owner of row warp*16 + r and no reduction is needed.
  float m_run = kHostNegInf;
  float l_run = 0.0f;
  const int my_row = warp * kM + lane;  // meaningful only for lane < kM
  const bool row_owner = lane < kM;

  wmma::fragment<wmma::matrix_a, kM, kM, kM, __nv_bfloat16, wmma::row_major> qf[kDTiles];
#pragma unroll
  for (int t = 0; t < kDTiles; ++t) {
    wmma::load_matrix_sync(qf[t], sm.qs + (warp * kM) * (D + kPadH) + t * kM, D + kPadH);
  }

  for (int k0 = 0; k0 < seq; k0 += kBc) {
    __syncthreads();  // previous iteration's PV is done reading ks/vs
    for (int i = tid; i < kBc * D; i += kThreads) {
      const int r = i / D;
      const int c = i % D;
      const int row = k0 + r;
      const bool live = row < seq;
      const size_t src = static_cast<size_t>(row) * kvld + kv_head * D + c;
      sm.ks[r * (D + kPadH) + c] = live ? k[src] : __float2bfloat16(0.0f);
      sm.vs[r * (D + kPadH) + c] = live ? __float2half(__bfloat162float(v[src])) : __float2half(0.0f);
    }
    __syncthreads();

    // S = Q K^T. K is [kBc][D] row-major in SMEM; reading it as a col_major B
    // fragment with ld = D + kPadH yields K^T without a transpose pass.
#pragma unroll
    for (int j = 0; j < kCTiles; ++j) {
      wmma::fragment<wmma::accumulator, kM, kM, kM, float> acc;
      wmma::fill_fragment(acc, 0.0f);
#pragma unroll
      for (int t = 0; t < kDTiles; ++t) {
        wmma::fragment<wmma::matrix_b, kM, kM, kM, __nv_bfloat16, wmma::col_major> kf;
        wmma::load_matrix_sync(kf, sm.ks + (j * kM) * (D + kPadH) + t * kM, D + kPadH);
        wmma::mma_sync(acc, qf[t], kf, acc);
      }
      wmma::store_matrix_sync(sm.st + (warp * kM) * (kBc + kPadF) + j * kM, acc, kBc + kPadF,
                              wmma::mem_row_major);
    }
    __syncwarp();

    // Online softmax over this warp's own 16 rows. Columns past the sequence end
    // are excluded here rather than by zeroing k: a zero key gives score 0,
    // whose exp is 1, which would silently inflate the denominator.
    float c_scale = 0.0f;
    if (row_owner) {
      const float* srow = sm.st + my_row * (kBc + kPadF);
      const int live = min(kBc, seq - k0);
      float tile_max = kHostNegInf;
      for (int j = 0; j < live; ++j) tile_max = fmaxf(tile_max, srow[j] * scale);
      const float m_new = fmaxf(m_run, tile_max);
      // m_new is -inf only if this warp has seen no live key at all, and then
      // every exp below is skipped; guarding keeps inf - inf out of the fp path.
      c_scale = (m_run == kHostNegInf || m_new == kHostNegInf) ? 0.0f : __expf(m_run - m_new);
      float sum = 0.0f;
      __half* prow = sm.ps + my_row * (kBc + kPadH);
      for (int j = 0; j < kBc; ++j) {
        float p = 0.0f;
        if (j < live && m_new != kHostNegInf) {
          p = __expf(srow[j] * scale - m_new);
          sum += p;
        }
        prow[j] = __float2half(p);
      }
      l_run = l_run * c_scale + sum;
      m_run = m_new;
      sm.cs[my_row] = c_scale;
    }
    __syncthreads();

    // O *= c, then O += P V. The rescale is a plain indexed loop precisely
    // because O lives in SMEM.
    for (int i = tid; i < kBr * D; i += kThreads) {
      const int r = i / D;
      sm.os[r * (D + kPadF) + (i % D)] *= sm.cs[r];
    }
    __syncthreads();

#pragma unroll
    for (int jj = 0; jj < kDTiles; ++jj) {
      wmma::fragment<wmma::accumulator, kM, kM, kM, float> acc;
      wmma::load_matrix_sync(acc, sm.os + (warp * kM) * (D + kPadF) + jj * kM, D + kPadF,
                             wmma::mem_row_major);
#pragma unroll
      for (int t = 0; t < kCTiles; ++t) {
        wmma::fragment<wmma::matrix_a, kM, kM, kM, __half, wmma::row_major> pf;
        wmma::fragment<wmma::matrix_b, kM, kM, kM, __half, wmma::row_major> vf;
        wmma::load_matrix_sync(pf, sm.ps + (warp * kM) * (kBc + kPadH) + t * kM, kBc + kPadH);
        wmma::load_matrix_sync(vf, sm.vs + (t * kM) * (D + kPadH) + jj * kM, D + kPadH);
        wmma::mma_sync(acc, pf, vf, acc);
      }
      wmma::store_matrix_sync(sm.os + (warp * kM) * (D + kPadF) + jj * kM, acc, D + kPadF,
                              wmma::mem_row_major);
    }
  }
  __syncthreads();

  if (row_owner && q0 + my_row < seq) {
    const float inv = l_run > 0.0f ? 1.0f / l_run : 0.0f;
    const float* orow = sm.os + my_row * (D + kPadF);
    __nv_bfloat16* dst = out + static_cast<size_t>(q0 + my_row) * qld + head * D;
    for (int d = 0; d < D; ++d) dst[d] = __float2bfloat16(orow[d] * inv);
  }
}

// >48 KB of shared memory per block is opt-in, and the opt-in is per function.
// Doing it once per (function, device) rather than per launch keeps it off the
// hot path; the flag is idempotent so a race between threads is harmless.
template <int D>
void ensure_smem_optin() {
  static thread_local bool done = false;
  if (done) return;
  VIDFAB_CUDA_CHECK(cudaFuncSetAttribute(fused_kernel<D>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem_bytes(D))));
  done = true;
}

template <int D>
void launch(cudaStream_t stream, const __nv_bfloat16* q, const __nv_bfloat16* k,
            const __nv_bfloat16* v, __nv_bfloat16* out, const AttentionConfig& cfg,
            int num_kv_heads) {
  ensure_smem_optin<D>();
  const dim3 grid(static_cast<unsigned>((cfg.seq_len + kBr - 1) / kBr),
                  static_cast<unsigned>(cfg.num_heads));
  fused_kernel<D><<<grid, kThreads, smem_bytes(D), stream>>>(
      q, k, v, out, cfg.seq_len, cfg.num_heads, num_kv_heads, cfg.effective_scale());
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
  if (cfg.head_dim == 64) {
    fused::launch<64>(stream, q, k, v, out, cfg, num_kv_heads);
  } else {
    fused::launch<128>(stream, q, k, v, out, cfg, num_kv_heads);
  }
}

}  // namespace

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
