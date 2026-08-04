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
// probabilities, 564 ms this way.
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
                                          __half* __restrict__ dst, size_t packs, size_t n) {
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
  (void)n;
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
                                                                                         packs, n);
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
