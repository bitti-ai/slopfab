#pragma once
#include "slopfab/cuda/attention.cuh"
#include "slopfab/cuda/sage_attention.cuh"
#include "slopfab/cuda/sol_attention.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"


namespace slopfab::cuda::attention_detail {
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

inline int effective_query_block(const AttentionConfig& cfg) {
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
inline int choose_key_block(const AttentionConfig& cfg) {
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

inline void check_config(const AttentionConfig& cfg, int num_kv_heads) {
  if (cfg.seq_len <= 0 || cfg.num_heads <= 0 || cfg.head_dim <= 0) {
    throw std::runtime_error("attention: seq_len, num_heads and head_dim must be positive");
  }
  if (num_kv_heads <= 0 || cfg.num_heads % num_kv_heads != 0) {
    throw std::runtime_error("attention: num_heads must be a multiple of num_kv_heads");
  }
}

void run_blocked(cublasHandle_t, cudaStream_t, const __nv_bfloat16*,
                 const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
                 const AttentionConfig&, int, Workspace&);
void run_fused(cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*,
               const __nv_bfloat16*, __nv_bfloat16*, const AttentionConfig&,
               int, int query_rows = 0);
inline bool fused_supported(const AttentionConfig& config) {
  return config.head_dim == 64 || config.head_dim == 128;
}
}  // namespace slopfab::cuda::attention_detail
