// Memory-bounded Sol-Attn reference kernel.
//
// Routing is performed at 64x64 physical-block granularity from mean-pooled Q
// and K. Selected blocks use exact attention; rejected blocks use the paper's
// zeroth-order numerator/denominator correction. A route map is deliberately
// not materialised: at H3's default shape it would consume about 20 MiB and
// add another global-memory pass. This first native path instead recomputes the
// inexpensive proxy in each query-row CTA. The exact and approximate terms
// share one online-softmax state, including tail-block multiplicity.

#include "vidfab/cuda/sol_attention.cuh"

#include <cmath>
#include <cfloat>
#include <stdexcept>
#include <string>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {
namespace {

constexpr int kBlock = 64;
constexpr int kDim = 128;
constexpr size_t kAlign = 256;

size_t aligned(size_t n) { return (n + kAlign - 1) & ~(kAlign - 1); }

__device__ float block_sum(float x, float* scratch) {
  const int lane = threadIdx.x;
  scratch[lane] = x;
  __syncthreads();
  for (int stride = kDim / 2; stride > 0; stride >>= 1) {
    if (lane < stride) scratch[lane] += scratch[lane + stride];
    __syncthreads();
  }
  return scratch[0];
}

__global__ void pool_kv(const __nv_bfloat16* k, const __nv_bfloat16* v,
                        __nv_bfloat16* k_mean, float* v_sum, int seq, int heads) {
  const int block = blockIdx.x;
  const int head = blockIdx.y;
  const int d = threadIdx.x;
  const int begin = block * kBlock;
  const int end = min(begin + kBlock, seq);
  const size_t width = static_cast<size_t>(heads) * kDim;
  const size_t pooled = (static_cast<size_t>(block) * heads + head) * kDim + d;
  float sk = 0.0f;
  float sv = 0.0f;
  for (int row = begin; row < end; ++row) {
    const size_t at = static_cast<size_t>(row) * width + head * kDim + d;
    sk += __bfloat162float(k[at]);
    sv += __bfloat162float(v[at]);
  }
  k_mean[pooled] = __float2bfloat16_rn(sk / static_cast<float>(end - begin));
  v_sum[pooled] = sv;
}

__global__ void sol_forward_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                   const __nv_bfloat16* v, const __nv_bfloat16* k_mean,
                                   const float* v_sum, __nv_bfloat16* out, int seq, int heads,
                                   int exact_prefix, float scale, float beta) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int d = threadIdx.x;
  const int blocks = (seq + kBlock - 1) / kBlock;
  const int q_begin = (row / kBlock) * kBlock;
  const int q_end = min(q_begin + kBlock, seq);
  const size_t width = static_cast<size_t>(heads) * kDim;
  const size_t q_at = static_cast<size_t>(row) * width + head * kDim + d;

  __shared__ float reduce[kDim];
  __shared__ float q_mean[kDim];
  __shared__ float route_mean;
  __shared__ float route_m2;
  __shared__ float threshold;
  __shared__ float online_m;
  __shared__ float online_l;
  __shared__ float alpha;
  __shared__ float weight;
  __shared__ int selected;

  float qm = 0.0f;
  for (int qr = q_begin; qr < q_end; ++qr) {
    qm += __bfloat162float(q[static_cast<size_t>(qr) * width + head * kDim + d]);
  }
  q_mean[d] = qm / static_cast<float>(q_end - q_begin);
  __syncthreads();

  if (d == 0) {
    route_mean = 0.0f;
    route_m2 = 0.0f;
  }
  __syncthreads();
  if (row >= exact_prefix) {
    for (int kb = 0; kb < blocks; ++kb) {
      const size_t pooled = (static_cast<size_t>(kb) * heads + head) * kDim + d;
      const float proxy = block_sum(q_mean[d] * __bfloat162float(k_mean[pooled]), reduce) * scale;
      if (d == 0) {
        const float n = static_cast<float>(kb + 1);
        const float delta = proxy - route_mean;
        route_mean += delta / n;
        route_m2 += delta * (proxy - route_mean);
      }
      __syncthreads();
    }
  }
  if (d == 0) {
    threshold = route_mean + beta * sqrtf(route_m2 / static_cast<float>(max(blocks, 1)));
    online_m = -FLT_MAX;
    online_l = 0.0f;
  }
  __syncthreads();

  const float qv = __bfloat162float(q[q_at]);
  float acc = 0.0f;
  for (int kb = 0; kb < blocks; ++kb) {
    const int begin = kb * kBlock;
    const int end = min(begin + kBlock, seq);
    const size_t pooled = (static_cast<size_t>(kb) * heads + head) * kDim + d;
    const float proxy = block_sum(q_mean[d] * __bfloat162float(k_mean[pooled]), reduce) * scale;
    if (d == 0) {
      selected = row < exact_prefix || begin < exact_prefix || proxy > threshold;
    }
    __syncthreads();

    if (selected) {
      for (int key_row = begin; key_row < end; ++key_row) {
        const size_t key_at = static_cast<size_t>(key_row) * width + head * kDim + d;
        const float score = block_sum(qv * __bfloat162float(k[key_at]), reduce) * scale;
        if (d == 0) {
          const float next_m = fmaxf(online_m, score);
          alpha = expf(online_m - next_m);
          weight = expf(score - next_m);
          online_l = online_l * alpha + weight;
          online_m = next_m;
        }
        __syncthreads();
        acc = acc * alpha + weight * __bfloat162float(v[key_at]);
        __syncthreads();
      }
    } else {
      const float score = block_sum(qv * __bfloat162float(k_mean[pooled]), reduce) * scale;
      if (d == 0) {
        const float next_m = fmaxf(online_m, score);
        alpha = expf(online_m - next_m);
        weight = expf(score - next_m);
        online_l = online_l * alpha + static_cast<float>(end - begin) * weight;
        online_m = next_m;
      }
      __syncthreads();
      acc = acc * alpha + weight * v_sum[pooled];
      __syncthreads();
    }
  }
  out[q_at] = __float2bfloat16_rn(acc / online_l);
}

void validate(const AttentionConfig& cfg) {
  if (cfg.seq_len <= 0 || cfg.num_heads <= 0)
    throw std::runtime_error("Sol-Attn: seq_len and num_heads must be positive");
  if (cfg.head_dim != kDim)
    throw std::runtime_error("Sol-Attn: native kernel requires head_dim 128");
  if (cfg.band_ranges != nullptr)
    throw std::runtime_error("Sol-Attn: frame banding cannot be combined with block routing");
  if (cfg.exact_prefix < 0 || cfg.exact_prefix > cfg.seq_len)
    throw std::runtime_error("Sol-Attn: exact_prefix is outside the sequence");
  if (!std::isfinite(cfg.sol_beta)) throw std::runtime_error("Sol-Attn: sol_beta must be finite");
}

}  // namespace

size_t sol_attention_workspace_bytes(const AttentionConfig& cfg) {
  if (cfg.seq_len <= 0 || cfg.num_heads <= 0 || cfg.head_dim != kDim) return 0;
  const size_t blocks = (static_cast<size_t>(cfg.seq_len) + kBlock - 1) / kBlock;
  const size_t values = blocks * cfg.num_heads * kDim;
  return aligned(values * sizeof(__nv_bfloat16)) + aligned(values * sizeof(float));
}

void sol_attention_forward(cudaStream_t stream, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v,
                           __nv_bfloat16* out, const AttentionConfig& cfg, Workspace& ws) {
  validate(cfg);
  Workspace::Scope scope(ws);
  const int blocks = (cfg.seq_len + kBlock - 1) / kBlock;
  const size_t values = static_cast<size_t>(blocks) * cfg.num_heads * kDim;
  auto* k_mean = ws.alloc_n<__nv_bfloat16>(values);
  auto* v_sum = ws.alloc_n<float>(values);
  const dim3 pool_grid(blocks, cfg.num_heads);
  pool_kv<<<pool_grid, kDim, 0, stream>>>(k, v, k_mean, v_sum, cfg.seq_len, cfg.num_heads);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
  const dim3 attention_grid(cfg.seq_len, cfg.num_heads);
  sol_forward_kernel<<<attention_grid, kDim, 0, stream>>>(
      q, k, v, k_mean, v_sum, out, cfg.seq_len, cfg.num_heads, cfg.exact_prefix,
      cfg.effective_scale(), cfg.sol_beta);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vidfab::cuda
