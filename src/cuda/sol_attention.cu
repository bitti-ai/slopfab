// Native memory-bounded Sol-Attn for H3 (BF16, D=128, physical blocks 64).
// One CTA owns a whole query block. Routing is shared by its 64 rows and never
// materialised globally; output fragments stay in registers across key blocks.

#include "vidfab/cuda/sol_attention.cuh"

#include <cfloat>
#include <cmath>
#include <stdexcept>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {
namespace {
constexpr int B = 64;
constexpr int D = 128;
constexpr int Threads = 256;
constexpr size_t Align = 256;
size_t aligned(size_t n) { return (n + Align - 1) & ~(Align - 1); }

__device__ float reduce128(float x, float* s) {
  const int t = threadIdx.x;
  if (t < D) s[t] = x;
  __syncthreads();
  for (int n = D / 2; n; n >>= 1) {
    if (t < n) s[t] += s[t + n];
    __syncthreads();
  }
  return s[0];
}

__global__ void pool_kv(const __nv_bfloat16* k, const __nv_bfloat16* v,
                        __nv_bfloat16* km, float* vs, int seq, int heads) {
  const int kb = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int lo = kb * B, hi = min(lo + B, seq);
  const size_t width = size_t(heads) * D;
  const size_t dst = (size_t(kb) * heads + h) * D + d;
  float sk = 0, sv = 0;
  for (int r = lo; r < hi; ++r) {
    const size_t p = size_t(r) * width + h * D + d;
    sk += __bfloat162float(k[p]);
    sv += __bfloat162float(v[p]);
  }
  km[dst] = __float2bfloat16_rn(sk / float(hi - lo));
  vs[dst] = sv;
}

// The diagonal-estimator threshold is produced once per (head, query block),
// rather than once per query row. Its small output is not a routing map.
__global__ void thresholds(const __nv_bfloat16* q, const __nv_bfloat16* km, float* tau,
                           int seq, int heads, float scale, float beta) {
  const int qb = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  const int qlo = qb * B, qhi = min(qlo + B, seq);
  const int nblocks = (seq + B - 1) / B;
  const size_t width = size_t(heads) * D;
  __shared__ float qm[D], scratch[D], mean, m2;
  if (t < D) {
    float x = 0;
    for (int r = qlo; r < qhi; ++r) x += __bfloat162float(q[size_t(r) * width + h * D + t]);
    qm[t] = x / float(qhi - qlo);
  }
  if (t == 0) mean = m2 = 0;
  __syncthreads();
  for (int kb = 0; kb < nblocks; ++kb) {
    float x = t < D ? qm[t] * __bfloat162float(km[(size_t(kb) * heads + h) * D + t]) : 0;
    const float proxy = reduce128(x, scratch) * scale;
    if (t == 0) {
      const float delta = proxy - mean;
      mean += delta / float(kb + 1);
      m2 += delta * (proxy - mean);
    }
    __syncthreads();
  }
  if (t == 0) tau[size_t(qb) * heads + h] = mean + beta * sqrtf(m2 / float(nblocks));
}

__global__ void sol(const __nv_bfloat16* q, const __nv_bfloat16* k,
                    const __nv_bfloat16* v, const __nv_bfloat16* km, const float* vs,
                    const float* tau, __nv_bfloat16* out, int seq, int heads,
                    int prefix, float scale) {
  const int qb = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  const int qlo = qb * B, qn = min(B, seq - qlo), nblocks = (seq + B - 1) / B;
  const size_t width = size_t(heads) * D;
  __shared__ float qm[D], red[D], score[B * B];
  __shared__ float om[B], ol[B], rescale[B], block_m[B];
  __shared__ int take;

  if (t < D) {
    float x = 0;
    for (int r = 0; r < qn; ++r) x += __bfloat162float(q[size_t(qlo + r) * width + h * D + t]);
    qm[t] = x / float(qn);
  }
  if (t < B) { om[t] = -FLT_MAX; ol[t] = 0; }
  __syncthreads();

  // 8192 output scalars / 256 threads = 32 register accumulators per thread.
  float acc[32];
#pragma unroll
  for (float& x : acc) x = 0;

  for (int kb = 0; kb < nblocks; ++kb) {
    const int klo = kb * B, kn = min(B, seq - klo);
    float px = t < D ? qm[t] * __bfloat162float(km[(size_t(kb) * heads + h) * D + t]) : 0;
    const float proxy = reduce128(px, red) * scale;
    if (t == 0) {
      // Prefix queries are dense. Prefix/sink keys and immediate physical
      // neighbours are invariant exact routes in the official H3 policy.
      take = qlo < prefix || klo < prefix || abs(qb - kb) <= 1 ||
             proxy > tau[size_t(qb) * heads + h];
    }
    __syncthreads();

    if (take) {
      for (int p = t; p < qn * kn; p += Threads) {
        const int qr = p / kn, kr = p % kn;
        float dot = 0;
#pragma unroll 4
        for (int d = 0; d < D; ++d) {
          dot += __bfloat162float(q[size_t(qlo + qr) * width + h * D + d]) *
                 __bfloat162float(k[size_t(klo + kr) * width + h * D + d]);
        }
        score[qr * B + kr] = dot * scale;
      }
      __syncthreads();
      if (t < qn) {
        float bm = -FLT_MAX;
        for (int kr = 0; kr < kn; ++kr) bm = fmaxf(bm, score[t * B + kr]);
        const float nm = fmaxf(om[t], bm);
        float sum = 0;
        for (int kr = 0; kr < kn; ++kr) sum += expf(score[t * B + kr] - nm);
        rescale[t] = expf(om[t] - nm);
        block_m[t] = nm;
        ol[t] = ol[t] * rescale[t] + sum;
        om[t] = nm;
      }
      __syncthreads();
      for (int slot = 0; slot < 32; ++slot) {
        const int od = t + slot * Threads;
        if (od >= qn * D) continue;
        const int qr = od / D, d = od % D;
        float add = 0;
        for (int kr = 0; kr < kn; ++kr)
          add += expf(score[qr * B + kr] - block_m[qr]) *
                 __bfloat162float(v[size_t(klo + kr) * width + h * D + d]);
        acc[slot] = acc[slot] * rescale[qr] + add;
      }
    } else {
      if (t < qn) {
        float dot = 0;
        for (int d = 0; d < D; ++d)
          dot += __bfloat162float(q[size_t(qlo + t) * width + h * D + d]) *
                 __bfloat162float(km[(size_t(kb) * heads + h) * D + d]);
        score[t * B] = dot * scale;
        const float nm = fmaxf(om[t], score[t * B]);
        rescale[t] = expf(om[t] - nm);
        block_m[t] = nm;
        ol[t] = ol[t] * rescale[t] + float(kn) * expf(score[t * B] - nm);
        om[t] = nm;
      }
      __syncthreads();
      for (int slot = 0; slot < 32; ++slot) {
        const int od = t + slot * Threads;
        if (od >= qn * D) continue;
        const int qr = od / D, d = od % D;
        acc[slot] = acc[slot] * rescale[qr] +
                    expf(score[qr * B] - block_m[qr]) * vs[(size_t(kb) * heads + h) * D + d];
      }
    }
    __syncthreads();
  }
  for (int slot = 0; slot < 32; ++slot) {
    const int od = t + slot * Threads;
    if (od < qn * D) {
      const int qr = od / D, d = od % D;
      out[size_t(qlo + qr) * width + h * D + d] = __float2bfloat16_rn(acc[slot] / ol[qr]);
    }
  }
}

void validate(const AttentionConfig& c) {
  if (c.seq_len <= 0 || c.num_heads <= 0) throw std::runtime_error("Sol-Attn: invalid shape");
  if (c.head_dim != D) throw std::runtime_error("Sol-Attn: requires head_dim 128");
  if (c.band_ranges) throw std::runtime_error("Sol-Attn: frame banding is incompatible");
  if (c.exact_prefix < 0 || c.exact_prefix > c.seq_len) throw std::runtime_error("Sol-Attn: invalid exact_prefix");
  if (!std::isfinite(c.sol_beta)) throw std::runtime_error("Sol-Attn: sol_beta must be finite");
}
}  // namespace

size_t sol_attention_workspace_bytes(const AttentionConfig& c) {
  if (c.seq_len <= 0 || c.num_heads <= 0 || c.head_dim != D) return 0;
  const size_t nb = (size_t(c.seq_len) + B - 1) / B;
  const size_t pooled = nb * c.num_heads * D;
  return aligned(pooled * sizeof(__nv_bfloat16)) + aligned(pooled * sizeof(float)) +
         aligned(nb * c.num_heads * sizeof(float));
}

void sol_attention_forward(cudaStream_t stream, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v,
                           __nv_bfloat16* out, const AttentionConfig& c, Workspace& ws) {
  validate(c);
  Workspace::Scope scope(ws);
  const int nb = (c.seq_len + B - 1) / B;
  const size_t pooled = size_t(nb) * c.num_heads * D;
  auto* km = ws.alloc_n<__nv_bfloat16>(pooled);
  auto* vs = ws.alloc_n<float>(pooled);
  auto* tau = ws.alloc_n<float>(size_t(nb) * c.num_heads);
  pool_kv<<<dim3(nb, c.num_heads), D, 0, stream>>>(k, v, km, vs, c.seq_len, c.num_heads);
  thresholds<<<dim3(nb, c.num_heads), Threads, 0, stream>>>(q, km, tau, c.seq_len, c.num_heads,
                                                            c.effective_scale(), c.sol_beta);
  sol<<<dim3(nb, c.num_heads), Threads, 0, stream>>>(q, k, v, km, vs, tau, out, c.seq_len,
                                                     c.num_heads, c.exact_prefix,
                                                     c.effective_scale());
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}
}  // namespace vidfab::cuda
