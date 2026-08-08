// Native memory-bounded Sol-Attn for H3 (BF16, D=128, physical blocks 64).
// One CTA owns a whole query block. Routing is shared by its 64 rows and never
// materialised globally; output fragments stay in registers across key blocks.

#include "vidfab/cuda/sol_attention.cuh"

#include <cfloat>
#include <cmath>
#include <mma.h>
#include <stdexcept>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {
namespace {
constexpr int B = 64;
constexpr int D = 128;
// SM120's warp MMA path uses the same four-warp CTA shape as the released
// CuTe kernel.  Besides reducing CTA-wide synchronization cost, this gives
// each thread stable ownership of 64 output values for the whole mainloop.
constexpr int Threads = 128;
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

// Released H3 diagonal estimator over pooled keys: population mean and
// diagonal population variance, once per head and request.
__global__ void key_stats(const __nv_bfloat16* km, float* key_mean, float* key_var,
                          int nblocks, int heads) {
  const int h = blockIdx.x, d = threadIdx.x;
  float mean = 0, m2 = 0;
  for (int kb = 0; kb < nblocks; ++kb) {
    const float x = __bfloat162float(km[(size_t(kb) * heads + h) * D + d]);
    const float delta = x - mean;
    mean += delta / float(kb + 1);
    m2 += delta * (x - mean);
  }
  key_mean[h * D + d] = mean;
  key_var[h * D + d] = m2 / float(nblocks);
}

// tau = scale * (qbar.meanK + beta * sqrt(qbar^2.diagVarK)).
__global__ void thresholds(const __nv_bfloat16* q, const float* key_mean,
                           const float* key_var, float* tau, int seq, int heads,
                           float scale, float beta) {
  const int qb = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  const int qlo = qb * B, qhi = min(qlo + B, seq);
  const size_t width = size_t(heads) * D;
  __shared__ float scratch[D];
  if (t < D) {
    float x = 0;
    for (int r = qlo; r < qhi; ++r) x += __bfloat162float(q[size_t(r) * width + h * D + t]);
    const float qm = x / float(qhi - qlo);
    scratch[t] = qm * key_mean[h * D + t];
  }
  __syncthreads();
  float mu = t < D ? scratch[t] : 0;
  mu = reduce128(mu, scratch);
  // Recompute qbar after the reduction because scratch is the reduction arena.
  float variance_term = 0;
  if (t < D) {
    float x = 0;
    for (int r = qlo; r < qhi; ++r) x += __bfloat162float(q[size_t(r) * width + h * D + t]);
    const float qm = x / float(qhi - qlo);
    variance_term = qm * qm * key_var[h * D + t];
  }
  const float var = reduce128(variance_term, scratch);
  if (t == 0) tau[size_t(qb) * heads + h] = scale * (mu + beta * sqrtf(fmaxf(var, 0.0f)));
}

__global__ void sol(const __nv_bfloat16* q, const __nv_bfloat16* k,
                    const __nv_bfloat16* v, const __nv_bfloat16* km, const float* vs,
                    const float* tau, __nv_bfloat16* out, int seq, int heads,
                    int prefix, float scale, unsigned long long* route_counts) {
  const int qb = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  const int qlo = qb * B, qn = min(B, seq - qlo), nblocks = (seq + B - 1) / B;
  const size_t width = size_t(heads) * D;
  const int warp = t / 32;
  __shared__ float qm[D], score[B * B];
  __shared__ float om[B], ol[B], rescale[B], block_m[B];
  // Evaluate a whole route group in parallel.  The old implementation used
  // a 128-thread reduction (and seven barriers) for every physical block.
  // This is the native equivalent of the reference kernel's CTA-local route
  // tile: one thread owns one pooled-key dot product, then the mainloop reads
  // the compact byte table without another reduction.
  __shared__ unsigned char route[Threads];

  if (t < D) {
    float x = 0;
    for (int r = 0; r < qn; ++r) x += __bfloat162float(q[size_t(qlo + r) * width + h * D + t]);
    qm[t] = x / float(qn);
  }
  if (t < B) { om[t] = -FLT_MAX; ol[t] = 0; }
  __syncthreads();

  // 8192 output scalars / 128 threads = 64 register accumulators per thread.
  float acc[64];
#pragma unroll
  for (int slot = 0; slot < 64; ++slot) acc[slot] = 0;

  for (int route_base = 0; route_base < nblocks; route_base += Threads) {
    const int route_kb = route_base + t;
    if (route_kb < nblocks) {
      float proxy = 0;
#pragma unroll 4
      for (int d = 0; d < D; ++d)
        proxy += qm[d] * __bfloat162float(
            km[(size_t(route_kb) * heads + h) * D + d]);
      const int klo = route_kb * B;
      route[t] = qlo < prefix || klo < prefix || abs(qb - route_kb) <= 1 ||
                 proxy * scale > tau[size_t(qb) * heads + h];
      if (route_counts) atomicAdd(route_counts + (route[t] ? 0 : 1), 1ull);
    }
    __syncthreads();

    const int route_end = min(route_base + Threads, nblocks);
    for (int kb = route_base; kb < route_end; ++kb) {
    const int klo = kb * B, kn = min(B, seq - klo);
    const bool take = route[kb - route_base] != 0;

    if (take) {
      if (qn == B && kn == B) {
        using namespace nvcuda;
        for (int tile = warp; tile < 16; tile += Threads / 32) {
          const int qr = (tile / 4) * 16, kr = (tile % 4) * 16;
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
          wmma::fill_fragment(c, 0.0f);
#pragma unroll
          for (int d = 0; d < D; d += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                           wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                           wmma::col_major> b;
            wmma::load_matrix_sync(a, q + size_t(qlo + qr) * width + h * D + d,
                                   width);
            wmma::load_matrix_sync(b, k + size_t(klo + kr) * width + h * D + d,
                                   width);
            wmma::mma_sync(c, a, b, c);
          }
          for (unsigned i = 0; i < c.num_elements; ++i) c.x[i] *= scale;
          wmma::store_matrix_sync(score + qr * B + kr, c, B,
                                  wmma::mem_row_major);
        }
      } else {
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
#pragma unroll 64
      for (int slot = 0; slot < 64; ++slot) {
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
#pragma unroll 64
      for (int slot = 0; slot < 64; ++slot) {
        const int od = t + slot * Threads;
        if (od >= qn * D) continue;
        const int qr = od / D, d = od % D;
        acc[slot] = acc[slot] * rescale[qr] +
                    expf(score[qr * B] - block_m[qr]) * vs[(size_t(kb) * heads + h) * D + d];
      }
    }
    __syncthreads();
    }
  }
#pragma unroll 64
  for (int slot = 0; slot < 64; ++slot) {
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
         2 * aligned(size_t(c.num_heads) * D * sizeof(float)) +
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
  auto* key_mean = ws.alloc_n<float>(size_t(c.num_heads) * D);
  auto* key_var = ws.alloc_n<float>(size_t(c.num_heads) * D);
  auto* tau = ws.alloc_n<float>(size_t(nb) * c.num_heads);
  pool_kv<<<dim3(nb, c.num_heads), D, 0, stream>>>(k, v, km, vs, c.seq_len, c.num_heads);
  key_stats<<<c.num_heads, D, 0, stream>>>(km, key_mean, key_var, nb, c.num_heads);
  thresholds<<<dim3(nb, c.num_heads), Threads, 0, stream>>>(
      q, key_mean, key_var, tau, c.seq_len, c.num_heads, c.effective_scale(), c.sol_beta);
  sol<<<dim3(nb, c.num_heads), Threads, 0, stream>>>(q, k, v, km, vs, tau, out, c.seq_len,
                                                     c.num_heads, c.exact_prefix,
                                                     c.effective_scale(), c.sol_route_counts);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}
}  // namespace vidfab::cuda
