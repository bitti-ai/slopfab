// Native memory-bounded Sol-Attn for H3 (BF16, D=128, physical blocks 64).
// One CTA owns a whole query block. Routing is shared by its 64 rows and never
// materialised globally; output fragments stay in registers across key blocks.

#include "vidfab/cuda/sol_attention.cuh"

#include <cuda.h>
#include <cfloat>
#include <cmath>
#include <mma.h>
#include <stdexcept>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {
namespace {
constexpr int B = 64;
constexpr int D = 128;
constexpr int Threads = 256;
constexpr size_t SolSharedBytes = (B * B + B * D) * sizeof(float) +
                                  (B * B + B * D) * sizeof(__nv_bfloat16);
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
                        __nv_bfloat16* km, __nv_bfloat16* vm, float* vs,
                        float* k_residual, float* v_residual,
                        int seq, int heads) {
  const int kb = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int lo = kb * B, hi = min(lo + B, seq);
  const size_t width = size_t(heads) * D;
  const size_t dst = (size_t(kb) * heads + h) * D + d;
  float sk=0,sv=0,sk2=0,sv2=0;
  for (int r = lo; r < hi; ++r) {
    const size_t p = size_t(r) * width + h * D + d;
    sk += __bfloat162float(k[p]);
    sv += __bfloat162float(v[p]);
    const float kx=__bfloat162float(k[p]),vx=__bfloat162float(v[p]);
    sk2+=kx*kx;sv2+=vx*vx;
  }
  km[dst] = __float2bfloat16_rn(sk / float(hi - lo));
  vm[dst] = __float2bfloat16_rn(sv / float(hi - lo));
  vs[dst] = sv;
  __shared__ float kr[D],vr[D];
  const float inv=1.0f/float(hi-lo),mk=sk*inv,mv0=sv*inv;
  kr[d]=fmaxf(0.0f,sk2*inv-mk*mk);
  vr[d]=fmaxf(0.0f,sv2*inv-mv0*mv0);
  __syncthreads();
  for(int n=D/2;n;n>>=1){if(d<n){kr[d]+=kr[d+n];vr[d]+=vr[d+n];}__syncthreads();}
  if(d==0){const size_t b=size_t(kb)*heads+h;
    k_residual[b]=sqrtf(kr[0]/D);v_residual[b]=sqrtf(vr[0]/D);}
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

__global__ void sol(const __grid_constant__ CUtensorMap q_map,
                    const __nv_bfloat16* q, const __nv_bfloat16* k,
                    const __nv_bfloat16* v, const __nv_bfloat16* km, const float* vs,
                    const float* tau, __nv_bfloat16* out, int seq, int heads,
                    int prefix, float scale, unsigned long long* route_counts) {
  const int qb = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  const int warp = t / 32;
  const int qlo = qb * B, qn = min(B, seq - qlo), nblocks = (seq + B - 1) / B;
  const size_t width = size_t(heads) * D;
  extern __shared__ __align__(128) unsigned char storage[];
  float* score = reinterpret_cast<float*>(storage);
  float* acc = score + B * B;
  __nv_bfloat16* prob = reinterpret_cast<__nv_bfloat16*>(acc + B * D);
  __nv_bfloat16* staged_q = prob + B * B;
  __shared__ float qm[D];
  __shared__ float om[B], ol[B], rescale[B], block_m[B], approx_weight[B];
  __shared__ int take;
  __shared__ uint8_t routes[Threads];
  __shared__ alignas(8) uint64_t q_barrier;

  if (t == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(staged_q));
    uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(&q_barrier));
    uint64_t state;
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 "fence.proxy.async.shared::cta;\n"
                 "mbarrier.arrive.expect_tx.shared::cta.b64 %1, [%0], %2;\n"
                 "cp.async.bulk.tensor.3d.shared::cta.global.tile."
                 "mbarrier::complete_tx::bytes [%3], [%4, {%5, %6, %7}], [%0];"
                 : "+r"(bar), "=l"(state)
                 : "r"(B * D * int(sizeof(__nv_bfloat16))), "r"(dst), "l"(&q_map),
                   "r"(0), "r"(h), "r"(qlo)
                 : "memory");
    uint32_t ready;
    do {
      asm volatile("{ .reg .pred p; mbarrier.test_wait.shared::cta.b64 p, [%1], %2; "
                   "selp.b32 %0, 1, 0, p; }"
                   : "=r"(ready) : "r"(bar), "l"(state) : "memory");
    } while (!ready);
#endif
  }
  __syncthreads();
  if (t < D) {
    float x = 0;
    for (int r = 0; r < qn; ++r) x += __bfloat162float(staged_q[r * D + t]);
    qm[t] = x / float(qn);
  }
  if (t < B) { om[t] = -FLT_MAX; ol[t] = 0; }
  __syncthreads();

  for (int od = t; od < B * D; od += Threads) acc[od] = 0;
  __syncthreads();

  for (int route_base = 0; route_base < nblocks; route_base += Threads) {
    const int route_kb = route_base + t;
    if (route_kb < nblocks) {
      float proxy = 0;
#pragma unroll 4
      for (int d = 0; d < D; ++d)
        proxy += qm[d] * __bfloat162float(km[(size_t(route_kb) * heads + h) * D + d]);
      const int route_klo = route_kb * B;
      routes[t] = qlo < prefix || route_klo < prefix || abs(qb - route_kb) <= 1 ||
                  proxy * scale > tau[size_t(qb) * heads + h];
      if (route_counts) atomicAdd(route_counts + (routes[t] ? 0 : 1), 1ull);
    }
    __syncthreads();
    const int route_end = min(route_base + Threads, nblocks);
    for (int kb = route_base; kb < route_end; ++kb) {
    const int klo = kb * B, kn = min(B, seq - klo);
    if (t == 0) {
      take = routes[kb - route_base];
    }
    __syncthreads();

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
            wmma::load_matrix_sync(a, staged_q + qr * D + d, D);
            wmma::load_matrix_sync(b, k + size_t(klo + kr) * width + h * D + d,
                                   width);
            wmma::mma_sync(c, a, b, c);
          }
          for (unsigned i = 0; i < c.num_elements; ++i) c.x[i] *= scale;
          wmma::store_matrix_sync(score + qr * B + kr, c, B, wmma::mem_row_major);
        }
      } else {
        for (int p = t; p < qn * kn; p += Threads) {
          const int qr = p / kn, kr = p % kn;
          float dot = 0;
          for (int d = 0; d < D; ++d)
            dot += __bfloat162float(staged_q[qr * D + d]) *
                   __bfloat162float(k[size_t(klo + kr) * width + h * D + d]);
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
      for (int od = t; od < qn * D; od += Threads) {
        const int qr = od / D;
        acc[od] *= rescale[qr];
      }
      if (qn == B && kn == B) {
        for (int p = t; p < B * B; p += Threads)
          prob[p] = __float2bfloat16_rn(expf(score[p] - block_m[p / B]));
        __syncthreads();
        using namespace nvcuda;
        for (int tile = warp; tile < 32; tile += Threads / 32) {
          const int qr = (tile / 8) * 16, d = (tile % 8) * 16;
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
          wmma::load_matrix_sync(c, acc + qr * D + d, D, wmma::mem_row_major);
#pragma unroll
          for (int kr = 0; kr < B; kr += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                           wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                           wmma::row_major> b;
            wmma::load_matrix_sync(a, prob + qr * B + kr, B);
            wmma::load_matrix_sync(b, v + size_t(klo + kr) * width + h * D + d,
                                   width);
            wmma::mma_sync(c, a, b, c);
          }
          wmma::store_matrix_sync(acc + qr * D + d, c, D, wmma::mem_row_major);
        }
      } else {
        for (int od = t; od < qn * D; od += Threads) {
          const int qr = od / D, d = od % D;
          float add = 0;
          for (int kr = 0; kr < kn; ++kr)
            add += expf(score[qr * B + kr] - block_m[qr]) *
                   __bfloat162float(v[size_t(klo + kr) * width + h * D + d]);
          acc[od] += add;
        }
      }
    } else {
      if (t < qn) {
        float dot = 0;
        for (int d = 0; d < D; ++d)
          dot += __bfloat162float(staged_q[t * D + d]) *
                 __bfloat162float(km[(size_t(kb) * heads + h) * D + d]);
        score[t * B] = dot * scale;
        const float nm = fmaxf(om[t], score[t * B]);
        rescale[t] = expf(om[t] - nm);
        block_m[t] = nm;
        approx_weight[t] = expf(score[t * B] - nm);
        ol[t] = ol[t] * rescale[t] + float(kn) * approx_weight[t];
        om[t] = nm;
      }
      __syncthreads();
      for (int od = t; od < qn * D; od += Threads) {
        const int qr = od / D, d = od % D;
        acc[od] = acc[od] * rescale[qr] +
                  approx_weight[qr] * vs[(size_t(kb) * heads + h) * D + d];
      }
    }
    __syncthreads();
    }
  }
  for (int od = t; od < qn * D; od += Threads) {
    if (od < qn * D) {
      const int qr = od / D, d = od % D;
      out[size_t(qlo + qr) * width + h * D + d] = __float2bfloat16_rn(acc[od] / ol[qr]);
    }
  }
}

void validate(const AttentionConfig& c) {
  if (c.seq_len <= 0 || c.num_heads <= 0) throw std::runtime_error("Sol-Attn: invalid shape");
  if (c.head_dim != D) throw std::runtime_error("Sol-Attn: requires head_dim 128");
  if (c.band_ranges) throw std::runtime_error("Sol-Attn: frame banding is incompatible");
  if (c.exact_prefix < 0 || c.exact_prefix > c.seq_len) throw std::runtime_error("Sol-Attn: invalid exact_prefix");
  if (!std::isfinite(c.sol_beta)) throw std::runtime_error("Sol-Attn: sol_beta must be finite");
  if (!std::isfinite(c.sol_error_k) || c.sol_error_k < 0.0f)
    throw std::runtime_error("Sol-Attn: sol_error_k must be finite and nonnegative");
  if (!std::isfinite(c.sol_error_v) || c.sol_error_v < 0.0f)
    throw std::runtime_error("Sol-Attn: sol_error_v must be finite and nonnegative");
}

CUtensorMap make_q_map(const __nv_bfloat16* q, const AttentionConfig& c) {
  CUtensorMap map{};
  const cuuint64_t dims[3] = {D, static_cast<cuuint64_t>(c.num_heads),
                              static_cast<cuuint64_t>(c.seq_len)};
  const cuuint64_t strides[2] = {D * sizeof(__nv_bfloat16),
                                 cuuint64_t(c.num_heads) * D * sizeof(__nv_bfloat16)};
  const cuuint32_t box[3] = {D, 1, B};
  const cuuint32_t elem[3] = {1, 1, 1};
  const CUresult status = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(q), dims,
      strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    throw std::runtime_error(std::string("Sol-Attn: Q tensor map: ") +
                             (message ? message : "CUDA driver error"));
  }
  return map;
}
}  // namespace

size_t sol_attention_workspace_bytes(const AttentionConfig& c) {
  if (c.seq_len <= 0 || c.num_heads <= 0 || c.head_dim != D) return 0;
  const size_t nb = (size_t(c.seq_len) + B - 1) / B;
  const size_t pooled = nb * c.num_heads * D;
  return 2*aligned(pooled * sizeof(__nv_bfloat16)) + aligned(pooled * sizeof(float)) +
         2*aligned(nb*c.num_heads*sizeof(float))+
         2 * aligned(size_t(c.num_heads) * D * sizeof(float)) +
         aligned(nb * c.num_heads * sizeof(float));
}

void sol_attention_forward(cudaStream_t stream, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v,
                           __nv_bfloat16* out, const AttentionConfig& c, Workspace& ws) {
  if (current_device_compute_capability() != 120) {
    throw std::runtime_error("Sol-Attn requires the shipped Blackwell sm_120 image");
  }
  validate(c);
  Workspace::Scope scope(ws);
  const int nb = (c.seq_len + B - 1) / B;
  const size_t pooled = size_t(nb) * c.num_heads * D;
  auto* km = ws.alloc_n<__nv_bfloat16>(pooled);
  auto* vm = ws.alloc_n<__nv_bfloat16>(pooled);
  auto* vs = ws.alloc_n<float>(pooled);
  auto* k_residual=ws.alloc_n<float>(size_t(nb)*c.num_heads);
  auto* v_residual=ws.alloc_n<float>(size_t(nb)*c.num_heads);
  auto* key_mean = ws.alloc_n<float>(size_t(c.num_heads) * D);
  auto* key_var = ws.alloc_n<float>(size_t(c.num_heads) * D);
  auto* tau = ws.alloc_n<float>(size_t(nb) * c.num_heads);
  const CUtensorMap q_map = make_q_map(q, c);
  cudaEvent_t phase[5]{};
  if (c.sol_phase_ms) {
    for (auto& event : phase) VIDFAB_CUDA_CHECK(cudaEventCreate(&event));
    VIDFAB_CUDA_CHECK(cudaEventRecord(phase[0], stream));
  }
  pool_kv<<<dim3(nb, c.num_heads), D, 0, stream>>>(k, v, km, vm, vs,k_residual,v_residual,
                                                   c.seq_len, c.num_heads);
  if (c.sol_phase_ms) VIDFAB_CUDA_CHECK(cudaEventRecord(phase[1], stream));
  key_stats<<<c.num_heads, D, 0, stream>>>(km, key_mean, key_var, nb, c.num_heads);
  if (c.sol_phase_ms) VIDFAB_CUDA_CHECK(cudaEventRecord(phase[2], stream));
  thresholds<<<dim3(nb, c.num_heads), Threads, 0, stream>>>(
      q, key_mean, key_var, tau, c.seq_len, c.num_heads, c.effective_scale(), c.sol_beta);
  if (c.sol_phase_ms) VIDFAB_CUDA_CHECK(cudaEventRecord(phase[3], stream));
  const bool pipeline_ran = c.sol_pipeline &&
      sol_pipeline_forward(stream,q,k,v,km,vm,vs,k_residual,v_residual,tau,out,c);
  if (c.sol_pipeline && !pipeline_ran) {
    throw std::runtime_error(
        "Sol-Attn experimental pipeline unavailable: requires SM120, head_dim=128, "
        "at most 1024 blocks, and aligned TMA-compatible Q/K/V tensors");
  }
  if (!pipeline_ran) {
    VIDFAB_CUDA_CHECK(cudaFuncSetAttribute(sol, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          int(SolSharedBytes)));
    sol<<<dim3(nb, c.num_heads), Threads, SolSharedBytes, stream>>>(
        q_map, q, k, v, km, vs, tau, out, c.seq_len, c.num_heads, c.exact_prefix,
        c.effective_scale(), c.sol_route_counts);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
  if (c.sol_phase_ms) {
    VIDFAB_CUDA_CHECK(cudaEventRecord(phase[4], stream));
    VIDFAB_CUDA_CHECK(cudaEventSynchronize(phase[4]));
    for (int i = 0; i < 4; ++i) {
      VIDFAB_CUDA_CHECK(cudaEventElapsedTime(c.sol_phase_ms + i, phase[i], phase[i + 1]));
      cudaEventDestroy(phase[i]);
    }
    cudaEventDestroy(phase[4]);
  }
}
}  // namespace vidfab::cuda
