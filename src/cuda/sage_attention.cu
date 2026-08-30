// Standalone SageAttention2.2 integration. The attention kernel and its CUDA
// primitives are adapted from THU-ML/SageAttention (Apache-2.0); see
// third_party/sageattention/LICENSE. The PyTorch wrapper is deliberately not
// included: vidfab supplies raw NHD pointers and owns the transient arena.
#include "vidfab/cuda/sage_attention.cuh"

#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/workspace.cuh"
#include "../../third_party/sageattention/qattn/qk_int_sv_f16_cuda_sm80.cuh"
#undef PACK_SIZE_QK
#undef PACK_SIZE_V
#undef PACK_SIZE_O
#undef MMA_QK_M
#undef MMA_QK_N
#undef MMA_QK_K
#undef MMA_SV_M
#undef MMA_SV_N
#undef MMA_SV_K
#include "../../third_party/sageattention/qattn/qk_int_sv_f8_cuda_sm89.cuh"

namespace vidfab::cuda {
namespace {

constexpr int kQBlock = 128;
constexpr int kQWarp = 32;
constexpr int kKBlock = 64;
constexpr int kThreads = 256;

size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }
int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct SageBuffers {
  int8_t* q;
  int8_t* k;
  void* v;
  float* qs;
  float* ks;
  float* vs;
  float* km;
};

size_t buffer_bytes(const AttentionConfig& c, int kvh, bool fp16_v) {
  const size_t qn = static_cast<size_t>(c.seq_len) * c.num_heads * c.head_dim;
  const size_t kn = static_cast<size_t>(c.seq_len) * kvh * c.head_dim;
  const size_t padded = static_cast<size_t>(ceil_div(c.seq_len, kKBlock)) * kKBlock;
  const size_t vn = padded * kvh * c.head_dim;
  const size_t qsn = static_cast<size_t>(c.num_heads) * ceil_div(c.seq_len, kQBlock) *
                     (kQBlock / kQWarp);
  const size_t ksn = static_cast<size_t>(kvh) * ceil_div(c.seq_len, kKBlock);
  const size_t channel = static_cast<size_t>(kvh) * c.head_dim;
  return align256(qn) + align256(kn) + align256(vn * (fp16_v ? sizeof(__half) : 1)) +
         align256(qsn * sizeof(float)) +
         align256(ksn * sizeof(float)) + 2 * align256(channel * sizeof(float));
}

SageBuffers carve(void* base, const AttentionConfig& c, int kvh, bool fp16_v) {
  auto* p = static_cast<uint8_t*>(base);
  auto take = [&](size_t n) { void* r = p; p += align256(n); return r; };
  const size_t qn = static_cast<size_t>(c.seq_len) * c.num_heads * c.head_dim;
  const size_t kn = static_cast<size_t>(c.seq_len) * kvh * c.head_dim;
  const size_t padded = static_cast<size_t>(ceil_div(c.seq_len, kKBlock)) * kKBlock;
  const size_t vn = padded * kvh * c.head_dim;
  const size_t qsn = static_cast<size_t>(c.num_heads) * ceil_div(c.seq_len, kQBlock) * 4;
  const size_t ksn = static_cast<size_t>(kvh) * ceil_div(c.seq_len, kKBlock);
  const size_t channel = static_cast<size_t>(kvh) * c.head_dim;
  SageBuffers b{};
  b.q = static_cast<int8_t*>(take(qn));
  b.k = static_cast<int8_t*>(take(kn));
  b.v = take(vn * (fp16_v ? sizeof(__half) : 1));
  b.qs = static_cast<float*>(take(qsn * sizeof(float)));
  b.ks = static_cast<float*>(take(ksn * sizeof(float)));
  b.vs = static_cast<float*>(take(channel * sizeof(float)));
  b.km = static_cast<float*>(take(channel * sizeof(float)));
  return b;
}

__device__ float warp_max(float x) {
  for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d));
  return x;
}

__device__ float warp_sum(float x) {
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
  return x;
}

// One block computes one (head, channel) sequence mean. K smoothing is exact
// in fp32 and does not alter softmax output: it subtracts one query-dependent
// constant from every logit in a row.
__global__ void key_mean(const __nv_bfloat16* k, float* mean, int seq, int heads, int dim) {
  const int hd = blockIdx.x;
  const int h = hd / dim, d = hd % dim;
  float sum = 0.0f;
  for (int s = threadIdx.x; s < seq; s += blockDim.x)
    sum += __bfloat162float(k[(static_cast<size_t>(s) * heads + h) * dim + d]);
  __shared__ float sm[8];
  sum = warp_sum(sum);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? sm[threadIdx.x] : 0.0f;
    for (int dlt = 16; dlt; dlt >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, dlt);
    if (threadIdx.x == 0) mean[hd] = sum / seq;
  }
}

template <int Rows, bool Smooth>
__global__ void quant_qk(const __nv_bfloat16* in, int8_t* out, float* scales,
                         const float* mean, int seq, int heads, int dim, int groups) {
  const int group = blockIdx.x;
  const int h = blockIdx.y;
  const int s0 = group * Rows;
  float amax = 0.0f;
  for (int i = threadIdx.x; i < Rows * dim; i += blockDim.x) {
    const int s = s0 + i / dim, d = i % dim;
    if (s < seq) {
      float x = __bfloat162float(in[(static_cast<size_t>(s) * heads + h) * dim + d]);
      if constexpr (Smooth) x -= mean[h * dim + d];
      amax = fmaxf(amax, fabsf(x));
    }
  }
  __shared__ float sm[8];
  amax = warp_max(amax);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    amax = threadIdx.x < 8 ? sm[threadIdx.x] : 0.0f;
    amax = warp_max(amax);
    if (threadIdx.x == 0) sm[0] = fmaxf(amax / 127.0f, 1.0e-8f);
  }
  __syncthreads();
  const float scale = sm[0];
  if (threadIdx.x == 0) scales[h * groups + group] = scale;
  for (int i = threadIdx.x; i < Rows * dim; i += blockDim.x) {
    const int s = s0 + i / dim, d = i % dim;
    if (s < seq) {
      float x = __bfloat162float(in[(static_cast<size_t>(s) * heads + h) * dim + d]);
      if constexpr (Smooth) x -= mean[h * dim + d];
      out[(static_cast<size_t>(s) * heads + h) * dim + d] =
          static_cast<int8_t>(__float2int_rn(fminf(127.0f, fmaxf(-127.0f, x / scale))));
    }
  }
}

// SageAttention2.2's sm120 route uses scale_max=2.25 for its FP32+FP16 PV
// accumulator. Output is transposed to [D,H,padded_sequence] as the official
// tensor-core kernel expects.
__global__ void quant_v(const __nv_bfloat16* v, int8_t* out, float* scales,
                        int seq, int padded, int heads, int dim) {
  const int hd = blockIdx.x;
  const int h = hd / dim, d = hd % dim;
  float amax = 0.0f;
  for (int s = threadIdx.x; s < seq; s += blockDim.x)
    amax = fmaxf(amax, fabsf(__bfloat162float(v[(static_cast<size_t>(s) * heads + h) * dim + d])));
  __shared__ float sm[8];
  amax = warp_max(amax);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    amax = threadIdx.x < 8 ? sm[threadIdx.x] : 0.0f;
    amax = warp_max(amax);
    if (threadIdx.x == 0) sm[0] = fmaxf(amax / 2.25f, 1.0e-8f);
  }
  __syncthreads();
  const float scale = sm[0];
  if (threadIdx.x == 0) scales[hd] = scale;
  int8_t* dst = out + (static_cast<size_t>(d) * heads + h) * padded;
  for (int s = threadIdx.x; s < padded; s += blockDim.x) {
    float x = s < seq ? __bfloat162float(v[(static_cast<size_t>(s) * heads + h) * dim + d]) / scale : 0.0f;
    // The upstream FP8 MMA expects the sequence dimension permuted inside
    // each 16-row group: 0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15.
    const int base = s & ~15;
    const int r = s & 15;
    const int perm = (r / 8) * 2 + ((r / 2) & 3) * 4 + (r & 1);
    dst[base + perm] = static_cast<int8_t>(__nv_fp8_e4m3(x).__x);
  }
}

// Ampere has INT8 and FP16 tensor cores but no FP8 tensor cores. Keep V in
// ordinary sequence-major FP16 for SageAttention's SM80 P*V kernel. Unlike the
// Blackwell preparation above this needs neither per-channel scales nor the
// FP8 lane permutation.
__global__ void convert_v_f16(const __nv_bfloat16* v, __half* out, size_t n) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x)
    out[i] = __float2half(__bfloat162float(v[i]));
}

template <int D>
void launch_official(cudaStream_t stream, const SageBuffers& b, __nv_bfloat16* out,
                     const AttentionConfig& c, int kvh, int padded) {
  constexpr int CTA_Q = 128, CTA_K = 64, WARP_Q = 32, WARP_K = 64;
  using KernelOut = nv_bfloat16;
  auto kernel = qk_int_sv_f8_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, WARP_K, D, DataType::kInt8,
      QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, float, true,
      KernelOut, ComputeUnit::kCudaCore, MaskMode::kNone, false, true, false, true>;
  const size_t smem = std::max<size_t>(CTA_Q * D + CTA_K * D + CTA_K * D,
                                      CTA_Q * D * sizeof(__half));
  VIDFAB_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem)));
  dim3 grid(ceil_div(c.seq_len, CTA_Q), c.num_heads, 1);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
  const int groups = c.num_heads / kvh;
  kernel<<<grid, block, smem, stream>>>(
      b.q, b.k, static_cast<int8_t*>(b.v), out, nullptr, b.qs, b.ks, b.vs, nullptr,
      c.seq_len, c.seq_len, groups,
      c.seq_len * c.num_heads * D, c.num_heads * D, D,
      c.seq_len * kvh * D, kvh * D, D,
      padded * kvh * D, padded, padded * kvh,
      c.seq_len * c.num_heads * D, c.num_heads * D, D,
      c.effective_scale());
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

template <int D>
void launch_ampere(cudaStream_t stream, const SageBuffers& b, __nv_bfloat16* out,
                   const AttentionConfig& c, int kvh) {
  constexpr int CTA_Q = 128, CTA_K = 64, WARP_Q = 32, WARP_K = 64;
  using KernelOut = nv_bfloat16;
  auto kernel = qk_int_sv_f16_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, WARP_K, D, DataType::kInt8,
      QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, float, true,
      KernelOut, ComputeUnit::kCudaCore, MaskMode::kNone, false, false>;
  const size_t smem = std::max<size_t>((CTA_Q + CTA_K) * D,
                                      CTA_K * D * sizeof(__half));
  VIDFAB_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem)));
  dim3 grid(ceil_div(c.seq_len, CTA_Q), c.num_heads, 1);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
  const int groups = c.num_heads / kvh;
  kernel<<<grid, block, smem, stream>>>(
      b.q, b.k, static_cast<__half*>(b.v), out, nullptr, b.qs, b.ks, nullptr,
      c.seq_len, c.seq_len, groups,
      c.seq_len * c.num_heads * D, c.num_heads * D, D,
      c.seq_len * kvh * D, kvh * D, D,
      c.seq_len * kvh * D, kvh * D, D,
      c.seq_len * c.num_heads * D, c.num_heads * D, D,
      c.effective_scale());
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

// Compute capability, cached for the process. A device's capability cannot
// change, but `cudaGetDeviceProperties` is a host driver round-trip, and
// `sage2_supported` is on the per-call attention path — one round-trip per
// block per step, 2500 in a fifty-step generation. A device whose query fails
// reports 0, which is the same "not supported" the direct call gave.
int compute_capability(int device) {
  // Only a successful query is cached. The table latches for the process, so
  // caching a failure would turn one transient driver error into "this card
  // does not support sage2" for the rest of the run — a failed query is
  // retried, and only the answer is permanent.
  //
  // A fixed table rather than a growable one on purpose: every write stores the
  // same value the query always returns for that device, and there is no
  // reallocation, so two threads racing here cannot observe a torn or moved
  // entry. A device index past the end simply pays the driver call each time.
  constexpr int kMaxCached = 64;
  static int caps[kMaxCached] = {0};
  if (device < 0) return 0;
  if (device < kMaxCached && caps[device] != 0) return caps[device];
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
  const int cap = prop.major * 10 + prop.minor;
  if (device < kMaxCached) caps[device] = cap;
  return cap;
}

}  // namespace

bool sage2_supported(const AttentionConfig& cfg, int device, const char** reason) {
  static const char* kDim = "head_dim must be 64 or 128";
  static const char* kBand = "frame-banded attention is not implemented for SageAttention2";
  static const char* kArch = "requires compute capability 8.0 or newer";
  const char* why = nullptr;
  if (cfg.head_dim != 64 && cfg.head_dim != 128) why = kDim;
  else if (cfg.band_ranges != nullptr) why = kBand;
  else if (compute_capability(device) < 80) why = kArch;
  if (reason) *reason = why;
  return why == nullptr;
}

size_t sage2_workspace_bytes(const AttentionConfig& cfg, int num_kv_heads) {
  if (cfg.seq_len <= 0 || cfg.num_heads <= 0 || cfg.head_dim <= 0) return 0;
  int device = 0;
  const bool fp16_v = cudaGetDevice(&device) == cudaSuccess && compute_capability(device) < 89;
  return buffer_bytes(cfg, num_kv_heads, fp16_v);
}

void sage2_attention_forward(cudaStream_t stream, const __nv_bfloat16* q,
                             const __nv_bfloat16* k, const __nv_bfloat16* v,
                             __nv_bfloat16* out, const AttentionConfig& cfg,
                             int num_kv_heads, Workspace& ws) {
  int device = 0;
  VIDFAB_CUDA_CHECK(cudaGetDevice(&device));
  const char* reason = nullptr;
  if (!sage2_supported(cfg, device, &reason))
    throw std::runtime_error(std::string("attention: sage2 ") + reason);
  Workspace::Scope scope(ws);
  const int capability = compute_capability(device);
  const bool ampere = capability < 89;
  SageBuffers b = carve(ws.alloc(buffer_bytes(cfg, num_kv_heads, ampere)), cfg,
                        num_kv_heads, ampere);
  const int padded = ceil_div(cfg.seq_len, kKBlock) * kKBlock;
  key_mean<<<num_kv_heads * cfg.head_dim, kThreads, 0, stream>>>(
      k, b.km, cfg.seq_len, num_kv_heads, cfg.head_dim);
  const int qgroups = ceil_div(cfg.seq_len, kQBlock) * (kQBlock / kQWarp);
  dim3 qgrid(qgroups, cfg.num_heads);
  quant_qk<kQWarp, false><<<qgrid, kThreads, 0, stream>>>(
      q, b.q, b.qs, nullptr, cfg.seq_len, cfg.num_heads, cfg.head_dim, qgroups);
  const int kgroups = ceil_div(cfg.seq_len, kKBlock);
  dim3 kgrid(kgroups, num_kv_heads);
  quant_qk<kKBlock, true><<<kgrid, kThreads, 0, stream>>>(
      k, b.k, b.ks, b.km, cfg.seq_len, num_kv_heads, cfg.head_dim, kgroups);
  if (ampere) {
    const size_t vn = static_cast<size_t>(cfg.seq_len) * num_kv_heads * cfg.head_dim;
    convert_v_f16<<<std::min<size_t>(ceil_div(static_cast<int>(vn), kThreads), 65535),
                    kThreads, 0, stream>>>(v, static_cast<__half*>(b.v), vn);
  } else {
    quant_v<<<num_kv_heads * cfg.head_dim, kThreads, 0, stream>>>(
        v, static_cast<int8_t*>(b.v), b.vs, cfg.seq_len, padded, num_kv_heads,
        cfg.head_dim);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
  if (ampere) {
    if (cfg.head_dim == 64) launch_ampere<64>(stream, b, out, cfg, num_kv_heads);
    else launch_ampere<128>(stream, b, out, cfg, num_kv_heads);
  } else {
    if (cfg.head_dim == 64) launch_official<64>(stream, b, out, cfg, num_kv_heads, padded);
    else launch_official<128>(stream, b, out, cfg, num_kv_heads, padded);
  }
}

}  // namespace vidfab::cuda
