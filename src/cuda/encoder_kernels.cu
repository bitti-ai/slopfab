// The Qwen3-VL conditioner's forward pass.
//
// Three things live here that could not be reused from elsewhere:
//
//   1. **Causal GQA attention.** `vidfab/cuda/attention.cuh` is unmasked by
//      design — the H3 DiT has no mask anywhere (transformer spec 2.2) — but
//      `Qwen3VLTextAttention` sets `is_causal = True` unconditionally and the
//      model calls `create_causal_mask`. The `attention_mask=ones_like(input_ids)`
//      in the reference pipeline is a *padding* mask meaning "nothing is
//      padded"; it does not disable causality. Bidirectional attention here
//      would produce a plausible, entirely wrong conditioning
//      (docs/text_encoder_spec.md section 3).
//
//      The problem is much smaller than the DiT's: L is a prompt length, a few
//      hundred to a few thousand and capped at 8192, not 37710. So this is the
//      same blocked online-softmax algorithm as attention.cu with two additions
//      — key blocks entirely past the diagonal are skipped, and the diagonal
//      block is masked per row — rather than a fused flash kernel.
//
//   2. **SwiGLU over two separate tensors.** Qwen3-VL ships `gate_proj` and
//      `up_proj` as independent weights, so `launch_swiglu`'s fused-halves
//      layout does not apply.
//
//   3. **The decoder layer and the encoder itself**, including both residency
//      modes. `Encoder`'s methods are defined here rather than in
//      src/text/encoder.cpp because that file is compiled by the host compiler
//      and everything below needs nvcc.

#include "vidfab/text/encoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nn_kernels.cuh"
#include "vidfab/cuda/workspace.cuh"

namespace vidfab::text {
namespace {

using vidfab::cuda::ComputeType;
using vidfab::cuda::DeviceBuffer;
using vidfab::cuda::PinnedBuffer;
using vidfab::cuda::QuantFormat;
using vidfab::cuda::QuantWeight;
using vidfab::cuda::Workspace;

// src/text/encoder.cpp works out the on-disk shapes from its own copy of this,
// because it is host-compiled and linear.cuh drags in cuBLAS. If the two ever
// disagree the scale tensor's declared width is wrong and nothing else notices.
static_assert(vidfab::cuda::kNVFP4BlockSize == 16,
              "the encoder's nvfp4 shapes and scale swizzle assume 16 elements per block");

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kSoftmaxThreads = 256;
constexpr int kConvRotGroup = 256;
constexpr int kDefaultQueryBlock = 256;

// Score tiles dominate the attention footprint: `heads * bq * bk` elements at
// fp32 plus bf16. A quarter of what attention.cu allows itself, because this
// module may be sharing the card with 24.4 GB of resident weights and its
// sequences are two orders of magnitude shorter.
constexpr size_t kScoreTileBudget = 256ull << 20;

// MSVC's INFINITY macro is a double expression, which nvcc warns about on every
// use in float context. Build the bit pattern instead.
__device__ inline float neg_inf() { return __int_as_float(0xFF800000); }
constexpr float kHostNegInf = -std::numeric_limits<float>::infinity();

inline size_t align_up(size_t n) { return (n + 255) / 256 * 256; }

inline int grid_1d(size_t n, int block) { return static_cast<int>((n + block - 1) / block); }

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("text encoder: " + message);
}

// --- elementwise ------------------------------------------------------------

// SiLU evaluates in fp32 and the product rounds once. Free on a kernel that is
// entirely bandwidth bound.
__global__ void swiglu_split_kernel(const __nv_bfloat16* __restrict__ gate,
                                    const __nv_bfloat16* __restrict__ up,
                                    __nv_bfloat16* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = __bfloat162float(gate[i]);
  const float u = __bfloat162float(up[i]);
  out[i] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
}

__global__ void residual_add_kernel(__nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ branch, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(branch[i]));
}

__global__ void fill_kernel(float* __restrict__ dst, float value, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = value;
}

// --- causal attention -------------------------------------------------------

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

// One block per (head, query row) of the current tile. Identical to
// attention.cu's online softmax except for `limit`: token `q0 + row` may only
// see keys `0 .. q0 + row`, so within a tile starting at `k0` the first `limit`
// columns are live and the rest are masked to zero probability.
//
// Masked columns still have their probability written, because the PV GEMM
// consumes the whole `[bq, bk]` tile and would otherwise multiply stale memory.
__global__ void causal_softmax_kernel(const float* __restrict__ scores,
                                      __nv_bfloat16* __restrict__ probs, float* __restrict__ acc,
                                      float* __restrict__ m_run, float* __restrict__ l_run,
                                      int rows_in_block, int key_block, int head_dim, int q0,
                                      int k0) {
  __shared__ float shared[kSoftmaxThreads / kWarp];
  const int idx = blockIdx.x;  // head * rows_in_block + row
  const int row = idx % rows_in_block;
  const size_t base = static_cast<size_t>(idx) * key_block;
  const float* s = scores + base;
  __nv_bfloat16* p = probs + base;
  float* a = acc + static_cast<size_t>(idx) * head_dim;

  int limit = q0 + row - k0 + 1;
  if (limit < 0) limit = 0;
  if (limit > key_block) limit = key_block;

  float local = neg_inf();
  for (int j = threadIdx.x; j < limit; j += blockDim.x) local = fmaxf(local, s[j]);
  const float tile_max = block_reduce_max(local, shared);
  __syncthreads();  // shared[] is reused by the sum reduction below

  const float m_old = m_run[idx];
  const float m_new = fmaxf(m_old, tile_max);
  // Three cases, and the guards matter because `limit` can be zero for a row
  // whose whole tile is above the diagonal:
  //   m_old = -inf  -> nothing accumulated yet, correction is 0
  //   tile empty    -> m unchanged, correction is 1, tile contributes nothing
  //   otherwise     -> the usual exp(m_old - m_new)
  const bool empty = (m_new == neg_inf());
  const float corr = (m_old == neg_inf()) ? 0.0f : (empty ? 1.0f : __expf(m_old - m_new));

  float sum = 0.0f;
  for (int j = threadIdx.x; j < key_block; j += blockDim.x) {
    const float e = (j < limit && !empty) ? __expf(s[j] - m_new) : 0.0f;
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
  // Row 0 sees exactly one key, so l is never zero: every query row has at
  // least key 0 inside the first tile.
  const float l = l_run[static_cast<size_t>(head) * rows_in_block + row];
  out[dst] = __float2bfloat16(acc[src] / l);
}

void check_causal_config(const CausalAttentionConfig& cfg) {
  require(cfg.seq_len > 0 && cfg.num_heads > 0 && cfg.head_dim > 0,
          "causal attention: seq_len, num_heads and head_dim must be positive");
  require(cfg.num_kv_heads > 0 && cfg.num_heads % cfg.num_kv_heads == 0,
          "causal attention: num_heads must be a multiple of num_kv_heads");
}

int effective_query_block(const CausalAttentionConfig& cfg) {
  const int bq = cfg.query_block > 0 ? cfg.query_block : kDefaultQueryBlock;
  return std::min(bq, cfg.seq_len);
}

int choose_key_block(const CausalAttentionConfig& cfg) {
  const int bq = effective_query_block(cfg);
  const size_t per_key = static_cast<size_t>(cfg.num_heads) * bq * 6;
  size_t bk = per_key > 0 ? kScoreTileBudget / per_key : static_cast<size_t>(cfg.seq_len);
  bk = bk / 256 * 256;
  if (bk < 256) bk = 256;
  if (bk > static_cast<size_t>(cfg.seq_len)) bk = static_cast<size_t>(cfg.seq_len);
  return static_cast<int>(bk);
}

// --- weight plumbing --------------------------------------------------------

QuantWeight int8_convrot(const uint8_t* base, const LayerLayout& layout, LayerTensor weight,
                         LayerTensor scale, int out_features, int in_features) {
  QuantWeight w;
  w.format = QuantFormat::kI8;
  w.data = base + layout.offset[static_cast<int>(weight)];
  w.out_features = out_features;
  w.in_features = in_features;
  w.weight_scale =
      reinterpret_cast<const float*>(base + layout.offset[static_cast<int>(scale)]);
  // Per output channel despite the format tag reading "int8_tensorwise", and
  // there is no input_scale anywhere in this checkpoint: the quantiser is
  // symmetric per row at /127, so `int8 * weight_scale` is exact
  // (spec section 5.1).
  w.per_channel_scale = true;
  w.input_scale = 0.0f;
  w.convrot = true;
  w.convrot_group = kConvRotGroup;
  return w;
}

// `pre_quant` is kCount where the layer has none — which is a positive
// statement that the quantiser folded it into the preceding norm, not that it
// is unknown. Five of the seven are like that; checked per tensor, never
// inferred from the name.
QuantWeight nvfp4_awq(const uint8_t* base, const LayerLayout& layout, LayerTensor weight,
                      LayerTensor scale, LayerTensor pre_quant, float global_scale,
                      int out_features, int in_features) {
  QuantWeight w;
  w.format = QuantFormat::kNVFP4;
  w.data = base + layout.offset[static_cast<int>(weight)];
  w.out_features = out_features;
  w.in_features = in_features;
  w.block_scale = base + layout.offset[static_cast<int>(scale)];
  w.global_scale = global_scale;
  // Every quantised linear of this build declares full_precision_matrix_mult,
  // so none may ever reach a native fp4 GEMM. The flag comes from the file and
  // validation has already insisted all 350 carry it.
  w.full_precision = true;
  // Not rotated, unlike the int8 build. Leaving this true would apply a
  // Hadamard nothing had undone.
  w.convrot = false;
  w.per_channel_scale = false;
  w.weight_scale = nullptr;
  if (pre_quant != LayerTensor::kCount &&
      layout.bytes[static_cast<int>(pre_quant)] != 0) {
    w.pre_quant_scale =
        reinterpret_cast<const __nv_bfloat16*>(base + layout.offset[static_cast<int>(pre_quant)]);
  }
  return w;
}

const __nv_bfloat16* norm_ptr(const uint8_t* base, const LayerLayout& layout, LayerTensor which) {
  return reinterpret_cast<const __nv_bfloat16*>(base + layout.offset[static_cast<int>(which)]);
}

CausalAttentionConfig attention_config(const LayerDims& dims) {
  CausalAttentionConfig cfg;
  cfg.seq_len = dims.num_tokens;
  cfg.num_heads = dims.num_heads;
  cfg.num_kv_heads = dims.num_kv_heads;
  cfg.head_dim = dims.head_dim;
  cfg.query_block = dims.attn_query_block;
  return cfg;
}

// The dequantisation scratch a projection of this shape will want. Shapes only:
// `linear_workspace_bytes` never dereferences the pointers.
//
// Both formats come to the same total — one dense bf16 copy of the weight plus
// one transformed copy of the activation — but for different reasons: the int8
// path rotates the activation, the nvfp4 path scales it. Sizing them separately
// keeps that a coincidence rather than an assumption.
size_t projection_workspace(WeightFormat format, int out_features, int in_features, int rows) {
  const size_t weight = align_up(static_cast<size_t>(out_features) * in_features *
                                 sizeof(__nv_bfloat16));
  const size_t activation = align_up(static_cast<size_t>(rows) * in_features *
                                     sizeof(__nv_bfloat16));
  if (format == WeightFormat::kNVFP4Awq) return weight + activation;

  QuantWeight w;
  w.format = QuantFormat::kI8;
  w.out_features = out_features;
  w.in_features = in_features;
  w.per_channel_scale = true;
  w.convrot = true;
  w.convrot_group = kConvRotGroup;
  return vidfab::cuda::linear_workspace_bytes(w, rows, ComputeType::kBF16);
}

}  // namespace

// --- causal attention, public ------------------------------------------------

float causal_attention_scale(const CausalAttentionConfig& cfg) {
  // 1/sqrt(128); nothing in Qwen3-VL overrides `self.scaling`.
  return cfg.scale > 0.0f ? cfg.scale : 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
}

size_t causal_attention_workspace_bytes(const CausalAttentionConfig& cfg) {
  if (cfg.seq_len <= 0 || cfg.num_heads <= 0 || cfg.head_dim <= 0) return 0;
  const int bq = effective_query_block(cfg);
  const int bk = choose_key_block(cfg);
  const size_t tile = static_cast<size_t>(cfg.num_heads) * bq * bk;
  const size_t stats = static_cast<size_t>(cfg.num_heads) * bq;

  size_t total = 0;
  total += align_up(tile * sizeof(float));                   // scores
  total += align_up(tile * sizeof(__nv_bfloat16));           // probabilities
  total += align_up(stats * cfg.head_dim * sizeof(float));   // accumulator
  total += align_up(stats * sizeof(float));                  // running max
  total += align_up(stats * sizeof(float));                  // running sum
  return total;
}

void causal_attention_forward(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                              const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                              const CausalAttentionConfig& cfg, Workspace& ws) {
  check_causal_config(cfg);

  const int S = cfg.seq_len;
  const int H = cfg.num_heads;
  const int D = cfg.head_dim;
  const int G = H / cfg.num_kv_heads;  // query heads per kv head
  const int qld = H * D;
  const int kvld = cfg.num_kv_heads * D;
  const int bq_max = effective_query_block(cfg);
  const int bk_max = choose_key_block(cfg);
  const float scale = causal_attention_scale(cfg);

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

    VIDFAB_CUDA_CHECK(cudaMemsetAsync(acc, 0, stat_n * D * sizeof(float), stream));
    VIDFAB_CUDA_CHECK(cudaMemsetAsync(l_run, 0, stat_n * sizeof(float), stream));
    fill_kernel<<<grid_1d(stat_n, 256), 256, 0, stream>>>(m_run, kHostNegInf, stat_n);
    VIDFAB_CUDA_CHECK(cudaGetLastError());

    // The whole point: the last query in this block is `q0 + bq - 1`, so no key
    // beyond it can ever be attended to and those tiles are never computed.
    // Roughly half the score matrix disappears.
    const int k_last = q0 + bq;

    for (int k0 = 0; k0 < k_last; k0 += bk_max) {
      const int bk = std::min(bk_max, k_last - k0);

      // S_tile[h] (row-major [bq, bk]) = Q[h] K[h]^T * scale.
      // Column-major: C[bk, bq] = op_T(K[D, bk]) * op_N(Q[D, bq]).
      if (G == 1) {
        VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, bk, bq, D, &scale,
            k + static_cast<size_t>(k0) * kvld, CUDA_R_16BF, kvld, D,
            q + static_cast<size_t>(q0) * qld, CUDA_R_16BF, qld, D, &zero, scores, CUDA_R_32F, bk,
            static_cast<long long>(bq) * bk, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else {
        // Query head h reads kv head h/G — contiguous blocks, not interleaved
        // (spec section 4.2), which is exactly what makes each kv head one
        // batched call with a zero stride on K.
        for (int kv = 0; kv < cfg.num_kv_heads; ++kv) {
          VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
              handle, CUBLAS_OP_T, CUBLAS_OP_N, bk, bq, D, &scale,
              k + static_cast<size_t>(k0) * kvld + static_cast<size_t>(kv) * D, CUDA_R_16BF, kvld,
              0, q + static_cast<size_t>(q0) * qld + static_cast<size_t>(kv) * G * D, CUDA_R_16BF,
              qld, D, &zero, scores + static_cast<size_t>(kv) * G * bq * bk, CUDA_R_32F, bk,
              static_cast<long long>(bq) * bk, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
      }

      causal_softmax_kernel<<<static_cast<int>(stat_n), kSoftmaxThreads, 0, stream>>>(
          scores, probs, acc, m_run, l_run, bq, bk, D, q0, k0);
      VIDFAB_CUDA_CHECK(cudaGetLastError());

      // acc[h] (row-major [bq, D]) += P[h] V[h], beta = 1.
      if (G == 1) {
        VIDFAB_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, D, bq, bk, &one,
            v + static_cast<size_t>(k0) * kvld, CUDA_R_16BF, kvld, D, probs, CUDA_R_16BF, bk,
            static_cast<long long>(bq) * bk, &one, acc, CUDA_R_32F, D,
            static_cast<long long>(bq) * D, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else {
        for (int kv = 0; kv < cfg.num_kv_heads; ++kv) {
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

// --- elementwise, public ------------------------------------------------------

void launch_swiglu_split(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                         size_t n, cudaStream_t stream) {
  if (n == 0) return;
  swiglu_split_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(gate, up, out, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_residual_add(__nv_bfloat16* x, const __nv_bfloat16* branch, size_t n,
                         cudaStream_t stream) {
  if (n == 0) return;
  residual_add_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(x, branch, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

// --- decoder layer -----------------------------------------------------------

LayerWeights layer_weights_from_blob(const uint8_t* base, const LayerLayout& layout,
                                     const EncoderConfig& config,
                                     const LayerGlobalScales& globals) {
  const int hidden = config.hidden_size;
  const int q_width = config.num_attention_heads * config.head_dim;
  const int kv_width = config.num_key_value_heads * config.head_dim;
  const int inner = config.intermediate_size;

  if (config.format == WeightFormat::kNVFP4Awq) {
    // The seven globals are in the order q, k, v, o, gate, up, down. Only
    // o_proj and down_proj carry a pre_quant_scale; the other five had it
    // folded into input_layernorm and post_attention_layernorm respectively,
    // which is why those two norms differ from the int8 build's while q_norm
    // and k_norm are bitwise identical to it.
    LayerWeights w;
    w.q_proj = nvfp4_awq(base, layout, LayerTensor::kQWeight, LayerTensor::kQScale,
                         LayerTensor::kCount, globals.value[0], q_width, hidden);
    w.k_proj = nvfp4_awq(base, layout, LayerTensor::kKWeight, LayerTensor::kKScale,
                         LayerTensor::kCount, globals.value[1], kv_width, hidden);
    w.v_proj = nvfp4_awq(base, layout, LayerTensor::kVWeight, LayerTensor::kVScale,
                         LayerTensor::kCount, globals.value[2], kv_width, hidden);
    w.o_proj = nvfp4_awq(base, layout, LayerTensor::kOWeight, LayerTensor::kOScale,
                         LayerTensor::kOPreQuantScale, globals.value[3], hidden, q_width);
    w.gate_proj = nvfp4_awq(base, layout, LayerTensor::kGateWeight, LayerTensor::kGateScale,
                            LayerTensor::kCount, globals.value[4], inner, hidden);
    w.up_proj = nvfp4_awq(base, layout, LayerTensor::kUpWeight, LayerTensor::kUpScale,
                          LayerTensor::kCount, globals.value[5], inner, hidden);
    w.down_proj = nvfp4_awq(base, layout, LayerTensor::kDownWeight, LayerTensor::kDownScale,
                            LayerTensor::kDownPreQuantScale, globals.value[6], hidden, inner);
    w.input_layernorm = norm_ptr(base, layout, LayerTensor::kInputLayerNorm);
    w.post_attention_layernorm = norm_ptr(base, layout, LayerTensor::kPostAttentionLayerNorm);
    w.q_norm = norm_ptr(base, layout, LayerTensor::kQNorm);
    w.k_norm = norm_ptr(base, layout, LayerTensor::kKNorm);
    return w;
  }

  LayerWeights w;
  w.q_proj = int8_convrot(base, layout, LayerTensor::kQWeight, LayerTensor::kQScale, q_width,
                          hidden);
  w.k_proj = int8_convrot(base, layout, LayerTensor::kKWeight, LayerTensor::kKScale, kv_width,
                          hidden);
  w.v_proj = int8_convrot(base, layout, LayerTensor::kVWeight, LayerTensor::kVScale, kv_width,
                          hidden);
  w.o_proj = int8_convrot(base, layout, LayerTensor::kOWeight, LayerTensor::kOScale, hidden,
                          q_width);
  w.gate_proj = int8_convrot(base, layout, LayerTensor::kGateWeight, LayerTensor::kGateScale, inner,
                             hidden);
  w.up_proj =
      int8_convrot(base, layout, LayerTensor::kUpWeight, LayerTensor::kUpScale, inner, hidden);
  w.down_proj = int8_convrot(base, layout, LayerTensor::kDownWeight, LayerTensor::kDownScale,
                             hidden, inner);
  w.input_layernorm = norm_ptr(base, layout, LayerTensor::kInputLayerNorm);
  w.post_attention_layernorm = norm_ptr(base, layout, LayerTensor::kPostAttentionLayerNorm);
  w.q_norm = norm_ptr(base, layout, LayerTensor::kQNorm);
  w.k_norm = norm_ptr(base, layout, LayerTensor::kKNorm);
  return w;
}

size_t layer_workspace_bytes(const LayerDims& d) {
  if (d.num_tokens <= 0) return 0;
  const size_t L = static_cast<size_t>(d.num_tokens);
  const size_t q_width = static_cast<size_t>(d.num_heads) * d.head_dim;
  const size_t kv_width = static_cast<size_t>(d.num_kv_heads) * d.head_dim;
  const size_t bf = sizeof(__nv_bfloat16);

  // Live across the whole layer, in the order `encoder_layer_forward` carves
  // them. Two [L, 25600] intermediates dominate: 419 MB each at L = 4096.
  size_t activations = 0;
  activations += align_up(L * d.hidden * bf);        // n
  activations += align_up(L * q_width * bf);        // q
  activations += align_up(L * kv_width * bf);       // k
  activations += align_up(L * kv_width * bf);       // v
  activations += align_up(L * q_width * bf);        // attention output
  activations += align_up(L * d.hidden * bf);       // projection output
  activations += align_up(L * d.intermediate * bf); // gate
  activations += align_up(L * d.intermediate * bf); // up

  // Transient, carved on top: one GEMM's dequantisation scratch, or the
  // attention tiles, whichever is larger. The seven GEMMs are strictly
  // sequential, so one weight scratch buffer suffices (spec section 7.1).
  size_t transient = causal_attention_workspace_bytes(attention_config(d));
  const int rows = d.num_tokens;
  const WeightFormat f = d.format;
  transient = std::max(transient, projection_workspace(f, static_cast<int>(q_width), d.hidden, rows));
  transient = std::max(transient, projection_workspace(f, static_cast<int>(kv_width), d.hidden, rows));
  transient = std::max(transient, projection_workspace(f, d.hidden, static_cast<int>(q_width), rows));
  transient = std::max(transient, projection_workspace(f, d.intermediate, d.hidden, rows));
  transient = std::max(transient, projection_workspace(f, d.hidden, d.intermediate, rows));

  return activations + transient;
}

void encoder_layer_forward(cublasHandle_t handle, cudaStream_t stream,
                           vidfab::cuda::LinearRunner& linear, const LayerWeights& w,
                           const LayerDims& d, const float* cos, const float* sin,
                           __nv_bfloat16* x, Workspace& ws) {
  require(d.num_tokens > 0, "encoder_layer_forward: num_tokens must be positive");
  const size_t L = static_cast<size_t>(d.num_tokens);
  const int rows = d.num_tokens;
  const size_t q_width = static_cast<size_t>(d.num_heads) * d.head_dim;
  const size_t kv_width = static_cast<size_t>(d.num_kv_heads) * d.head_dim;

  Workspace::Scope scope(ws);
  __nv_bfloat16* n = ws.alloc_n<__nv_bfloat16>(L * d.hidden);
  __nv_bfloat16* q = ws.alloc_n<__nv_bfloat16>(L * q_width);
  __nv_bfloat16* k = ws.alloc_n<__nv_bfloat16>(L * kv_width);
  __nv_bfloat16* v = ws.alloc_n<__nv_bfloat16>(L * kv_width);
  __nv_bfloat16* attn = ws.alloc_n<__nv_bfloat16>(L * q_width);
  __nv_bfloat16* proj = ws.alloc_n<__nv_bfloat16>(L * d.hidden);
  __nv_bfloat16* gate = ws.alloc_n<__nv_bfloat16>(L * d.intermediate);
  __nv_bfloat16* up = ws.alloc_n<__nv_bfloat16>(L * d.intermediate);

  // --- attention half. Pre-norm: the residual carries the *unnormalised*
  // stream and is never gated or scaled (spec section 4.4).
  vidfab::cuda::launch_rmsnorm(x, w.input_layernorm, n, rows, d.hidden, d.rms_norm_eps, stream);

  // ConvRot rotates the contraction axis, so it belongs to the activation, not
  // to the GEMM: q/k/v share one rotation of `n`, and LinearRunner applies it
  // per call. The rotation must see the *complete* RMSNorm output — H does not
  // commute with diag(w) (spec section 5.3).
  linear.forward(w.q_proj, n, rows, q, ws);
  linear.forward(w.k_proj, n, rows, k, ws);
  linear.forward(w.v_proj, n, rows, v, ws);

  // QK-norm BEFORE RoPE. Reversing the two is a silent quality bug: RMSNorm
  // scales channel j by w[j], RoPE mixes j with j+64, and those two weights
  // differ by up to 440x on k_norm (spec section 4.2).
  vidfab::cuda::launch_head_rmsnorm(q, w.q_norm, rows, d.num_heads, d.head_dim, d.rms_norm_eps,
                                    stream);
  vidfab::cuda::launch_head_rmsnorm(k, w.k_norm, rows, d.num_kv_heads, d.head_dim, d.rms_norm_eps,
                                    stream);
  // v is not normalised. Only q and k.

  // All 128 head dims rotate, pairing j with j + 64 — unlike the H3 DiT, which
  // rotates 96 of 128 and pairs j with j + 48 (spec section 2.4).
  vidfab::cuda::launch_rope_neox(q, cos, sin, rows, d.num_heads, d.head_dim, stream);
  vidfab::cuda::launch_rope_neox(k, cos, sin, rows, d.num_kv_heads, d.head_dim, stream);

  causal_attention_forward(handle, stream, q, k, v, attn, attention_config(d), ws);
  linear.forward(w.o_proj, attn, rows, proj, ws);
  launch_residual_add(x, proj, L * d.hidden, stream);

  // --- MLP half.
  vidfab::cuda::launch_rmsnorm(x, w.post_attention_layernorm, n, rows, d.hidden, d.rms_norm_eps,
                               stream);
  linear.forward(w.gate_proj, n, rows, gate, ws);
  linear.forward(w.up_proj, n, rows, up, ws);
  // gate_proj goes through SiLU; up_proj does not.
  launch_swiglu_split(gate, up, gate, L * d.intermediate, stream);
  linear.forward(w.down_proj, gate, rows, proj, ws);
  launch_residual_add(x, proj, L * d.hidden, stream);
}

// --- Encoder ------------------------------------------------------------------

struct Encoder::Impl {
  EncoderConfig cfg;
  EncoderStats stats;
  Residency mode = Residency::kResident;
  bool loaded = false;

  const SafeTensors* checkpoint = nullptr;
  const TensorView* embed = nullptr;
  // Non-null only for the nvfp4 build, whose embedding table is I8 with a
  // per-row F32 scale rather than BF16.
  const TensorView* embed_scale = nullptr;
  LayerLayout layout;
  // One set of seven `weight_scale_2` values per layer. Host floats, read once
  // at load: 350 scalars is not worth a device allocation or a re-read per
  // encode.
  std::vector<LayerGlobalScales> globals;

  // kResident: one blob per layer. kStreaming: two, ping-ponged.
  std::vector<DeviceBuffer<uint8_t>> resident;
  DeviceBuffer<uint8_t> ping[2];
  PinnedBuffer<uint8_t> staging[2];
  // Set when the whole checkpoint mapping is page-locked, which lets each
  // weight DMA straight out of it instead of being memcpy'd into a pinned
  // staging buffer first. See `try_register_mapping`.
  bool mapping_registered = false;
  const void* registered_base = nullptr;

  cublasHandle_t cublas = nullptr;
  cudaStream_t compute = nullptr;
  cudaStream_t transfer = nullptr;
  cudaEvent_t upload_done[2] = {nullptr, nullptr};
  cudaEvent_t compute_done[2] = {nullptr, nullptr};

  Workspace ws;
  vidfab::cuda::LinearRunner linear;

  void open_device() {
    if (cublas != nullptr) return;
    VIDFAB_CUBLAS_CHECK(cublasCreate(&cublas));
    // Non-blocking rather than the legacy default stream: the streaming path
    // needs the upload stream to run concurrently with compute, and the legacy
    // default stream serialises against every other blocking stream.
    VIDFAB_CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    VIDFAB_CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
      VIDFAB_CUDA_CHECK(cudaEventCreateWithFlags(&upload_done[i], cudaEventDisableTiming));
      VIDFAB_CUDA_CHECK(cudaEventCreateWithFlags(&compute_done[i], cudaEventDisableTiming));
    }
    linear.init(cublas, compute);
  }

  void close_device() {
    for (int i = 0; i < 2; ++i) {
      if (upload_done[i] != nullptr) cudaEventDestroy(upload_done[i]);
      if (compute_done[i] != nullptr) cudaEventDestroy(compute_done[i]);
      upload_done[i] = nullptr;
      compute_done[i] = nullptr;
    }
    if (transfer != nullptr) cudaStreamDestroy(transfer);
    if (compute != nullptr) cudaStreamDestroy(compute);
    if (cublas != nullptr) cublasDestroy(cublas);
    transfer = nullptr;
    compute = nullptr;
    cublas = nullptr;
  }

  // Page-locks the entire checkpoint mapping so weights can be uploaded
  // without a host-side copy.
  //
  // The staging path this replaces was, measured, 96% of a streaming encode:
  // 57-62 ms per layer of single-threaded `memcpy` against 9.6 ms of DMA. Most
  // of that was not memcpy bandwidth but soft page faults on the 27 GB mapping
  // (~119k pages per layer). Registering the range up front pays those faults
  // once and lifts H2D from 8.7 GB/s pageable to ~42 GB/s.
  //
  // Best-effort by design: registering tens of gigabytes can fail on a machine
  // short of physical memory or lockable pages, and that is not a reason to
  // refuse to run. On failure the staging path still works, just slower.
  void try_register_mapping() {
    mapping_registered = false;
    registered_base = nullptr;
    if (checkpoint == nullptr) return;
    const void* base = checkpoint->mapping_base();
    if (base == nullptr) return;

    // cudaHostRegister wants a page-aligned range. The mapping base is already
    // allocation-granularity aligned, and a file mapping always covers whole
    // pages, so rounding the length up stays inside it.
    constexpr size_t kPage = 4096;
    const size_t bytes = (checkpoint->file_size() + kPage - 1) / kPage * kPage;

    const cudaError_t status =
        cudaHostRegister(const_cast<void*>(base), bytes, cudaHostRegisterReadOnly);
    if (status == cudaSuccess) {
      mapping_registered = true;
      registered_base = base;
    } else {
      // Clear the sticky error so the next real call is not misattributed.
      cudaGetLastError();
    }
  }

  void unregister_mapping() {
    if (!mapping_registered || registered_base == nullptr) return;
    cudaHostUnregister(const_cast<void*>(registered_base));
    mapping_registered = false;
    registered_base = nullptr;
  }

  // Packs layer `layer` into pinned slot `slot` and starts its upload to
  // `dst`. Pageable memory would make cudaMemcpyAsync synchronous and force a
  // staging copy inside the driver, which is exactly what the pinned buffers
  // are for (spec section 7.2).
  void stage_upload(int layer, int slot, uint8_t* dst) {
    // The previous upload out of this pinned buffer must have landed before it
    // is overwritten.
    if (mapping_registered) {
      // No host copy at all: eighteen DMAs straight out of the page-locked
      // mapping into the layer arena. Nothing is written on the host, so the
      // pinned slot is unused and there is nothing to wait for beyond the
      // previous upload into this arena, which the caller's ping-pong event
      // already orders.
      VIDFAB_CUDA_CHECK(cudaEventSynchronize(upload_done[slot]));
      upload_layer_direct(*checkpoint, cfg, layer, layout, dst, transfer);
      VIDFAB_CUDA_CHECK(cudaEventRecord(upload_done[slot], transfer));
      return;
    }

    // The previous upload out of this pinned buffer must have landed before it
    // is overwritten.
    VIDFAB_CUDA_CHECK(cudaEventSynchronize(upload_done[slot]));
    pack_layer(*checkpoint, cfg, layer, layout, staging[slot].get());
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(dst, staging[slot].get(), layout.total_bytes,
                                      cudaMemcpyHostToDevice, transfer));
    VIDFAB_CUDA_CHECK(cudaEventRecord(upload_done[slot], transfer));
  }

  void free_weights() {
    unregister_mapping();
    resident.clear();
    for (int i = 0; i < 2; ++i) {
      ping[i].reset();
      staging[i].reset();
    }
  }
};

Encoder::Encoder() : impl_(new Impl()) {}

Encoder::~Encoder() {
  unload();
  impl_->close_device();
}

const EncoderConfig& Encoder::config() const { return impl_->cfg; }

size_t Encoder::weight_bytes() const { return impl_->stats.weight_bytes; }

Residency Encoder::residency() const { return impl_->mode; }

WeightFormat Encoder::format() const { return impl_->cfg.format; }

const EncoderStats& Encoder::stats() const { return impl_->stats; }

void Encoder::unload() {
  if (impl_->compute != nullptr) cudaStreamSynchronize(impl_->compute);
  if (impl_->transfer != nullptr) cudaStreamSynchronize(impl_->transfer);
  impl_->free_weights();
  impl_->ws = Workspace();
  impl_->loaded = false;
  impl_->checkpoint = nullptr;
  impl_->embed = nullptr;
  impl_->stats = EncoderStats();
}

void Encoder::load(const SafeTensors& checkpoint, const EncoderConfig& config) {
  unload();
  const auto t0 = std::chrono::steady_clock::now();

  Impl& s = *impl_;
  s.cfg = config;
  // Detection is per file and there is no flag: the two builds differ in tensor
  // count, dtype and shape, so validation catches a mismatch rather than
  // letting it become wrong numbers. An explicitly set format is checked
  // against the file rather than trusted.
  s.cfg.format = detect_weight_format(checkpoint);
  if (config.format != WeightFormat::kAuto && config.format != s.cfg.format) {
    throw std::runtime_error(
        "text encoder: the checkpoint declares the other weight format; detection is per file");
  }
  validate_checkpoint(checkpoint, s.cfg);
  s.checkpoint = &checkpoint;
  s.embed = &checkpoint.at("model.embed_tokens.weight");
  s.embed_scale = checkpoint.find("model.embed_tokens.weight_scale");
  s.layout = make_layer_layout(s.cfg);
  s.globals.resize(static_cast<size_t>(s.cfg.num_layers));
  for (int i = 0; i < s.cfg.num_layers; ++i) {
    s.globals[static_cast<size_t>(i)] = read_global_scales(checkpoint, s.cfg, i);
  }
  s.open_device();

  const size_t layer_bytes = s.layout.total_bytes;
  const size_t weight_bytes = layer_bytes * static_cast<size_t>(s.cfg.num_layers);

  Residency mode = config.residency;
  if (mode == Residency::kAuto) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    VIDFAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // The working set at the 8192-token bound is ~2.4 GB of activations plus
    // the 262 MB dequantisation scratch; ask for 3 GB of room on top of the
    // weights before committing to residency.
    const size_t headroom = 3ull << 30;
    mode = (free_bytes > weight_bytes + headroom) ? Residency::kResident : Residency::kStreaming;
  }

  // Page-lock the mapping if we can, which removes the host copy from every
  // upload in both modes. Best effort: if it fails we fall back to staging.
  s.try_register_mapping();

  // The staging buffers are only needed when registration failed. Allocating
  // 930 MB of pinned memory that nothing will ever touch would be a waste of
  // exactly the resource that made registration fail in the first place.
  if (!s.mapping_registered) {
    s.staging[0].allocate(layer_bytes);
    s.staging[1].allocate(layer_bytes);
    s.stats.host_pinned_bytes = 2 * layer_bytes;
  } else {
    s.stats.host_pinned_bytes = 0;
  }

  if (mode == Residency::kResident) {
    try {
      s.resident.resize(static_cast<size_t>(config.num_layers));
      for (int i = 0; i < config.num_layers; ++i) s.resident[static_cast<size_t>(i)].allocate(layer_bytes);
    } catch (const std::exception&) {
      s.resident.clear();
      if (config.residency != Residency::kAuto) throw;
      // Only kAuto is allowed to change its mind: an explicit kResident that
      // does not fit is a sizing error the caller wants to hear about.
      std::printf(
          "  text encoder: %.2f GB of layer weights did not fit; falling back to streaming\n",
          static_cast<double>(weight_bytes) / (1 << 30));
      mode = Residency::kStreaming;
    }
  }

  if (mode == Residency::kResident) {
    for (int i = 0; i < config.num_layers; ++i) {
      s.stage_upload(i, i % 2, s.resident[static_cast<size_t>(i)].get());
    }
    VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.transfer));
    for (int i = 0; i < 2; ++i) s.staging[i].reset();
    s.stats.host_pinned_bytes = 0;
    s.stats.weight_bytes = weight_bytes;
  } else {
    // Two device buffers, ping-ponged: layer i+1 uploads while layer i
    // computes. 0.98 GB instead of 24.39 GB.
    for (int i = 0; i < 2; ++i) s.ping[i].allocate(layer_bytes);
    s.stats.weight_bytes = 2 * layer_bytes;
  }

  s.mode = mode;
  s.stats.load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  s.loaded = true;
}

PromptEmbedding Encoder::encode(const std::vector<int32_t>& token_ids) {
  Impl& s = *impl_;
  require(s.loaded, "encode: load() has not been called");
  require(!token_ids.empty(),
          "encode: the prompt tokenised to zero tokens. The reference does not handle an empty "
          "prompt and neither does this port");
  require(static_cast<int>(token_ids.size()) <= s.cfg.max_prompt_tokens,
          "encode: " + std::to_string(token_ids.size()) + " tokens exceeds max_prompt_tokens " +
              std::to_string(s.cfg.max_prompt_tokens) +
              ". Truncating would be silently observable to the user, so this is an error");

  const auto t0 = std::chrono::steady_clock::now();
  const int L = static_cast<int>(token_ids.size());
  const int hidden = s.cfg.hidden_size;
  const size_t stream_elems = static_cast<size_t>(L) * hidden;

  // hidden_states[0] is the embedding lookup, unscaled and with no positional
  // add. Gathered on the host: only L of 151936 rows are ever read.
  std::vector<uint16_t> host_embed;
  gather_embedding_rows(*s.embed, s.embed_scale, token_ids, host_embed);
  DeviceBuffer<uint16_t> x(stream_elems);
  x.copy_from_host(host_embed.data(), host_embed.size(), s.compute);
  __nv_bfloat16* xp = reinterpret_cast<__nv_bfloat16*>(x.get());

  // One cos/sin pair for the whole request, shared by all 50 layers and by both
  // q and k.
  const std::vector<float> inv_freq = rope_inv_freq(s.cfg.head_dim, s.cfg.rope_theta);
  std::vector<float> cos_host;
  std::vector<float> sin_host;
  build_rope_tables(L, inv_freq, cos_host, sin_host);
  DeviceBuffer<float> cos(cos_host.size());
  DeviceBuffer<float> sin(sin_host.size());
  cos.copy_from_host(cos_host.data(), cos_host.size(), s.compute);
  sin.copy_from_host(sin_host.data(), sin_host.size(), s.compute);

  LayerDims dims;
  dims.format = s.cfg.format;
  dims.num_tokens = L;
  dims.hidden = hidden;
  dims.num_heads = s.cfg.num_attention_heads;
  dims.num_kv_heads = s.cfg.num_key_value_heads;
  dims.head_dim = s.cfg.head_dim;
  dims.intermediate = s.cfg.intermediate_size;
  dims.rms_norm_eps = s.cfg.rms_norm_eps;
  s.ws.reserve(layer_workspace_bytes(dims));

  const int N = s.cfg.num_layers;
  if (s.mode == Residency::kResident) {
    for (int i = 0; i < N; ++i) {
      const LayerWeights w =
          layer_weights_from_blob(s.resident[static_cast<size_t>(i)].get(), s.layout, s.cfg,
                                  s.globals[static_cast<size_t>(i)]);
      encoder_layer_forward(s.cublas, s.compute, s.linear, w, dims, cos.get(), sin.get(), xp,
                            s.ws);
    }
  } else {
    s.stage_upload(0, 0, s.ping[0].get());
    for (int i = 0; i < N; ++i) {
      const int slot = i % 2;
      VIDFAB_CUDA_CHECK(cudaStreamWaitEvent(s.compute, s.upload_done[slot], 0));
      const LayerWeights w = layer_weights_from_blob(s.ping[slot].get(), s.layout, s.cfg,
                                                    s.globals[static_cast<size_t>(i)]);
      encoder_layer_forward(s.cublas, s.compute, s.linear, w, dims, cos.get(), sin.get(), xp,
                            s.ws);
      VIDFAB_CUDA_CHECK(cudaEventRecord(s.compute_done[slot], s.compute));

      if (i + 1 < N) {
        const int next = (i + 1) % 2;
        // The buffer layer i+1 lands in is the one layer i-1 computed from, so
        // that compute must finish first. Without this the upload would race
        // ahead and rewrite weights mid-GEMM — silently, and only under load.
        if (i >= 1) VIDFAB_CUDA_CHECK(cudaStreamWaitEvent(s.transfer, s.compute_done[next], 0));
        s.stage_upload(i + 1, next, s.ping[next].get());
      }
    }
  }

  // fp32 on the host. NO final norm and NO lm_head: the wanted tensor is the
  // raw output of the last layer present (spec section 1.4).
  DeviceBuffer<float> out(stream_elems);
  vidfab::cuda::launch_widen_bf16(xp, out.get(), stream_elems, s.compute);

  PromptEmbedding result;
  result.num_tokens = L;
  result.hidden_size = hidden;
  result.data.resize(stream_elems);
  out.copy_to_host(result.data.data(), stream_elems, s.compute);
  VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.compute));

  s.stats.workspace_bytes = s.ws.capacity();
  s.stats.activation_bytes = x.nbytes() + cos.nbytes() + sin.nbytes() + out.nbytes();
  s.stats.peak_device_bytes =
      s.stats.weight_bytes + s.stats.workspace_bytes + s.stats.activation_bytes;
  s.stats.last_num_tokens = L;
  s.stats.last_encode_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return result;
}

PromptEmbedding Encoder::encode(const Tokenizer& tokenizer, const std::string& prompt) {
  require(!prompt.empty(), "encode: the prompt is empty");
  // add_special_tokens=False: no BOS, no EOS, no chat template. One extra
  // leading token shifts every RoPE position and, because attention is causal,
  // changes every row (spec section 1.2).
  return encode(tokenizer.encode(prompt));
}

}  // namespace vidfab::text
