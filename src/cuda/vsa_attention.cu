// Native FastH3 V2 tile-64 attention. Contract:
// https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/attention/backends/video_sparse_attn_h3.py
// The sparse MMA loop follows Slopfab's existing Flash2 implementation.
#include "slopfab/cuda/vsa_attention.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace slopfab::cuda {
namespace {
constexpr int kWarp = 32, kWarps = 4, kThreads = 128;
constexpr int kBr = 64, kBc = 64, kMmaN = 8, kPadH = 8, kPadV = 2;
constexpr float kHostNegInf = -std::numeric_limits<float>::infinity();

__global__ void pool_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                            float* pooled, VsaConfig c) {
  const int tile = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
  if (d >= c.head_dim)
    return;
  float sq = 0, sk = 0, sv = 0;
  for (int i = 0; i < c.sizes[tile]; ++i) {
    const size_t off = (size_t(c.rows[tile * 64 + i]) * c.heads + head) * c.head_dim + d;
    sq += __bfloat162float(q[off]);
    sk += __bfloat162float(k[off]);
    sv += __bfloat162float(v[off]);
  }
  const size_t stride = size_t(c.heads) * c.tiles * c.head_dim;
  const size_t dst = (size_t(head) * c.tiles + tile) * c.head_dim + d;
  const float inv = 1.0f / c.sizes[tile];
  pooled[dst] = sq * inv;
  pooled[stride + dst] = sk * inv;
  pooled[2 * stride + dst] = sv * inv;
}

__global__ void score_kernel(const float* pooled, float* scores, VsaConfig c) {
  const int tile = blockIdx.x, head = blockIdx.y;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const size_t stride = size_t(c.heads) * c.tiles * c.head_dim;
  const float* q = pooled + (size_t(head) * c.tiles + tile) * c.head_dim;
  const float* k = pooled + stride + size_t(head) * c.tiles * c.head_dim;
  for (int key = warp; key < c.tiles; key += 8) {
    float dot = 0;
    for (int d = lane; d < c.head_dim; d += 32)
      dot += q[d] * k[key * c.head_dim + d];
    for (int delta = 16; delta > 0; delta /= 2)
      dot += __shfl_down_sync(0xffffffff, dot, delta);
    if (lane == 0)
      scores[(size_t(head) * c.tiles + tile) * c.tiles + key] = dot / sqrtf(float(c.head_dim));
  }
}

// Sort video tile scores in shared memory. Prefix keys are always retained;
// prefix queries attend every key. Ties use the lower tile index consistently.
__global__ void route_kernel(const float* scores, uint8_t* mask, VsaConfig c, int padded,
                             int keep) {
  const int tile = blockIdx.x, head = blockIdx.y;
  const size_t base = (size_t(head) * c.tiles + tile) * c.tiles;
  for (int i = threadIdx.x; i < c.tiles; i += blockDim.x)
    mask[base + i] = tile < c.prefix_tiles || i < c.prefix_tiles;
  if (tile < c.prefix_tiles)
    return;
  extern __shared__ char storage[];
  float* values = reinterpret_cast<float*>(storage);
  int* ids = reinterpret_cast<int*>(values + padded);
  for (int i = threadIdx.x; i < padded; i += blockDim.x) {
    const int key = c.prefix_tiles + i;
    values[i] = key < c.tiles ? scores[base + key] : kHostNegInf;
    ids[i] = key;
  }
  __syncthreads();
  for (int width = 2; width <= padded; width *= 2)
    for (int stride = width / 2; stride > 0; stride /= 2) {
      for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        const int j = i ^ stride;
        if (j <= i)
          continue;
        const bool before = values[i] > values[j] || (values[i] == values[j] && ids[i] < ids[j]);
        const bool descending = (i & width) == 0;
        if (before != descending) {
          const float v = values[i];
          values[i] = values[j];
          values[j] = v;
          const int id = ids[i];
          ids[i] = ids[j];
          ids[j] = id;
        }
      }
      __syncthreads();
    }
  for (int i = threadIdx.x; i < keep; i += blockDim.x)
    mask[base + ids[i]] = 1;
}

__global__ void compress_kernel(const float* pooled, const float* scores, __nv_bfloat16* compressed,
                                VsaConfig c) {
  const int tile = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
  extern __shared__ float probabilities[];
  const float* row = scores + (size_t(head) * c.tiles + tile) * c.tiles;
  // Every channel uses the same softmax. One thread forms it, then all head
  // channels evaluate their pooled-value dot product in parallel.
  if (d == 0) {
    float maximum = kHostNegInf, sum = 0;
    for (int i = 0; i < c.tiles; ++i)
      maximum = fmaxf(maximum, row[i]);
    for (int i = 0; i < c.tiles; ++i) {
      probabilities[i] = expf(row[i] - maximum);
      sum += probabilities[i];
    }
    for (int i = 0; i < c.tiles; ++i)
      probabilities[i] /= sum;
  }
  __syncthreads();
  if (d >= c.head_dim)
    return;
  const float* vp = pooled + (size_t(2 * c.heads + head) * c.tiles) * c.head_dim;
  float value = 0;
  for (int i = 0; i < c.tiles; ++i)
    value += probabilities[i] * vp[i * c.head_dim + d];
  compressed[(size_t(tile) * c.heads + head) * c.head_dim + d] = __float2bfloat16(value);
}

__global__ void add_kernel(__nv_bfloat16* out, const __nv_bfloat16* gate,
                           const __nv_bfloat16* compressed, int offset, int count, VsaConfig c) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int inner = c.heads * c.head_dim;
  if (i >= size_t(count) * inner)
    return;
  const int tile = c.row_tiles[offset + i / inner];
  const float branch = __bfloat162float(__float2bfloat16(
      __bfloat162float(gate[i]) * __bfloat162float(compressed[size_t(tile) * inner + i % inner])));
  out[i] = __float2bfloat16(__bfloat162float(out[i]) + branch);
}

__device__ inline uint32_t pack_h2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ inline void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ inline void mma_f16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ inline uint32_t ld32(const void* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

template <int D>
__global__ __launch_bounds__(kThreads) void sparse_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out, int heads, float scale,
    const int32_t* rows, const int32_t* sizes, const uint8_t* mask, int tiles) {
  constexpr int kKStride = D + kPadH;
  constexpr int kVStride = kBc + kPadV;
  constexpr int kDSteps = D / 16;      // k-steps of the QK product
  constexpr int kSTiles = kBc / kMmaN; // n-tiles of S
  constexpr int kOTiles = D / kMmaN;   // n-tiles of O
  constexpr int kPSteps = kBc / 16;    // k-steps of the PV product

  extern __shared__ __align__(16) char raw_smem[];
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(raw_smem);
  __half* vt = reinterpret_cast<__half*>(ks + kBc * kKStride);

  const int q0 = blockIdx.x * kBr;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;
  const int gid = lane >> 2; // 0..7, selects the row pair
  const int tig = lane & 3;  // 0..3, selects the column pair

  const size_t qld = static_cast<size_t>(heads) * D;
  const size_t kvld = static_cast<size_t>(heads) * D;

  const int row_a = warp * 16 + gid;
  const int row_b = row_a + 8;

  const int packed_a = rows[q0 + row_a];
  const int packed_b = rows[q0 + row_b];
  const bool q_live_a = packed_a >= 0;
  const bool q_live_b = packed_b >= 0;
  const __nv_bfloat16* qsrc_a =
      q + static_cast<size_t>(q_live_a ? packed_a : 0) * qld + head * D + tig * 2;
  const __nv_bfloat16* qsrc_b =
      q + static_cast<size_t>(q_live_b ? packed_b : 0) * qld + head * D + tig * 2;
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

  const uint8_t* selected = mask + (size_t(head) * tiles + blockIdx.x) * tiles;
  for (int tile = 0; tile < tiles; ++tile) {
    if (!selected[tile])
      continue;
    const int valid_keys = sizes[tile];
    __syncthreads();
    for (int i = tid; i < 64 * D / 8; i += kThreads) {
      const int row = i / (D / 8), col = i % (D / 8) * 8;
      const int packed = rows[tile * 64 + row];
      uint4 kw = {}, vw = {};
      if (packed >= 0) {
        const size_t off = size_t(packed) * kvld + head * D + col;
        kw = *reinterpret_cast<const uint4*>(k + off);
        vw = *reinterpret_cast<const uint4*>(v + off);
      }
      *reinterpret_cast<uint4*>(ks + row * kKStride + col) = kw;
      const auto* values = reinterpret_cast<const __nv_bfloat16*>(&vw);
      for (int j = 0; j < 8; ++j)
        vt[(col + j) * kVStride + row] = __float2half(__bfloat162float(values[j]));
    }
    __syncthreads();

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

    const int col0 = tig * 2;
    float ma = kHostNegInf, mb = kHostNegInf;
#pragma unroll
    for (int j = 0; j < kSTiles; ++j) {
      const int c = col0 + j * kMmaN;
      if (c < valid_keys) {
        ma = fmaxf(ma, s[j][0] * scale);
        mb = fmaxf(mb, s[j][2] * scale);
      }
      if (c + 1 < valid_keys) {
        ma = fmaxf(ma, s[j][1] * scale);
        mb = fmaxf(mb, s[j][3] * scale);
      }
    }
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
      const bool l0 = c < valid_keys && new_a != kHostNegInf;
      const bool l1 = c + 1 < valid_keys && new_a != kHostNegInf;
      const bool r0 = c < valid_keys && new_b != kHostNegInf;
      const bool r1 = c + 1 < valid_keys && new_b != kHostNegInf;
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

#pragma unroll
    for (int j = 0; j < kOTiles; ++j) {
      o[j][0] *= c_a;
      o[j][1] *= c_a;
      o[j][2] *= c_b;
      o[j][3] *= c_b;
    }

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
  }

  const float inv_a = l_a > 0.0f ? 1.0f / l_a : 0.0f;
  const float inv_b = l_b > 0.0f ? 1.0f / l_b : 0.0f;
  const int out_a = packed_a;
  const int out_b = packed_b;
  __nv_bfloat16* base = out + head * D;
#pragma unroll
  for (int j = 0; j < kOTiles; ++j) {
    const int c = j * kMmaN + tig * 2;
    if (q_live_a) {
      base[static_cast<size_t>(out_a) * qld + c] = __float2bfloat16(o[j][0] * inv_a);
      base[static_cast<size_t>(out_a) * qld + c + 1] = __float2bfloat16(o[j][1] * inv_a);
    }
    if (q_live_b) {
      base[static_cast<size_t>(out_b) * qld + c] = __float2bfloat16(o[j][2] * inv_b);
      base[static_cast<size_t>(out_b) * qld + c + 1] = __float2bfloat16(o[j][3] * inv_b);
    }
  }
}

} // namespace

size_t vsa_attention_workspace_bytes(int tiles, int heads, int dim) {
  if (tiles <= 0 || tiles > 4096 || heads <= 0 || (dim != 64 && dim != 128))
    throw std::invalid_argument("VSA-H3 requires 1..4096 tiles and head dimension 64 or 128");
  auto align = [](size_t n) {
    return (n + 255) / 256 * 256;
  };
  return align(size_t(3) * heads * tiles * dim * sizeof(float)) +
         align(size_t(heads) * tiles * tiles * sizeof(float)) +
         align(size_t(heads) * tiles * tiles);
}

void vsa_attention_forward(cudaStream_t stream, const __nv_bfloat16* q, const __nv_bfloat16* k,
                           const __nv_bfloat16* v, __nv_bfloat16* out, __nv_bfloat16* compressed,
                           const VsaConfig& c, Workspace& ws) {
  (void)vsa_attention_workspace_bytes(c.tiles, c.heads, c.head_dim);
  if (c.prefix_tiles < 0 || c.prefix_tiles >= c.tiles || !c.rows || !c.sizes || !c.row_tiles)
    throw std::invalid_argument("VSA-H3 invalid tile maps");
  Workspace::Scope scope(ws);
  float* pooled = ws.alloc_n<float>(size_t(3) * c.heads * c.tiles * c.head_dim);
  float* scores = ws.alloc_n<float>(size_t(c.heads) * c.tiles * c.tiles);
  auto* mask = ws.alloc_n<uint8_t>(size_t(c.heads) * c.tiles * c.tiles);
  const dim3 grid(c.tiles, c.heads);
  pool_kernel<<<grid, 128, 0, stream>>>(q, k, v, pooled, c);
  score_kernel<<<grid, 256, 0, stream>>>(pooled, scores, c);
  int padded = 1;
  while (padded < c.tiles - c.prefix_tiles)
    padded *= 2;
  // ceil((1 - 0.8) * n) without floating-point boundary artifacts.
  const int keep = (c.tiles - c.prefix_tiles + 4) / 5;
  route_kernel<<<grid, 256, padded * 8, stream>>>(scores, mask, c, padded, keep);
  compress_kernel<<<grid, 128, c.tiles * sizeof(float), stream>>>(pooled, scores, compressed, c);
  const size_t smem = 2 * 64 * (c.head_dim + 8) + 2 * c.head_dim * (64 + 2);
  if (c.head_dim == 128)
    sparse_kernel<128><<<grid, kThreads, smem, stream>>>(
        q, k, v, out, c.heads, 1.0f / sqrtf(128.0f), c.rows, c.sizes, mask, c.tiles);
  else
    sparse_kernel<64><<<grid, kThreads, smem, stream>>>(q, k, v, out, c.heads, 1.0f / sqrtf(64.0f),
                                                        c.rows, c.sizes, mask, c.tiles);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void vsa_add_compression(cudaStream_t stream, __nv_bfloat16* output, const __nv_bfloat16* gate,
                         const __nv_bfloat16* compressed, int offset, int count,
                         const VsaConfig& c) {
  const size_t n = size_t(count) * c.heads * c.head_dim;
  add_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(output, gate, compressed,
                                                                         offset, count, c);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

} // namespace slopfab::cuda
