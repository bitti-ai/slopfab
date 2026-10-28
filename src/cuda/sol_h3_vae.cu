// Sol-H3-VAE math/geometry: Efficient-Large-Model/Sol-H3-VAE,
// revision 6a8a33fc61e092281f32bf28cde236760e2caaa4 (Apache-2.0 reference).
#include "slopfab/cuda/sol_h3_vae.h"
#include "slopfab/cuda/attention.cuh"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/nn_kernels.cuh"
#include "slopfab/tensor_convert.h"
#include "slopfab/vae/sol_h3.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <math_constants.h>

namespace slopfab::cuda {
namespace {
using BF = __nv_bfloat16;
constexpr int D = 2048, Inner = 8192, Heads = 32, HD = 64, Chunk = 512;
__device__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ float sum_warp(float x) {
  for (int d = 16; d; d /= 2) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
__device__ float sum_block(float x) {
  __shared__ float sums[8];
  x = sum_warp(x);
  if ((threadIdx.x & 31) == 0) sums[threadIdx.x / 32] = x;
  __syncthreads();
  float y = threadIdx.x < 8 ? sums[threadIdx.x] : 0;
  y = sum_warp(y);
  if (threadIdx.x == 0) sums[0] = y;
  __syncthreads();
  const float result = sums[0];
  __syncthreads();
  return result;
}
__global__ void narrow(const float* x, BF* out, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16_rn(x[i]);
}
__global__ void norm(const float* x, const float* w, const float* bias, BF* out,
                     float* fpout, int dim, bool layernorm) {
  const int row = blockIdx.x;
  float mean = 0;
  if (layernorm) {
    float v = 0;
    for (int c = threadIdx.x; c < dim; c += 256) v += x[size_t(row) * dim + c];
    mean = sum_block(v) / dim;
  }
  float ss = 0;
  for (int c = threadIdx.x; c < dim; c += 256) {
    const float a = x[size_t(row) * dim + c] - mean;
    ss += a * a;
  }
  const float inv = rsqrtf(sum_block(ss) / dim + 1e-5f);
  for (int c = threadIdx.x; c < dim; c += 256) {
    float v = (x[size_t(row) * dim + c] - mean) * inv * w[c];
    if (layernorm) v += bias[c];
    if (out) out[size_t(row) * dim + c] = __float2bfloat16_rn(v);
    if (fpout) fpout[size_t(row) * dim + c] = v;
  }
}
__global__ void bias_round(const float* x, const float* bias, float* out, int n, int dim,
                           bool rounded_bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * dim) out[i] = bf(x[i] + (rounded_bias ? bf(bias[i % dim]) : bias[i % dim]));
}
__global__ void residual(float* x, const float* y, const float* bias, const float* scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * D) x[i] += bf(y[i] + bf(bias[i % D])) * scale[i % D];
}
__global__ void swiglu(const float* x, const float* bias, BF* out, int rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * Inner) return;
  const int c = i % Inner, r = i / Inner;
  const float a = bf(x[size_t(r) * 2 * Inner + c] + bf(bias[c]));
  const float b = bf(x[size_t(r) * 2 * Inner + c + Inner] + bf(bias[c + Inner]));
  out[i] = __float2bfloat16_rn(bf(a / (1 + expf(-a))) * b);
}
__global__ void input_rows(const float* z, BF* raw, const float* mean, const float* stddev,
                           int patches, int begin, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count * 24) {
    const int c = i % 24, row = begin + i / 24;
    raw[i] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(z[size_t(c) * patches + row], stddev[c]), mean[c]));
  }
}
__global__ void init_globals(float* x, const float* reg, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count * D) x[i] = (i / D) % 5 == 4 ? 0 : bf(reg[((i / D) % 5) * D + i % D]);
}
// One warp per head, preserving the reference's BF16 rounding around QK norm/RoPE.
__global__ void qkv_post(const float* input, const float* bias, BF* q, BF* k, BF* v,
                         const float* cosine, const float* sine, int begin, int patches) {
  const int local = blockIdx.x / Heads, head = blockIdx.x % Heads, lane = threadIdx.x;
  const int row = begin + local;
  float qq[2], kk[2], vv[2];
  float sq = 0, sk = 0;
  for (int j = 0; j < 2; ++j) {
    const int c = lane + j * 32, off = head * 3 * HD + c;
    qq[j] = bf(input[size_t(local) * 3 * D + off] + bf(bias[off]));
    kk[j] = bf(input[size_t(local) * 3 * D + off + HD] + bf(bias[off + HD]));
    vv[j] = bf(input[size_t(local) * 3 * D + off + 2 * HD] + bf(bias[off + 2 * HD]));
    sq += qq[j] * qq[j]; sk += kk[j] * kk[j];
  }
  const float rq = rsqrtf(sum_warp(sq) / HD + 1e-5f);
  const float rk = rsqrtf(sum_warp(sk) / HD + 1e-5f);
  for (int j = 0; j < 2; ++j) { qq[j] = bf(qq[j] * rq); kk[j] = bf(kk[j] * rk); }
  for (int j = 0; j < 2; ++j) {
    const int c = lane + j * 32;
    const int pair = c < 24 ? c + 24 : (c < 48 ? c - 24 : c);
    // All lanes participate, including unrotated dimensions.
    const float qp0 = __shfl_sync(0xffffffff, qq[0], pair % 32);
    const float qp1 = __shfl_sync(0xffffffff, qq[1], pair % 32);
    const float kp0 = __shfl_sync(0xffffffff, kk[0], pair % 32);
    const float kp1 = __shfl_sync(0xffffffff, kk[1], pair % 32);
    float qv = qq[j], kv = kk[j];
    if (row < patches && c < 48) {
      const float co = bf(cosine[size_t(row) * 48 + c]);
      const float si = bf(sine[size_t(row) * 48 + c]) * (c < 24 ? -1.f : 1.f);
      qv = bf(bf(qv * co) + bf((pair < 32 ? qp0 : qp1) * si));
      kv = bf(bf(kv * co) + bf((pair < 32 ? kp0 : kp1) * si));
    }
    const size_t i = size_t(row) * D + head * HD + c;
    q[i] = __float2bfloat16_rn(qv); k[i] = __float2bfloat16_rn(kv);
    v[i] = __float2bfloat16_rn(vv[j]);
  }
}
// Only five queries per block: online softmax avoids a quadratic global score matrix.
__global__ void global_attention(const BF* q, const BF* k, const BF* v, BF* out,
                                 int T, int HW, int patches) {
  const int head = blockIdx.x % Heads, gr = blockIdx.x / Heads;
  const int block = gr / 5, row = patches + gr, lane = threadIdx.x;
  const int start = block * 5 * HW, count = min(5, T - block * 5) * HW;
  const size_t qi = size_t(row) * D + head * HD + lane;
  const float q0 = __bfloat162float(q[qi]), q1 = __bfloat162float(q[qi + 32]);
  float m = -CUDART_INF_F, denom = 0, a0 = 0, a1 = 0;
  for (int n = 0; n < count + 5; ++n) {
    const int key = n < count ? start + n : patches + block * 5 + n - count;
    const size_t ki = size_t(key) * D + head * HD + lane;
    float score = sum_warp(q0 * __bfloat162float(k[ki]) + q1 * __bfloat162float(k[ki + 32])) * .125f;
    const float next = fmaxf(m, score), alpha = expf(m - next), p = expf(score - next);
    a0 = a0 * alpha + p * __bfloat162float(v[ki]);
    a1 = a1 * alpha + p * __bfloat162float(v[ki + 32]);
    denom = denom * alpha + p; m = next;
  }
  out[qi] = __float2bfloat16_rn(a0 / denom);
  out[qi + 32] = __float2bfloat16_rn(a1 / denom);
}
__global__ void pixels(const float* patch, const float* bias, float* output,
                       int begin, int count, int H, int W, int F) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * 3072) return;
  const int row = begin + i / 3072, c = i % 3072;
  const int t = row / (H * W), y = (row / W) % H, x = row % W;
  const int channel = c / 1024, phase = (c / 256) % 4, py = (c / 16) % 16, px = c % 16;
  if (t % 5 == 0 && phase < 3) return;
  const int frame = t * 4 + phase - 3 * (t / 5 + 1);
  const float means[] = {.485f, .456f, .406f}, stds[] = {.229f, .224f, .225f};
  const size_t index = ((size_t(channel) * F + frame) * H * 16 + y * 16 + py) * W * 16 + x * 16 + px;
  output[index] = (patch[i] + bias[c]) * stds[channel] + means[channel];
}
struct Linear {
  DeviceBuffer<BF> w;
  DeviceBuffer<float> bias;
};
struct Block {
  DeviceBuffer<float> n1, n2, s1, s2;
  Linear qkv, out, w1, w2;
};
struct Window {
  DeviceBuffer<int32_t> indices;
  int patches = 0, keys = 0;
};
} // namespace

struct SolH3Decoder::Impl {
  Stream stream;
  cublasHandle_t blas = nullptr;
  size_t bytes = 0;
  bool loaded = false;
  std::vector<Block> blocks;
  Linear pq;
  DeviceBuffer<float> embed, embed_bias, reg, norm_w, norm_b, head, head_bias;
  Impl() {
    SLOPFAB_CUBLAS_CHECK(cublas_create(&blas));
    SLOPFAB_CUBLAS_CHECK(cublas_set_stream(blas, stream.get()));
  }
  ~Impl() { if (blas) cublas_destroy(blas); }
  DeviceBuffer<float> load_float(const SafeTensors& st, const std::string& name) {
    const auto& view = st.at(name);
    DeviceBuffer<float> result(view.numel());
    if (view.dtype == DType::kF32)
      result.copy_from_host(static_cast<const float*>(view.data), view.numel(), stream.get());
    else {
      const auto values = to_f32(view);
      result.copy_from_host(values.data(), values.size(), stream.get());
      SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    }
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    bytes += result.size() * sizeof(float);
    return result;
  }
  Linear load_linear(const SafeTensors& st, const std::string& name) {
    Linear l;
    auto f = load_float(st, name + ".weight");
    l.w.allocate(f.size());
    narrow<<<(f.size() + 255) / 256, 256, 0, stream.get()>>>(f.get(), l.w.get(), f.size());
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    bytes -= f.size() * 2;
    l.bias = load_float(st, name + ".bias");
    return l;
  }
  void mm(const BF* a, const Linear& b, float* out, int m, int n, int k) {
    const float alpha = 1, beta = 0;
    SLOPFAB_CUBLAS_CHECK(cublas_gemm_ex(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        &alpha, b.w.get(), CUDA_R_16BF, k, a, CUDA_R_16BF, k, &beta, out, CUDA_R_32F,
        n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
};
SolH3Decoder::SolH3Decoder() : impl_(std::make_unique<Impl>()) {}
SolH3Decoder::~SolH3Decoder() = default;
SolH3Decoder::SolH3Decoder(SolH3Decoder&&) noexcept = default;
SolH3Decoder& SolH3Decoder::operator=(SolH3Decoder&&) noexcept = default;
size_t SolH3Decoder::weight_bytes() const { return impl_->bytes; }
void SolH3Decoder::load(const SafeTensors& st) {
  vae::validate_sol_h3_vae(st);
  auto next = std::make_unique<Impl>();
  auto& d = *next;
  st.prefetch();
  d.pq = d.load_linear(st, "post_quant_conv");
  d.embed = d.load_float(st, "x_embedder.weight");
  d.embed_bias = d.load_float(st, "x_embedder.bias");
  d.reg = d.load_float(st, "register_tokens");
  d.norm_w = d.load_float(st, "norm_out.weight");
  d.norm_b = d.load_float(st, "norm_out.bias");
  d.head = d.load_float(st, "proj_out.weight");
  d.head_bias = d.load_float(st, "proj_out.bias");
  for (int i = 0; i < 24; ++i) {
    Block b;
    const auto p = "blocks." + std::to_string(i) + ".";
    b.n1 = d.load_float(st, p + "norm1.weight");
    b.n2 = d.load_float(st, p + "norm2.weight");
    b.s1 = d.load_float(st, p + "scale1");
    b.s2 = d.load_float(st, p + "scale2");
    b.qkv = d.load_linear(st, p + "attn.to_qkv");
    b.out = d.load_linear(st, p + "attn.to_out");
    b.w1 = d.load_linear(st, p + "ff.w1");
    b.w2 = d.load_linear(st, p + "ff.w2");
    d.blocks.push_back(std::move(b));
  }
  d.loaded = true;
  impl_ = std::move(next);
}

vae::DecodedVideo SolH3Decoder::decode(const float* z, int T, int H, int W,
    const std::vector<float>& mean, const std::vector<float>& stddev, const vae::DecodeSchedule&) {
  auto& d = *impl_;
  if (!d.loaded || !z || mean.size() != 24 || stddev.size() != 24)
    throw std::invalid_argument("Sol-H3-VAE: load weights and provide 24-channel latents/statistics");
  const auto geom = vae::build_sol_h3_geometry(T, H, W);
  const int N = geom.patches, M = geom.rows, nb = (T + 4) / 5;
  const auto s = d.stream.get();
  DeviceBuffer<float> x(size_t(M) * D), tmp(size_t(Chunk) * 2 * Inner);
  DeviceBuffer<BF> h(size_t(Chunk) * Inner);
  {
    DeviceBuffer<float> dz(size_t(N) * 24), dm(24), ds(24), pq(size_t(Chunk) * 24);
    dz.copy_from_host(z, dz.size(), s); dm.copy_from_host(mean.data(), 24, s);
    ds.copy_from_host(stddev.data(), 24, s);
    for (int first = 0; first < N; first += Chunk) {
      const int count = std::min(Chunk, N - first);
      input_rows<<<(count * 24 + 255) / 256, 256, 0, s>>>(dz.get(), h.get(), dm.get(), ds.get(), N, first, count);
      d.mm(h.get(), d.pq, pq.get(), count, 24, 24);
      bias_round<<<(count * 24 + 255) / 256, 256, 0, s>>>(pq.get(), d.pq.bias.get(), pq.get(), count, 24, true);
      gemm_nt(d.blas, pq.get(), d.embed.get(), x.get() + size_t(first) * D, count, D, 24);
      bias_round<<<(count * D + 255) / 256, 256, 0, s>>>(x.get() + size_t(first) * D,
          d.embed_bias.get(), x.get() + size_t(first) * D, count, D, false);
    }
    init_globals<<<(nb * 5 * D + 255) / 256, 256, 0, s>>>(x.get() + size_t(N) * D, d.reg.get(), nb * 5);
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s));
  }
  {
    DeviceBuffer<BF> q(size_t(M) * D), k(size_t(M) * D), v(size_t(M) * D), out(size_t(M) * D);
    DeviceBuffer<float> co[2], si[2];
    std::vector<Window> windows[2];
    int max_keys = 0;
    for (int parity = 0; parity < 2; ++parity) {
      co[parity].allocate(geom.rope[parity].cosine.size()); si[parity].allocate(geom.rope[parity].sine.size());
      co[parity].copy_from_host(geom.rope[parity].cosine.data(), co[parity].size(), s);
      si[parity].copy_from_host(geom.rope[parity].sine.data(), si[parity].size(), s);
      for (const auto& win : geom.windows[parity]) {
        Window w;
        w.patches = win.patches; w.keys = int(win.rows.size());
        max_keys = std::max(max_keys, w.keys);
        w.indices.allocate(win.rows.size());
        w.indices.copy_from_host(win.rows.data(), win.rows.size(), s);
        windows[parity].push_back(std::move(w));
      }
    }
    DeviceBuffer<BF> wq(size_t(max_keys) * D), wk(size_t(max_keys) * D), wv(size_t(max_keys) * D), wo(size_t(max_keys) * D);
    Workspace workspace;
    for (int layer = 0; layer < 24; ++layer) {
      if (progress && !progress(layer, 24)) throw std::runtime_error("Sol-H3-VAE decode cancelled");
      const auto& b = d.blocks[layer];
      const int parity = layer % 2;
      for (int first = 0; first < M; first += Chunk) {
        const int count = std::min(Chunk, M - first);
        norm<<<count, 256, 0, s>>>(x.get() + size_t(first) * D, b.n1.get(), nullptr, h.get(), nullptr, D, false);
        d.mm(h.get(), b.qkv, tmp.get(), count, 3 * D, D);
        qkv_post<<<count * Heads, 32, 0, s>>>(tmp.get(), b.qkv.bias.get(), q.get(), k.get(), v.get(), co[parity].get(), si[parity].get(), first, N);
      }
      for (const auto& w : windows[parity]) {
        launch_gather_rows(q.get(), w.indices.get(), wq.get(), w.keys, D, s);
        launch_gather_rows(k.get(), w.indices.get(), wk.get(), w.keys, D, s);
        launch_gather_rows(v.get(), w.indices.get(), wv.get(), w.keys, D, s);
        AttentionConfig cfg; cfg.seq_len = w.keys; cfg.num_heads = Heads; cfg.head_dim = HD;
        const auto backend = attention_preferred_backend(cfg);
        workspace.reserve(attention_workspace_bytes(cfg, backend)); workspace.clear();
        attention_forward(d.blas, s, wq.get(), wk.get(), wv.get(), wo.get(), cfg, backend, workspace);
        launch_scatter_rows(wo.get(), w.indices.get(), out.get(), w.patches, D, s);
      }
      global_attention<<<nb * 5 * Heads, 32, 0, s>>>(q.get(), k.get(), v.get(), out.get(), T, H * W, N);
      for (int first = 0; first < M; first += Chunk) {
        const int count = std::min(Chunk, M - first);
        auto* rows = x.get() + size_t(first) * D;
        d.mm(out.get() + size_t(first) * D, b.out, tmp.get(), count, D, D);
        residual<<<(count * D + 255) / 256, 256, 0, s>>>(rows, tmp.get(), b.out.bias.get(), b.s1.get(), count);
        norm<<<count, 256, 0, s>>>(rows, b.n2.get(), nullptr, h.get(), nullptr, D, false);
        d.mm(h.get(), b.w1, tmp.get(), count, 2 * Inner, D);
        swiglu<<<(count * Inner + 255) / 256, 256, 0, s>>>(tmp.get(), b.w1.bias.get(), h.get(), count);
        d.mm(h.get(), b.w2, tmp.get(), count, D, Inner);
        residual<<<(count * D + 255) / 256, 256, 0, s>>>(rows, tmp.get(), b.w2.bias.get(), b.s2.get(), count);
      }
      SLOPFAB_CUDA_CHECK(cudaGetLastError());
      SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s));
    }
  } // Release Q/K/V before allocating full-resolution pixels.
  vae::DecodedVideo result;
  result.frames = vae::sol_h3_frames(T); result.height = H * 16; result.width = W * 16;
  result.data.resize(size_t(3) * result.frames * result.height * result.width);
  DeviceBuffer<float> output(result.data.size()), normalized(size_t(Chunk) * D);
  for (int first = 0; first < N; first += Chunk) {
    const int count = std::min(Chunk, N - first);
    norm<<<count, 256, 0, s>>>(x.get() + size_t(first) * D, d.norm_w.get(), d.norm_b.get(), nullptr, normalized.get(), D, true);
    gemm_nt(d.blas, normalized.get(), d.head.get(), tmp.get(), count, 3072, D);
    pixels<<<(count * 3072 + 255) / 256, 256, 0, s>>>(tmp.get(), d.head_bias.get(), output.get(), first, count, H, W, result.frames);
  }
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  output.copy_to_host(result.data.data(), result.data.size(), s);
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s));
  if (progress && !progress(24, 24)) throw std::runtime_error("Sol-H3-VAE decode cancelled");
  return result;
}
} // namespace slopfab::cuda
