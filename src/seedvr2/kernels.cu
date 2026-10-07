#include "runtime.cuh"
#include "slopfab/tensor_convert.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include "vae_ops.cuh"

namespace slopfab::seedvr2 {
namespace {
__device__ float val(BFloat x) {
  return __bfloat162float(x);
}

__device__ BFloat bf(float x) {
  return __float2bfloat16_rn(x);
}

__device__ float rounded(float x) {
  return val(bf(x));
}

__global__ void upload_kernel(const float* x, BFloat* y, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = bf(x[i]);
}

__global__ void half_kernel(const __half* x, BFloat* y, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = bf(__half2float(x[i]));
}

__global__ void fp8_kernel(const uint8_t* x, BFloat* y, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int bits = x[i], exp = (bits >> 3) & 15, mantissa = bits & 7;
  float value = exp ? ldexpf(1.0f + mantissa * 0.125f, exp - 7)
                    : ldexpf(float(mantissa), -9);
  if (exp == 15 && mantissa == 7) value = nanf("");
  y[i] = bf(bits & 128 ? -value : value);
}

__global__ void download_kernel(const BFloat* x, float* y, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = val(x[i]);
}

__global__ void bias_kernel(BFloat* x, const BFloat* b, size_t n, int c) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = bf(val(x[i]) + val(b[i % c]));
}

__global__ void linear_epilogue(const float* x, const BFloat* b, BFloat* y, size_t n, int c) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = bf(x[i] + val(b[i % c]));
}

__global__ void act_kernel(BFloat* x, const BFloat* gate, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = val(gate ? gate[i] : x[i]);
    float s = rounded(g / (1.0f + expf(-g)));
    x[i] = bf(gate ? val(x[i]) * s : s);
  }
}

__global__ void add_kernel(BFloat* x, const BFloat* y, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = bf(val(x[i]) + val(y[i]));
}

__global__ void mod_kernel(BFloat* x, const BFloat* emb, const BFloat* shift, const BFloat* scale,
                           size_t n, int c, int layer, bool gate) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int ch = int(i % c), e = ch * 6 + layer * 3;
  if (gate)
    x[i] = bf(val(x[i]) * rounded(val(emb[e + 2]) + val(scale[ch])));
  else
    x[i] = bf(rounded(val(x[i]) * rounded(val(emb[e + 1]) + val(scale[ch]))) +
              rounded(val(emb[e]) + val(shift[ch])));
}

__device__ float sum_block(float v) {
  __shared__ float sums[256];
  sums[threadIdx.x] = v;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d /= 2) {
    if (threadIdx.x < d)
      sums[threadIdx.x] += sums[threadIdx.x + d];
    __syncthreads();
  }
  return sums[0];
}

__global__ void rms_kernel(const BFloat* x, BFloat* y, const BFloat* weight, int c) {
  size_t offset = size_t(blockIdx.x) * c;
  float s = 0;
  for (int i = threadIdx.x; i < c; i += blockDim.x) {
    float a = val(x[offset + i]);
    s += a * a;
  }
  float inv = rsqrtf(sum_block(s) / c + 1e-5f);
  for (int i = threadIdx.x; i < c; i += blockDim.x)
    y[offset + i] = bf(val(x[offset + i]) * inv * (weight ? val(weight[i]) : 1));
}

__global__ void gn_kernel(const BFloat* x, BFloat* y, const BFloat* weight, const BFloat* bias,
                          int spatial, int c, bool silu) {
  int frame = blockIdx.x / 32, group = blockIdx.x % 32, gc = c / 32;
  int count = spatial * gc;
  float s = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    s += val(x[(size_t(frame) * spatial + i / gc) * c + group * gc + i % gc]);
  float mean = sum_block(s) / count;
  __syncthreads();
  s = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    float a = val(x[(size_t(frame) * spatial + i / gc) * c + group * gc + i % gc]) - mean;
    s += a * a;
  }
  float inv = rsqrtf(sum_block(s) / count + 1e-6f);
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    int ch = group * gc + i % gc;
    size_t p = (size_t(frame) * spatial + i / gc) * c + ch;
    float a = rounded((val(x[p]) - mean) * inv * val(weight[ch]) + val(bias[ch]));
    y[p] = bf(silu ? a / (1 + expf(-a)) : a);
  }
}

__global__ void col_kernel(const BFloat* x, BFloat* col, int t, int h, int w, int c, int kt, int kh,
                           int kw, int ss, int ts, int oh, int ow, bool down, int first, int rows) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int k = c * kt * kh * kw;
  if (i >= size_t(rows) * k)
    return;
  int r = int(i % k), p = int(i / k) + first;
  int kx = r % kw;
  r /= kw;
  int ky = r % kh;
  r /= kh;
  int kz = r % kt, ch = r / kt;
  int sx = (p % ow) * ss + kx - (down ? 0 : kw / 2);
  int sy = (p / ow % oh) * ss + ky - (down ? 0 : kh / 2);
  int st = max(0, p / (oh * ow) * ts + kz - (kt - 1));
  col[i] = sx < 0 || sx >= w || sy < 0 || sy >= h || st >= t
               ? bf(0)
               : x[((size_t(st) * h + sy) * w + sx) * c + ch];
}

__global__ void up_kernel(const BFloat* x, BFloat* y, int t, int h, int w, int c, int tr) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int oh = h * 2, ow = w * 2, ot = (t - 1) * tr + 1;
  if (i >= size_t(ot) * oh * ow * c)
    return;
  int ch = int(i % c), xx = int(i / c % ow), yy = int(i / c / ow % oh), zz = int(i / c / ow / oh);
  // Reference remove_head keeps frame zero and drops the second duplicate.
  int z = zz == 0 ? 0 : zz + (tr - 1);
  int channel = (((yy % 2) * 2 + xx % 2) * tr + z % tr) * c + ch;
  y[i] = x[((size_t(z / tr) * h + yy / 2) * w + xx / 2) * (c * 4 * tr) + channel];
}

__global__ void pack_kernel(const BFloat* vid, const BFloat* txt, BFloat* q, BFloat* k, BFloat* v,
                            const BFloat* vqn, const BFloat* vkn, const BFloat* tqn,
                            const BFloat* tkn, const float* freq, int h, int w, int dim, int nt,
                            Window win) {
  const int head = blockIdx.x % (dim / 128), token = blockIdx.x / (dim / 128);
  const int wt = win.t1 - win.t0, wh = win.y1 - win.y0, ww = win.x1 - win.x0, nv = wt * wh * ww;
  const bool text = token >= nv;
  const int local = text ? token - nv : token;
  const int source = text ? local
                          : ((local / (wh * ww) + win.t0) * h + local / ww % wh + win.y0) * w +
                                local % ww + win.x0;
  const BFloat* src = (text ? txt : vid) + size_t(source) * 3 * dim + head * 128;
  const BFloat* qw = text ? tqn : vqn;
  const BFloat* kw = text ? tkn : vkn;
  const int j = threadIdx.x;
  float a = val(src[j]), b = val(src[dim + j]);
  float qi = rsqrtf(sum_block(a * a) / 128 + 1e-5f);
  __syncthreads();
  float ki = rsqrtf(sum_block(b * b) / 128 + 1e-5f);
  __shared__ float qn[128], kn[128];
  qn[j] = rounded(a * qi * val(qw[j]));
  kn[j] = rounded(b * ki * val(kw[j]));
  __syncthreads();
  float qa = qn[j], ka = kn[j];
  if (j < 126) {
    int axis = j / 42, pair = (j % 42) / 2;
    int pos = text        ? local
              : axis == 0 ? local / (wh * ww) + nt
              : axis == 1 ? local / ww % wh
                          : local % ww;
    float angle = pos * freq[pair], co = cosf(angle), si = sinf(angle);
    qa = qn[j] * co + (j % 2 ? qn[j - 1] : -qn[j + 1]) * si;
    ka = kn[j] * co + (j % 2 ? kn[j - 1] : -kn[j + 1]) * si;
  }
  size_t dst = size_t(token) * dim + head * 128 + j;
  q[dst] = bf(qa);
  k[dst] = bf(ka);
  v[dst] = src[2 * dim + j];
}

__global__ void unpack_kernel(const BFloat* in, BFloat* video, float* text, int h, int w, int dim,
                              int nt, Window win) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int wh = win.y1 - win.y0, ww = win.x1 - win.x0, nv = (win.t1 - win.t0) * wh * ww;
  if (i >= size_t(nv + nt) * dim)
    return;
  int token = int(i / dim), ch = int(i % dim);
  if (token >= nv)
    text[size_t(token - nv) * dim + ch] += val(in[i]);
  else {
    int p = ((token / (wh * ww) + win.t0) * h + token / ww % wh + win.y0) * w + token % ww + win.x0;
    video[size_t(p) * dim + ch] = in[i];
  }
}

__global__ void pool_kernel(const float* in, BFloat* out, size_t n, int windows) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = bf(in[i] / windows);
}

int blocks(size_t n) {
  return int((n + 255) / 256);
}

void checked() {
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}
}

Runtime::Runtime() {
  SLOPFAB_CUBLAS_CHECK(cuda::cublas_create(&blas));
  const char* pool = std::getenv("SLOPFAB_SEEDVR2_POOL");
  if (!pool || std::string(pool) != "0") {
    activations = std::make_shared<cuda::ReferenceBufferPool>();
    temporaries = std::make_shared<cuda::ReferenceBufferPool>();
  }
  size_t free = 0, total = 0;
  SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free, &total));
  constexpr size_t gib = size_t(1) << 30;
  // Leave most available memory to spatial activations and other applications.
  resident_budget = free >= 8 * gib ? std::min(8 * gib, free / 3) : 0;
  if (const char* budget = std::getenv("SLOPFAB_SEEDVR2_CACHE_MIB")) {
    const long long mib = std::stoll(budget);
    if (mib < 0 || mib > 1048576) throw std::invalid_argument("SeedVR2 cache budget");
    resident_budget = std::min(size_t(mib) * 1048576, free / 2);
  }
}

Runtime::~Runtime() {
  cudaStreamSynchronize(nullptr);
  if (blas)
    cuda::cublas_destroy(blas);
}

void Runtime::end_segment() {
  weights.clear();
}

void Runtime::clear(SafeTensors& f) {
  weights.clear();
  file = &f;
}

Tensor Runtime::upload(const std::vector<float>& v, int t, int h, int w, int c) {
  Tensor out = tensor(t, h, w, c);
  if (v.size() != out.size())
    throw std::runtime_error("SeedVR2: upload size mismatch");
  Buffer<float> tmp(v.size(), temporaries);
  tmp.copy_from_host(v.data(), v.size());
  upload_kernel<<<blocks(v.size()), 256>>>(tmp.get(), out.data.get(), v.size());
  checked();
  return out;
}

std::vector<float> Runtime::download(const Tensor& x) {
  Buffer<float> tmp(x.size(), temporaries);
  download_kernel<<<blocks(x.size()), 256>>>(x.data.get(), tmp.get(), x.size());
  checked();
  std::vector<float> out(x.size());
  tmp.copy_to_host(out.data(), out.size());
  return out;
}

const Tensor& Runtime::weight(const std::string& name) {
  const auto key = std::make_pair(file, name);
  auto cached = resident.find(key);
  if (cached != resident.end()) return cached->second;
  auto it = weights.find(name);
  if (it != weights.end())
    return it->second;
  const auto& v = file->at(name);
  if (v.numel() < 1 || v.numel() > INT_MAX)
    throw std::runtime_error("SeedVR2: invalid tensor size: " + name);
  if (v.dtype != DType::kF16 && v.dtype != DType::kBF16 && v.dtype != DType::kF32 &&
      v.dtype != DType::kF8E4M3)
    throw std::runtime_error("SeedVR2: unsupported weight dtype: " + name +
                             "; use Comfy-Org FP16 or FP8 weights");
  Tensor x(1, 1, 1, int(v.numel()));
  if (v.dtype == DType::kBF16) {
    SLOPFAB_CUDA_CHECK(cudaMemcpy(x.data.get(), v.data, v.nbytes, cudaMemcpyHostToDevice));
  } else if (v.dtype == DType::kF16) {
    Buffer<__half> tmp(size_t(v.numel()), temporaries);
    SLOPFAB_CUDA_CHECK(cudaMemcpy(tmp.get(), v.data, v.nbytes, cudaMemcpyHostToDevice));
    half_kernel<<<blocks(x.size()), 256>>>(tmp.get(), x.data.get(), x.size());
    checked();
  } else if (v.dtype == DType::kF8E4M3) {
    Buffer<uint8_t> tmp(size_t(v.numel()), temporaries);
    tmp.copy_from_host(static_cast<const uint8_t*>(v.data), tmp.size());
    fp8_kernel<<<blocks(x.size()), 256>>>(tmp.get(), x.data.get(), x.size());
    checked();
  } else {
    Buffer<float> tmp(size_t(v.numel()), temporaries);
    tmp.copy_from_host(static_cast<const float*>(v.data), tmp.size());
    upload_kernel<<<blocks(x.size()), 256>>>(tmp.get(), x.data.get(), x.size());
    checked();
  }
  const size_t bytes = x.size() * sizeof(BFloat);
  if (bytes <= resident_budget - std::min(resident_bytes, resident_budget)) {
    resident_bytes += bytes;
    return resident.emplace(key, std::move(x)).first->second;
  }
  return weights.emplace(name, std::move(x)).first->second;
}

Tensor Runtime::linear(const Tensor& x, const std::string& name) {
  const auto& shape = file->at(name + ".weight").shape;
  if (shape.size() != 2 || shape[1] != x.c)
    throw std::runtime_error("SeedVR2: linear shape mismatch: " + name);
  int out = int(shape[0]);
  Tensor y = tensor(x.t, x.h, x.w, out);
  const auto& w = weight(name + ".weight");
  const bool has_bias = file->find(name + ".bias") != nullptr;
  Buffer<float> accumulator(has_bias ? y.size() : 0, temporaries);
  float alpha = 1, beta = 0;
  SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(
      blas, CUBLAS_OP_T, CUBLAS_OP_N, out, x.rows(), x.c, &alpha, w.data.get(), CUDA_R_16BF, x.c,
      x.data.get(), CUDA_R_16BF, x.c, &beta,
      has_bias ? static_cast<void*>(accumulator.get()) : static_cast<void*>(y.data.get()),
      has_bias ? CUDA_R_32F : CUDA_R_16BF, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  if (file->find(name + ".bias")) {
    if (file->at(name + ".bias").shape != std::vector<int64_t>{out})
      throw std::runtime_error("SeedVR2: linear bias shape mismatch: " + name);
    const auto& b = weight(name + ".bias");
    linear_epilogue<<<blocks(y.size()), 256>>>(accumulator.get(), b.data.get(), y.data.get(),
                                               y.size(), out);
    checked();
  }
  return y;
}

Tensor Runtime::conv(const Tensor& x, const std::string& name, bool down, int ts) {
  const auto& s = file->at(name + ".weight").shape;
  if (s.size() != 5 || s[1] != x.c || s[0] < 1 || s[0] > 4096 || (s[2] != 1 && s[2] != 3) ||
      (s[3] != 1 && s[3] != 3) || (s[4] != 1 && s[4] != 3))
    throw std::runtime_error("SeedVR2: convolution shape mismatch: " + name);
  int co = int(s[0]), kt = int(s[2]), kh = int(s[3]), kw = int(s[4]), ss = down ? 2 : 1;
  int ot = (x.t - 1) / ts + 1, oh = down ? x.h / 2 : x.h, ow = down ? x.w / 2 : x.w;
  Tensor y = tensor(ot, oh, ow, co);
  int k = x.c * kt * kh * kw;
  const auto& weight_data = weight(name + ".weight");
  const int tile = std::min(2048, y.rows());
  Buffer<BFloat> col(size_t(tile) * k, temporaries);
  float a = 1, b = 0;
  for (int start = 0; start < y.rows(); start += tile) {
    int count = std::min(tile, y.rows() - start);
    col_kernel<<<blocks(size_t(count) * k), 256>>>(x.data.get(), col.get(), x.t, x.h, x.w, x.c, kt,
                                                   kh, kw, ss, ts, oh, ow, down, start, count);
    checked();
    SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, co, count, k, &a, weight_data.data.get(), CUDA_R_16BF, k,
        col.get(), CUDA_R_16BF, k, &b, y.data.get() + size_t(start) * co, CUDA_R_16BF, co,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  if (file->find(name + ".bias")) {
    if (file->at(name + ".bias").shape != std::vector<int64_t>{co})
      throw std::runtime_error("SeedVR2: convolution bias shape mismatch: " + name);
    const auto& bias = weight(name + ".bias");
    bias_kernel<<<blocks(y.size()), 256>>>(y.data.get(), bias.data.get(), y.size(), co);
    checked();
  }
  return y;
}

Tensor Runtime::groupnorm(const Tensor& x, const std::string& name, bool silu) {
  if (x.c % 32 || file->at(name + ".weight").shape != std::vector<int64_t>{x.c} ||
      file->at(name + ".bias").shape != std::vector<int64_t>{x.c})
    throw std::runtime_error("SeedVR2: group norm shape mismatch: " + name);
  Tensor y = tensor(x.t, x.h, x.w, x.c);
  const auto& w = weight(name + ".weight");
  const auto& b = weight(name + ".bias");
  gn_kernel<<<x.t * 32, 256>>>(x.data.get(), y.data.get(), w.data.get(), b.data.get(), x.h * x.w,
                               x.c, silu);
  checked();
  return y;
}

Tensor Runtime::rms(const Tensor& x, const std::string& name) {
  if (!name.empty() && file->at(name).shape != std::vector<int64_t>{x.c})
    throw std::runtime_error("SeedVR2: RMS norm shape mismatch: " + name);
  Tensor y = tensor(x.t, x.h, x.w, x.c);
  const BFloat* w = name.empty() ? nullptr : weight(name).data.get();
  rms_kernel<<<x.rows(), 256>>>(x.data.get(), y.data.get(), w, x.c);
  checked();
  return y;
}

void Runtime::activation(Tensor& x, const Tensor* gate) {
  act_kernel<<<blocks(x.size()), 256>>>(x.data.get(), gate ? gate->data.get() : nullptr, x.size());
  checked();
}

void Runtime::add(Tensor& x, const Tensor& y) {
  if (x.size() != y.size())
    throw std::runtime_error("SeedVR2: residual shape mismatch");
  add_kernel<<<blocks(x.size()), 256>>>(x.data.get(), y.data.get(), x.size());
  checked();
}

void Runtime::modulate(Tensor& x, const Tensor& emb, const std::string& name, int layer,
                       bool gate) {
  if (emb.size() != size_t(x.c) * 6 || layer < 0 || layer > 1 ||
      file->at(name + (gate ? "_gate" : "_scale")).shape != std::vector<int64_t>{x.c} ||
      (!gate && file->at(name + "_shift").shape != std::vector<int64_t>{x.c}))
    throw std::runtime_error("SeedVR2: modulation shape mismatch: " + name);
  const BFloat* sh = gate ? nullptr : weight(name + "_shift").data.get();
  const BFloat* sc = weight(name + (gate ? "_gate" : "_scale")).data.get();
  mod_kernel<<<blocks(x.size()), 256>>>(x.data.get(), emb.data.get(), sh, sc, x.size(), x.c, layer,
                                        gate);
  checked();
}

Tensor Runtime::upsample(const Tensor& x, int tr) {
  Tensor y = tensor((x.t - 1) * tr + 1, x.h * 2, x.w * 2, x.c / (4 * tr));
  up_kernel<<<blocks(y.size()), 256>>>(x.data.get(), y.data.get(), x.t, x.h, x.w, y.c, tr);
  checked();
  return y;
}

void Runtime::mlp(Tensor& x, const Tensor& emb, const std::string& p, const std::string& branch) {
  int chunk = 4096;
  if (const char* value = std::getenv("SLOPFAB_SEEDVR2_MLP_ROWS")) {
    chunk = std::stoi(value);
    if (chunk < 0) throw std::invalid_argument("SeedVR2 MLP rows must be nonnegative");
  }
  if (!chunk) chunk = x.rows();
  for (int first = 0; first < x.rows(); first += chunk) {
    auto part = x.rows_view(first, std::min(chunk, x.rows() - first));
    auto norm = rms(part);
    modulate(norm, emb, p + "ada." + branch + ".mlp", 1, false);
    auto hidden = linear(norm, p + "mlp." + branch + ".proj_in");
    auto gate = linear(norm, p + "mlp." + branch + ".proj_in_gate");
    norm = Tensor();
    activation(hidden, &gate);
    gate = Tensor();
    hidden = linear(hidden, p + "mlp." + branch + ".proj_out");
    modulate(hidden, emb, p + "ada." + branch + ".mlp", 1, true);
    add(part, hidden);
  }
}

Tensor Runtime::attention(const Tensor& q, const Tensor& k, const Tensor& v, int heads, int dim) {
  Tensor out = tensor(q.t, q.h, q.w, q.c);
  cuda::AttentionConfig cfg;
  cfg.seq_len = q.rows();
  cfg.num_heads = heads;
  cfg.head_dim = dim;
  const auto backend = cuda::attention_preferred_backend(cfg);
  scratch.reserve(cuda::attention_workspace_bytes(cfg, backend));
  scratch.clear();
  cuda::attention_forward(blas, nullptr, q.data.get(), k.data.get(), v.data.get(), out.data.get(),
                          cfg, backend, scratch);
  return out;
}

void Runtime::window_attention(const Tensor& vqkv, const Tensor& tqkv, Tensor& video, Tensor& text,
                               const std::string& p, const std::string& vb, const std::string& tb,
                               const std::vector<Window>& windows) {
  int d = video.c, nt = text.rows();
  for (const auto& branch : {vb, tb})
    for (const auto& norm : {"norm_q.", "norm_k."})
      if (file->at(p + norm + branch + ".weight").shape != std::vector<int64_t>{128})
        throw std::runtime_error("SeedVR2: attention norm shape mismatch: " + p);
  if (file->at(p + "rope.rope.freqs").shape != std::vector<int64_t>{21})
    throw std::runtime_error("SeedVR2: RoPE shape mismatch: " + p);
  const auto* vqn = weight(p + "norm_q." + vb + ".weight").data.get();
  const auto* vkn = weight(p + "norm_k." + vb + ".weight").data.get();
  const auto* tqn = weight(p + "norm_q." + tb + ".weight").data.get();
  const auto* tkn = weight(p + "norm_k." + tb + ".weight").data.get();
  const auto frequencies = to_f32(file->at(p + "rope.rope.freqs"));
  Buffer<float> freq(frequencies.size(), temporaries);
  freq.copy_from_host(frequencies.data(), frequencies.size());
  Buffer<float> pool(text.size(), temporaries);
  pool.zero();
  for (auto win : windows) {
    int n = (win.t1 - win.t0) * (win.y1 - win.y0) * (win.x1 - win.x0) + nt;
    Tensor q = tensor(1, 1, n, d), k = tensor(1, 1, n, d), v = tensor(1, 1, n, d);
    pack_kernel<<<n*(d / 128), 128>>>(vqkv.data.get(), tqkv.data.get(), q.data.get(), k.data.get(),
                                      v.data.get(), vqn, vkn, tqn, tkn, freq.get(), video.h,
                                      video.w, d, nt, win);
    checked();
    auto out = attention(q, k, v, d / 128, 128);
    unpack_kernel<<<blocks(out.size()), 256>>>(out.data.get(), video.data.get(), pool.get(),
                                               video.h, video.w, d, nt, win);
    checked();
  }
  pool_kernel<<<blocks(text.size()), 256>>>(pool.get(), text.data.get(), text.size(),
                                            int(windows.size()));
  checked();
}
} // namespace slopfab::seedvr2
