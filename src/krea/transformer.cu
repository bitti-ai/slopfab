#include "runtime.cuh"
#include "slopfab/tensor_convert.h"
#include <cmath>
#include <stdexcept>

namespace slopfab::krea {
namespace {
__device__ float f(BFloat x) {
  return __bfloat162float(x);
}

__device__ float bf(float x) {
  return f(__float2bfloat16(x));
}

// Krea's learned RMS scales are zero-centered and kept in FP32.
__global__ void rms(const BFloat* x, const float* weight, BFloat* y, int width) {
  __shared__ float sums[256];
  int t = threadIdx.x;
  const BFloat* row = x + size_t(blockIdx.x) * width;
  float sum = 0;
  for (int c = t; c < width; c += 256)
    sum += f(row[c]) * f(row[c]);
  sums[t] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (t < d)
      sums[t] += sums[t + d];
    __syncthreads();
  }
  float inv = rsqrtf(sums[0] / width + 1e-5f);
  for (int c = t; c < width; c += 256)
    y[size_t(blockIdx.x) * width + c] = __float2bfloat16(f(row[c]) * inv * (1 + weight[c]));
}

Tensor norm(Runtime& rt, const Tensor& x, const std::string& name, int width = 0) {
  if (!width)
    width = x.cols;
  if (x.cols % width || rt.checkpoint.at(name).numel() != width)
    throw std::invalid_argument("Krea 2 RMS shape: " + name);
  Tensor y(x.rows, x.cols);
  rms<<<x.rows * x.cols / width, 256>>>(x.ptr(), rt.float_weight(name), y.ptr(), width);
  return y;
}

__global__ void modulate(BFloat* x, const BFloat* mod, int n, int width, int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int c = i % width;
    x[i] = __float2bfloat16(bf(f(x[i]) * bf(1 + f(mod[offset * width + c]))) +
                            f(mod[(offset + 1) * width + c]));
  }
}

// Attention across the twelve encoder layers, independently for each token.
// Flattening these batches into a single sequence would mix unrelated tokens.
__global__ void layer_attention(const BFloat* q, const BFloat* k, const BFloat* v, BFloat* out) {
  __shared__ float reduce[128], scores[12];
  int head = blockIdx.x % 20, row = blockIdx.x / 20, c = threadIdx.x;
  int batch = row / 12;
  float query = f(q[size_t(row) * 2560 + head * 128 + c]);
  for (int j = 0; j < 12; ++j) {
    reduce[c] = query * f(k[size_t(batch * 12 + j) * 2560 + head * 128 + c]);
    __syncthreads();
    for (int d = 64; d; d /= 2) {
      if (c < d)
        reduce[c] += reduce[c + d];
      __syncthreads();
    }
    if (c == 0)
      scores[j] = reduce[0] * .08838834764831845f;
    __syncthreads();
  }
  if (c == 0) {
    float maximum = scores[0], sum = 0;
    for (int j = 1; j < 12; ++j)
      maximum = fmaxf(maximum, scores[j]);
    for (int j = 0; j < 12; ++j) {
      scores[j] = expf(scores[j] - maximum);
      sum += scores[j];
    }
    for (int j = 0; j < 12; ++j)
      scores[j] /= sum;
  }
  __syncthreads();
  float value = 0;
  for (int j = 0; j < 12; ++j)
    value += scores[j] * f(v[size_t(batch * 12 + j) * 2560 + head * 128 + c]);
  out[size_t(row) * 2560 + head * 128 + c] = __float2bfloat16(value);
}

__global__ void collapse(const BFloat* x, const BFloat* weight, BFloat* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int row = i / 2560, col = i % 2560;
  float value = 0;
  for (int j = 0; j < 12; ++j)
    value += f(x[(size_t(row) * 12 + j) * 2560 + col]) * f(weight[j]);
  y[i] = __float2bfloat16(value);
}

Tensor attention(Runtime& rt, const Tensor& x, const std::string& p, int heads, int kvheads,
                 bool layerwise, const std::vector<float>& c, const std::vector<float>& s) {
  auto q = norm(rt, rt.project(x, p + ".wq"), p + ".qknorm.qnorm.scale", 128);
  auto k = norm(rt, rt.project(x, p + ".wk"), p + ".qknorm.knorm.scale", 128);
  auto v = rt.project(x, p + ".wv");
  if (!c.empty()) {
    fibo::rope(q, heads, c, s, true);
    fibo::rope(k, kvheads, c, s, true);
  }
  Tensor a;
  if (layerwise) {
    a = Tensor(x.rows, x.cols);
    layer_attention<<<x.rows * 20, 128>>>(q.ptr(), k.ptr(), v.ptr(), a.ptr());
  } else
    a = rt.attend(q, k, v, heads, kvheads);
  auto gate = rt.project(x, p + ".gate");
  return rt.project(moss::activate(a, &gate, false, true), p + ".wo");
}

Tensor mlp(Runtime& rt, const Tensor& x, const std::string& p) {
  auto gate = rt.project(x, p + ".gate"), up = rt.project(x, p + ".up");
  return rt.project(moss::activate(up, &gate), p + ".down");
}
}

Tensor fuse_text(Runtime& rt, const Tensor& context) {
  if (context.cols != 2560 || context.rows < 12 || context.rows % 12 || context.rows > 512 * 12)
    throw std::invalid_argument("Krea 2 context requires [1..512,12,2560]");
  auto x = fibo::rows(context, 0, context.rows);
  for (int stage = 0; stage < 2; ++stage) {
    for (int i = 0; i < 2; ++i) {
      const std::string p = std::string("txtfusion.") +
                            (stage ? "refiner_blocks." : "layerwise_blocks.") + std::to_string(i);
      auto a =
          attention(rt, norm(rt, x, p + ".prenorm.scale"), p + ".attn", 20, 20, !stage, {}, {});
      moss::residual(x, a);
      auto m = mlp(rt, norm(rt, x, p + ".postnorm.scale"), p + ".mlp");
      moss::residual(x, m);
    }
    if (!stage) {
      Tensor y(x.rows / 12, 2560);
      collapse<<<(y.rows * y.cols + 255) / 256, 256>>>(
          x.ptr(), rt.weight("txtfusion.projector.weight"), y.ptr(), y.rows * y.cols);
      x = std::move(y);
    }
  }
  x = rt.project(fibo::gelu(rt.project(norm(rt, x, "txtmlp.0.scale"), "txtmlp.1"), true),
                 "txtmlp.3");
  rt.release_prefix("txtfusion.");
  rt.release_prefix("txtmlp.");
  return x;
}

Tensor predict(Runtime& rt, const Tensor& latent, const Tensor& text, int height, int width,
               float sigma, bool streaming, const std::function<void()>& check_cancel) {
  if (height < 1 || width < 1 || latent.rows != height * width || latent.cols != 64 ||
      text.cols != 6144)
    throw std::invalid_argument("Krea 2 prediction dimensions");
  std::vector<float> time(256);
  sigma = __bfloat162float(__float2bfloat16(sigma));
  for (int i = 0; i < 128; ++i) {
    float angle = sigma * 1000.f * float(std::exp(-std::log(10000.) * i / 128));
    time[i] = std::cos(angle);
    time[128 + i] = std::sin(angle);
  }
  auto t = rt.project(fibo::gelu(rt.project(from_host(time, 1, 256), "tmlp.0"), true), "tmlp.2");
  auto vec = rt.project(fibo::gelu(t, true), "tproj.1");
  auto base = to_host(vec), last = to_host(t);
  auto x = fibo::join(text, rt.project(latent, "first"));
  std::vector<float> cosine, sine;
  cosine.reserve(size_t(x.rows) * 64);
  sine.reserve(size_t(x.rows) * 64);
  const int axes[] = {32, 48, 48};
  for (int r = 0; r < x.rows; ++r) {
    int pos[] = {0, r < text.rows ? 0 : (r - text.rows) / width,
                 r < text.rows ? 0 : (r - text.rows) % width};
    for (int a = 0; a < 3; ++a)
      for (int j = 0; j < axes[a] / 2; ++j) {
        double angle = pos[a] * std::pow(1000., -2. * j / axes[a]);
        cosine.push_back(float(std::cos(angle)));
        sine.push_back(float(std::sin(angle)));
      }
  }
  for (int i = 0; i < 28; ++i) {
    check_cancel();
    const std::string p = "blocks." + std::to_string(i);
    auto values = to_f32(rt.checkpoint.at(p + ".mod.lin"));
    if (values.size() != base.size())
      throw std::invalid_argument("Krea 2 modulation shape");
    for (size_t j = 0; j < values.size(); ++j)
      values[j] += base[j];
    auto mod = from_host(values, 1, 36864);
    auto y = norm(rt, x, p + ".prenorm.scale");
    modulate<<<(y.rows * y.cols + 255) / 256, 256>>>(y.ptr(), mod.ptr(), y.rows * y.cols, y.cols,
                                                     0);
    auto a = attention(rt, y, p + ".attn", 48, 12, false, cosine, sine);
    moss::residual(x, a, &mod, 2);
    y = norm(rt, x, p + ".postnorm.scale");
    modulate<<<(y.rows * y.cols + 255) / 256, 256>>>(y.ptr(), mod.ptr(), y.rows * y.cols, y.cols,
                                                     3);
    auto m = mlp(rt, y, p + ".mlp");
    moss::residual(x, m, &mod, 5);
    if (streaming)
      rt.release_prefix(p + ".");
  }
  auto values = to_f32(rt.checkpoint.at("last.modulation.lin"));
  for (size_t j = 0; j < values.size(); ++j)
    values[j] += last[j % 6144];
  auto mod = from_host(values, 1, 12288);
  x = norm(rt, fibo::rows(x, text.rows, latent.rows), "last.norm.scale");
  modulate<<<(x.rows * x.cols + 255) / 256, 256>>>(x.ptr(), mod.ptr(), x.rows * x.cols, x.cols, 0);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  return rt.project(x, "last.linear");
}
}
