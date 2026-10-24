#include "runtime.cuh"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab::fibo {
namespace {
__device__ float f(BFloat a) {
  return __bfloat162float(a);
}

__device__ float bf(float a) {
  return f(__float2bfloat16(a));
}

__global__ void modulate_kernel(BFloat* x, const BFloat* m, int n, int width, int start, int end,
                                int field, bool gate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int row = i / width, c = i % width;
  float v = f(m[((row >= start && row < end) ? 4 : 0) * width + field * width + c]);
  x[i] = __float2bfloat16(f(x[i]) * (gate ? bf(tanhf(v)) : bf(1 + v)));
}

__global__ void scale_kernel(BFloat* x, const BFloat* scale, int n, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = __float2bfloat16(f(x[i]) * bf(1 + f(scale[i % width])));
}

__global__ void gated_residual(BFloat* x, const BFloat* y, float gate, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = __float2bfloat16(fmaf(f(y[i]), gate, f(x[i])));
}

void modulate(Tensor& x, const Tensor& m, int start, int end, int field, bool gate) {
  int n = x.rows * x.cols;
  modulate_kernel<<<(n + 255) / 256, 256>>>(x.ptr(), m.ptr(), n, x.cols, start, end, field, gate);
}

Tensor norm(Runtime& rt, const Tensor& x, const std::string& p, int width = 0) {
  return rt.norm(x, p + ".weight", width, false, nullptr, 0, nullptr, 1e-5f);
}

Tensor ff(Runtime& rt, const Tensor& x, const std::string& p) {
  auto both = rt.project(x, p + ".w13");
  auto a = moss::slice(both, 0, both.cols / 2), b = moss::slice(both, both.cols / 2, both.cols / 2);
  return rt.project(moss::activate(b, &a), p + ".w2");
}

Tensor cross(Runtime& rt, const Tensor& x, const Tensor& context, const std::string& p) {
  auto q = norm(rt, rt.project(x, p + ".to_q"), p + ".norm_q", 128);
  auto kv = rt.project(context, p + ".to_kv");
  auto k = norm(rt, moss::slice(kv, 0, 3584), p + ".norm_k", 128), v = moss::slice(kv, 3584, 3584);
  return rt.project(rt.attend(q, k, v, 28, 28), p + ".to_out");
}

void block(Runtime& rt, Tensor& x, const std::string& p, const Tensor& time, int start, int end,
           const std::vector<float>& cos, const std::vector<float>& sin, const Tensor* context) {
  auto m = rt.project(time, p + ".adaLN_modulation.0");
  auto y = norm(rt, x, p + ".attention_norm1");
  modulate(y, m, start, end, 0, false);
  auto qkv = rt.project(y, p + ".attention.qkv");
  auto q = norm(rt, moss::slice(qkv, 0, 3584), p + ".attention.q_norm", 128);
  auto k = norm(rt, moss::slice(qkv, 3584, 3584), p + ".attention.k_norm", 128),
       v = moss::slice(qkv, 7168, 3584);
  rope(q, 28, cos, sin, true);
  rope(k, 28, cos, sin, true);
  y = norm(rt, rt.project(rt.attend(q, k, v, 28, 28), p + ".attention.out"),
           p + ".attention_norm2");
  modulate(y, m, start, end, 1, true);
  moss::residual(x, y);
  if (context) {
    y = norm(rt, cross(rt, norm(rt, x, p + ".cross_attn_norm1"), *context, p + ".cross_attention"),
             p + ".cross_attn_norm2");
    const auto gate = to_f32(rt.checkpoint.at(p + ".cross_attn_gate"));
    int n = x.rows * x.cols;
    gated_residual<<<(n + 255) / 256, 256>>>(
        x.ptr(), y.ptr(), __bfloat162float(__float2bfloat16(std::tanh(gate.at(0)))), n);
  }
  y = norm(rt, x, p + ".ffn_norm1");
  modulate(y, m, start, end, 2, false);
  y = norm(rt, ff(rt, y, p + ".feed_forward"), p + ".ffn_norm2");
  modulate(y, m, start, end, 3, true);
  moss::residual(x, y);
}

void append_rope(std::vector<float>& c, std::vector<float>& s, int plane, int height, int width) {
  const int dims[] = {32, 48, 48};
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      int pos[] = {plane, y, x};
      for (int axis = 0; axis < 3; ++axis)
        for (int i = 0; i < dims[axis]; i += 2) {
          float angle = float(pos[axis] * std::pow(256., -double(i) / dims[axis]));
          c.push_back(std::cos(angle));
          s.push_back(std::sin(angle));
        }
    }
}
}

Tensor perceiver(Runtime& rt, const Tensor& context, bool streaming) {
  const int n = gist_count(context.rows);
  Tensor x(n, 3584);
  SLOPFAB_CUDA_CHECK(cudaMemcpy(x.ptr(), rt.weight("perceiver.latent_queries"),
                                size_t(n) * 3584 * 2, cudaMemcpyDeviceToDevice));
  for (int i = 0; i < 2; ++i) {
    std::string p = "perceiver.layers." + std::to_string(i);
    auto y = cross(rt, norm(rt, x, p + ".cross_attn_norm"), context, p + ".cross_attention");
    moss::residual(x, y);
    y = norm(rt, x, p + ".self_attn_norm");
    y = cross(rt, y, y, p + ".self_attention");
    moss::residual(x, y);
    y = ff(rt, norm(rt, x, p + ".ffn_norm"), p + ".ffn");
    moss::residual(x, y);
    if (streaming)
      rt.release_prefix(p + ".");
  }
  return norm(rt, x, "perceiver.norm_out");
}

Tensor predict(Runtime& rt, const Tensor& latent, int height, int width,
               const std::vector<ImageLatent>& refs, const std::vector<Tensor>& context,
               const Tensor& gist, float sigma, bool streaming,
               const std::function<void()>& check_cancel) {
  if (context.size() != 6 || latent.rows != height * width || latent.cols != 128)
    throw std::invalid_argument("Fibo transformer input shape");
  std::vector<float> embedding(512);
  for (int r = 0; r < 2; ++r)
    for (int j = 0; j < 128; ++j) {
      float angle = (r ? sigma * 1000 : 0) * std::exp(-std::log(10000.f) * j / 128);
      embedding[r * 256 + j] = std::cos(angle);
      embedding[r * 256 + 128 + j] = std::sin(angle);
    }
  auto time =
      rt.project(moss::activate(rt.project(from_host(embedding, 2, 256), "t_embedder.mlp.0")),
                 "t_embedder.mlp.2");
  Tensor img = rows(latent, 0, latent.rows);
  std::vector<float> c, s;
  append_rope(c, s, 513, height, width);
  for (size_t i = 0; i < refs.size(); ++i) {
    const auto& r = refs[i];
    img = join(img, from_host(r.values, r.height * r.width, 128));
    append_rope(c, s, 514 + int(i), r.height, r.width);
  }
  img = rt.project(img, "x_embedder");
  for (int i = 0; i < 2; ++i) {
    check_cancel();
    std::string p = "noise_refiner." + std::to_string(i);
    block(rt, img, p, time, 0, latent.rows, c, s, nullptr);
    if (streaming)
      rt.release_prefix(p + ".");
  }
  std::vector<float> all_c, all_s;
  for (int i = 0; i < gist.rows; ++i)
    append_rope(all_c, all_s, i + 1, 1, 1);
  all_c.insert(all_c.end(), c.begin(), c.end());
  all_s.insert(all_s.end(), s.begin(), s.end());
  auto x = join(gist, img);
  img = Tensor();
  int bundle = 1;
  for (int i = 0; i < 44; ++i) {
    check_cancel();
    std::string p = "layers." + std::to_string(i);
    bool inject = i >= 6 && (i - 6) % 8 == 0;
    block(rt, x, p, time, gist.rows, gist.rows + latent.rows, all_c, all_s,
          inject ? &context.at(bundle++) : nullptr);
    if (streaming)
      rt.release_prefix(p + ".");
  }
  auto target = layer_norm(rt, rows(x, gist.rows, latent.rows), "");
  auto scale = rt.project(moss::activate(rows(time, 1, 1)), "final_layer.adaLN_modulation.1");
  int n = target.rows * target.cols;
  scale_kernel<<<(n + 255) / 256, 256>>>(target.ptr(), scale.ptr(), n, target.cols);
  return rt.project(target, "final_layer.linear");
}
}
