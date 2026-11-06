#include "vulkan_runtime.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace slopfab::seedvr2_vulkan {
using krea_vulkan::dispatch;

namespace {
uint32_t blocks(size_t n) {
  return uint32_t((n + 255) / 256);
}

void multiply(Runtime& rt, vk::DeviceTensor& x, vk::DeviceTensor& w, vk::DeviceTensor& y, int rows,
              int inner, int cols, int output_offset = 0, vk::DeviceTensor* bias = nullptr) {
  DenseGemmPlanDesc d;
  d.max_rows = rows;
  d.in_features = inner;
  d.out_features = cols;
  d.mode = DenseGemmMode::kBFloat16;
  d.bias = bias ? DenseGemmBias::kBFloat16Fused : DenseGemmBias::kNone;
  auto plan = vk::DenseGemmPlan::create(rt.engine->context, d);
  plan.record(rt.engine->batch(), x, w, y, rows, 0, output_offset, bias);
}
}

Tensor Tensor::rows_view(int offset, int count) const {
  Tensor out;
  out.w = count;
  out.c = c;
  out.values = krea_vulkan::ops::rows(values, offset, count);
  return out;
}

void Runtime::clear(SafeTensors& f) {
  engine->flush();
  weights.clear();
  file = &f;
}

void Runtime::end_segment() {
  engine->flush();
  weights.clear();
}

vk::DeviceTensor& Runtime::weight(const std::string& name, int rows, int cols, bool vector) {
  auto it = weights.find(name);
  if (it != weights.end())
    return it->second;
  const auto& v = file->at(name);
  if (v.numel() != int64_t(rows) * cols || (v.dtype != DType::kF32 && v.dtype != DType::kF16 &&
                                            v.dtype != DType::kBF16 && v.dtype != DType::kF8E4M3))
    throw std::invalid_argument("SeedVR2 Vulkan: unsupported weight: " + name);
  return weights.emplace(name, engine->upload(to_f32(v), rows, cols, false, vector)).first->second;
}

Tensor Runtime::upload(const std::vector<float>& v, int t, int h, int w, int c) {
  Tensor out;
  out.t = t;
  out.h = h;
  out.w = w;
  out.c = c;
  out.values = music_gpu::from_host(v, t * h * w, c);
  return out;
}

std::vector<float> Runtime::download(const Tensor& x) {
  return music_gpu::to_host(x.values);
}

void Runtime::copy_rows(const Tensor& x, Tensor& y, int src, int dst, int count) {
  engine->batch().copy_rows(x.values.data, y.values.data, src, dst, count);
}

Tensor Runtime::linear(const Tensor& x, const std::string& name) {
  const auto& s = file->at(name + ".weight").shape;
  if (s.size() != 2 || s[1] != x.c || s[0] < 1)
    throw std::invalid_argument("SeedVR2 Vulkan linear shape: " + name);
  int co = int(s[0]);
  auto& w = weight(name + ".weight", co, x.c);
  auto* b = file->find(name + ".bias") ? &weight(name + ".bias", 1, co, true) : nullptr;
  Tensor y(x.t, x.h, x.w, co);
  multiply(*this, x.values.data, w, y.values.data, x.rows(), x.c, co, 0, b);
  return y;
}

Tensor Runtime::conv(const Tensor& x, const std::string& name, bool down, int ts) {
  const auto& s = file->at(name + ".weight").shape;
  if (s.size() != 5 || s[1] != x.c || s[0] < 1 || s[0] > 4096 || (s[2] != 1 && s[2] != 3) ||
      (s[3] != 1 && s[3] != 3) || (s[4] != 1 && s[4] != 3) || (ts != 1 && ts != 2))
    throw std::invalid_argument("SeedVR2 Vulkan convolution shape: " + name);
  int co = int(s[0]), kt = int(s[2]), kh = int(s[3]), kw = int(s[4]), ss = down ? 2 : 1;
  Tensor y((x.t - 1) / ts + 1, x.h / ss, x.w / ss, co);
  int inner = x.c * kt * kh * kw;
  auto& w = weight(name + ".weight", co, inner);
  for (int first = 0; first < y.rows(); first += 2048) {
    int count = std::min(2048, y.rows() - first);
    auto col = engine->allocate(count, inner);
    uint32_t n = count * inner;
    dispatch(*engine, x.values.data, x.values.data, x.values.data, col,
             {Patches, n, uint32_t(inner), uint32_t(first), uint32_t(x.t), uint32_t(x.h),
              uint32_t(x.w), uint32_t(x.c), uint32_t(kt), uint32_t(kh), uint32_t(kw), uint32_t(ss),
              uint32_t(ts), uint32_t(y.h), uint32_t(y.w)},
             blocks(n));
    multiply(*this, col, w, y.values.data, count, inner, co, first);
  }
  if (file->find(name + ".bias")) {
    auto& b = weight(name + ".bias", 1, co, true);
    dispatch(*engine, y.values.data, b, b, y.values.data, {Bias, uint32_t(y.size()), uint32_t(co)},
             blocks(y.size()));
  }
  // VAE weights are streamed; activation memory dominates large spatial tiles.
  weights.erase(name + ".weight");
  weights.erase(name + ".bias");
  return y;
}

Tensor Runtime::groupnorm(const Tensor& x, const std::string& name, bool silu) {
  if (x.c % 32)
    throw std::invalid_argument("SeedVR2 Vulkan group norm channels");
  auto& w = weight(name + ".weight", 1, x.c);
  auto& b = weight(name + ".bias", 1, x.c);
  Tensor y(x.t, x.h, x.w, x.c);
  dispatch(*engine, x.values.data, w, b, y.values.data,
           {GroupNorm, uint32_t(x.h * x.w), uint32_t(x.c), uint32_t(silu)}, x.t * 32);
  return y;
}

Tensor Runtime::rms(const Tensor& x, const std::string& name) {
  Tensor y(x.t, x.h, x.w, x.c);
  auto& w = name.empty() ? x.values.data : weight(name, 1, x.c);
  dispatch(*engine, x.values.data, w, w, y.values.data,
           {moss_vulkan::LayerNorm, uint32_t(x.rows()), uint32_t(x.c), uint32_t(!name.empty()), 0,
            1, moss_vulkan::float_bits(1e-5f)},
           x.rows());
  return y;
}

void Runtime::activation(Tensor& x, const Tensor* gate) {
  x.values = moss_vulkan::activate(x.values, gate ? &gate->values : nullptr);
}

void Runtime::add(Tensor& x, const Tensor& y) {
  music_gpu::residual(x.values, y.values);
}

void Runtime::modulate(Tensor& x, const Tensor& emb, const std::string& name, int layer,
                       bool gate) {
  if (layer < 0 || layer > 1 || emb.size() != size_t(x.c) * 6)
    throw std::invalid_argument("SeedVR2 Vulkan modulation shape");
  std::vector<float> params;
  if (!gate)
    params = to_f32(file->at(name + "_shift"));
  auto scales = to_f32(file->at(name + (gate ? "_gate" : "_scale")));
  if (scales.size() != size_t(x.c) || (!gate && params.size() != scales.size()))
    throw std::invalid_argument("SeedVR2 Vulkan modulation weights");
  params.insert(params.end(), scales.begin(), scales.end());
  auto w = engine->upload(params, 1, int(params.size()));
  dispatch(*engine, x.values.data, emb.values.data, w, x.values.data,
           {Modulate, uint32_t(x.size()), uint32_t(x.c), uint32_t(layer), uint32_t(gate)},
           blocks(x.size()));
}

void Runtime::mlp(Tensor& x, const Tensor& emb, const std::string& p, const std::string& branch) {
  int chunk = 4096;
  if (const char* v = std::getenv("SLOPFAB_SEEDVR2_MLP_ROWS"))
    chunk = std::stoi(v);
  if (chunk < 0)
    throw std::invalid_argument("SeedVR2 MLP rows must be nonnegative");
  if (!chunk)
    chunk = x.rows();
  for (int first = 0; first < x.rows(); first += chunk) {
    auto part = x.rows_view(first, std::min(chunk, x.rows() - first));
    auto norm = rms(part);
    modulate(norm, emb, p + "ada." + branch + ".mlp", 1, false);
    auto hidden = linear(norm, p + "mlp." + branch + ".proj_in");
    auto gate = linear(norm, p + "mlp." + branch + ".proj_in_gate");
    activation(hidden, &gate);
    hidden = linear(hidden, p + "mlp." + branch + ".proj_out");
    modulate(hidden, emb, p + "ada." + branch + ".mlp", 1, true);
    add(part, hidden);
    copy_rows(part, x, 0, first, part.rows());
  }
}

Tensor Runtime::upsample(const Tensor& x, int tr) {
  if ((tr != 1 && tr != 2) || x.c % (4 * tr))
    throw std::invalid_argument("SeedVR2 Vulkan upsample shape");
  Tensor y((x.t - 1) * tr + 1, x.h * 2, x.w * 2, x.c / (4 * tr));
  dispatch(
      *engine, x.values.data, x.values.data, x.values.data, y.values.data,
      {Upsample, uint32_t(y.size()), uint32_t(y.c), uint32_t(x.h), uint32_t(x.w), uint32_t(tr)},
      blocks(y.size()));
  return y;
}

Tensor Runtime::attention(const Tensor& q, const Tensor& k, const Tensor& v, int heads, int dim) {
  if (q.c != heads * dim)
    throw std::invalid_argument("SeedVR2 Vulkan attention shape");
  Tensor y;
  y.t = q.t;
  y.h = q.h;
  y.w = q.w;
  y.c = q.c;
  y.values = moss_vulkan::Runtime::attend(q.values, k.values, v.values, heads, heads);
  return y;
}

void Runtime::window_attention(const Tensor& vqkv, const Tensor& tqkv, Tensor& video, Tensor& text,
                               const std::string& p, const std::string& vb, const std::string& tb,
                               const std::vector<Window>& windows) {
  int d = video.c, nt = text.rows();
  if (d % 128 || vqkv.c != 3 * d || tqkv.c != 3 * d || windows.empty())
    throw std::invalid_argument("SeedVR2 Vulkan window attention shape");
  std::vector<float> params;
  for (const auto& branch : {vb, tb})
    for (const auto* norm : {"norm_q.", "norm_k."}) {
      auto v = to_f32(file->at(p + norm + branch + ".weight"));
      if (v.size() != 128)
        throw std::invalid_argument("SeedVR2 Vulkan attention norm shape");
      for (auto& a : v)
        a = music_gpu::round_bf(a);
      params.insert(params.end(), v.begin(), v.end());
    }
  auto freq = to_f32(file->at(p + "rope.rope.freqs"));
  if (freq.size() != 21)
    throw std::invalid_argument("SeedVR2 Vulkan RoPE shape");
  params.insert(params.end(), freq.begin(), freq.end());
  auto norms = engine->upload(params, 1, int(params.size()), true);
  auto pool = engine->upload(std::vector<float>(text.size(), 0), nt, d, true);
  for (const auto& win : windows) {
    int wh = win.y1 - win.y0, ww = win.x1 - win.x0, nv = (win.t1 - win.t0) * wh * ww, n = nv + nt;
    Tensor packed(1, 1, n, 3 * d);
    dispatch(*engine, vqkv.values.data, tqkv.values.data, norms, packed.values.data,
             {PackWindow, uint32_t(n), uint32_t(d), uint32_t(nv), uint32_t(nt), uint32_t(video.h),
              uint32_t(video.w), uint32_t(win.t0), uint32_t(win.y0), uint32_t(win.x0), uint32_t(wh),
              uint32_t(ww)},
             n * (d / 128));
    Tensor q, k, v;
    for (auto* a : {&q, &k, &v}) {
      a->w = n;
      a->c = d;
    }
    q.values = music_gpu::slice(packed.values, 0, d);
    k.values = music_gpu::slice(packed.values, d, d);
    v.values = music_gpu::slice(packed.values, 2 * d, d);
    auto out = attention(q, k, v, d / 128, 128);
    // Windows do not overlap for video. Text is accumulated in FP32 then averaged.
    dispatch(*engine, out.values.data, pool, pool, video.values.data,
             {UnpackWindow, uint32_t(nv * d), uint32_t(d), uint32_t(nv), uint32_t(video.h),
              uint32_t(video.w), uint32_t(win.t0), uint32_t(win.y0), uint32_t(win.x0), uint32_t(wh),
              uint32_t(ww)},
             blocks(size_t(nv) * d));
    dispatch(*engine, out.values.data, pool, pool, pool,
             {PoolText, uint32_t(text.size()), uint32_t(d), uint32_t(nv), 0}, blocks(text.size()));
  }
  dispatch(*engine, pool, pool, pool, text.values.data,
           {PoolText, uint32_t(text.size()), uint32_t(d), 0, uint32_t(windows.size())},
           blocks(text.size()));
}
}
