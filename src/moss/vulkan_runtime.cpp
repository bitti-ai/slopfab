#include "vulkan_runtime.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace slopfab::moss_vulkan {
uint32_t float_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

Tensor Runtime::norm(const Tensor& x, const std::string& name, int width, bool offset,
                     const Tensor* mod, int shift, const Tensor* gate, float epsilon) {
  if (!width)
    width = x.cols;
  if (width <= 0 || x.cols % width || checkpoint.at(name).numel() != width ||
      (mod && (shift < 0 || (shift + 2) * x.cols > mod->rows * mod->cols)) ||
      (gate && (gate->rows != x.rows || gate->cols != x.cols)))
    throw std::invalid_argument("Vulkan image normalization shape: " + name);
  Tensor y(x.rows, x.cols);
  auto& w = weight(name);
  uint32_t rows = x.rows * x.cols / width, n = x.rows * x.cols;
  dispatch(*engine, x.data, w, w, y.data,
           {Rms, rows, uint32_t(width), uint32_t(offset), float_bits(epsilon)}, rows);
  if (mod)
    dispatch(*engine, y.data, mod->data, mod->data, y.data,
             {Modulate, n, uint32_t(x.cols), uint32_t(shift)}, (n + 255) / 256);
  if (gate) // RMS-gated SiLU has no intermediate rounding of the gate.
    dispatch(*engine, y.data, gate->data, gate->data, y.data, {Activation, n, 1, 0, 2},
             (n + 255) / 256);
  return y;
}

Tensor Runtime::attend(const Tensor& q, const Tensor& k, const Tensor& v, int heads, int kv_heads,
                       bool causal, bool fp32) {
  if (!causal && !fp32 && heads > 0 && q.cols == heads * 128 && q.rows == k.rows)
    return krea_vulkan::Runtime::attend(q, k, v, heads, kv_heads);
  return music_gpu::Runtime::attend(q, k, v, heads, kv_heads, causal, fp32);
}

void residual(Tensor& x, const Tensor& y, const Tensor* mod, int gate) {
  krea_vulkan::ops::residual(x, y, mod, gate);
}

Tensor activate(const Tensor& x, const Tensor* gate, bool situ, bool sigmoid) {
  if (gate && (gate->rows != x.rows || gate->cols != x.cols))
    throw std::invalid_argument("Vulkan image activation shape");
  Tensor y(x.rows, x.cols);
  uint32_t n = x.rows * x.cols;
  auto& w = gate ? gate->data : x.data;
  dispatch(*x.engine, x.data, w, w, y.data,
           {Activation, n, uint32_t(gate != nullptr), uint32_t(situ), uint32_t(sigmoid)},
           (n + 255) / 256);
  return y;
}

Tensor delta_attention(Runtime& rt, const Tensor& x, const std::string& p) {
  auto mixed = rt.project(x, p + "in_proj_qkv");
  Tensor convolved(x.rows, 8192);
  auto& conv = rt.weight(p + "conv1d.weight");
  uint32_t n = x.rows * 8192;
  dispatch(*rt.engine, mixed.data, conv, conv, convolved.data, {CausalConv, n, 8192},
           (n + 255) / 256);
  auto a = rt.project(x, p + "in_proj_a"), b = rt.project(x, p + "in_proj_b");
  auto params = to_f32(rt.checkpoint.at(p + "A_log"));
  auto bias = to_f32(rt.checkpoint.at(p + "dt_bias"));
  if (params.size() != 32 || bias.size() != 32 || a.cols != 32 || b.cols != 32)
    throw std::invalid_argument("MOSS delta attention dimensions");
  for (float v : bias)
    params.push_back(music_gpu::round_bf(v));
  auto weights = rt.engine->upload(params, 1, 64, true);
  auto decay = rt.engine->allocate(x.rows, 64, ScalarType::kFloat32);
  n = x.rows * 32;
  dispatch(*rt.engine, a.data, b.data, weights, decay, {DeltaParameters, n, 32}, (n + 255) / 256);
  Tensor out(x.rows, 4096);
  dispatch(*rt.engine, convolved.data, decay, decay, out.data, {RecurrentDelta, uint32_t(x.rows)},
           32);
  auto z = rt.project(x, p + "in_proj_z");
  return rt.project(rt.norm(out, p + "norm.weight", 128, false, nullptr, 0, &z), p + "out_proj");
}

Tensor image_rope(int height, int width, int length, bool text_only) {
  const int pixels = text_only ? 0 : height * width, rows = pixels + length;
  std::vector<float> table(size_t(rows) * 128);
  for (int r = 0; r < rows; ++r) {
    const float pos[] = {0.f, float(r < pixels ? length : r - pixels),
                         r < pixels ? r / width - (height - 1) * .5f : 0.f,
                         r < pixels ? r % width - (width - 1) * .5f : 0.f};
    for (int axis = 0; axis < 4; ++axis)
      for (int i = 0; i < 16; ++i) {
        float phase = pos[axis] / std::pow(10000.f, i / 16.f);
        table[r * 128 + axis * 16 + i] = std::sin(phase);
        table[r * 128 + 64 + axis * 16 + i] = std::cos(phase);
      }
  }
  return from_host(table, rows, 128);
}

Tensor text_rope(int length) {
  std::vector<float> table(size_t(length) * 64);
  for (int r = 0; r < length; ++r)
    for (int i = 0; i < 32; ++i) {
      float phase = r / std::pow(10000000.f, i / 32.f);
      table[r * 64 + i] = std::sin(phase);
      table[r * 64 + 32 + i] = std::cos(phase);
    }
  return from_host(table, length, 64);
}
}
