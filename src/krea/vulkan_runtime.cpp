#include "vulkan_runtime.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab::krea_vulkan {
Runtime::Runtime(const std::string& path) : music_gpu::Runtime(path) {
  engine->flush();
  engine->context.prepare_pipeline_sets(engine->device, vk::TensorPipelineSet::kKrea |
                                                            vk::TensorPipelineSet::kFlashAttention);
}

void dispatch(music_gpu::Engine& e, vk::DeviceTensor& x, vk::DeviceTensor& w, vk::DeviceTensor& b,
              vk::DeviceTensor& y, std::initializer_list<uint32_t> values, uint32_t groups) {
  if (values.size() > 15 || !groups)
    throw std::invalid_argument("Krea Vulkan dispatch dimensions");
  uint32_t p[16] = {};
  std::copy(values.begin(), values.end(), p);
  e.batch().krea_operation(x, w, b, y, p, groups);
}

Tensor Runtime::attend(const Tensor& q, const Tensor& k, const Tensor& v, int heads, int kv_heads,
                       bool causal) {
  if (heads < 1 || kv_heads < 1 || heads % kv_heads || q.cols != heads * 128 ||
      k.cols != kv_heads * 128 || v.cols != k.cols || q.rows != k.rows || k.rows != v.rows)
    throw std::invalid_argument("Krea Vulkan attention shape");
  // The optimized text plan is specialized for 64 query heads; Krea uses 32.
  if (causal)
    return music_gpu::Runtime::attend(q, k, v, heads, kv_heads, true, false);
  Tensor out(q.rows, q.cols);
  if (engine->context.h3_attention_supported(AttentionMode::kFlash2)) {
    Tensor kk, vv;
    auto* kp = &k.data;
    auto* vp = &v.data;
    if (heads != kv_heads) {
      kk = Tensor(q.rows, q.cols);
      vv = Tensor(q.rows, q.cols);
      const uint32_t n = q.rows * q.cols;
      dispatch(*engine, k.data, k.data, k.data, kk.data,
               {RepeatHeads, n, uint32_t(q.cols), uint32_t(kv_heads), uint32_t(heads)},
               (n + 255) / 256);
      dispatch(*engine, v.data, v.data, v.data, vv.data,
               {RepeatHeads, n, uint32_t(q.cols), uint32_t(kv_heads), uint32_t(heads)},
               (n + 255) / 256);
      kp = &kk.data;
      vp = &vv.data;
    }
    vk::H3AttentionPlanDesc d;
    d.sequence = q.rows;
    d.heads = heads;
    d.head_dim = 128;
    d.scale = 1 / std::sqrt(128.f);
    d.mode = AttentionMode::kFlash2;
    auto plan = vk::H3AttentionPlan::create(engine->context, d);
    plan.record(engine->batch(), q.data, *kp, *vp, out.data);
  } else {
    return music_gpu::Runtime::attend(q, k, v, heads, kv_heads, false, true);
  }
  return out;
}

Tensor norm(Runtime& rt, const Tensor& x, const std::string& name, int width) {
  if (!width)
    width = x.cols;
  if (width < 1 || x.cols % width || rt.checkpoint.at(name).numel() != width)
    throw std::invalid_argument("Krea Vulkan RMS shape");
  Tensor y(x.rows, x.cols);
  auto& w = rt.float_weight(name);
  const uint32_t rows = x.rows * x.cols / width;
  dispatch(*x.engine, x.data, w, w, y.data, {Rms, rows, uint32_t(width)}, rows);
  return y;
}

void gpu_modulate(Tensor& x, const Tensor& mod, int offset) {
  if (offset < 0 || mod.rows != 1 || mod.cols < (offset + 2) * x.cols)
    throw std::invalid_argument("Krea Vulkan modulation shape");
  uint32_t n = x.rows * x.cols;
  dispatch(*x.engine, x.data, mod.data, mod.data, x.data,
           {Modulate, n, uint32_t(x.cols), uint32_t(offset)}, (n + 255) / 256);
}

Tensor gpu_layer_attention(const Tensor& q, const Tensor& k, const Tensor& v) {
  if (q.rows % 12 || q.cols != 2560 || q.rows != k.rows || q.cols != k.cols || k.rows != v.rows ||
      k.cols != v.cols)
    throw std::invalid_argument("Krea Vulkan text fusion shape");
  Tensor y(q.rows, q.cols);
  dispatch(*q.engine, q.data, k.data, v.data, y.data, {LayerAttention, uint32_t(q.rows * 20)},
           q.rows * 20);
  return y;
}

Tensor gpu_collapse(Runtime& rt, const Tensor& x) {
  Tensor y(x.rows / 12, 2560);
  auto& w = rt.weight("txtfusion.projector.weight");
  uint32_t n = y.rows * y.cols;
  dispatch(*x.engine, x.data, w, w, y.data, {Collapse, n}, (n + 255) / 256);
  return y;
}

namespace ops {
Tensor rows(const Tensor& x, int first, int count) {
  if (first < 0 || count < 1 || first + count > x.rows)
    throw std::invalid_argument("Krea Vulkan row slice");
  Tensor y(count, x.cols);
  x.engine->batch().copy_rows(x.data, y.data, first, 0, count);
  return y;
}

Tensor join(const Tensor& a, const Tensor& b) {
  return music_gpu::concatenate(a, b);
}

Tensor gelu(const Tensor& x, bool approximate) {
  if (!approximate)
    throw std::invalid_argument("Krea uses approximate GELU");
  Tensor y(x.rows, x.cols);
  uint32_t n = x.rows * x.cols;
  dispatch(*x.engine, x.data, x.data, x.data, y.data, {Activation, n, 0, 0}, (n + 255) / 256);
  return y;
}

Tensor activate(const Tensor& x, const Tensor* gate, bool situ, bool sigmoid) {
  if (situ)
    throw std::invalid_argument("Krea does not use SiTU");
  if (!sigmoid)
    return music_gpu::activate(x, gate);
  if (!gate || gate->rows != x.rows || gate->cols != x.cols)
    throw std::invalid_argument("Krea gate dimensions");
  Tensor y(x.rows, x.cols);
  uint32_t n = x.rows * x.cols;
  dispatch(*x.engine, x.data, gate->data, gate->data, y.data, {Activation, n, 0, 1},
           (n + 255) / 256);
  return y;
}

void residual(Tensor& x, const Tensor& y, const Tensor* mod, int gate) {
  if (x.rows != y.rows || x.cols != y.cols ||
      (mod && (gate < 0 || mod->cols < (gate + 1) * x.cols)))
    throw std::invalid_argument("Krea Vulkan residual dimensions");
  uint32_t n = x.rows * x.cols;
  dispatch(*x.engine, x.data, y.data, mod ? mod->data : y.data, x.data,
           {Residual, n, uint32_t(x.cols), uint32_t(mod != nullptr), uint32_t(gate)},
           (n + 255) / 256);
}

void rope(Tensor& x, int heads, const std::vector<float>& cosine, const std::vector<float>& sine,
          bool adjacent) {
  if (heads < 1 || x.cols % heads || x.cols / heads % 2 ||
      cosine.size() != size_t(x.rows) * x.cols / heads / 2 || sine.size() != cosine.size())
    throw std::invalid_argument("Krea Vulkan RoPE dimensions");
  auto c = x.engine->upload(cosine, x.rows, x.cols / heads / 2, true);
  auto s = x.engine->upload(sine, x.rows, x.cols / heads / 2, true);
  uint32_t n = x.rows * x.cols / 2;
  dispatch(*x.engine, x.data, c, s, x.data,
           {Rotary, n, uint32_t(x.cols / heads), uint32_t(heads), uint32_t(adjacent)},
           (n + 255) / 256);
}
}
}
