#define SLOPFAB_IMAGE_VULKAN 1
#include "runtime.cuh"
#include <set>
#include <stdexcept>

namespace slopfab::fibo_vulkan {
namespace mv = moss_vulkan;

Tensor rows(const Tensor& x, int first, int count) {
  return krea_vulkan::ops::rows(x, first, count);
}

Tensor join(const Tensor& a, const Tensor& b) {
  return music_gpu::concatenate(a, b);
}

Tensor reshape(const Tensor& x, int rows, int cols) {
  if (rows <= 0 || cols <= 0 || int64_t(rows) * cols != int64_t(x.rows) * x.cols)
    throw std::invalid_argument("Fibo reshape dimensions");
  Tensor y(rows, cols);
  uint32_t n = rows * cols;
  mv::dispatch(*x.engine, x.data, x.data, x.data, y.data,
               {mv::CopyColumns, n, uint32_t(cols), uint32_t(cols), 0}, (n + 255) / 256);
  return y;
}

void copy_columns(const Tensor& x, Tensor& y, int offset) {
  if (x.rows != y.rows || offset < 0 || offset + x.cols > y.cols)
    throw std::invalid_argument("Fibo column copy dimensions");
  uint32_t n = x.rows * x.cols;
  mv::dispatch(*x.engine, x.data, x.data, x.data, y.data,
               {mv::CopyColumns, n, uint32_t(x.cols), uint32_t(y.cols), uint32_t(offset)},
               (n + 255) / 256);
}

Tensor layer_norm(Runtime& rt, const Tensor& x, const std::string& p, bool rms, float eps) {
  Tensor y(x.rows, x.cols);
  auto& w = p.empty() ? x.data : rt.weight(p + ".weight");
  auto& b = p.empty() || rms ? w : rt.weight(p + ".bias");
  if ((!p.empty() && w.layout().elements() != uint64_t(x.cols)) ||
      (!p.empty() && !rms && b.layout().elements() != uint64_t(x.cols)))
    throw std::invalid_argument("Fibo layer norm dimensions");
  mv::dispatch(*x.engine, x.data, w, b, y.data,
               {mv::LayerNorm, uint32_t(x.rows), uint32_t(x.cols), uint32_t(!p.empty()),
                uint32_t(!p.empty() && !rms), uint32_t(rms), mv::float_bits(eps)},
               x.rows);
  return y;
}

Tensor gelu(const Tensor& x, bool approximate) {
  Tensor y(x.rows, x.cols);
  uint32_t n = x.rows * x.cols;
  mv::dispatch(*x.engine, x.data, x.data, x.data, y.data, {mv::Gelu, n, uint32_t(approximate)},
               (n + 255) / 256);
  return y;
}

void rope(Tensor& x, int heads, const std::vector<float>& c, const std::vector<float>& s,
          bool adjacent) {
  krea_vulkan::ops::rope(x, heads, c, s, adjacent);
}

void scatter(Tensor& x, const Tensor& source, const std::vector<int32_t>& indices, bool add) {
  if (x.cols != source.cols || indices.size() != size_t(source.rows))
    throw std::invalid_argument("Fibo scatter dimensions");
  std::set<int32_t> unique;
  std::vector<float> values;
  for (auto i : indices) {
    if (i < 0 || i >= x.rows || !unique.insert(i).second)
      throw std::invalid_argument("Fibo scatter index");
    values.push_back(float(i));
  }
  auto idx = x.engine->upload(values, source.rows, 1, true);
  uint32_t n = source.rows * source.cols;
  mv::dispatch(*x.engine, source.data, idx, idx, x.data,
               {mv::Scatter, n, uint32_t(x.cols), uint32_t(add)}, (n + 255) / 256);
}

void modulate(Tensor& x, const Tensor& m, int start, int end, int field, bool gate) {
  if (start < 0 || end < start || end > x.rows || field < 0 || field > 3 ||
      m.rows * m.cols != 8 * x.cols)
    throw std::invalid_argument("Fibo modulation dimensions");
  uint32_t n = x.rows * x.cols;
  mv::dispatch(*x.engine, x.data, m.data, m.data, x.data,
               {mv::FiboModulate, n, uint32_t(x.cols), uint32_t(start), uint32_t(end),
                uint32_t(field), uint32_t(gate)},
               (n + 255) / 256);
}

void scale(Tensor& x, const Tensor& s) {
  uint32_t n = x.rows * x.cols;
  mv::dispatch(*x.engine, x.data, s.data, s.data, x.data, {mv::Scale, n, uint32_t(x.cols)},
               (n + 255) / 256);
}

void gated_residual(Tensor& x, const Tensor& y, float gate) {
  uint32_t n = x.rows * x.cols;
  mv::dispatch(*x.engine, x.data, y.data, y.data, x.data,
               {mv::GatedResidual, n, 0, mv::float_bits(music_gpu::round_bf(gate))},
               (n + 255) / 256);
}
}
