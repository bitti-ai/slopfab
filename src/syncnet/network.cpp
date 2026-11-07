#include "network.h"
#include "../demucs/parallel.h"
#if SLOPFAB_WITH_VULKAN
#include "../music/vulkan_runtime.h"
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace slopfab::syncnet {
Image::Image(int height, int width, int channels) : h(height), w(width), c(channels) {
  if (h <= 0 || w <= 0 || c <= 0 || uint64_t(h) * w * c > 64 * 1024 * 1024)
    throw std::runtime_error("SyncNet: invalid or excessive image dimensions");
  data.resize(size_t(h) * w * c);
}
struct Network::Impl {
  Weights weights;
  std::function<void()> check;
#if SLOPFAB_WITH_VULKAN
  std::unique_ptr<music_gpu::Engine> gpu;
  std::map<std::string, vulkan::DeviceTensor> gpu_weights;
#endif
};
Network::Network(Weights weights, const std::string& backend, std::function<void()> check)
    : impl_(std::make_unique<Impl>()) {
  impl_->weights = std::move(weights);
  impl_->check = std::move(check);
  if (backend != "auto" && backend != "cpu" && backend != "vulkan")
    throw std::invalid_argument("SyncNet backend must be auto, cpu or vulkan");
#if SLOPFAB_WITH_VULKAN
  if (backend != "cpu") {
    try {
      impl_->gpu = std::make_unique<music_gpu::Engine>();
    } catch (...) {
      if (backend == "vulkan")
        throw;
    }
  }
#else
  if (backend == "vulkan")
    throw std::runtime_error("SyncNet: Vulkan is not built");
#endif
}
Network::~Network() = default;
std::string Network::backend() const {
#if SLOPFAB_WITH_VULKAN
  if (impl_->gpu)
    return "vulkan";
#endif
  return "cpu";
}
Image Network::conv(const Image& x, const std::string& name, int stride, int pad,
                    int dilation, bool relu) {
  if (impl_->check)
    impl_->check();
  const auto& weight = impl_->weights.at(name + ".weight");
  const auto& bias = impl_->weights.at(name + ".bias").data;
  const int rank = int(weight.shape.size());
  const int kh = rank > 2 ? weight.shape[rank - 2] : 1;
  const int kw = rank > 2 ? weight.shape[rank - 1] : 1;
  const int out = weight.shape[0], k = int(weight.data.size()) / out;
  if (k != kh * kw * x.c || bias.size() != size_t(out))
    throw std::runtime_error("SyncNet: convolution shape mismatch: " + name);
  Image y((x.h + 2 * pad - dilation * (kh - 1) - 1) / stride + 1,
          (x.w + 2 * pad - dilation * (kw - 1) - 1) / stride + 1, out);
  // Bound im2col to 16 MiB, independent of input resolution.
  const int tile = std::max(1, std::min(512, 4 * 1024 * 1024 / k));
#if SLOPFAB_WITH_VULKAN
  vulkan::DenseGemmPlan plan;
  if (impl_->gpu) {
    auto& g = *impl_->gpu;
    if (!impl_->gpu_weights.count(name))
      impl_->gpu_weights.emplace(name, g.upload(weight.data, out, k, true));
    DenseGemmPlanDesc d;
    d.max_rows = tile;
    d.out_features = out;
    d.in_features = k;
    d.mode = DenseGemmMode::kFloat32;
    plan = vulkan::DenseGemmPlan::create(g.context, d);
  }
#endif
  for (int start = 0; start < y.h * y.w; start += tile) {
    if (impl_->check)
      impl_->check();
    int rows = std::min(tile, y.h * y.w - start);
    std::vector<float> col(size_t(rows) * k);
    for (int r = 0; r < rows; ++r) {
      int oy = (start + r) / y.w * stride - pad;
      int ox = (start + r) % y.w * stride - pad;
      int j = 0;
      for (int c = 0; c < x.c; ++c)
        for (int dy = 0; dy < kh; ++dy)
          for (int dx = 0; dx < kw; ++dx, ++j) {
            int iy = oy + dy * dilation, ix = ox + dx * dilation;
            if (iy >= 0 && ix >= 0 && iy < x.h && ix < x.w)
              col[size_t(r) * k + j] = x.data[(size_t(iy) * x.w + ix) * x.c + c];
          }
    }
    float* dest = y.data.data() + size_t(start) * out;
#if SLOPFAB_WITH_VULKAN
    if (impl_->gpu) {
      auto& g = *impl_->gpu;
      auto a = g.upload(col, rows, k, true);
      auto b = g.allocate(rows, out, ScalarType::kFloat32);
      plan.record(g.batch(), a, impl_->gpu_weights.at(name), b, rows);
      g.flush();
      g.context.download_bytes(b, dest, size_t(rows) * out * sizeof(float));
    } else
#endif
    {
      demucs::parallel_for(rows, [&](int r) {
        for (int o = 0; o < out; ++o) {
          float sum = 0;
          for (int j = 0; j < k; ++j)
            sum += col[size_t(r) * k + j] * weight.data[size_t(o) * k + j];
          dest[size_t(r) * out + o] = sum;
        }
      });
    }
    for (int r = 0; r < rows; ++r)
      for (int o = 0; o < out; ++o) {
        auto& v = dest[size_t(r) * out + o];
        v += bias[o];
        if (relu)
          v = std::max(0.f, v);
      }
  }
  return y;
}
void Network::norm(Image& x, const std::string& name) {
  const auto& w = impl_->weights.at(name + ".weight").data;
  const auto& b = impl_->weights.at(name + ".bias").data;
  const auto& mean = impl_->weights.at(name + ".running_mean").data;
  const auto& var = impl_->weights.at(name + ".running_var").data;
  for (int c = 0; c < x.c; ++c) {
    float scale = w.at(c) / std::sqrt(var.at(c) + 1e-5f);
    for (int p = 0; p < x.h * x.w; ++p) {
      auto& v = x.data[size_t(p) * x.c + c];
      v = std::max(0.f, (v - mean.at(c)) * scale + b.at(c));
    }
  }
}
void Network::l2norm(Image& x, const std::string& name) {
  const auto& w = impl_->weights.at(name + ".weight").data;
  for (int p = 0; p < x.h * x.w; ++p) {
    float sum = 0;
    for (int c = 0; c < x.c; ++c) {
      float v = x.data[size_t(p) * x.c + c];
      sum += v * v;
    }
    float norm = std::sqrt(sum) + 1e-10f;
    for (int c = 0; c < x.c; ++c)
      x.data[size_t(p) * x.c + c] *= w.at(c) / norm;
  }
}
Image Network::pool(const Image& x, int kh, int kw, int sh, int sw, int pad, bool ceil) {
  int h = (x.h + 2 * pad - kh + (ceil ? sh - 1 : 0)) / sh + 1;
  int w = (x.w + 2 * pad - kw + (ceil ? sw - 1 : 0)) / sw + 1;
  if ((h - 1) * sh >= x.h + pad) --h;
  if ((w - 1) * sw >= x.w + pad) --w;
  Image y(h, w, x.c);
  for (int r = 0; r < h; ++r)
    for (int s = 0; s < w; ++s)
      for (int c = 0; c < x.c; ++c) {
        float v = -std::numeric_limits<float>::infinity();
        for (int dy = 0; dy < kh; ++dy)
          for (int dx = 0; dx < kw; ++dx) {
            int iy = r * sh + dy - pad, ix = s * sw + dx - pad;
            if (iy >= 0 && ix >= 0 && iy < x.h && ix < x.w)
              v = std::max(v, x.data[(size_t(iy) * x.w + ix) * x.c + c]);
          }
        y.data[(size_t(r) * w + s) * x.c + c] = v;
      }
  return y;
}
std::vector<float> Network::embedding(Image x, bool audio) {
  const std::string prefix = audio ? "netcnnaud." : "netcnnlip.";
  x = conv(x, prefix + "0", audio ? 1 : 2, audio ? 1 : 0);
  norm(x, prefix + "1");
  if (!audio) x = pool(x, 3, 3, 2, 2);
  x = conv(x, prefix + "4", audio ? 1 : 2, 1);
  norm(x, prefix + "5");
  x = pool(x, 3, 3, audio ? 1 : 2, 2, audio ? 0 : 1);
  for (int i : {8, 11, 14}) {
    x = conv(x, prefix + std::to_string(i), 1, 1);
    norm(x, prefix + std::to_string(i + 1));
  }
  x = pool(x, 3, 3, 2, 2);
  x = conv(x, prefix + "18");
  norm(x, prefix + "19");
  if (x.h != 1 || x.w != 1)
    throw std::runtime_error("SyncNet: embedding input must be 224x224x15 or 13x20x1");
  const std::string fc = audio ? "netfcaud." : "netfclip.";
  x = conv(x, fc + "0");
  norm(x, fc + "1");
  x = conv(x, fc + "3");
  return x.data;
}
} // namespace slopfab::syncnet
