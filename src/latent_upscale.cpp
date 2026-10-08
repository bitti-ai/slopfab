#include "slopfab/latent_upscale.h"
#include "slopfab/upscale.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vae/vit_decoder.h"
#include "upscale/backend.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace slopfab {
namespace {
using namespace upscale_detail;

float silu(float x) {
  return x / (1 + std::exp(-x));
}

class Network {
  std::unique_ptr<Backend> backend_;
  std::map<std::string, Buffer> weights_;
  std::function<bool(int, int)> progress_;
  int done_ = 0, total_;

public:
  Network(const std::string& path, DeviceBackend device, float scale, int segments,
          const std::function<bool(int, int)>& progress)
      : progress_(progress), total_(segments * 38) {
    check();
    SafeTensors file;
    file.open(path);
    validate_latent_upscale_checkpoint(file);
    switch (device) {
#if SLOPFAB_WITH_CUDA
    case DeviceBackend::kCuda:
      backend_ = make_cuda_backend();
      break;
#endif
#if SLOPFAB_WITH_VULKAN
    case DeviceBackend::kVulkan:
      backend_ = make_vulkan_backend();
      break;
#endif
    default:
      throw std::invalid_argument("latent upscale: backend not compiled in");
    }
    // Scale embedding is shared by all frames and segments. Fold its FiLM
    // scale/shift into each output GroupNorm affine, in FP32.
    std::map<std::string, std::vector<float>> host;
    auto read = [&](const std::string& name) {
      auto v = to_f32(file.at(name));
      for (float x : v)
        if (!std::isfinite(x))
          throw std::invalid_argument("latent upscale: non-finite weight " + name);
      return v;
    };
    auto linear = [&](const std::string& name, const std::vector<float>& x) {
      const auto w = read(name + ".weight");
      auto y = read(name + ".bias");
      for (size_t i = 0; i < y.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j)
          y[i] += w[i * x.size() + j] * x[j];
      return y;
    };
    auto emb = linear("embed.0", {scale - 1});
    for (auto& v : emb)
      v = silu(v);
    emb = linear("embed.2", emb);
    for (auto& v : emb)
      v = silu(v);
    for (const char* stage : {"in_blocks.", "out_blocks."})
      for (int i = 0; i < 18; ++i) {
        if (i % 3 == 1)
          continue;
        const std::string p = std::string(stage) + std::to_string(i);
        const auto e = linear(p + ".emb_layers.1", emb);
        auto w = read(p + ".out_norm.weight"), b = read(p + ".out_norm.bias");
        for (int c = 0; c < 512; ++c) {
          w[c] *= 1 + e[c];
          b[c] = b[c] * (1 + e[c]) + e[c + 512];
        }
        host[p + ".out_norm.weight"] = std::move(w);
        host[p + ".out_norm.bias"] = std::move(b);
      }
    for (const auto& entry : file.tensors()) {
      check();
      const auto& name = entry.first;
      if (name.compare(0, 6, "embed.") == 0 || name.find(".emb_layers.") != std::string::npos)
        continue;
      const auto it = host.find(name);
      weights_[name] = backend_->upload(it == host.end() ? read(name) : it->second);
    }
  }

  void check() {
    if (progress_ && !progress_(done_, total_))
      throw UpscaleCancelled();
  }

  std::vector<float> segment(const std::vector<float>& input, int t, int h, int w, int oh, int ow) {
    auto operation = [&](Operation op, const Buffer& x, const Buffer& y, const Buffer& bias, int ci,
                         int co, int kernel = 3) {
      const size_t n = size_t(t) * h * w * co;
      auto out = backend_->allocate(n);
      Parameters p{op, uint32_t(h), uint32_t(w), uint32_t(ci), uint32_t(co), uint32_t(n), 0, 1};
      p.frames = t;
      p.kernel = kernel;
      backend_->run(p, x, y, bias, out);
      return out;
    };
    auto conv = [&](const Buffer& x, const std::string& name, int ci, int co, int k = 3,
                    Operation op = kConv3d) {
      return operation(op, x, weights_.at(name + ".weight"), weights_.at(name + ".bias"), ci, co,
                       k);
    };
    auto norm = [&](const Buffer& x, const std::string& name) {
      return operation(kGroupNormSilu, x, weights_.at(name + ".weight"),
                       weights_.at(name + ".bias"), 512, 512);
    };
    auto x = conv(backend_->upload(input), "conv_in", 24, 512);
    ++done_;
    check();
    for (const char* stage : {"in_blocks.", "out_blocks."}) {
      if (stage[0] == 'o') {
        const size_t n = size_t(t) * oh * ow * 512;
        auto out = backend_->allocate(n);
        Parameters p{kBilinear, uint32_t(oh), uint32_t(ow), 512, 512, uint32_t(n), 0, 0};
        p.frames = t;
        p.source_height = h;
        p.source_width = w;
        backend_->run(p, x, x, x, out);
        x = std::move(out);
        h = oh;
        w = ow;
      }
      for (int i = 0; i < 18; ++i) {
        const std::string p = std::string(stage) + std::to_string(i);
        Buffer branch;
        if (i % 3 == 1) {
          branch = norm(x, p + ".norm");
          branch = conv(branch, p + ".dwconv", 512, 512, 5, kDepthwiseTemporal);
          branch = conv(branch, p + ".pwconv", 512, 512, 1);
        } else {
          branch = norm(x, p + ".in_layers.0");
          branch = conv(branch, p + ".in_layers.2", 512, 512);
          branch = norm(branch, p + ".out_norm");
          branch = conv(branch, p + ".out_layers.2", 512, 512);
        }
        x = operation(kResidual, x, branch, branch, 512, 512);
        ++done_;
        check();
      }
    }
    x = conv(norm(x, "norm_out"), "conv_out", 512, 24);
    auto out = backend_->download(x, size_t(t) * oh * ow * 24);
    ++done_;
    check();
    return out;
  }
};
} // namespace

std::vector<float> upscale_latents(const std::vector<float>& input, int frames, int h, int w,
                                   const std::string& checkpoint, DeviceBackend backend,
                                   const LatentUpscaleOptions& options,
                                   const std::function<bool(int, int)>& progress) {
  const auto dims = latent_upscale_dimensions(h, w, options);
  const int oh = dims.first, ow = dims.second;
  if (frames <= 0 || frames > std::numeric_limits<int>::max() - 32 ||
      uint64_t(frames) > std::numeric_limits<size_t>::max() / (24 * sizeof(float)) / h / w ||
      input.size() != size_t(frames) * h * w * 24)
    throw std::invalid_argument("latent upscale: expected [24,T,H,W] input");
  for (float v : input)
    if (!std::isfinite(v))
      throw std::invalid_argument("latent upscale: non-finite input");
  if (progress && !progress(0, 0))
    throw UpscaleCancelled();
  if (oh == h && ow == w)
    return input;
  const bool chunked = options.temporal_chunking && frames > 32;
  const int segments = chunked ? (frames - 1) / 32 + 1 : 1;
  const int max_t = chunked ? int(std::min(int64_t(frames) + 10, int64_t(52))) : frames;
  if (uint64_t(max_t) > uint64_t(std::numeric_limits<int>::max()) / 512 / oh / ow ||
      uint64_t(frames) > std::numeric_limits<size_t>::max() / (24 * sizeof(float)) / oh / ow)
    throw std::length_error(
        "latent upscale: activation or output dimensions exceed indexing limits");
  Network network(checkpoint, backend, options.scale, segments, progress);
  // The companion node applies these statistics to already-normalized H3
  // sampler latents, then reverses them on output. Preserve that extra affine
  // transform: feeding sampler latents straight into the network corrupts them.
  const auto& mean = vae::default_video_latents_mean();
  const auto& std_dev = vae::default_video_latents_std();
  const size_t src_plane = size_t(h) * w, dst_plane = size_t(oh) * ow;
  std::vector<float> output(size_t(frames) * dst_plane * 24, 0), weights(frames, 0);
  for (int segment = 0; segment < segments; ++segment) {
    const int start = chunked ? segment * 32 : 0,
              end = chunked ? std::min(frames, start + 32) : frames;
    const int first = chunked ? std::max(0, start - 5) : 0;
    const int last = chunked ? std::min(frames, end + 5) : frames;
    const int lo = chunked ? std::max(0, first - 5) : 0;
    const int hi = chunked ? std::min(frames + 10, last + 5) : frames;
    const int t = hi - lo;
    std::vector<float> packed(size_t(t) * src_plane * 24);
    for (int f = 0; f < t; ++f) {
      const int source = chunked ? std::clamp(lo + f - 5, 0, frames - 1) : f;
      for (size_t p = 0; p < src_plane; ++p)
        for (int c = 0; c < 24; ++c)
          packed[(size_t(f) * src_plane + p) * 24 + c] =
              (input[(size_t(c) * frames + source) * src_plane + p] - mean[c]) / std_dev[c];
    }
    const auto out = network.segment(packed, t, h, w, oh, ow);
    for (int f = first; f < last; ++f) {
      const float weight = f < start  ? float(f - first + 1) / (start - first + 1)
                           : f >= end ? float(last - f) / (last - end + 1)
                                      : 1.0f;
      weights[f] += weight;
      const int local = chunked ? f + 5 - lo : f;
      for (size_t p = 0; p < dst_plane; ++p)
        for (int c = 0; c < 24; ++c)
          output[(size_t(c) * frames + f) * dst_plane + p] +=
              out[(size_t(local) * dst_plane + p) * 24 + c] * weight;
    }
  }
  for (int c = 0; c < 24; ++c)
    for (int f = 0; f < frames; ++f)
      for (size_t p = 0; p < dst_plane; ++p) {
        auto& v = output[(size_t(c) * frames + f) * dst_plane + p];
        v = (v / weights[f]) * std_dev[c] + mean[c];
        if (!std::isfinite(v))
          throw std::runtime_error("latent upscale: non-finite output");
      }
  return output;
}
} // namespace slopfab
