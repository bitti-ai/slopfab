#include "slopfab/vulkan/reference_encoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>

#include "slopfab/reference_conditioning.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vae/audio_primitives.h"
#include "slopfab/vae/keyframe_encoder.h"
#include "slopfab/vulkan/tensor.h"

namespace slopfab::vulkan {
namespace {
size_t count(const DeviceTensor& t) { return t.layout().elements(); }
}  // namespace
struct ReferenceEncoder::Impl {
  TensorContext context;
  std::map<std::string, DeviceTensor> weights;
  std::vector<float> mean, stddev;
  bool audio;
  uint64_t peak_used = 0;
  DeviceTensor allocate(uint64_t n) {
    if (!n || n > UINT32_MAX)
      throw std::invalid_argument(
          "Vulkan reference: tensor size exceeds shader indexing");
    return context.allocate(TensorLayout::contiguous(&n, 1));
  }
  DeviceTensor upload(const std::vector<float>& values) {
    auto t = allocate(values.size());
    context.upload_transient(t, values.data(), values.size());
    return t;
  }
  void put(const std::string& name, const std::vector<float>& values) {
    weights.emplace(name, upload(values));
  }
  DeviceTensor& at(const std::string& n) { return weights.at(n); }
  std::vector<float> download(DeviceTensor& t) {
    std::vector<float> result(count(t));
    context.download(t, result.data(), result.size());
    return result;
  }
  DeviceTensor run(const DeviceTensor& x, const DeviceTensor& w,
                   const DeviceTensor& b, std::array<uint32_t, 16> p,
                   uint64_t n, uint32_t groups = 0, uint32_t batches = 1,
                   DeviceTensor* previous = nullptr,
                   DeviceTensor* earliest = nullptr) {
    auto out = allocate(n);
    peak_used = std::max(peak_used, context.pooled_used_bytes());
    p[12] = uint32_t(n);
    auto batch = context.begin_batch();
    batch.reference_operation(const_cast<DeviceTensor&>(x),
                              const_cast<DeviceTensor&>(w),
                              const_cast<DeviceTensor&>(b), out, p.data(),
                              groups ? groups : uint32_t((n + 255) / 256),
                              batches, previous, earliest);
    batch.submit().wait();
    context.collect();
    return out;
  }
  using Video = std::vector<DeviceTensor>;
  Video conv3(Video& x, const std::string& name, int ci, int co, int h,
              int width, int k, int stride = 1, int ts = 1, bool down = false) {
    auto& w = at(name + ".weight");
    auto& b = at(name + ".bias");
    if (count(w) != size_t(co) * ci * k * k * k || count(b) != size_t(co))
      throw std::invalid_argument("Vulkan reference: video weight shape");
    const uint32_t n = (h / stride) * (width / stride);
    Video out;
    for (size_t t = 0; t < x.size(); t += ts) {
      auto* previous = &x[t ? t - 1 : 0];
      auto* earliest = &x[t > 1 ? t - 2 : 0];
      out.push_back(run(
          x[t], w, b,
          {0, uint32_t(ci), uint32_t(co), uint32_t(std::min(t + 1, size_t(k))),
           uint32_t(h), uint32_t(width), uint32_t(k), uint32_t(stride), 1,
           uint32_t(down ? 0 : k / 2), 1, n, 0, 0, 1},
          uint64_t(co) * n, ((n + 15) / 16) * ((co + 15) / 16), 1, previous,
          earliest));
    }
    return out;
  }
  DeviceTensor conv1(const DeviceTensor& x, const std::string& name, int ci,
                     int co, int len, int k, int stride = 1, int pad = 0,
                     int dil = 1) {
    uint32_t n = (len + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    return run(x, at(name + ".weight"), at(name + ".bias"),
               {1, uint32_t(ci), uint32_t(co), uint32_t(len), 0, 0, uint32_t(k),
                uint32_t(stride), 0, uint32_t(pad), uint32_t(dil), n, 0, 0, 1},
               uint64_t(2) * co * n, ((n + 15) / 16) * ((co + 15) / 16), 2);
  }
  DeviceTensor norm3(const DeviceTensor& x, const std::string& name, int c,
                     int t, int h, int w) {
    return run(x, at(name + ".weight"), at(name + ".bias"),
               {3, uint32_t(c), 0, uint32_t(t), uint32_t(h), uint32_t(w)},
               count(x), 32 * t);
  }
  DeviceTensor norm(const DeviceTensor& x, const std::string& name, int rows,
                    int dim) {
    return run(x, at(name + ".weight"), at(name + ".bias"),
               {4, uint32_t(dim), 0, 1}, count(x), rows);
  }
  DeviceTensor linear(const DeviceTensor& x, const std::string& name, int rows,
                      int ci, int co) {
    return run(x, at(name + ".weight"), at(name + ".bias"),
               {2, uint32_t(ci), uint32_t(co), 0, 0, 0, 0, 0, 0, 0, 0,
                uint32_t(rows), 0, 0, 1},
               uint64_t(rows) * co, ((rows + 15) / 16) * ((co + 15) / 16));
  }
  DeviceTensor snake(const DeviceTensor& x, const std::string& name,
                     int channels, int length) {
    return run(x, at(name), x, {5, uint32_t(channels), 0, uint32_t(length)},
               count(x));
  }
  DeviceTensor transpose(const DeviceTensor& x, int rows, int cols) {
    return run(x, x, x, {6, uint32_t(rows), 0, uint32_t(cols)}, count(x));
  }
  void add(DeviceTensor& x, const DeviceTensor& y) {
    x = run(x, y, x, {7}, count(x));
  }
  void geglu(DeviceTensor& x, const DeviceTensor& gate) {
    x = run(x, gate, x, {8}, count(x));
  }
  DeviceTensor attention(const DeviceTensor& q, int length) {
    auto heads = run(q, q, q, {9, 0, 0, uint32_t(length)},
                     uint64_t(2) * length * 2048, 2 * length * 8);
    return run(heads, heads, heads, {10}, uint64_t(2) * length * 32);
  }
  Impl(const Device& device, const SafeTensors& checkpoint, bool is_audio);
};

ReferenceEncoder::ReferenceEncoder(const Device& device,
                                   const SafeTensors& checkpoint, bool audio)
    : impl_(std::make_unique<Impl>(device, checkpoint, audio)) {}
ReferenceEncoder::~ReferenceEncoder() = default;

void ReferenceEncoder::report_memory() const {
  constexpr double gib = 1024.0 * 1024 * 1024;
  std::printf("references  Vulkan encoder tensors: peak %.2f GiB, reserved %.2f GiB\n",
              impl_->peak_used / gib, impl_->context.reserved_bytes() / gib);
}

ReferenceEncoder::Impl::Impl(const Device& device,
                             const SafeTensors& checkpoint, bool is_audio)
    : context(device, [] {
        TensorContextOptions options;
        options.enable_reference_encoder = true;
        return options;
      }()), audio(is_audio) {
  if (!audio) {
    vae::validate_keyframe_encoder_weights(checkpoint);
    for (const auto& entry : checkpoint.tensors()) {
      if (entry.first.rfind("encoder.", 0) != 0 &&
          entry.first.rfind("quant_conv.", 0) != 0)
        continue;
      if (entry.second.dtype != DType::kF16 &&
          entry.second.dtype != DType::kF32 &&
          entry.second.dtype != DType::kBF16)
        throw std::invalid_argument(
            "Vulkan reference video requires floating-point encoder weights");
      put(entry.first, to_f32(entry.second));
    }
    return;
  }
  auto tensor = [&](const std::string& name, std::vector<int64_t> shape) {
    put(name, vae::load_audio_f32_tensor(checkpoint, name, shape));
  };
  auto conv = [&](const std::string& name, int ci, int co, int k) {
    const auto weight =
        vae::load_audio_conv_weights(checkpoint, name, {co, ci, k}, co, true);
    put(name + ".weight", weight.weight);
    put(name + ".bias", weight.bias);
  };
  conv("encoder.block.0", 1, 64, 7);
  int channels = 64;
  const int strides[] = {2, 4, 4, 5, 5};
  for (int stage = 0; stage < 5; ++stage) {
    const std::string prefix =
        "encoder.block." + std::to_string(stage + 1) + ".block.";
    for (int block = 0; block < 3; ++block) {
      const std::string p = prefix + std::to_string(block) + ".block.";
      tensor(p + "0.alpha", {1, channels, 1});
      conv(p + "1", channels, channels, 7);
      tensor(p + "2.alpha", {1, channels, 1});
      conv(p + "3", channels, channels, 1);
    }
    tensor(prefix + "3.alpha", {1, channels, 1});
    conv(prefix + "4", channels, channels * 2, 2 * strides[stage]);
    channels *= 2;
  }
  tensor("encoder.block.6.alpha", {1, 2048, 1});
  conv("encoder.block.7", 2048, 2048, 3);
  for (const auto& name : {"pre_block.norm1", "pre_block.norm3"}) {
    tensor(std::string(name) + ".weight", {2048});
    tensor(std::string(name) + ".bias", {2048});
  }
  for (const auto& name : {"pre_block.norm2", "pre_block.mlp.norm"}) {
    tensor(std::string(name) + ".weight", {32});
    tensor(std::string(name) + ".bias", {32});
  }
  tensor("pre_block.attn.qkv.weight", {6144, 2048});
  for (const auto& name : {"q_bias", "zero_k_bias", "v_bias"})
    tensor(std::string("pre_block.attn.") + name, {2048});
  auto linear = [&](const std::string& name, int ci, int co) {
    tensor(name + ".weight", {co, ci});
    tensor(name + ".bias", {co});
  };
  linear("pre_block.attn.proj", 32, 32);
  linear("pre_block.proj", 2048, 32);
  linear("pre_block.mlp.w0", 32, 64);
  linear("pre_block.mlp.w1", 32, 64);
  linear("pre_block.mlp.w2", 64, 32);
  conv("mean_proj", 32, 32, 1);
  mean = vae::load_audio_f32_tensor(checkpoint, "latents_mean", {32});
  stddev = vae::load_audio_f32_tensor(checkpoint, "latents_std", {32});
  for (int i = 0; i < 32; ++i)
    if (!std::isfinite(mean[i]) || !std::isfinite(stddev[i]) || stddev[i] <= 0)
      throw std::runtime_error("audio encoder: invalid latent statistics");
  std::vector<float> bias;
  for (const auto& name : {"q_bias", "zero_k_bias", "v_bias"}) {
    auto part = vae::load_audio_f32_tensor(
        checkpoint, std::string("pre_block.attn.") + name, {2048});
    bias.insert(bias.end(), part.begin(), part.end());
  }
  put("pre_block.attn.qkv.bias", bias);
}
std::vector<float> ReferenceEncoder::encode_mean(const float* stereo,
                                                 int samples) {
  if (!impl_->audio || !stereo || samples <= 0 || samples > 15 * 32000)
    throw std::invalid_argument(
        "audio encoder: expected 1..480000 stereo samples");
  const int padded = ((samples + 799) / 800) * 800;
  std::vector<float> host(size_t(2) * padded, 0);
  for (int b = 0; b < 2; ++b)
    std::copy_n(stereo + size_t(b) * samples, samples,
                host.data() + size_t(b) * padded);
  auto x = impl_->upload(host);
  auto& ops = *impl_;
  auto conv = [&](const DeviceTensor& input, const std::string& n, int ci,
                  int co, int len, int k, int stride = 1, int pad = 0,
                  int dil = 1) {
    return ops.conv1(input, n, ci, co, len, k, stride, pad, dil);
  };
  int length = padded, channels = 64;
  x = conv(x, "encoder.block.0", 1, 64, length, 7, 1, 3);
  const int strides[] = {2, 4, 4, 5, 5}, dilations[] = {1, 3, 9};
  for (int stage = 0; stage < 5; ++stage) {
    const std::string prefix =
        "encoder.block." + std::to_string(stage + 1) + ".block.";
    for (int block = 0; block < 3; ++block) {
      const std::string p = prefix + std::to_string(block) + ".block.";
      auto branch = ops.snake(x, p + "0.alpha", channels, length);
      branch = conv(branch, p + "1", channels, channels, length, 7, 1,
                    3 * dilations[block], dilations[block]);
      branch = ops.snake(branch, p + "2.alpha", channels, length);
      branch = conv(branch, p + "3", channels, channels, length, 1);
      ops.add(x, branch);
    }
    x = ops.snake(x, prefix + "3.alpha", channels, length);
    const int stride = strides[stage], pad = (stride + 1) / 2, k = 2 * stride;
    x = conv(x, prefix + "4", channels, channels * 2, length, k, stride, pad);
    length = (length + 2 * pad - k) / stride + 1;
    channels *= 2;
  }
  x = ops.snake(x, "encoder.block.6.alpha", 2048, length);
  x = conv(x, "encoder.block.7", 2048, 2048, length, 3, 1, 1);
  x = ops.transpose(x, 2048, length);
  const int rows = 2 * length;
  auto norm = [&](const DeviceTensor& input, const std::string& n, int width) {
    return ops.norm(input, n, rows, width);
  };
  auto linear = [&](const DeviceTensor& input, const std::string& n, int ci,
                    int co) { return ops.linear(input, n, rows, ci, co); };
  auto q = norm(x, "pre_block.norm1", 2048);
  q = ops.linear(q, "pre_block.attn.qkv", rows, 2048, 6144);
  q = ops.attention(q, length);
  q = linear(q, "pre_block.attn.proj", 32, 32);
  x = norm(x, "pre_block.norm3", 2048);
  x = linear(x, "pre_block.proj", 2048, 32);
  ops.add(x, q);
  q = {};
  auto branch = norm(x, "pre_block.norm2", 32);
  branch = norm(branch, "pre_block.mlp.norm", 32);
  auto gate = linear(branch, "pre_block.mlp.w1", 32, 64);
  branch = linear(branch, "pre_block.mlp.w0", 32, 64);
  ops.geglu(branch, gate);
  branch = linear(branch, "pre_block.mlp.w2", 64, 32);
  ops.add(x, branch);
  x = ops.transpose(x, length, 32);
  x = conv(x, "mean_proj", 32, 32, length, 1);
  if (length != (samples + 799) / 800)
    throw std::logic_error("audio encoder: latent length mismatch");
  return impl_->download(x);
}
std::vector<float> ReferenceEncoder::encode_reference(const float* stereo,
                                                      int samples) {
  auto mean = encode_mean(stereo, samples);
  int length = (samples + 799) / 800;
  std::vector<float> rows(mean.size());
  for (int b = 0; b < 2; ++b)
    for (int t = 0; t < length; ++t)
      for (int c = 0; c < 32; ++c)
        rows[(size_t(b) * length + t) * 32 + c] =
            (mean[(size_t(b) * 32 + c) * length + t] - impl_->mean[c]) /
            impl_->stddev[c];
  return rows;
}
std::vector<float> ReferenceEncoder::encode_temporal_moments(
    const float* pixels, int frames, int height, int width) {
  if (impl_->audio || !pixels || frames <= 0 || frames > 17 || height <= 0 ||
      width <= 0 || height % 16 || width % 16)
    throw std::invalid_argument(
        "video encoder: expected 1..17 frames and dimensions divisible by 16");
  auto& ops = *impl_;
  int h = height, w = width;
  using Video = Impl::Video;
  auto conv = [&](Video& x, const std::string& name, int ci, int co, int k,
                  int ss = 1, int ts = 1, bool down = false) {
    return ops.conv3(x, name, ci, co, h, w, k, ss, ts, down);
  };
  auto norm = [&](const Video& x, const std::string& name, int c) {
    Video out;
    for (const auto& frame : x)
      out.push_back(ops.norm3(frame, name, c, 1, h, w));
    return out;
  };
  Video x;
  const size_t plane = size_t(h) * w;
  for (int t = 0; t < frames; ++t) {
    std::vector<float> host(3 * plane);
    for (int c = 0; c < 3; ++c)
      std::copy_n(pixels + (size_t(c) * frames + t) * plane, plane,
                  host.data() + c * plane);
    x.push_back(ops.upload(host));
  }
  x = conv(x, "encoder.conv_in", 3, 128, 3);
  const int channels[] = {128, 256, 256, 512, 512, 1024},
            spatial[] = {2, 2, 2, 2, 1, 1}, temporal[] = {1, 2, 2, 1, 1, 1};
  int current = 128;
  for (int level = 0; level < 6; ++level) {
    for (int block = 0; block < 2; ++block) {
      std::string prefix = "encoder.down." + std::to_string(level) + ".block." +
                           std::to_string(block);
      auto branch = norm(x, prefix + ".norm1", current);
      branch = conv(branch, prefix + ".conv1", current, channels[level], 3);
      branch = norm(branch, prefix + ".norm2", channels[level]);
      branch =
          conv(branch, prefix + ".conv2", channels[level], channels[level], 3);
      if (current != channels[level])
        x = conv(x, prefix + ".nin_shortcut", current, channels[level], 1);
      for (size_t t = 0; t < x.size(); ++t) ops.add(x[t], branch[t]);
      current = channels[level];
    }
    if (spatial[level] > 1 || temporal[level] > 1) {
      x = conv(x, "encoder.down." + std::to_string(level) + ".downsample.conv",
               current, current, 3, spatial[level], temporal[level], true);
      h /= spatial[level];
      w /= spatial[level];
    }
  }
  x = norm(x, "encoder.norm_out", 1024);
  x = conv(x, "encoder.conv_out", 1024, 48, 3);
  x = conv(x, "quant_conv", 48, 48, 1);
  const size_t latent_plane = size_t(h) * w;
  std::vector<float> moments(48 * x.size() * latent_plane);
  for (size_t t = 0; t < x.size(); ++t) {
    auto frame = ops.download(x[t]);
    for (int c = 0; c < 48; ++c)
      std::copy_n(frame.data() + c * latent_plane, latent_plane,
                  moments.data() + (c * x.size() + t) * latent_plane);
  }
  return moments;
}

std::vector<float> ReferenceEncoder::encode_reference_video(
    const std::vector<RGBImage>& frames, int count,
    const std::vector<float>& mean, const std::vector<float>& stddev) {
  if (count < 22 || (count - 5) % 17 || size_t(count) > frames.size())
    throw std::invalid_argument("reference video: input must be 17*n+5 frames");
  const int h = frames.front().height, w = frames.front().width;
  const int latent_frames = (count - 5) / 17 * 5 + 2;
  const size_t plane = size_t(h) * w, latent_plane = size_t(h / 16) * (w / 16);
  std::vector<float> moments(size_t(48) * latent_frames * latent_plane);
  std::vector<float> pixels(size_t(3) * 17 * plane);
  for (int start = 0; start < count; start += 17) {
    for (int t = 0; t < 17; ++t) {
      const auto& image = frames[std::min(start + t, count - 1)];
      if (image.height != h || image.width != w)
        throw std::invalid_argument("reference video: changing dimensions");
      const auto prepared = vae::prepare_keyframe_pixels(image);
      for (int c = 0; c < 3; ++c)
        std::copy_n(prepared.data() + size_t(c) * plane, plane,
                    pixels.data() + (size_t(c) * 17 + t) * plane);
    }
    const auto chunk = encode_temporal_moments(pixels.data(), 17, h, w);
    const int latent_start = start / 17 * 5;
    const int keep = std::min(5, latent_frames - latent_start);
    for (int c = 0; c < 48; ++c)
      std::copy_n(chunk.data() + size_t(c) * 5 * latent_plane,
                  size_t(keep) * latent_plane,
                  moments.data() + (size_t(c) * latent_frames + latent_start) *
                                       latent_plane);
  }
  auto normal =
      vae::torch_cpu_normal_seed42(size_t(24) * latent_frames * latent_plane);
  auto latent = vae::sample_keyframe_latents(moments.data(), normal.data(),
                                             latent_frames * (h / 16), w / 16,
                                             mean, stddev);
  return patchify_reference_video(latent.data(), latent_frames, h / 16, w / 16);
}

}  // namespace slopfab::vulkan
