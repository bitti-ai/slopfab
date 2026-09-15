#include "slopfab/vae/audio_encoder.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

#include "slopfab/cuda/reference_encoder.cuh"
#include "slopfab/vae/audio_primitives.h"

namespace slopfab::vae {
using cuda::DeviceBuffer;
struct AudioEncoder::Impl {
  cuda::Stream stream;
  cuda::ReferenceEncoderOps ops{stream.get()};
  std::map<std::string, DeviceBuffer<float>> weights;
  std::vector<float> mean, stddev;
  const float* at(const std::string& name) const {
    return weights.at(name).get();
  }
  void put(const std::string& name, const std::vector<float>& values) {
    DeviceBuffer<float> buffer(values.size());
    buffer.copy_from_host(values.data(), values.size(), stream.get());
    // Host vectors can be temporary; finish their upload before returning.
    stream.synchronize();
    weights.emplace(name, std::move(buffer));
  }
  explicit Impl(const SafeTensors& checkpoint) {
    auto tensor = [&](const std::string& name, std::vector<int64_t> shape) {
      put(name, load_audio_f32_tensor(checkpoint, name, shape));
    };
    auto conv = [&](const std::string& name, int ci, int co, int k) {
      const auto weight =
          load_audio_conv_weights(checkpoint, name, {co, ci, k}, co, true);
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
    mean = load_audio_f32_tensor(checkpoint, "latents_mean", {32});
    stddev = load_audio_f32_tensor(checkpoint, "latents_std", {32});
    for (int i = 0; i < 32; ++i)
      if (!std::isfinite(mean[i]) || !std::isfinite(stddev[i]) ||
          stddev[i] <= 0)
        throw std::runtime_error("audio encoder: invalid latent statistics");
  }
};

AudioEncoder::AudioEncoder(const SafeTensors& checkpoint)
    : impl_(std::make_unique<Impl>(checkpoint)) {}
AudioEncoder::~AudioEncoder() = default;
std::vector<float> AudioEncoder::encode_mean(const float* stereo, int samples) {
  if (!stereo || samples <= 0 || samples > 15 * 32000)
    throw std::invalid_argument(
        "audio encoder: expected 1..480000 stereo samples");
  const int padded = ((samples + 799) / 800) * 800;
  std::vector<float> host(size_t(2) * padded, 0);
  for (int b = 0; b < 2; ++b)
    std::copy_n(stereo + size_t(b) * samples, samples,
                host.data() + size_t(b) * padded);
  auto& ops = impl_->ops;
  auto x = ops.allocate<float>(host.size());
  x.copy_from_host(host.data(), host.size(), impl_->stream.get());
  auto w = [&](const std::string& n) { return impl_->at(n); };
  auto conv = [&](const cuda::ReferenceBuffer<float>& input, const std::string& n,
                  int ci, int co, int len, int k, int stride = 1, int pad = 0,
                  int dil = 1) {
    return ops.conv1d(input.get(), w(n + ".weight"), w(n + ".bias"), 2, ci, co,
                      len, k, stride, pad, dil);
  };
  int length = padded, channels = 64;
  x = conv(x, "encoder.block.0", 1, 64, length, 7, 1, 3);
  const int strides[] = {2, 4, 4, 5, 5}, dilations[] = {1, 3, 9};
  for (int stage = 0; stage < 5; ++stage) {
    const std::string prefix =
        "encoder.block." + std::to_string(stage + 1) + ".block.";
    for (int block = 0; block < 3; ++block) {
      const std::string p = prefix + std::to_string(block) + ".block.";
      auto branch = ops.allocate<float>(x.size());
      SLOPFAB_CUDA_CHECK(
          cudaMemcpyAsync(branch.get(), x.get(), x.size() * sizeof(float),
                          cudaMemcpyDeviceToDevice, impl_->stream.get()));
      ops.snake(branch.get(), w(p + "0.alpha"), 2, channels, length);
      branch = conv(branch, p + "1", channels, channels, length, 7, 1,
                    3 * dilations[block], dilations[block]);
      ops.snake(branch.get(), w(p + "2.alpha"), 2, channels, length);
      branch = conv(branch, p + "3", channels, channels, length, 1);
      ops.add(x.get(), branch.get(), x.size());
    }
    ops.snake(x.get(), w(prefix + "3.alpha"), 2, channels, length);
    const int stride = strides[stage], pad = (stride + 1) / 2, k = 2 * stride;
    x = conv(x, prefix + "4", channels, channels * 2, length, k, stride, pad);
    length = (length + 2 * pad - k) / stride + 1;
    channels *= 2;
  }
  ops.snake(x.get(), w("encoder.block.6.alpha"), 2, 2048, length);
  x = conv(x, "encoder.block.7", 2048, 2048, length, 3, 1, 1);
  x = ops.transpose(x.get(), 2, 2048, length);
  const int rows = 2 * length;
  auto norm = [&](const float* input, const std::string& n, int width) {
    return ops.norm(input, w(n + ".weight"), w(n + ".bias"), rows, width);
  };
  auto linear = [&](const float* input, const std::string& n, int ci, int co) {
    return ops.linear(input, w(n + ".weight"), w(n + ".bias"), rows, ci, co);
  };
  auto q = norm(x.get(), "pre_block.norm1", 2048);
  q = ops.linear(q.get(), w("pre_block.attn.qkv.weight"), nullptr, rows, 2048,
                 6144);
  q = ops.attention(q.get(), w("pre_block.attn.q_bias"),
                    w("pre_block.attn.zero_k_bias"), w("pre_block.attn.v_bias"),
                    2, length);
  q = linear(q.get(), "pre_block.attn.proj", 32, 32);
  x = norm(x.get(), "pre_block.norm3", 2048);
  x = linear(x.get(), "pre_block.proj", 2048, 32);
  ops.add(x.get(), q.get(), x.size());
  q.reset();
  auto branch = norm(x.get(), "pre_block.norm2", 32);
  branch = norm(branch.get(), "pre_block.mlp.norm", 32);
  auto gate = linear(branch.get(), "pre_block.mlp.w1", 32, 64);
  branch = linear(branch.get(), "pre_block.mlp.w0", 32, 64);
  ops.geglu(branch.get(), gate.get(), branch.size());
  branch = linear(branch.get(), "pre_block.mlp.w2", 64, 32);
  ops.add(x.get(), branch.get(), x.size());
  x = ops.transpose(x.get(), 2, length, 32);
  x = conv(x, "mean_proj", 32, 32, length, 1);
  if (length != (samples + 799) / 800)
    throw std::logic_error("audio encoder: latent length mismatch");
  std::vector<float> result(x.size());
  x.copy_to_host(result.data(), result.size(), impl_->stream.get());
  impl_->stream.synchronize();
  ops.report_memory("audio encoder");
  return result;
}
std::vector<float> AudioEncoder::encode_reference(const float* stereo,
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
}  // namespace slopfab::vae
