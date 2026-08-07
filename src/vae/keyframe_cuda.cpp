#include "vidfab/vae/keyframe_encoder.h"

#include <cuda_fp16.h>

#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/keyframe_encoder.cuh"

namespace vidfab::vae {
namespace {

using cuda::DeviceBuffer;

struct ConvWeight {
  DeviceBuffer<__half> weight;
  DeviceBuffer<__half> bias;
};

ConvWeight upload_conv(const SafeTensors& checkpoint, const std::string& name) {
  const TensorView& w = checkpoint.at(name + ".weight");
  const TensorView& b = checkpoint.at(name + ".bias");
  ConvWeight result;
  result.weight.allocate(static_cast<size_t>(w.numel()));
  result.bias.allocate(static_cast<size_t>(b.numel()));
  result.weight.copy_from_host(static_cast<const __half*>(w.data), result.weight.size());
  result.bias.copy_from_host(static_cast<const __half*>(b.data), result.bias.size());
  return result;
}

struct NormWeight {
  DeviceBuffer<__half> weight;
  DeviceBuffer<__half> bias;
};

NormWeight upload_norm(const SafeTensors& checkpoint, const std::string& name) {
  const TensorView& w = checkpoint.at(name + ".weight");
  const TensorView& b = checkpoint.at(name + ".bias");
  NormWeight result;
  result.weight.allocate(static_cast<size_t>(w.numel()));
  result.bias.allocate(static_cast<size_t>(b.numel()));
  result.weight.copy_from_host(static_cast<const __half*>(w.data), result.weight.size());
  result.bias.copy_from_host(static_cast<const __half*>(b.data), result.bias.size());
  return result;
}

}  // namespace

struct KeyframeEncoder::Impl {
  std::map<std::string, ConvWeight> convs;
  std::map<std::string, NormWeight> norms;
  cuda::Stream stream;

  explicit Impl(const SafeTensors& checkpoint) {
    validate_keyframe_encoder_weights(checkpoint);
    constexpr int channels[] = {128, 256, 256, 512, 512, 1024};
    constexpr int down[] = {2, 2, 2, 2, 1, 1};
    convs.emplace("encoder.conv_in", upload_conv(checkpoint, "encoder.conv_in"));
    int previous = 128;
    for (int level = 0; level < 6; ++level) {
      const int output = channels[level];
      for (int block = 0; block < 2; ++block) {
        const int input = block == 0 ? previous : output;
        const std::string p = "encoder.down." + std::to_string(level) + ".block." +
                              std::to_string(block);
        norms.emplace(p + ".norm1", upload_norm(checkpoint, p + ".norm1"));
        norms.emplace(p + ".norm2", upload_norm(checkpoint, p + ".norm2"));
        convs.emplace(p + ".conv1", upload_conv(checkpoint, p + ".conv1"));
        convs.emplace(p + ".conv2", upload_conv(checkpoint, p + ".conv2"));
        if (input != output)
          convs.emplace(p + ".nin_shortcut", upload_conv(checkpoint, p + ".nin_shortcut"));
      }
      if (down[level] == 2) {
        const std::string p = "encoder.down." + std::to_string(level) + ".downsample.conv";
        convs.emplace(p, upload_conv(checkpoint, p));
      }
      previous = output;
    }
    norms.emplace("encoder.norm_out", upload_norm(checkpoint, "encoder.norm_out"));
    convs.emplace("encoder.conv_out", upload_conv(checkpoint, "encoder.conv_out"));
    convs.emplace("quant_conv", upload_conv(checkpoint, "quant_conv"));
    stream.synchronize();
  }

  DeviceBuffer<float> conv(const DeviceBuffer<float>& x, const std::string& name, int cin,
                           int cout, int h, int w, int kernel, int stride = 1,
                           bool asymmetric = false) {
    const int oh = stride == 2 ? h / 2 : h;
    const int ow = stride == 2 ? w / 2 : w;
    DeviceBuffer<float> y(static_cast<size_t>(cout) * oh * ow);
    const ConvWeight& cw = convs.at(name);
    cuda::launch_keyframe_conv3d(x.get(), cw.weight.get(), cw.bias.get(), y.get(), cin, cout, h,
                                 w, kernel, stride, true, asymmetric, stream.get());
    return y;
  }

  DeviceBuffer<float> norm(const DeviceBuffer<float>& x, const std::string& name, int channels,
                           int h, int w) {
    DeviceBuffer<float> y(static_cast<size_t>(channels) * h * w);
    const NormWeight& nw = norms.at(name);
    cuda::launch_keyframe_groupnorm_silu(x.get(), nw.weight.get(), nw.bias.get(), y.get(),
                                         channels, h, w, 32, 1e-6f, stream.get());
    return y;
  }
};

KeyframeEncoder::KeyframeEncoder(const SafeTensors& checkpoint)
    : impl_(std::make_unique<Impl>(checkpoint)) {}
KeyframeEncoder::~KeyframeEncoder() = default;
KeyframeEncoder::KeyframeEncoder(KeyframeEncoder&&) noexcept = default;
KeyframeEncoder& KeyframeEncoder::operator=(KeyframeEncoder&&) noexcept = default;

std::vector<float> KeyframeEncoder::encode_moments(const float* pixels, int height, int width) {
  if (!pixels || height <= 0 || width <= 0 || height % 16 || width % 16)
    throw std::runtime_error("keyframe encoder: dimensions must be positive multiples of 16");
  DeviceBuffer<float> hbuf(static_cast<size_t>(3) * height * width);
  hbuf.copy_from_host(pixels, hbuf.size(), impl_->stream.get());
  hbuf = impl_->conv(hbuf, "encoder.conv_in", 3, 128, height, width, 3);

  constexpr int channels[] = {128, 256, 256, 512, 512, 1024};
  constexpr int down[] = {2, 2, 2, 2, 1, 1};
  int current = 128;
  int h = height, w = width;
  for (int level = 0; level < 6; ++level) {
    const int output = channels[level];
    for (int block = 0; block < 2; ++block) {
      const int input = current;
      const std::string p = "encoder.down." + std::to_string(level) + ".block." +
                            std::to_string(block);
      DeviceBuffer<float> residual;
      if (input != output) residual = impl_->conv(hbuf, p + ".nin_shortcut", input, output, h, w, 1);
      DeviceBuffer<float> tmp = impl_->norm(hbuf, p + ".norm1", input, h, w);
      tmp = impl_->conv(tmp, p + ".conv1", input, output, h, w, 3);
      tmp = impl_->norm(tmp, p + ".norm2", output, h, w);
      tmp = impl_->conv(tmp, p + ".conv2", output, output, h, w, 3);
      DeviceBuffer<float> sum(static_cast<size_t>(output) * h * w);
      const DeviceBuffer<float>& skip = input == output ? hbuf : residual;
      cuda::launch_keyframe_add(skip.get(), tmp.get(), sum.get(), sum.size(), impl_->stream.get());
      hbuf = std::move(sum);
      current = output;
    }
    if (down[level] == 2) {
      const std::string p = "encoder.down." + std::to_string(level) + ".downsample.conv";
      hbuf = impl_->conv(hbuf, p, current, current, h, w, 3, 2, true);
      h /= 2;
      w /= 2;
    }
  }
  hbuf = impl_->norm(hbuf, "encoder.norm_out", 1024, h, w);
  hbuf = impl_->conv(hbuf, "encoder.conv_out", 1024, 48, h, w, 3);
  hbuf = impl_->conv(hbuf, "quant_conv", 48, 48, h, w, 1);
  std::vector<float> moments(hbuf.size());
  hbuf.copy_to_host(moments.data(), moments.size(), impl_->stream.get());
  impl_->stream.synchronize();
  return moments;
}

}  // namespace vidfab::vae
