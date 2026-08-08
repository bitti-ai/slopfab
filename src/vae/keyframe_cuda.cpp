#include "vidfab/vae/keyframe_encoder.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/keyframe_encoder.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nf4_weight.cuh"
#include "vidfab/nf4.h"
#include "vidfab/tensor_convert.h"

namespace vidfab::vae {
namespace {

using cuda::DeviceBuffer;

struct ConvWeight {
  cuda::F16Weight weight;
  DeviceBuffer<__half> bias;
};

DeviceBuffer<__half> upload_half(const SafeTensors& checkpoint, const std::string& name,
                                 cudaStream_t stream) {
  const TensorView& view = checkpoint.at(name);
  if (is_nf4_weight(checkpoint, name)) {
    const NF4State state = read_nf4_state(checkpoint, name, "keyframe encoder");
    size_t logical = 1;
    for (int64_t dim : state.shape) logical *= static_cast<size_t>(dim);
    const TensorView& absmax = checkpoint.at(name + ".absmax");
    const TensorView& qmap = checkpoint.at(name + ".quant_map");
    const TensorView& nested_map = checkpoint.at(name + ".nested_quant_map");
    const TensorView& nested_absmax = checkpoint.at(name + ".nested_absmax");
    DeviceBuffer<uint8_t> codes(view.nbytes), scales(absmax.nbytes);
    DeviceBuffer<float> qm(static_cast<size_t>(qmap.numel()));
    DeviceBuffer<float> nm(static_cast<size_t>(nested_map.numel()));
    DeviceBuffer<float> na(static_cast<size_t>(nested_absmax.numel()));
    DeviceBuffer<__half> out(logical);
    codes.copy_from_host(static_cast<const uint8_t*>(view.data), view.nbytes, stream);
    scales.copy_from_host(static_cast<const uint8_t*>(absmax.data), absmax.nbytes, stream);
    qm.copy_from_host(static_cast<const float*>(qmap.data), qm.size(), stream);
    nm.copy_from_host(static_cast<const float*>(nested_map.data), nm.size(), stream);
    na.copy_from_host(static_cast<const float*>(nested_absmax.data), na.size(), stream);
    cuda::launch_dequant_nf4_f16(codes.get(), scales.get(), qm.get(), nm.get(), na.get(),
                                  state.block_size, state.nested_block_size, state.nested_offset,
                                  out.get(), logical, stream);
    VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream));
    return out;
  }
  DeviceBuffer<__half> out(static_cast<size_t>(view.numel()));
  if (view.dtype == DType::kF16) {
    out.copy_from_host(static_cast<const __half*>(view.data), out.size(), stream);
  } else {
    const std::vector<float> f = to_f32(view);
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half_rn(f[i]);
    out.copy_from_host(h.data(), h.size(), stream);
  }
  VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream));
  return out;
}

ConvWeight upload_conv(const SafeTensors& checkpoint, const std::string& name,
                       cudaStream_t stream) {
  const TensorView& w = checkpoint.at(name + ".weight");
  ConvWeight result;
  size_t elements = static_cast<size_t>(w.numel());
  if (is_nf4_weight(checkpoint, name + ".weight")) {
    elements = 1;
    for (int64_t dim : read_nf4_state(checkpoint, name + ".weight", "keyframe encoder").shape)
      elements *= static_cast<size_t>(dim);
  }
  result.weight.load(checkpoint, name + ".weight", elements, stream, "keyframe encoder");
  result.bias = upload_half(checkpoint, name + ".bias", stream);
  return result;
}

struct NormWeight {
  DeviceBuffer<__half> weight;
  DeviceBuffer<__half> bias;
};

NormWeight upload_norm(const SafeTensors& checkpoint, const std::string& name, cudaStream_t stream) {
  const TensorView& w = checkpoint.at(name + ".weight");
  const TensorView& b = checkpoint.at(name + ".bias");
  NormWeight result;
  result.weight = upload_half(checkpoint, name + ".weight", stream);
  result.bias = upload_half(checkpoint, name + ".bias", stream);
  return result;
}

}  // namespace

struct KeyframeEncoder::Impl {
  std::map<std::string, ConvWeight> convs;
  std::map<std::string, NormWeight> norms;
  cuda::Stream stream;
  DeviceBuffer<__half> weight_workspace;
  size_t weight_workspace_elements = 0;

  explicit Impl(const SafeTensors& checkpoint) {
    validate_keyframe_encoder_weights(checkpoint);
    constexpr int channels[] = {128, 256, 256, 512, 512, 1024};
    constexpr int down[] = {2, 2, 2, 2, 1, 1};
    convs.emplace("encoder.conv_in", upload_conv(checkpoint, "encoder.conv_in", stream.get()));
    int previous = 128;
    for (int level = 0; level < 6; ++level) {
      const int output = channels[level];
      for (int block = 0; block < 2; ++block) {
        const int input = block == 0 ? previous : output;
        const std::string p = "encoder.down." + std::to_string(level) + ".block." +
                              std::to_string(block);
        norms.emplace(p + ".norm1", upload_norm(checkpoint, p + ".norm1", stream.get()));
        norms.emplace(p + ".norm2", upload_norm(checkpoint, p + ".norm2", stream.get()));
        convs.emplace(p + ".conv1", upload_conv(checkpoint, p + ".conv1", stream.get()));
        convs.emplace(p + ".conv2", upload_conv(checkpoint, p + ".conv2", stream.get()));
        if (input != output)
          convs.emplace(p + ".nin_shortcut", upload_conv(checkpoint, p + ".nin_shortcut", stream.get()));
      }
      if (down[level] == 2) {
        const std::string p = "encoder.down." + std::to_string(level) + ".downsample.conv";
        convs.emplace(p, upload_conv(checkpoint, p, stream.get()));
      }
      previous = output;
    }
    norms.emplace("encoder.norm_out", upload_norm(checkpoint, "encoder.norm_out", stream.get()));
    convs.emplace("encoder.conv_out", upload_conv(checkpoint, "encoder.conv_out", stream.get()));
    convs.emplace("quant_conv", upload_conv(checkpoint, "quant_conv", stream.get()));
    for (const auto& item : convs)
      weight_workspace_elements = std::max(weight_workspace_elements, item.second.weight.elements());
    weight_workspace.allocate(weight_workspace_elements);
    stream.synchronize();
  }

  DeviceBuffer<float> conv(const DeviceBuffer<float>& x, const std::string& name, int cin,
                           int cout, int h, int w, int kernel, int stride = 1,
                           bool asymmetric = false) {
    const int oh = stride == 2 ? h / 2 : h;
    const int ow = stride == 2 ? w / 2 : w;
    DeviceBuffer<float> y(static_cast<size_t>(cout) * oh * ow);
    const ConvWeight& cw = convs.at(name);
    const __half* weight = cw.weight.materialize(weight_workspace.get(), weight_workspace_elements,
                                                 stream.get());
    cuda::launch_keyframe_conv3d(x.get(), weight, cw.bias.get(), y.get(), cin, cout, h,
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

std::vector<float> KeyframeEncoder::encode_condition_rows(
    const RGBImage& image, const float* normal, const std::vector<float>& latents_mean,
    const std::vector<float>& latents_std) {
  if (!normal) throw std::runtime_error("keyframe encoder: missing posterior normal field");
  const std::vector<float> pixels = prepare_keyframe_pixels(image);
  const std::vector<float> moments = encode_moments(pixels.data(), image.height, image.width);
  const int latent_h = image.height / 16;
  const int latent_w = image.width / 16;
  const std::vector<float> latents = sample_keyframe_latents(
      moments.data(), normal, latent_h, latent_w, latents_mean, latents_std);
  return patchify_keyframe_latents(latents.data(), latent_h, latent_w);
}

std::vector<float> KeyframeEncoder::encode_reference_image(
    const RGBImage& image, const std::vector<float>& latents_mean,
    const std::vector<float>& latents_std) {
  if (image.height <= 0 || image.width <= 0 || image.height % 16 || image.width % 16)
    throw std::runtime_error("keyframe encoder: reference dimensions must be multiples of 16");
  const size_t count = static_cast<size_t>(24) * (image.height / 16) * (image.width / 16);
  const std::vector<float> normal = torch_cpu_normal_seed42(count);
  return encode_condition_rows(image, normal.data(), latents_mean, latents_std);
}

}  // namespace vidfab::vae
