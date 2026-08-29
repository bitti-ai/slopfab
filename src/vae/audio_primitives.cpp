#include "vidfab/vae/audio_primitives.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "vidfab/tensor_convert.h"

namespace vidfab::vae {
namespace {

uint64_t checked_product(uint64_t a, uint64_t b, const char* what) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
    throw std::overflow_error(std::string("audio VAE: ") + what + " overflow");
  return a * b;
}

void require_shape(const TensorView& tensor, const std::string& name,
                   const std::vector<int64_t>& shape) {
  if (tensor.dtype != DType::kF32 && tensor.dtype != DType::kF16 &&
      tensor.dtype != DType::kBF16) {
    throw std::runtime_error("audio VAE: " + name + " is not floating point");
  }
  if (tensor.shape != shape)
    throw std::runtime_error("audio VAE: invalid shape for " + name);
}

}  // namespace

uint64_t AudioConv1DDesc::input_elements() const {
  return checked_product(checked_product(batch, in_channels, "conv input"),
                         length_in, "conv input");
}
uint64_t AudioConv1DDesc::output_elements() const {
  return checked_product(checked_product(batch, out_channels, "conv output"),
                         length_out, "conv output");
}
uint64_t AudioConv1DDesc::weight_elements() const {
  return checked_product(checked_product(out_channels, in_channels, "conv weight"),
                         kernel, "conv weight");
}
void AudioConv1DDesc::validate() const {
  if (batch == 0 || in_channels == 0 || out_channels == 0 ||
      length_in == 0 || length_out == 0 || kernel == 0 || dilation == 0)
    throw std::invalid_argument("audio VAE: zero Conv1D extent");
  const uint64_t padded = static_cast<uint64_t>(length_in) + 2ull * padding;
  const uint64_t reach = static_cast<uint64_t>(dilation) * (kernel - 1);
  if (padded <= reach || padded - reach != length_out)
    throw std::invalid_argument("audio VAE: inconsistent Conv1D output length");
  (void)input_elements(); (void)output_elements(); (void)weight_elements();
}

uint64_t AudioConvTranspose1DDesc::input_elements() const {
  return checked_product(checked_product(batch, in_channels, "transpose input"),
                         length_in, "transpose input");
}
uint64_t AudioConvTranspose1DDesc::output_elements() const {
  return checked_product(checked_product(batch, out_channels, "transpose output"),
                         length_out, "transpose output");
}
uint64_t AudioConvTranspose1DDesc::weight_elements() const {
  return checked_product(checked_product(in_channels, out_channels,
                                         "transpose weight"),
                         kernel, "transpose weight");
}
void AudioConvTranspose1DDesc::validate() const {
  if (batch == 0 || in_channels == 0 || out_channels == 0 ||
      length_in == 0 || length_out == 0 || kernel == 0 || stride == 0)
    throw std::invalid_argument("audio VAE: zero ConvTranspose1D extent");
  const uint64_t expanded = static_cast<uint64_t>(length_in - 1) * stride + kernel;
  if (expanded < 2ull * padding || expanded - 2ull * padding != length_out)
    throw std::invalid_argument(
        "audio VAE: inconsistent ConvTranspose1D output length");
  (void)input_elements(); (void)output_elements(); (void)weight_elements();
}

std::vector<float> load_audio_f32_tensor(
    const SafeTensors& checkpoint, const std::string& name,
    const std::vector<int64_t>& shape) {
  const TensorView& tensor = checkpoint.at(name);
  require_shape(tensor, name, shape);
  return to_f32(tensor);
}

AudioConvWeights load_audio_conv_weights(
    const SafeTensors& checkpoint, const std::string& name,
    const std::vector<int64_t>& weight_shape, uint32_t bias_channels,
    bool require_bias) {
  if (weight_shape.size() != 3 || weight_shape[0] <= 0 ||
      weight_shape[1] <= 0 || weight_shape[2] <= 0)
    throw std::invalid_argument("audio VAE: convolution weight shape must be positive rank 3");
  AudioConvWeights result;
  if (const TensorView* plain = checkpoint.find(name + ".weight")) {
    require_shape(*plain, name + ".weight", weight_shape);
    result.weight = to_f32(*plain);
  } else {
    const TensorView& v = checkpoint.at(name + ".weight_v");
    const TensorView& g = checkpoint.at(name + ".weight_g");
    require_shape(v, name + ".weight_v", weight_shape);
    const std::vector<int64_t> g_shape{weight_shape[0], 1, 1};
    require_shape(g, name + ".weight_g", g_shape);
    const std::vector<float> vf = to_f32(v);
    const std::vector<float> gf = to_f32(g);
    const size_t channels = static_cast<size_t>(weight_shape[0]);
    const size_t per_channel = vf.size() / channels;
    result.weight.resize(vf.size());
    for (size_t channel = 0; channel < channels; ++channel) {
      double square_sum = 0.0;
      for (size_t i = 0; i < per_channel; ++i) {
        const float value = vf[channel * per_channel + i];
        square_sum += static_cast<double>(value) * value;
      }
      const float scale = gf[channel] /
          static_cast<float>(std::sqrt(std::max(square_sum, 1.0e-30)));
      for (size_t i = 0; i < per_channel; ++i)
        result.weight[channel * per_channel + i] =
            vf[channel * per_channel + i] * scale;
    }
    result.folded_weight_norm = true;
  }
  if (const TensorView* bias = checkpoint.find(name + ".bias")) {
    if (bias_channels == 0)
      throw std::runtime_error("audio VAE: unexpected bias for " + name);
    require_shape(*bias, name + ".bias", {bias_channels});
    result.bias = to_f32(*bias);
  } else if (require_bias) {
    throw std::runtime_error("audio VAE: missing bias for " + name);
  }
  return result;
}

}  // namespace vidfab::vae
