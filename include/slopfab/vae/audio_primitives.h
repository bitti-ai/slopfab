// Backend-neutral contracts and typed checkpoint loading for the audio VAE's
// exact CUDA/Vulkan primitive substrate.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "slopfab/safetensors.h"

namespace slopfab::vae {

struct AudioConv1DDesc {
  uint32_t batch = 0;
  uint32_t in_channels = 0;
  uint32_t out_channels = 0;
  uint32_t length_in = 0;
  uint32_t length_out = 0;
  uint32_t kernel = 0;
  uint32_t padding = 0;
  uint32_t dilation = 1;

  uint64_t input_elements() const;
  uint64_t output_elements() const;
  uint64_t weight_elements() const;
  void validate() const;
};

struct AudioConvTranspose1DDesc {
  uint32_t batch = 0;
  uint32_t in_channels = 0;
  uint32_t out_channels = 0;
  uint32_t length_in = 0;
  uint32_t length_out = 0;
  uint32_t kernel = 0;
  uint32_t stride = 0;
  uint32_t padding = 0;

  uint64_t input_elements() const;
  uint64_t output_elements() const;
  uint64_t weight_elements() const;
  void validate() const;
};

struct AudioConvWeights {
  // Conv1D: [Cout,Cin,K]. ConvTranspose1D: [Cin,Cout,K].
  std::vector<float> weight;
  std::vector<float> bias;
  bool folded_weight_norm = false;
};

// Loads a plain `name.weight`, or folds legacy `weight_g`/`weight_v` along
// dimension zero with one shared implementation for both backends. Bias is
// optional only when `require_bias` is false.
AudioConvWeights load_audio_conv_weights(const SafeTensors& checkpoint, const std::string& name,
                                         const std::vector<int64_t>& weight_shape,
                                         uint32_t bias_channels, bool require_bias);

std::vector<float> load_audio_f32_tensor(const SafeTensors& checkpoint, const std::string& name,
                                         const std::vector<int64_t>& shape);

} // namespace slopfab::vae
