// The MiniMax H3 audio VAE decoder: DAC front end plus a BigVGAN vocoder.
//
// 32 latent channels at 40 latents/s become 32 kHz mono; the two stereo
// channels decode as two independent batch items. Upsampling is
// [5, 5, 2, 2, 2, 2, 2] = 800x, which is exactly 32000/40.
//
// The checkpoint is fp32 throughout (605 MB), so there is no quantisation
// story here at all — this is the one stage that can be implemented straight.
// What is not straight is BigVGAN's anti-aliased Snake activation: each one is
// upsample -> snake(x) = x + (1/alpha)*sin^2(alpha*x), with alpha and beta
// per channel -> lowpass -> downsample, using 12-tap filters that ship as
// weights. Getting the filter phase or the padding wrong produces audible
// aliasing rather than an error.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vidfab/safetensors.h"

namespace vidfab::vae {

struct AudioVAEConfig {
  int latent_channels = 32;
  int latent_dim = 2048;
  int decoder_dim = 1024;
  int output_channels = 2;
  int sample_rate = 32000;
  int num_attention_heads = 8;
  bool attn_proj = true;
  std::vector<int> decoder_rates = {5, 5, 2, 2, 2, 2, 2};
  std::vector<int> decoder_kernel_sizes = {9, 9, 4, 4, 4, 4, 4};
  std::vector<int> resblock_kernel_sizes = {3, 7, 11};

  int total_upsample() const {
    int r = 1;
    for (int v : decoder_rates) r *= v;
    return r;
  }
};

// Interleaved stereo float samples in [-1, 1], as the WAV writer and the muxer
// both want them.
struct DecodedAudio {
  int channels = 2;
  int sample_rate = 32000;
  std::vector<float> samples;  // [num_frames * channels], interleaved

  int64_t num_frames() const {
    return channels > 0 ? static_cast<int64_t>(samples.size()) / channels : 0;
  }
};

class AudioDecoder {
 public:
  AudioDecoder();
  ~AudioDecoder();
  AudioDecoder(const AudioDecoder&) = delete;
  AudioDecoder& operator=(const AudioDecoder&) = delete;

  void load(const SafeTensors& checkpoint, const AudioVAEConfig& config = {});
  const AudioVAEConfig& config() const;
  size_t weight_bytes() const;
  void unload();

  // Per-channel latent statistics from the checkpoint's `latents_mean` and
  // `latents_std` tensors, for the caller's de-normalisation step.
  const std::vector<float>& latents_mean() const;
  const std::vector<float>& latents_std() const;

  // `latents` is `[2, 32, A]` fp32, already de-normalised
  // (`z * latents_std + latents_mean`). Returns interleaved stereo.
  DecodedAudio decode(const float* latents, int num_latents);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vidfab::vae
