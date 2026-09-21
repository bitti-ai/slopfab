#pragma once
#include <memory>
#include <vector>

#include "slopfab/safetensors.h"

namespace slopfab::vae {
class AudioEncoder {
public:
  explicit AudioEncoder(const SafeTensors& checkpoint);
  ~AudioEncoder();
  AudioEncoder(const AudioEncoder&) = delete;
  AudioEncoder& operator=(const AudioEncoder&) = delete;
  // Planar stereo at 32 kHz. Pads each channel independently to a multiple
  // of 800. Returns unnormalized posterior mean [2,32,ceil(samples/800)].
  std::vector<float> encode_mean(const float* stereo, int samples);
  // H3's normalized channel-major [2*A,32] fixed condition rows.
  std::vector<float> encode_reference(const float* stereo, int samples);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace slopfab::vae
