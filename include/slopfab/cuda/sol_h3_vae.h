#pragma once
#include "slopfab/vae/vit_decoder.h"
#include <functional>

namespace slopfab::cuda {
// Native CUDA Sol-H3-VAE. Whole-clip window attention, no tile cross-fades.
class SolH3Decoder {
public:
  SolH3Decoder();
  ~SolH3Decoder();
  SolH3Decoder(SolH3Decoder&&) noexcept;
  SolH3Decoder& operator=(SolH3Decoder&&) noexcept;
  void load(const SafeTensors& checkpoint);
  size_t weight_bytes() const;
  vae::DecodedVideo decode(const float* normalized, int time, int height, int width,
                           const std::vector<float>& mean, const std::vector<float>& std_dev,
                           const vae::DecodeSchedule& unused = {});
  // Returning false cancels between transformer layers.
  std::function<bool(int, int)> progress;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace slopfab::cuda
