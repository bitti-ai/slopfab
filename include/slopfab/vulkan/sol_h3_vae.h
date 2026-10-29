#pragma once
#include "slopfab/vae/vit_decoder.h"
#include "slopfab/vulkan/runtime.h"
#include <functional>

namespace slopfab::vulkan {
class SolH3Decoder {
public:
  explicit SolH3Decoder(const Device& device);
  ~SolH3Decoder();
  SolH3Decoder(SolH3Decoder&&) noexcept;
  SolH3Decoder& operator=(SolH3Decoder&&) noexcept;
  void load(const SafeTensors& checkpoint);
  size_t weight_bytes() const;
  vae::DecodedVideo decode(const float* normalized, int time, int height, int width,
                           const std::vector<float>& mean, const std::vector<float>& std_dev,
                           const vae::DecodeSchedule& unused = {});
  std::function<bool(int, int)> progress;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace slopfab::vulkan
