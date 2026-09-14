#pragma once
#include <memory>
#include <vector>

#include "slopfab/image.h"
#include "slopfab/safetensors.h"
#include "slopfab/vulkan/runtime.h"

namespace slopfab::vulkan {
// FP32 Vulkan reference graphs. All neural operations execute on this device;
// host work is limited to checkpoint loading, preprocessing and latent packing.
// Video weights follow the FP16/BF16 keyframe checkpoint contract. Audio
// weights may be FP32/FP16/BF16; neither graph accepts NF4 weights.
class ReferenceEncoder {
 public:
  ReferenceEncoder(const Device& device, const SafeTensors& checkpoint,
                   bool audio);
  ~ReferenceEncoder();
  std::vector<float> encode_temporal_moments(const float*, int frames,
                                             int height, int width);
  std::vector<float> encode_reference_video(const std::vector<RGBImage>&,
                                            int frames,
                                            const std::vector<float>& mean,
                                            const std::vector<float>& stddev);
  std::vector<float> encode_mean(const float* stereo, int samples);
  std::vector<float> encode_reference(const float* stereo, int samples);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace slopfab::vulkan
