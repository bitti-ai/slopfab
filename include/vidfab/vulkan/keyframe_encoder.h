#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vidfab/image.h"
#include "vidfab/safetensors.h"
#include "vidfab/vulkan/runtime.h"

namespace vidfab::vulkan {

struct KeyframeEncoderStats {
  double last_encode_seconds = 0.0;
  uint64_t persistent_bytes = 0;
  uint64_t activation_bytes = 0;
  uint64_t allocator_peak_used_bytes = 0;
  uint64_t allocator_used_bytes = 0;
  uint64_t allocator_reserved_bytes = 0;
  uint64_t descriptor_set_allocations = 0;
  uint32_t operators = 0;
};

// Complete exact Vulkan moments encoder for the H3 video VAE. Checkpoint
// weights are immutable and uploaded once; a three-buffer flat activation
// arena is reused through all six levels without per-block allocation or
// submission. No CUDA symbol or fallback is reachable from this type.
class KeyframeEncoder {
 public:
  KeyframeEncoder();
  ~KeyframeEncoder();
  KeyframeEncoder(KeyframeEncoder&&) noexcept;
  KeyframeEncoder& operator=(KeyframeEncoder&&) noexcept;
  KeyframeEncoder(const KeyframeEncoder&) = delete;
  KeyframeEncoder& operator=(const KeyframeEncoder&) = delete;

  static KeyframeEncoder create(const Device& device);
  void load(const SafeTensors& checkpoint);
  void unload() noexcept;
  bool loaded() const noexcept;

  // ImageNet-normalized planar [3,H,W] -> moments [48,H/16,W/16].
  std::vector<float> encode_moments(const float* pixels, int height, int width);
  std::vector<float> encode_condition_rows(
      const RGBImage& image, const float* normal,
      const std::vector<float>& latents_mean,
      const std::vector<float>& latents_std);
  std::vector<float> encode_reference_image(
      const RGBImage& image, const std::vector<float>& latents_mean,
      const std::vector<float>& latents_std);

  const KeyframeEncoderStats& stats() const noexcept;

 private:
  struct Impl;
  explicit KeyframeEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
