#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vidfab/safetensors.h"
#include "vidfab/vae/vit_decoder.h"
#include "vidfab/vulkan/runtime.h"

namespace vidfab::vulkan {

// Complete exact Vulkan window decoder around the 36-block video-VAE graph.
// It owns no CUDA resource and fails during construction when the exact Vulkan
// arithmetic contract is unavailable.
class VideoVaeDecoder final : public vae::VideoVaeWindowBackend {
 public:
  VideoVaeDecoder();
  ~VideoVaeDecoder();
  VideoVaeDecoder(VideoVaeDecoder&&) noexcept;
  VideoVaeDecoder& operator=(VideoVaeDecoder&&) noexcept;
  VideoVaeDecoder(const VideoVaeDecoder&) = delete;
  VideoVaeDecoder& operator=(const VideoVaeDecoder&) = delete;

  static VideoVaeDecoder create(const Device& device,
                                const vae::ViTConfig& config);
  void load(const SafeTensors& checkpoint);
  const vae::ViTConfig& config() const override;

  void forward_window(const float* latent, int time, int height, int width,
                      std::vector<float>& output);
  void forward_windows(const float* latent, int batch, int time, int height,
                       int width, std::vector<std::vector<float>>& output,
                       const size_t* slots) override;
  void release_host_registrations() override {}
  void denormalize_latents(
      const float* normalized, int channels, uint64_t voxels,
      const std::vector<float>& mean, const std::vector<float>& std_dev,
      std::vector<float>& output) override;
  vae::DecodedVideo decode(
      const float* normalized_latent, int time, int height, int width,
      const std::vector<float>& mean, const std::vector<float>& std_dev,
      const vae::DecodeSchedule& schedule = {});

  uint64_t persistent_bytes() const noexcept;
  uint64_t peak_device_bytes() const noexcept;
  uint64_t allocator_reserved_bytes() const noexcept;
  uint64_t allocator_used_bytes() const noexcept;
  uint64_t descriptor_set_allocations() const noexcept;
  uint32_t cached_shapes() const noexcept;
  uint32_t operators_per_document() const noexcept;

 private:
  struct Impl;
  explicit VideoVaeDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
