#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "slopfab/safetensors.h"
#include "slopfab/vae/audio_decoder.h"
#include "slopfab/vulkan/runtime.h"

namespace slopfab::vulkan {

// Complete exact Vulkan implementation of the MiniMax H3 audio VAE decoder.
// The graph and all 779 decoder tensors remain device resident; one decode is
// recorded as a single bounded transaction and returns interleaved stereo.
class AudioDecoder {
public:
  AudioDecoder();
  ~AudioDecoder();
  AudioDecoder(AudioDecoder&&) noexcept;
  AudioDecoder& operator=(AudioDecoder&&) noexcept;
  AudioDecoder(const AudioDecoder&) = delete;
  AudioDecoder& operator=(const AudioDecoder&) = delete;

  static AudioDecoder create(const Device& device);
  void load(const SafeTensors& checkpoint, const vae::AudioVAEConfig& config = {});
  void unload();
  const vae::AudioVAEConfig& config() const;
  const std::vector<float>& latents_mean() const;
  const std::vector<float>& latents_std() const;
  vae::DecodedAudio decode(const float* latents, int num_latents,
                           vae::AudioDecodeTrace* trace = nullptr);

  uint64_t weight_bytes() const noexcept;
  uint64_t peak_device_bytes() const noexcept;
  uint64_t allocator_used_bytes() const noexcept;
  uint64_t allocator_reserved_bytes() const noexcept;
  uint64_t load_pool_high_water_bytes() const noexcept;
  uint64_t decode_pool_high_water_bytes() const noexcept;
  uint64_t staging_capacity_bytes() const noexcept;
  uint64_t host_loader_peak_bytes() const noexcept;
  uint64_t descriptor_set_allocations() const noexcept;
  uint32_t recorded_operators() const noexcept;

private:
  struct Impl;
  explicit AudioDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace slopfab::vulkan
