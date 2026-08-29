#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "vidfab/dit/packing.h"
#include "vidfab/sampler/scheduler.h"
#include "vidfab/vulkan/dit_transformer.h"

namespace vidfab::vulkan {

// Exact pruned-table T2VA denoise orchestration. The transformer, latent rows,
// velocity rows, RoPE and attention ranges remain on one Vulkan context for
// the complete trajectory. Only small per-step control tensors are uploaded;
// fp32 modality rows cross the host boundary once at prepare and once at the
// final result.
struct ExactH3DenoiseConfig {
  ExactH3TransformerConfig transformer;
  dit::SequenceLayout layout;
  dit::PackedIndices indices;
  std::vector<double> position_ids;
  int attention_band = 0;
};

struct ExactH3DenoiseResult {
  std::vector<float> video_rows;
  std::vector<float> audio_rows;
  uint32_t steps_completed = 0;
  bool cancelled = false;
};

using ExactH3DenoiseProgress = std::function<bool(uint32_t step,
                                                  uint32_t total_steps)>;

class ExactH3Denoiser {
 public:
  ExactH3Denoiser();
  ~ExactH3Denoiser();
  ExactH3Denoiser(ExactH3Denoiser&&) noexcept;
  ExactH3Denoiser& operator=(ExactH3Denoiser&&) noexcept;
  ExactH3Denoiser(const ExactH3Denoiser&) = delete;
  ExactH3Denoiser& operator=(const ExactH3Denoiser&) = delete;

  static ExactH3Denoiser create(TensorContext& context,
                                const ExactH3DenoiseConfig& config);
  void load(const SafeTensors& checkpoint);
  void unload() noexcept;
  bool loaded() const noexcept;
  bool prepared() const noexcept;
  const ExactH3DenoiseConfig& config() const noexcept;

  // All host spans must exactly match the configured text/video/audio rows.
  // This is the only input activation boundary in a trajectory.
  void prepare(const float* prompt, uint64_t prompt_elements,
               const float* video_rows, uint64_t video_elements,
               const float* audio_rows, uint64_t audio_elements);

  // Only exact Euler is accepted. The two schedules must describe the same
  // number of evaluations. Progress is called after each completed update;
  // false returns the current, consistently updated device trajectory.
  ExactH3DenoiseResult run(const sampler::FlowScheduler& video,
                           const sampler::FlowScheduler& audio,
                           const ExactH3DenoiseProgress& progress = {});

  uint64_t persistent_bytes() const noexcept;
  uint64_t scratch_bytes() const noexcept;
  uint64_t peak_device_bytes() const noexcept;

 private:
  struct Impl;
  explicit ExactH3Denoiser(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
