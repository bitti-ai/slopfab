#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "slopfab/dit/packing.h"
#include "slopfab/dit/motion_cache.h"
#include "slopfab/sampler/scheduler.h"
#include "slopfab/vulkan/dit_transformer.h"

namespace slopfab::vulkan {

// Exact pruned-table H3 denoise orchestration. The transformer, latent rows,
// velocity rows, RoPE and attention ranges remain on one Vulkan context for
// the complete trajectory. Only small per-step control tensors are uploaded;
// fp32 modality rows cross the host boundary once at prepare and once at the
// final result unless MotionCache is enabled. Its host estimator transfers
// target rows during sampling. A still sequence may contain zero audio rows.
struct ExactH3DenoiseConfig {
  ExactH3TransformerConfig transformer;
  dit::SequenceLayout layout;
  dit::PackedIndices indices;
  std::vector<double> position_ids;
  int attention_band = 0;
  // Keep the prepared target audio rows fixed and condition them at t=1.
  bool pin_target_audio = false;
  // Opt-in host estimator. Downloads target latents/velocities while active.
  dit::MotionCacheConfig motion_cache;
  // Optional canonical four-int range record per global 128-query tile.
  // Captured replay can supply the CUDA table verbatim; production normally
  // derives it from attention_band. The two forms are exclusive.
  std::vector<int32_t> attention_ranges;
};

struct ExactH3DenoiseResult {
  std::vector<float> video_rows;
  std::vector<float> audio_rows;
  uint32_t steps_completed = 0;
  uint32_t steps_computed = 0;
  uint32_t steps_skipped = 0;
  bool cancelled = false;
};

using ExactH3DenoiseProgress = std::function<bool(uint32_t step,
                                                  uint32_t total_steps)>;
using ExactH3DenoiseBoundary = std::function<void(
    uint32_t step, const std::vector<float>& video_rows,
    const std::vector<float>& audio_rows)>;

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
               const float* audio_rows, uint64_t audio_elements,
               const H3TransformerTextReplayTaps* taps = nullptr);

  // Only exact Euler is accepted. The two schedules must describe the same
  // number of evaluations. Progress is called after each completed update;
  // false returns the current, consistently updated device trajectory.
  ExactH3DenoiseResult run(const sampler::FlowScheduler& video,
                           const sampler::FlowScheduler& audio,
                           const ExactH3DenoiseProgress& progress = {},
                           const ExactH3DenoiseBoundary& boundary = {},
                           const H3TransformerForwardReplayTaps* taps = nullptr);

  uint64_t persistent_bytes() const noexcept;
  uint64_t scratch_bytes() const noexcept;
  uint64_t peak_device_bytes() const noexcept;
  uint32_t required_step_operators(
      const H3TransformerForwardReplayTaps* taps = nullptr) const;

 private:
  struct Impl;
  explicit ExactH3Denoiser(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace slopfab::vulkan
