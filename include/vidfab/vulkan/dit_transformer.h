#pragma once

#include <cstdint>
#include <memory>

#include "vidfab/vulkan/dit_graph.h"

namespace vidfab::vulkan {

// One exact H3 transformer evaluation around the accepted 50-block main
// graph. Packing/index construction and timestep lookup stay backend-neutral
// host work; every neural operation and packed residual remains on Vulkan.
struct ExactH3TransformerConfig {
  H3MainGraphConfig main;
  uint32_t text_rows = 0;
  uint32_t video_rows = 0;
  uint32_t audio_rows = 0;
  uint32_t text_dim = 5120;
  uint32_t video_dim = 96;
  uint32_t audio_dim = 32;
  uint32_t refiner_layers = 2;
};

class ExactH3Transformer {
 public:
  ExactH3Transformer();
  ~ExactH3Transformer();
  ExactH3Transformer(ExactH3Transformer&&) noexcept;
  ExactH3Transformer& operator=(ExactH3Transformer&&) noexcept;
  ExactH3Transformer(const ExactH3Transformer&) = delete;
  ExactH3Transformer& operator=(const ExactH3Transformer&) = delete;

  static ExactH3Transformer create(TensorContext& context,
                                   const ExactH3TransformerConfig& config);
  void load(const SafeTensors& checkpoint);
  void unload() noexcept;
  bool loaded() const noexcept;
  bool text_prepared() const noexcept;
  const ExactH3TransformerConfig& config() const noexcept;

  // prompt is contiguous fp32 [text_rows,text_dim]. The cached final-refiner
  // BF16 stream remains device-resident for every subsequent evaluation.
  // One bounded submission; marks the cache ready only after completion.
  void prepare_text(DeviceTensor& prompt);
  uint32_t required_prepare_text_operators() const;

  // Inputs/outputs are contiguous fp32 modality rows. Indices are the trusted
  // backend-neutral packed permutation. main_selectors are [S] AdaLN table
  // rows, code is [T,rank], and final selectors are timestep-only [V]/[A].
  void record_forward(TensorBatch& batch,
                      DeviceTensor& video_latents,
                      DeviceTensor& audio_latents,
                      DeviceTensor& text_indices,
                      DeviceTensor& video_indices,
                      DeviceTensor& audio_indices,
                      DeviceTensor& main_selectors,
                      DeviceTensor& code,
                      DeviceTensor& cosine,
                      DeviceTensor& sine,
                      DeviceTensor& video_timestep_indices,
                      DeviceTensor& audio_timestep_indices,
                      DeviceTensor& video_velocity,
                      DeviceTensor& audio_velocity,
                      const H3AttentionRanges* ranges = nullptr,
                      const H3MainGraphReplayTaps* main_taps = nullptr);
  uint32_t required_forward_operators(
      const H3MainGraphReplayTaps* main_taps = nullptr) const;

  uint64_t persistent_bytes() const noexcept;
  uint64_t scratch_bytes() const noexcept;
  uint64_t peak_device_bytes() const noexcept;

 private:
  struct Impl;
  explicit ExactH3Transformer(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
