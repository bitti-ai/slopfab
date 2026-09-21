#pragma once

#include <cstdint>
#include <memory>

#include "slopfab/vulkan/dit_graph.h"

namespace slopfab::vulkan {

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
  // Total projected modality rows may include fixed Ref2VA anchors. Final
  // heads emit only the generated contiguous suffix at these packed offsets.
  // Zero output rows preserves the text-to-video/audio all-row contract.
  uint32_t video_output_rows = 0;
  uint32_t audio_output_rows = 0;
  uint32_t video_output_start = 0;
  uint32_t audio_output_start = 0;
};

struct H3TransformerTextReplayTaps {
  // Six distinct BF16 [text_rows,hidden] tensors: condition, refiner0
  // attention/final, refiner1 attention/final, final norm.
  DeviceTensor* boundaries = nullptr;
  uint32_t count = 0;
};

struct H3TransformerForwardReplayTaps {
  DeviceTensor* packed_input = nullptr;
  DeviceTensor* main_final = nullptr;
  const H3MainGraphReplayTaps* main_boundaries = nullptr;
};

class ExactH3Transformer {
public:
  ExactH3Transformer();
  ~ExactH3Transformer();
  ExactH3Transformer(ExactH3Transformer&&) noexcept;
  ExactH3Transformer& operator=(ExactH3Transformer&&) noexcept;
  ExactH3Transformer(const ExactH3Transformer&) = delete;
  ExactH3Transformer& operator=(const ExactH3Transformer&) = delete;

  static ExactH3Transformer create(TensorContext& context, const ExactH3TransformerConfig& config);
  void load(const SafeTensors& checkpoint);
  void unload() noexcept;
  bool loaded() const noexcept;
  bool text_prepared() const noexcept;
  const ExactH3TransformerConfig& config() const noexcept;

  // prompt is contiguous fp32 [text_rows,text_dim]. The cached final-refiner
  // BF16 stream remains device-resident for every subsequent evaluation.
  // One bounded submission; marks the cache ready only after completion.
  void prepare_text(DeviceTensor& prompt, const H3TransformerTextReplayTaps* taps = nullptr);
  uint32_t required_prepare_text_operators(const H3TransformerTextReplayTaps* taps = nullptr) const;

  // Inputs/outputs are contiguous fp32 modality rows. Audio tensors may be
  // empty when `audio_rows == 0`; that modality's projection and head are not
  // recorded. The backend-neutral H3
  // packing invariant is [text|audio|video]; exact row-range transfers build
  // and gather that packed stream without a shader or host boundary.
  // main_selectors are [S] AdaLN table rows, code is [T,rank], and final
  // selectors are timestep-only [V]/[A].
  void record_forward(TensorBatch& batch, DeviceTensor& video_latents, DeviceTensor& audio_latents,
                      DeviceTensor& main_selectors, DeviceTensor& code, DeviceTensor& cosine,
                      DeviceTensor& sine, DeviceTensor& video_timestep_indices,
                      DeviceTensor& audio_timestep_indices, DeviceTensor& video_velocity,
                      DeviceTensor& audio_velocity, const H3AttentionRanges* ranges = nullptr,
                      const H3TransformerForwardReplayTaps* taps = nullptr,
                      DeviceTensor* video_row_indices = nullptr,
                      DeviceTensor* audio_row_indices = nullptr);
  uint32_t required_forward_operators(const H3TransformerForwardReplayTaps* taps = nullptr) const;

  uint64_t persistent_bytes() const noexcept;
  uint64_t scratch_bytes() const noexcept;
  uint64_t peak_device_bytes() const noexcept;

private:
  struct Impl;
  explicit ExactH3Transformer(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

} // namespace slopfab::vulkan
