#pragma once

#include <cstdint>
#include <memory>

#include "slopfab/vulkan/dit_block.h"

namespace slopfab::vulkan {

struct H3MainGraphConfig {
  H3BlockConfig block;
  uint32_t layers = 50;
};

// Optional device-only final residual after every recorded layer. `boundaries`
// points to `count` distinct contiguous BF16 [S,H] tensors. Production passes
// null and pays no copy operations.
struct H3MainGraphReplayTaps {
  DeviceTensor* boundaries = nullptr;
  uint32_t count = 0;
};

// Complete exact main transformer stack. The graph owns immutable per-layer
// weights and one reusable activation/cache arena. It records into a caller's
// batch without allocation, submission, staging, or CUDA fallback.
class ExactH3MainGraph {
public:
  ExactH3MainGraph();
  ~ExactH3MainGraph();
  ExactH3MainGraph(ExactH3MainGraph&&) noexcept;
  ExactH3MainGraph& operator=(ExactH3MainGraph&&) noexcept;
  ExactH3MainGraph(const ExactH3MainGraph&) = delete;
  ExactH3MainGraph& operator=(const ExactH3MainGraph&) = delete;

  static ExactH3MainGraph create(TensorContext& context, const H3MainGraphConfig& config);
  // Scratch is allocated lazily here. Reloading an active graph is rejected
  // to avoid a transient second copy of all layer weights; call unload first.
  void load(const SafeTensors& checkpoint);
  // Releases all layer weights and the shared scratch/cache arena.
  void unload() noexcept;
  bool loaded() const noexcept;
  uint32_t layers() const noexcept;
  const H3MainGraphConfig& config() const noexcept;

  void record(TensorBatch& batch, DeviceTensor& tokens, DeviceTensor& selectors,
              DeviceTensor& adaln_code, DeviceTensor& cosine, DeviceTensor& sine,
              const H3AttentionRanges* ranges = nullptr,
              const H3MainGraphReplayTaps* taps = nullptr) const;
  // Records a contiguous stack span into the same caller batch. This is the
  // seam used by denoise block-cache orchestration to compute or skip its
  // configured middle span without cloning weights or scratch.
  void record_layers(TensorBatch& batch, DeviceTensor& tokens, DeviceTensor& selectors,
                     DeviceTensor& adaln_code, DeviceTensor& cosine, DeviceTensor& sine,
                     uint32_t first_layer, uint32_t layer_count,
                     const H3AttentionRanges* ranges = nullptr,
                     const H3MainGraphReplayTaps* taps = nullptr) const;

  // Complete non-recording validation for orchestrators that must reject a
  // malformed graph call before recording any surrounding endpoint op.
  uint32_t preflight(DeviceTensor& tokens, DeviceTensor& selectors, DeviceTensor& adaln_code,
                     DeviceTensor& cosine, DeviceTensor& sine,
                     const H3AttentionRanges* ranges = nullptr,
                     const H3MainGraphReplayTaps* taps = nullptr) const;
  uint32_t preflight_layers(DeviceTensor& tokens, DeviceTensor& selectors, DeviceTensor& adaln_code,
                            DeviceTensor& cosine, DeviceTensor& sine, uint32_t first_layer,
                            uint32_t layer_count, const H3AttentionRanges* ranges = nullptr,
                            const H3MainGraphReplayTaps* taps = nullptr) const;

  uint32_t required_operators(const H3MainGraphReplayTaps* taps = nullptr) const;
  uint32_t required_operators(uint32_t first_layer, uint32_t layer_count,
                              const H3MainGraphReplayTaps* taps = nullptr) const;
  uint64_t persistent_bytes() const noexcept;
  uint64_t scratch_bytes() const noexcept;
  uint64_t peak_device_bytes() const noexcept;

private:
  struct Impl;
  explicit ExactH3MainGraph(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

} // namespace slopfab::vulkan
