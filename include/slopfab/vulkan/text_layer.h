#pragma once

#include <cstdint>
#include <memory>

#include "slopfab/safetensors.h"
#include "slopfab/text/encoder.h"
#include "slopfab/vulkan/gemm.h"
#include "slopfab/vulkan/tensor.h"

namespace slopfab::vulkan {

struct QwenTextLayerConfig {
  uint32_t sequence = 0;
  text::EncoderConfig encoder;
};

// Optional device-only replay destinations. Production passes null. Taps are
// distinct contiguous BF16 buffers and never introduce host activation seams.
struct QwenTextLayerTaps {
  DeviceTensor* input_norm = nullptr;
  DeviceTensor* query = nullptr;
  DeviceTensor* key = nullptr;
  DeviceTensor* value = nullptr;
  DeviceTensor* attention = nullptr;
  DeviceTensor* attention_residual = nullptr;
  DeviceTensor* post_attention_norm = nullptr;
  DeviceTensor* gate = nullptr;
  DeviceTensor* up = nullptr;
  DeviceTensor* activation = nullptr;
  DeviceTensor* final_residual = nullptr;
};

// Caller-owned, reusable activation and materialization storage. One largest
// BF16 matrix slot is overwritten sequentially for all seven projections; no
// expanded checkpoint matrix is retained by a stage.
class ExactQwenTextLayerScratch {
 public:
  ExactQwenTextLayerScratch();
  ~ExactQwenTextLayerScratch();
  ExactQwenTextLayerScratch(ExactQwenTextLayerScratch&&) noexcept;
  ExactQwenTextLayerScratch& operator=(ExactQwenTextLayerScratch&&) noexcept;
  ExactQwenTextLayerScratch(const ExactQwenTextLayerScratch&) = delete;
  ExactQwenTextLayerScratch& operator=(const ExactQwenTextLayerScratch&) = delete;

  static ExactQwenTextLayerScratch create(
      TensorContext& context, const QwenTextLayerConfig& config);
  uint64_t reserved_bytes() const noexcept;
  uint64_t dense_cache_bytes() const noexcept;

 private:
  struct Impl;
  explicit ExactQwenTextLayerScratch(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class ExactQwenTextLayerStage;
};

// One complete Qwen3-VL decoder layer. The stage owns only compressed weights
// and four BF16 norm vectors. load() validates the complete typed checkpoint
// before allocating, then transactionally replaces the active layer. record()
// only appends work to the caller's batch and performs no allocation/submit.
class ExactQwenTextLayerStage {
 public:
  ExactQwenTextLayerStage();
  ~ExactQwenTextLayerStage();
  ExactQwenTextLayerStage(ExactQwenTextLayerStage&&) noexcept;
  ExactQwenTextLayerStage& operator=(ExactQwenTextLayerStage&&) noexcept;
  ExactQwenTextLayerStage(const ExactQwenTextLayerStage&) = delete;
  ExactQwenTextLayerStage& operator=(const ExactQwenTextLayerStage&) = delete;

  static ExactQwenTextLayerStage create(
      TensorContext& context, const QwenTextLayerConfig& config);
  static void validate_checkpoint(const SafeTensors& checkpoint,
                                  uint32_t layer,
                                  const QwenTextLayerConfig& config);
  // Validates the exact embedding/global manifest and every layer in one
  // archive pass. Unlike calling validate_checkpoint() 50 times, this does
  // not rescan all visual and layer names once per target layer.
  static void validate_archive(const SafeTensors& checkpoint,
                               const QwenTextLayerConfig& config);
  void load(const SafeTensors& checkpoint, uint32_t layer);
  void unload() noexcept;
  bool loaded() const noexcept;

  // tokens [L,H] BF16 is updated in place. cosine/sine are device-resident
  // fp32 [L,128] tables in the canonical NeoX duplicated-half layout.
  void record(TensorBatch& batch, DeviceTensor& tokens,
              DeviceTensor& cosine, DeviceTensor& sine,
              ExactQwenTextLayerScratch& scratch,
              const QwenTextLayerTaps* taps = nullptr) const;
  uint32_t required_operators(
      const QwenTextLayerTaps* taps = nullptr) const;
  const QwenTextLayerConfig& config() const noexcept;
  text::WeightFormat format() const noexcept;
  uint64_t persistent_bytes() const noexcept;
  uint64_t peak_device_bytes(
      const ExactQwenTextLayerScratch& scratch) const noexcept;

 private:
  struct Impl;
  explicit ExactQwenTextLayerStage(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace slopfab::vulkan
