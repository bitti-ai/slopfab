#pragma once

#include <cstdint>
#include <memory>

#include "vidfab/text/qwen_vision.h"
#include "vidfab/vulkan/gemm.h"
#include "vidfab/vulkan/tensor.h"

namespace vidfab::vulkan {

struct QwenVisionStageConfig {
  uint32_t sequence = 0;
  text::QwenVisionConfig vision;
};

// Caller-owned activation arena and immutable plans for one visual sequence.
// It is shared while the stage streams each of the 27 block weight sets.
class ExactQwenVisionScratch {
 public:
  ExactQwenVisionScratch();
  ~ExactQwenVisionScratch();
  ExactQwenVisionScratch(ExactQwenVisionScratch&&) noexcept;
  ExactQwenVisionScratch& operator=(ExactQwenVisionScratch&&) noexcept;
  ExactQwenVisionScratch(const ExactQwenVisionScratch&) = delete;
  ExactQwenVisionScratch& operator=(const ExactQwenVisionScratch&) = delete;
  static ExactQwenVisionScratch create(TensorContext& context,
                                       const QwenVisionStageConfig& config);
  uint64_t reserved_bytes() const noexcept;

 private:
  struct Impl;
  explicit ExactQwenVisionScratch(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class ExactQwenVisionBlockStage;
};

// One streamed visual transformer block. load() consumes a descriptor returned
// by the strict 351-tensor archive validator and transactionally replaces the
// active BF16 weights. record() appends no allocation or submission.
class ExactQwenVisionBlockStage {
 public:
  ExactQwenVisionBlockStage();
  ~ExactQwenVisionBlockStage();
  ExactQwenVisionBlockStage(ExactQwenVisionBlockStage&&) noexcept;
  ExactQwenVisionBlockStage& operator=(ExactQwenVisionBlockStage&&) noexcept;
  ExactQwenVisionBlockStage(const ExactQwenVisionBlockStage&) = delete;
  ExactQwenVisionBlockStage& operator=(const ExactQwenVisionBlockStage&) = delete;
  static ExactQwenVisionBlockStage create(
      TensorContext& context, const QwenVisionStageConfig& config);

  void load(const text::QwenVisionCheckpoint& checkpoint, uint32_t block);
  void unload() noexcept;
  bool loaded() const noexcept;
  uint32_t block() const;
  uint32_t required_operators() const noexcept;
  uint64_t persistent_bytes() const noexcept;
  void record(TensorBatch& batch, DeviceTensor& residual,
              DeviceTensor& cosine, DeviceTensor& sine,
              ExactQwenVisionScratch& scratch) const;

 private:
  struct Impl;
  explicit ExactQwenVisionBlockStage(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
