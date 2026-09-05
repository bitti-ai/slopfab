#pragma once

#include <cstdint>
#include <memory>

#include "slopfab/text/qwen_vision.h"
#include "slopfab/vulkan/gemm.h"
#include "slopfab/vulkan/tensor.h"

namespace slopfab::vulkan {

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
  friend class ExactQwenVisionPatchStage;
  friend class ExactQwenVisionMergerStage;
};

class ExactQwenVisionPatchStage {
 public:
  ExactQwenVisionPatchStage();
  ~ExactQwenVisionPatchStage();
  ExactQwenVisionPatchStage(ExactQwenVisionPatchStage&&) noexcept;
  ExactQwenVisionPatchStage& operator=(ExactQwenVisionPatchStage&&) noexcept;
  ExactQwenVisionPatchStage(const ExactQwenVisionPatchStage&) = delete;
  ExactQwenVisionPatchStage& operator=(const ExactQwenVisionPatchStage&) = delete;
  static ExactQwenVisionPatchStage create(
      TensorContext& context, const QwenVisionStageConfig& config);
  void load(const text::QwenVisionCheckpoint& checkpoint);
  void unload() noexcept;
  bool loaded() const noexcept;
  uint32_t required_operators() const noexcept;
  uint64_t persistent_bytes() const noexcept;
  // pixel_rows [S,1536], learned_index [S], output [S,1152].
  void record(TensorBatch& batch, DeviceTensor& pixel_rows,
              DeviceTensor& learned_index, DeviceTensor& output,
              ExactQwenVisionScratch& scratch) const;
 private:
  struct Impl;
  explicit ExactQwenVisionPatchStage(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
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

// slot -1 is the main merger (norm before merge); slots 0..2 are the three
// DeepStack mergers (merge before norm).
class ExactQwenVisionMergerStage {
 public:
  ExactQwenVisionMergerStage();
  ~ExactQwenVisionMergerStage();
  ExactQwenVisionMergerStage(ExactQwenVisionMergerStage&&) noexcept;
  ExactQwenVisionMergerStage& operator=(ExactQwenVisionMergerStage&&) noexcept;
  ExactQwenVisionMergerStage(const ExactQwenVisionMergerStage&) = delete;
  ExactQwenVisionMergerStage& operator=(const ExactQwenVisionMergerStage&) = delete;
  static ExactQwenVisionMergerStage create(
      TensorContext& context, const QwenVisionStageConfig& config);
  void load(const text::QwenVisionCheckpoint& checkpoint, int slot);
  void unload() noexcept;
  bool loaded() const noexcept;
  int slot() const;
  uint32_t required_operators() const noexcept;
  uint64_t persistent_bytes() const noexcept;
  void record(TensorBatch& batch, DeviceTensor& visual_residual,
              DeviceTensor& output, ExactQwenVisionScratch& scratch) const;
 private:
  struct Impl;
  explicit ExactQwenVisionMergerStage(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

struct ExactQwenVisionStats {
  double last_encode_seconds = 0.0;
  uint32_t patch_rows = 0;
  uint64_t scratch_bytes = 0;
  uint64_t activation_bytes = 0;
  uint64_t max_streamed_weight_bytes = 0;
  uint64_t allocator_peak_used_bytes = 0;
  uint64_t allocator_peak_nonstaging_bytes = 0;
  uint64_t allocator_used_bytes = 0;
  uint64_t allocator_reserved_bytes = 0;
  uint64_t descriptor_set_allocations = 0;
};

// Complete one-image visual tower. Host patchification/BF16 conversion and
// canonical position/RoPE construction are one input boundary; residual and
// all four outputs remain device-resident through the 27 streamed blocks.
class ExactQwenVisionEncoder {
 public:
  ExactQwenVisionEncoder();
  ~ExactQwenVisionEncoder();
  ExactQwenVisionEncoder(ExactQwenVisionEncoder&&) noexcept;
  ExactQwenVisionEncoder& operator=(ExactQwenVisionEncoder&&) noexcept;
  ExactQwenVisionEncoder(const ExactQwenVisionEncoder&) = delete;
  ExactQwenVisionEncoder& operator=(const ExactQwenVisionEncoder&) = delete;
  static ExactQwenVisionEncoder create(TensorContext& context);
  void load(const SafeTensors& checkpoint);
  void unload() noexcept;
  bool loaded() const noexcept;
  void encode(const text::QwenPixelValues& image,
              text::QwenVisionTrace* trace = nullptr);
  DeviceTensor& main_output();
  DeviceTensor& deepstack_output(uint32_t slot);
  uint32_t output_tokens() const noexcept;
  const ExactQwenVisionStats& stats() const noexcept;
 private:
  struct Impl;
  explicit ExactQwenVisionEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace slopfab::vulkan
