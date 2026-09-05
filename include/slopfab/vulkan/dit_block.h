#pragma once

#include <cstdint>
#include <memory>

#include "slopfab/safetensors.h"
#include "slopfab/vulkan/gemm.h"
#include "slopfab/vulkan/tensor.h"

namespace slopfab::vulkan {

struct H3BlockConfig {
  uint32_t sequence = 0;
  uint32_t hidden = 5376;
  uint32_t heads = 56;
  uint32_t head_dim = 128;
  uint32_t ffn = 14336;
  uint32_t timesteps = 1;
  uint32_t modalities = 3;
  uint32_t adaln_rank = 8;
  float epsilon = 1.0e-5f;
};

// Optional device-only diagnostic taps. Each non-null destination must match
// the corresponding contiguous BF16 stage tensor and be distinct from every
// input and other tap. Production passes null and records no copies.
struct H3BlockReplayTaps {
  DeviceTensor* q = nullptr;
  DeviceTensor* k = nullptr;
  DeviceTensor* v = nullptr;
  DeviceTensor* attention = nullptr;
  DeviceTensor* attention_residual = nullptr;
  DeviceTensor* final_residual = nullptr;
};

class ExactH3BlockScratch {
 public:
  ExactH3BlockScratch();
  ~ExactH3BlockScratch();
  ExactH3BlockScratch(ExactH3BlockScratch&&) noexcept;
  ExactH3BlockScratch& operator=(ExactH3BlockScratch&&) noexcept;
  ExactH3BlockScratch(const ExactH3BlockScratch&) = delete;
  ExactH3BlockScratch& operator=(const ExactH3BlockScratch&) = delete;
  static ExactH3BlockScratch create(TensorContext& context,
                                    const H3BlockConfig& config);
  uint64_t reserved_bytes() const noexcept;
 private:
  struct Impl;
  explicit ExactH3BlockScratch(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class ExactH3BlockStage;
};

// One complete main-transformer block. Loading is transactional: all typed
// weights are uploaded and non-NVFP4 matrices materialized before the active
// state is replaced. The production record seam neither submits nor stages.
class ExactH3BlockStage {
 public:
  ExactH3BlockStage();
  ~ExactH3BlockStage();
  ExactH3BlockStage(ExactH3BlockStage&&) noexcept;
  ExactH3BlockStage& operator=(ExactH3BlockStage&&) noexcept;
  ExactH3BlockStage(const ExactH3BlockStage&) = delete;
  ExactH3BlockStage& operator=(const ExactH3BlockStage&) = delete;
  static ExactH3BlockStage create(TensorContext& context,
                                  const H3BlockConfig& config);
  // Complete host-only archive validation. Performs no context allocation or
  // upload and is used by multi-layer graphs before transactional loading.
  static void validate_checkpoint(const SafeTensors& checkpoint,
                                  uint32_t layer,
                                  const H3BlockConfig& config);
  // Token-refiner variant: same exact attention/MLP arithmetic and projection
  // formats, but weights live under token_refiner.blocks.N and there is no
  // learned AdaLN module.
  static void validate_refiner_checkpoint(const SafeTensors& checkpoint,
                                           uint32_t layer,
                                           const H3BlockConfig& config);
  void load(const SafeTensors& checkpoint, uint32_t layer);
  void load_refiner(const SafeTensors& checkpoint, uint32_t layer);
  // Allocate only the optional AWQ/ConvRot activation buffers required by
  // the loaded weight metadata. Call after load and before opening a batch.
  void prepare(ExactH3BlockScratch& scratch) const;
  void unload() noexcept;
  bool loaded() const noexcept;
  // tokens [S,H], selectors [S] in [0,T*M), code [T,R], tables [S,96].
  void record(TensorBatch& batch, DeviceTensor& tokens,
              DeviceTensor& selectors, DeviceTensor& adaln_code,
              DeviceTensor& cosine, DeviceTensor& sine,
              ExactH3BlockScratch& scratch,
              const H3AttentionRanges* ranges = nullptr,
              const H3BlockReplayTaps* taps = nullptr) const;
  const H3BlockConfig& config() const noexcept;
  uint64_t persistent_bytes() const noexcept;
  uint64_t peak_device_bytes(const ExactH3BlockScratch& scratch) const noexcept;
  uint32_t required_operators(const H3BlockReplayTaps* taps = nullptr) const;
 private:
  struct Impl;
  explicit ExactH3BlockStage(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace slopfab::vulkan
