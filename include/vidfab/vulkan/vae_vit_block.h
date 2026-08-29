#pragma once

#include <memory>

#include "vidfab/vae/vit_block.h"
#include "vidfab/vulkan/runtime.h"
#include "vidfab/vulkan/tensor.h"

namespace vidfab::vulkan {

class ExactViTBlockScratch {
 public:
  ExactViTBlockScratch();
  ~ExactViTBlockScratch();
  ExactViTBlockScratch(ExactViTBlockScratch&&) noexcept;
  ExactViTBlockScratch& operator=(ExactViTBlockScratch&&) noexcept;
  ExactViTBlockScratch(const ExactViTBlockScratch&) = delete;
  ExactViTBlockScratch& operator=(const ExactViTBlockScratch&) = delete;
  static ExactViTBlockScratch create(TensorContext& context,
                                     const vae::ViTBlockConfig& config);
  uint64_t reserved_bytes() const noexcept;
 private:
  struct Impl;
  explicit ExactViTBlockScratch(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class ExactViTBlockStage;
};

class ExactViTBlockStage final : public vae::ExactViTBlockStage {
 public:
  ExactViTBlockStage();
  ~ExactViTBlockStage() override;
  ExactViTBlockStage(ExactViTBlockStage&&) noexcept;
  ExactViTBlockStage& operator=(ExactViTBlockStage&&) noexcept;
  ExactViTBlockStage(const ExactViTBlockStage&) = delete;
  ExactViTBlockStage& operator=(const ExactViTBlockStage&) = delete;
  static ExactViTBlockStage create(TensorContext& context,
                                   const vae::ViTBlockConfig& config);
  DeviceBackend backend() const noexcept override;
  const vae::ViTBlockConfig& config() const noexcept override;
  void load(const vae::ViTBlockWeightsView& weights) override;
  void forward(const float* tokens, const float* cosine, const float* sine,
               float* output) override;
  uint64_t persistent_bytes() const noexcept override;
  uint64_t peak_device_bytes() const noexcept override;

  // Production seam. `tokens` is updated in place and may be consumed by the
  // next block in the same batch. Scratch is externally owned so all 36 blocks
  // can share one bounded activation arena; no submission or host boundary is
  // introduced here.
  void record(TensorBatch& batch, DeviceTensor& tokens, DeviceTensor& cosine,
              DeviceTensor& sine, ExactViTBlockScratch& scratch) const;
 private:
  struct Impl;
  explicit ExactViTBlockStage(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class ExactViTBlockGraph;
};

class ExactViTBlockGraph {
 public:
  ExactViTBlockGraph();
  ~ExactViTBlockGraph();
  ExactViTBlockGraph(ExactViTBlockGraph&&) noexcept;
  ExactViTBlockGraph& operator=(ExactViTBlockGraph&&) noexcept;
  ExactViTBlockGraph(const ExactViTBlockGraph&) = delete;
  ExactViTBlockGraph& operator=(const ExactViTBlockGraph&) = delete;
  static ExactViTBlockGraph create(TensorContext& context,
                                   const vae::ViTBlockConfig& config,
                                   uint32_t layers);
  void load(const SafeTensors& checkpoint);
  void load_layer(uint32_t layer,
                  const vae::ViTBlockWeightsView& weights);
  void record(TensorBatch& batch, DeviceTensor& tokens, DeviceTensor& cosine,
              DeviceTensor& sine) const;
  // Verification seam; production uses record() once for the complete stack.
  void record_layer(uint32_t layer, TensorBatch& batch, DeviceTensor& tokens,
                    DeviceTensor& cosine, DeviceTensor& sine) const;
  void forward(const float* tokens, const float* cosine, const float* sine,
               float* output);
  uint32_t layers() const noexcept;
  uint64_t persistent_bytes() const noexcept;
  uint64_t peak_device_bytes() const noexcept;
 private:
  struct Impl;
  explicit ExactViTBlockGraph(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
