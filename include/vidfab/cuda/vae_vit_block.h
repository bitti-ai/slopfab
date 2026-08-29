#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "vidfab/vae/vit_block.h"
#include "vidfab/safetensors.h"

namespace vidfab::cuda {

std::unique_ptr<vae::ExactViTBlockStage> create_exact_vae_vit_block_stage(
    const vae::ViTBlockConfig& config);

// Device-resident exact transformer stack used by the real ViTDecoder exact
// path. All blocks share one activation arena; forward_device introduces no
// host transfer and accepts the decoder's existing stream.
class ExactViTBlockGraph {
 public:
  ExactViTBlockGraph();
  ~ExactViTBlockGraph();
  ExactViTBlockGraph(ExactViTBlockGraph&&) noexcept;
  ExactViTBlockGraph& operator=(ExactViTBlockGraph&&) noexcept;
  ExactViTBlockGraph(const ExactViTBlockGraph&) = delete;
  ExactViTBlockGraph& operator=(const ExactViTBlockGraph&) = delete;
  static ExactViTBlockGraph create(const vae::ViTBlockConfig& config,
                                   uint32_t layers);
  void load(const SafeTensors& checkpoint);
  void load_layer(uint32_t layer,
                  const vae::ViTBlockWeightsView& weights);
  void forward_device(float* tokens, const float* cosine, const float* sine,
                      cudaStream_t stream) const;
  // Verification seam. Production uses forward_device() for the whole stack.
  void forward_layer_device(uint32_t layer, float* tokens,
                            const float* cosine, const float* sine,
                            cudaStream_t stream) const;
  void forward(const float* tokens, const float* cosine, const float* sine,
               float* output);
  uint32_t layers() const noexcept;
  uint64_t persistent_bytes() const noexcept;
  uint64_t peak_device_bytes() const noexcept;
 private:
  struct Impl;
  explicit ExactViTBlockGraph(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace vidfab::cuda
