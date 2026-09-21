#pragma once

#include "slopfab/dit/vsa.h"
#include "slopfab/vulkan/tensor.h"

namespace slopfab::vulkan {

// Device-resident FastH3 V2 attention. Three dispatches pool, route/compress,
// and evaluate sparse attention. The gate is a separate learned projection.
class VsaAttentionPlan {
public:
  VsaAttentionPlan();
  ~VsaAttentionPlan();
  VsaAttentionPlan(VsaAttentionPlan&&) noexcept;
  VsaAttentionPlan& operator=(VsaAttentionPlan&&) noexcept;
  VsaAttentionPlan(const VsaAttentionPlan&) = delete;
  VsaAttentionPlan& operator=(const VsaAttentionPlan&) = delete;
  static VsaAttentionPlan create(TensorContext& context, const dit::VsaTiles& tiles, uint32_t heads,
                                 uint32_t head_dim);
  uint64_t workspace_bytes() const noexcept;
  void record(TensorBatch& batch, DeviceTensor& query, DeviceTensor& key, DeviceTensor& value,
              DeviceTensor& output) const;
  void add_compression(TensorBatch& batch, DeviceTensor& gate, DeviceTensor& output) const;

private:
  struct Impl;
  explicit VsaAttentionPlan(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

} // namespace slopfab::vulkan
