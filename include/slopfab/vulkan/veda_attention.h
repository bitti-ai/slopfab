#pragma once
#include "slopfab/dit/veda.h"
#include "slopfab/vulkan/tensor.h"

namespace slopfab::vulkan {
class VedaAttentionPlan {
public:
  VedaAttentionPlan() = default;
  static VedaAttentionPlan create(TensorContext& context,
                                  std::shared_ptr<dit::VedaRequest> request);
  uint64_t workspace_bytes() const noexcept;
  // Packed BF16 [2, heads, 3*dim, dim], Q projection followed by K.
  static DeviceTensor upload_projection(TensorContext& context, const dit::VedaBundle& bundle,
                                        int layer);
  void record(TensorBatch& batch, int layer, DeviceTensor& projection, DeviceTensor& query,
              DeviceTensor& key, DeviceTensor& value, DeviceTensor& output) const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
} // namespace slopfab::vulkan
