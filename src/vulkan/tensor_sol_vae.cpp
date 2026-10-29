#include "tensor_recording.h"

namespace slopfab::vulkan {
void TensorBatch::sol_vae_operation(DeviceTensor* const* values, uint32_t writes,
                                     const uint32_t* parameters, uint32_t groups) {
  if (!impl_ || impl_->poisoned || !values || !parameters || !groups || !writes ||
      (writes & ~255u) || !impl_->owner->sol_vae_pipeline)
    throw std::invalid_argument("Vulkan Sol-H3-VAE: invalid dispatch");
  std::array<std::shared_ptr<DeviceTensor::Impl>, 8> tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (!values[i]) throw std::invalid_argument("Vulkan Sol-H3-VAE: null binding");
    tensors[i] = impl_->owner->require(*values[i]);
  }
  uint32_t p[16];
  std::copy_n(parameters, 16, p);
  p[15] = std::min(groups, impl_->owner->max_dispatch_x);
  const uint32_t gy = (groups + p[15] - 1) / p[15];
  if (gy > impl_->owner->max_dispatch_y)
    throw std::out_of_range("Vulkan Sol-H3-VAE: dispatch exceeds device limits");
  try {
    impl_->count_operator();
    std::vector<StorageBinding> bindings;
    for (uint32_t i = 0; i < 8; ++i) {
      bool seen = false, modified = false;
      for (uint32_t j = 0; j < 8; ++j) {
        seen |= j < i && tensors[i] == tensors[j];
        modified |= (writes & (1u << j)) && tensors[i] == tensors[j];
      }
      if (!seen)
        impl_->transition(tensors[i], modified ? BufferAccess::kComputeReadWrite
                                               : BufferAccess::kComputeRead);
      bindings.push_back({i, &tensors[i]->buffer, 0, tensors[i]->buffer.size()});
    }
    impl_->commands.bind_compute(impl_->owner->sol_vae_pipeline, bindings);
    impl_->commands.push_constants(p, sizeof(p));
    impl_->commands.dispatch(p[15], gy, 1);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}
} // namespace slopfab::vulkan
