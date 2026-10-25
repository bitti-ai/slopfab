#include "tensor_recording.h"

namespace slopfab::vulkan {
void TensorBatch::music_operation(DeviceTensor& input, DeviceTensor& weight, DeviceTensor& bias,
                                  DeviceTensor& output, const uint32_t* parameters,
                                  uint32_t groups) {
  if (!impl_ || impl_->poisoned || !parameters || !groups || !impl_->owner->music_pipeline)
    throw std::invalid_argument("Vulkan music: invalid dispatch");
  std::array<std::shared_ptr<DeviceTensor::Impl>, 4> tensors = {
      impl_->owner->require(input), impl_->owner->require(weight), impl_->owner->require(bias),
      impl_->owner->require(output)};
  uint32_t p[16];
  std::copy_n(parameters, 16, p);
  p[15] = std::min(groups, impl_->owner->max_dispatch_x);
  uint32_t gy = (groups + p[15] - 1) / p[15];
  if (gy > impl_->owner->max_dispatch_y)
    throw std::out_of_range("Vulkan music: dispatch exceeds device limits");
  try {
    impl_->count_operator();
    std::vector<StorageBinding> bindings;
    for (uint32_t i = 0; i < 4; ++i) {
      bool already = false;
      for (uint32_t j = 0; j < i; ++j)
        already |= tensors[i] == tensors[j];
      if (!already)
        impl_->transition(tensors[i], tensors[i] == tensors[3] ? BufferAccess::kComputeReadWrite
                                                               : BufferAccess::kComputeRead);
      bindings.push_back({i, &tensors[i]->buffer, 0, tensors[i]->buffer.size()});
    }
    impl_->commands.bind_compute(impl_->owner->music_pipeline, bindings);
    impl_->commands.push_constants(p, sizeof(p));
    impl_->commands.dispatch(p[15], gy, 1);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}
}
