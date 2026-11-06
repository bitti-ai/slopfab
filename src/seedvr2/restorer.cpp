#include "model_backend.h"
#include <stdexcept>

namespace slopfab::seedvr2 {
struct Restorer::Impl {
  std::unique_ptr<ModelBackend> model;
};
Restorer::Restorer(const Options& o) : impl_(std::make_unique<Impl>()) {
  validate(o);
  switch (o.backend) {
#if SLOPFAB_WITH_CUDA
  case DeviceBackend::kCuda: impl_->model = make_cuda_model(o); break;
#endif
#if SLOPFAB_WITH_VULKAN
  case DeviceBackend::kVulkan: impl_->model = seedvr2_vulkan::make_vulkan_model(o); break;
#endif
  default: throw std::invalid_argument("SeedVR2: requested backend was not compiled in");
  }
}
Restorer::~Restorer() = default;
std::vector<Frame> Restorer::restore(const std::vector<Frame>& frames, uint64_t first) {
  impl_->model->progress = progress;
  impl_->model->cancelled = cancelled;
  return impl_->model->restore(frames, first);
}
}
