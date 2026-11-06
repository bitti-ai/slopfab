#pragma once
#include "slopfab/seedvr2.h"

namespace slopfab::seedvr2 {
struct ModelBackend {
  virtual ~ModelBackend() = default;
  virtual std::vector<Frame> restore(const std::vector<Frame>&, uint64_t) = 0;
  std::function<void(const std::string&)> progress;
  std::function<bool()> cancelled;
};

std::unique_ptr<ModelBackend> make_cuda_model(const Options&);
}

namespace slopfab::seedvr2_vulkan {
std::unique_ptr<seedvr2::ModelBackend> make_vulkan_model(const seedvr2::Options&);
}
