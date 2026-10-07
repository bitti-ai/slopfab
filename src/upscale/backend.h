#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace slopfab::upscale_detail {

struct ConvSpec {
  std::string name;
  int input, output;
};

std::vector<ConvSpec> model_convolutions();

struct Tensor {
  virtual ~Tensor() = default;
};

using Buffer = std::shared_ptr<Tensor>;

// Activations are contiguous HWC; weights retain PyTorch OIHW order.
// Non-convolution operations: concat channels, x + scale*y, nearest 2x.
enum Operation : uint32_t { kConv, kConcat, kResidual, kNearest };

struct Parameters {
  uint32_t op, height, width, input, output, count, leaky;
  float scale;
};

static_assert(sizeof(Parameters) == 32);

class Backend {
public:
  virtual ~Backend() = default;
  virtual Buffer allocate(size_t count) = 0;
  virtual Buffer upload(const std::vector<float>& values) = 0;
  virtual std::vector<float> download(const Buffer& tensor, size_t count) = 0;
  virtual void run(const Parameters& p, const Buffer& x, const Buffer& y, const Buffer& bias,
                   const Buffer& output) = 0;
};

std::unique_ptr<Backend> make_cuda_backend();
std::unique_ptr<Backend> make_vulkan_backend();

} // namespace slopfab::upscale_detail
