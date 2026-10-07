#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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
  // p describes the convolution at the doubled resolution. Backends may
  // read the original pixels during convolution instead of materializing
  // nearest-neighbor expansion. The fallback preserves existing kernels.
  virtual Buffer nearest_conv(Buffer input, const Buffer& weight, const Buffer& bias,
                              const Parameters& p) {
    const uint32_t count = p.height * p.width * p.input;
    auto expanded = allocate(count);
    run({kNearest, p.height, p.width, p.input, p.input, count, 0, 0}, input, input, input,
        expanded);
    input = std::move(expanded);
    auto output = allocate(p.count);
    run(p, input, weight, bias, output);
    return output;
  }
};

std::unique_ptr<Backend> make_cuda_backend();
std::unique_ptr<Backend> make_vulkan_backend();

} // namespace slopfab::upscale_detail
