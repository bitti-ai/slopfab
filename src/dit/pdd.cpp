#include "slopfab/dit/pdd.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab::dit {
namespace {
double sigma(int node, double shift) {
  const double u = 1.0 - double(node) / kOmniRefHeads;
  return .999 * shift * u / (1 + (shift - 1) * u);
}
}

std::vector<float> omniref_sigmas(float shift) {
  if (!(shift > 0) || !std::isfinite(shift))
    throw std::invalid_argument("OmniRef requires a positive sigma shift");
  std::vector<float> result;
  for (int i = 0; i <= kOmniRefHeads; i += 4)
    result.push_back(static_cast<float>(sigma(i, shift)));
  return result;
}

int omniref_step(float t) {
  const auto grid = omniref_sigmas(12);
  for (int i = 0; i < kOmniRefSteps; ++i)
    if (std::abs(t - (1.f - grid[i])) < 2e-6f)
      return i;
  throw std::invalid_argument("OmniRef evaluation is outside its trained eight-step grid");
}

std::array<float, kOmniRefHeads> omniref_head_weights(int step, float shift) {
  if (step < 0 || step >= kOmniRefSteps || !(shift > 0) || !std::isfinite(shift))
    throw std::invalid_argument("invalid OmniRef head fusion block");
  std::array<float, kOmniRefHeads> weights{};
  const int start = 4 * step, end = start + 4;
  const double total = sigma(start, shift) - sigma(end, shift);
  weights[0] = 1;
  for (int j = std::max(start, 1); j < end; ++j)
    weights[j] = static_cast<float>((sigma(j, shift) - sigma(j + 1, shift)) / total);
  return weights;
}

std::vector<float> fuse_omniref_heads(const TensorView& tensor, float shift) {
  if (tensor.shape.empty() || tensor.shape.size() > 2 || tensor.shape[0] % kOmniRefHeads)
    throw std::invalid_argument("invalid OmniRef output projection shape");
  const auto source = to_f32(tensor);
  const size_t stride = source.size() / kOmniRefHeads;
  if (!stride || source.size() % kOmniRefHeads)
    throw std::invalid_argument("empty or incomplete OmniRef head bank");
  std::vector<float> result(kOmniRefSteps * stride);
  for (int step = 0; step < kOmniRefSteps; ++step) {
    const auto weights = omniref_head_weights(step, shift);
    for (size_t i = 0; i < stride; ++i) {
      float residual = 0;
      for (int j = std::max(4 * step, 1); j < 4 * step + 4; ++j)
        residual += weights[j] * source[j * stride + i];
      result[step * stride + i] = source[i] + residual;
    }
  }
  return result;
}
}
