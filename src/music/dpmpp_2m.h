#pragma once
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace slopfab::music {

// ComfyUI sample_dpmpp_2m with YuE2's denoised = x - sigma * velocity.
// Keep the integration/history in FP32; the network narrows inputs to BF16.
// One instance per flow chunk: histories must never cross chunk boundaries.
class Dpmpp2M {
public:
  void step(std::vector<float>& state, const std::vector<float>& velocity,
            float sigma, float next_sigma) {
    if (state.size() != velocity.size() ||
        (!previous_.empty() && previous_.size() != state.size()) ||
        !std::isfinite(sigma) || !std::isfinite(next_sigma) ||
        sigma <= 0 || next_sigma < 0 || next_sigma >= sigma ||
        (!previous_.empty() && previous_sigma_ <= sigma))
      throw std::invalid_argument("invalid YuE2 DPM++ 2M step");
    const double ratio = double(next_sigma) / sigma;
    // Terminal denoising avoids log(0), and the first step has no history.
    const double correction = previous_.empty() || next_sigma == 0 ? 0 :
        std::log(double(sigma) / next_sigma) /
        (2 * std::log(double(previous_sigma_) / sigma));
    std::vector<float> denoised(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
      denoised[i] = state[i] - sigma * velocity[i];
      const double prediction = correction == 0 ? denoised[i] :
          (1 + correction) * denoised[i] - correction * previous_[i];
      state[i] = float(ratio * state[i] + (1 - ratio) * prediction);
      if (!std::isfinite(state[i]))
        throw std::runtime_error("non-finite YuE2 flow state");
    }
    previous_ = std::move(denoised);
    previous_sigma_ = sigma;
  }

private:
  std::vector<float> previous_;
  float previous_sigma_ = 0;
};
} // namespace slopfab::music
