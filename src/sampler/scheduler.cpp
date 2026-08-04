#include "vidfab/sampler/scheduler.h"

#include <stdexcept>
#include <string>

namespace vidfab::sampler {

FlowScheduler::FlowScheduler(float shift) : shift_(shift) {
  if (shift <= 0.0f) {
    throw std::runtime_error("scheduler: shift must be positive, got " + std::to_string(shift));
  }
}

void FlowScheduler::set_timesteps(int num_inference_steps) {
  if (num_inference_steps < 2) {
    throw std::runtime_error("scheduler: num_inference_steps must be >= 2, got " +
                             std::to_string(num_inference_steps));
  }

  sigmas_.clear();
  timesteps_.clear();

  // The grid is built in fp32 on the host so the schedule never depends on the
  // accelerator. linspace(1, 0, n) — note the terminal zero is a grid point,
  // and the shift maps 0 to exactly 0.
  const int n = num_inference_steps;
  sigmas_.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const float base = 1.0f - static_cast<float>(i) / static_cast<float>(n - 1);
    const float shifted = shift_ * base / (1.0f + (shift_ - 1.0f) * base);

    // The shift compresses the grid near sigma = 1 hard enough to create
    // exact fp32 collisions; consecutive duplicates are collapsed, which is
    // why the realised step count can be lower than requested.
    if (!sigmas_.empty() && shifted == sigmas_.back()) continue;
    sigmas_.push_back(shifted);
  }

  timesteps_.reserve(sigmas_.size() - 1);
  for (size_t i = 0; i + 1 < sigmas_.size(); ++i) {
    timesteps_.push_back(1.0f - sigmas_[i]);
  }
}

void FlowScheduler::step(int step_index, const float* sample, const float* velocity, size_t count,
                         float* out) const {
  if (step_index < 0 || static_cast<size_t>(step_index) + 1 >= sigmas_.size()) {
    throw std::runtime_error("scheduler: step index " + std::to_string(step_index) +
                             " out of range for a schedule of " + std::to_string(sigmas_.size()) +
                             " sigmas");
  }

  // The sigma used to recover x0 comes from the *timestep the transformer was
  // conditioned on*, while the Euler ratio uses the sigma grid. For sigma <
  // 0.5 the round trip 1 - (1 - sigma) is not exact in fp32, and the reference
  // deliberately keeps the two sources apart. Reconstructing both from
  // sigmas_[i] would silently change the result.
  const float t = timesteps_[static_cast<size_t>(step_index)];
  const float sigma_from_timestep = 1.0f - t;

  const float sigma = sigmas_[static_cast<size_t>(step_index)];
  const float sigma_next = sigmas_[static_cast<size_t>(step_index) + 1];
  const float ratio = sigma_next / sigma;

  for (size_t i = 0; i < count; ++i) {
    // Data-ward velocity: x0 = x_t + sigma*v, a plus.
    const float denoised = sample[i] + sigma_from_timestep * velocity[i];
    out[i] = ratio * sample[i] + (1.0f - ratio) * denoised;
  }
}

void FlowScheduler::scale_noise(const float* x0, const float* noise, float t, size_t count,
                                float* out) {
  for (size_t i = 0; i < count; ++i) {
    out[i] = t * x0[i] + (1.0f - t) * noise[i];
  }
}

}  // namespace vidfab::sampler
