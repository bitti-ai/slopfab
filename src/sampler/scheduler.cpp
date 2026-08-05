#include "vidfab/sampler/scheduler.h"

#include <stdexcept>
#include <string>

namespace vidfab::sampler {

FlowScheduler::FlowScheduler(float shift) : shift_(shift) {
  if (shift <= 0.0f) {
    throw std::runtime_error("scheduler: shift must be positive, got " + std::to_string(shift));
  }
}

void FlowScheduler::clear_history() {
  previous_velocity_.clear();
  has_previous_ = false;
  expected_step_ = -1;
}

void FlowScheduler::reset() { clear_history(); }

void FlowScheduler::set_sampler(SamplerKind kind) {
  sampler_ = kind;
  clear_history();
}

void FlowScheduler::set_timesteps(int num_inference_steps) {
  // A new schedule invalidates any velocity carried over from the old one.
  clear_history();

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
                         float* out) {
  if (step_index < 0 || static_cast<size_t>(step_index) + 1 >= sigmas_.size()) {
    throw std::runtime_error("scheduler: step index " + std::to_string(step_index) +
                             " out of range for a schedule of " + std::to_string(sigmas_.size()) +
                             " sigmas");
  }

  // Ordering. kAb2 carries v_{n-1} across calls, so an index that is neither a
  // restart nor the successor of the last one would extrapolate from a
  // velocity belonging to some other point of the trajectory — plausible
  // output, wrong integrator. Index 0 restarts rather than throwing, which is
  // what lets one scheduler serve several runs.
  if (step_index == 0 || expected_step_ < 0) {
    // Either the top of a trajectory, or the first call after a reset — both
    // start fresh, at whatever index the caller names, with no history.
    clear_history();
  } else if (expected_step_ != step_index) {
    throw std::runtime_error(
        "scheduler: step index " + std::to_string(step_index) + " out of order; expected " +
        std::to_string(expected_step_) +
        " (or 0 to restart, or reset()). The sampler carries a velocity history, so steps must "
        "run in sequence");
  }
  const bool extrapolate = sampler_ == SamplerKind::kAb2 && has_previous_;
  if (extrapolate && previous_velocity_.size() != count) {
    throw std::runtime_error("scheduler: sample length changed from " +
                             std::to_string(previous_velocity_.size()) + " to " +
                             std::to_string(count) +
                             " inside one trajectory; the velocity history does not describe this "
                             "buffer");
  }
  expected_step_ = step_index + 1;

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

  if (!extrapolate) {
    // Euler, byte for byte what this function has always computed. kAb2's
    // first step lands here too, because there is no v_{n-1} to extrapolate
    // from and a second-order start would have to invent one.
    for (size_t i = 0; i < count; ++i) {
      // Data-ward velocity: x0 = x_t + sigma*v, a plus.
      const float denoised = sample[i] + sigma_from_timestep * velocity[i];
      out[i] = ratio * sample[i] + (1.0f - ratio) * denoised;
    }
  } else {
    // AB2 in this scheduler's ratio parameterisation. The working:
    //
    //   denoised = x + s*v                       s = sigma_from_timestep
    //   x_next   = r*x + (1-r)*denoised          r = sigmas[i+1]/sigmas[i]
    //            = r*x + (1-r)*x + (1-r)*s*v
    //            = x + (1-r)*s*v
    //
    // so Euler here is x_{n+1} = x_n + h_n*v_n with an effective step size
    // h_n = (1-r)*s. When s == sigma_n exactly, h_n = sigma_n - sigma_{n+1},
    // which is the step in the variable the ODE is actually integrated in:
    // v = x0 - noise is dx/d(-sigma), so the schedule marches -sigma forward.
    //
    // AB2 is that same step with v_n replaced by the linear extrapolation of
    // the velocity field to the midpoint of the interval,
    //
    //   v_hat = 1.5*v_n - 0.5*v_{n-1}
    //
    // and nothing else changes. Writing it as a substitution rather than as a
    // second expression is deliberate: `s` and `r` are then consumed exactly
    // as Euler consumes them, from the two different sources spec 7.3 insists
    // on keeping apart — `s` from the timestep the transformer was
    // conditioned on, `r` from the stored sigma grid — and the whole update
    // stays fp32. It also makes the reduction to Euler exact rather than
    // approximate: when v_{n-1} == v_n, v_hat == v_n bitwise.
    //
    // The coefficients are the fixed-step ones on a grid that is not fixed
    // (sigma is shifted by 12 or 3), so the local error carries a
    // (h_n - h_{n-1}) factor rather than h_n. That is the brief; see the
    // linear-ODE test for what it buys in practice.
    //
    // NOTE, if step caching is ever enabled alongside this: v_{n-1} would then
    // sometimes be a *reused* velocity from an earlier step rather than a
    // fresh evaluation, and the extrapolation would be through a point the
    // model never actually visited. The two features are not composable as
    // written and their step savings are not multiplicative. Nothing here
    // tries to detect or fix that.
    for (size_t i = 0; i < count; ++i) {
      const float v_hat = 1.5f * velocity[i] - 0.5f * previous_velocity_[i];
      const float denoised = sample[i] + sigma_from_timestep * v_hat;
      out[i] = ratio * sample[i] + (1.0f - ratio) * denoised;
    }
  }

  if (sampler_ == SamplerKind::kAb2) {
    // After the update, so `velocity` is still the caller's buffer and the
    // aliasing rule (`out` may alias `sample`) is untouched. One buffer copy
    // per step against a transformer forward pass measured in seconds.
    previous_velocity_.assign(velocity, velocity + count);
    has_previous_ = true;
  }
}

void FlowScheduler::scale_noise(const float* x0, const float* noise, float t, size_t count,
                                float* out) {
  for (size_t i = 0; i < count; ++i) {
    out[i] = t * x0[i] + (1.0f - t) * noise[i];
  }
}

}  // namespace vidfab::sampler
