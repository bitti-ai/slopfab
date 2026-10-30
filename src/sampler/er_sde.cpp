#include "slopfab/sampler/scheduler.h"
#include <algorithm>
#include <cmath>

namespace slopfab::sampler {
bool stochastic_sampler(SamplerKind kind) noexcept {
  return kind == SamplerKind::kRenoise || kind == SamplerKind::kErSde;
}

void FlowScheduler::adjust_first_sigma() {
  if (sigmas_.empty()) return;
  sigmas_[0] = original_first_sigma_;
  if (sampler_ == SamplerKind::kErSde && sigmas_[0] == 1) {
    // Rectified flow has alpha=1-sigma, so sigma=1 has infinite noise/data ratio.
    // Offset the unshifted start by 1e-4, as in ComfyUI's flow ER-SDE path.
    const double t = 1. - 1e-4;
    float first = float(shift_ * t / (1 + (shift_ - 1.) * t));
    first = std::min(first, std::nextafter(1.f, 0.f));
    sigmas_[0] = std::max(first, std::nextafter(sigmas_[1], 1.f));
  }
  timesteps_[0] = 1 - sigmas_[0];
}

void FlowScheduler::er_sde_step(int step, const float* sample, const float* velocity,
                                size_t count, float* out, const float* noise) {
  // ER-SDE-Solver-3 (Cui et al., arXiv:2309.06169), expressed for rectified
  // flow: alpha=1-sigma, lambda=sigma/alpha, x0=x+sigma*velocity (H3 sign).
  // Use log(phi) for stable ratios even very near the all-noise endpoint.
  const double sigma = sigmas_[step], next = sigmas_[step + 1];
  const double lambda = sigma / (1 - sigma);
  const auto log_phi = [](double x) {
    const double p = std::pow(x, .3);
    return std::log(x) + p + std::log1p(10 * std::exp(-p));
  };
  double carry = 0, clean = 1, first = 0, second = 0, noise_scale = 0;
  if (next > 0) {
    const double target = next / (1 - next), alpha = 1 - next;
    const double log_target = log_phi(target);
    const double ratio = std::exp(log_target - log_phi(lambda));
    carry = alpha / (1 - sigma) * ratio;
    clean = alpha * (1 - ratio);
    if (er_history_) {
      // The reference uses a 200-point left Riemann sum. Integrate ratios to
      // avoid forming phi(lambda) and its reciprocal as separate huge values.
      const double dt = target - lambda, h = -dt / 200;
      double integral = 0, moment = 0;
      for (int j = 0; j < 200; ++j) {
        const double position = target + j * h;
        const double weight = std::exp(log_target - log_phi(position));
        integral += weight;
        moment += (position - lambda) * weight;
      }
      first = alpha * (dt + integral * h);
      second = alpha * (dt * dt / 2 + moment * h);
    }
    noise_scale = alpha * std::sqrt(std::max(0., target * target - lambda * lambda * ratio * ratio));
  }
  if (!er_history_) {
    previous_denoised_.resize(count);
    previous_derivative_.resize(count);
  }
  for (size_t j = 0; j < count; ++j) {
    const float denoised = sample[j] + float(sigma) * velocity[j];
    double value = carry * sample[j] + clean * denoised, derivative = 0;
    if (next > 0 && er_history_) {
      derivative = (double(denoised) - previous_denoised_[j]) / (lambda - previous_lambda_);
      value += first * derivative;
      if (er_history_ >= 2)
        value += second * (derivative - previous_derivative_[j]) / ((lambda - older_lambda_) / 2);
    }
    if (next > 0) value += noise_scale * noise[j];
    out[j] = float(value);
    previous_denoised_[j] = denoised;
    previous_derivative_[j] = float(derivative);
  }
  older_lambda_ = previous_lambda_;
  previous_lambda_ = lambda;
  er_history_ = std::min(2, er_history_ + 1);
}
}
