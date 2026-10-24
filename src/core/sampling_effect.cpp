#include "slopfab/sampling_effect.h"
#include <cmath>
#include <stdexcept>

namespace slopfab {
bool apply_sampling_effect(const SamplingEffect& effect, SamplingStep step, const float* velocity,
                           std::vector<float>& modified, std::vector<float>& prediction) {
  if (!effect || !step.count || !step.latent || !velocity || !std::isfinite(step.sigma) ||
      step.sigma <= 0)
    throw std::invalid_argument("sampling effect: invalid prediction or sigma");
  prediction.resize(step.count);
  modified.assign(velocity, velocity + step.count);
  for (size_t i = 0; i < step.count; ++i)
    prediction[i] = step.latent[i] + step.sigma * velocity[i];
  const auto original = prediction;
  step.denoised = prediction.data();
  if (!effect(step))
    return false;
  for (size_t i = 0; i < step.count; ++i) {
    if (!std::isfinite(prediction[i]))
      throw std::runtime_error("sampling effect returned a non-finite prediction");
    if (prediction[i] != original[i]) {
      modified[i] += (prediction[i] - original[i]) / step.sigma;
      if (!std::isfinite(modified[i]))
        throw std::runtime_error("sampling effect overflowed the velocity");
    }
  }
  return true;
}
} // namespace slopfab
