// Rectified-flow Euler scheduler for MiniMax H3.
//
// Three things differ from a conventional flow-match scheduler, and all three
// change the numbers:
//
//   1. The transformer predicts a *data-ward* velocity, so the denoised
//      estimate is x0 = x_t + sigma*v — a plus, not the usual minus.
//   2. Timesteps are t = 1 - sigma in [0, 1], with t = 1 meaning clean. The
//      AdaLN conditioning consumes this convention directly.
//   3. The sigma grid is linspace(1, 0, steps) with the terminal zero included
//      in the requested count, then shifted and de-duplicated.
//
// A request runs two of these, one per modality: shift 12.0 for video latents,
// 3.0 for audio.
#pragma once

#include <cstddef>
#include <vector>

namespace vidfab::sampler {

class FlowScheduler {
 public:
  // `shift` is the exponential sigma shift: sigma' = s*sigma / (1 + (s-1)*sigma).
  explicit FlowScheduler(float shift = 12.0f);

  // Builds the schedule. `num_inference_steps` counts sigma grid points
  // *including* the terminal zero, so the model is evaluated
  // `timesteps().size()` times, which is one fewer.
  void set_timesteps(int num_inference_steps);

  // Sigma grid, strictly decreasing, ending at exactly 0.
  const std::vector<float>& sigmas() const { return sigmas_; }

  // t = 1 - sigma for every sigma except the terminal zero. One entry per
  // model evaluation.
  const std::vector<float>& timesteps() const { return timesteps_; }

  size_t num_steps() const { return timesteps_.size(); }
  float shift() const { return shift_; }

  // One Euler step (eta = 0). `step_index` selects the sigma pair; `sample` is
  // x_t and `velocity` the transformer output. Both spans must be the same
  // length. Writes x_{t+1} into `out`, which may alias `sample`.
  void step(int step_index, const float* sample, const float* velocity, size_t count,
            float* out) const;

  // Rectified-flow forward process in H3's convention: x_t = t*x0 + (1-t)*noise.
  // Used to noise conditioning anchors, where t is a noise-augmentation level
  // rather than a schedule entry.
  static void scale_noise(const float* x0, const float* noise, float t, size_t count, float* out);

 private:
  float shift_;
  std::vector<float> sigmas_;
  std::vector<float> timesteps_;
};

}  // namespace vidfab::sampler
