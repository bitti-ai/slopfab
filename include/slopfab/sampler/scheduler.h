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

namespace slopfab::sampler {

enum class ScheduleKind { kDefault, kTaoMate3Step, kFastH3V2 };

// Which integrator advances the trajectory. Both cost exactly one model
// evaluation per step; the difference is what they do with the velocity they
// were given.
enum class SamplerKind {
  // First order. The reference's update, and the default — nothing else in
  // this file changes its arithmetic by so much as an ulp.
  kEuler,
  // Adams-Bashforth 2 with the fixed-step coefficients 3/2 and -1/2, applied
  // to the same effective step size Euler uses. Second order at one evaluation
  // per step, so the point of it is being able to *lower* --steps, not a
  // cheaper step. The first step of a trajectory has no history and falls back
  // to Euler.
  kAb2,
};

// Canonical fp32 arithmetic used by the production Euler path on both CUDA
// (through FlowScheduler) and Vulkan. Subnormal operands and results become
// signed zero. Any NaN, infinity, or non-finite arithmetic result becomes the
// canonical quiet NaN 0x7fc00000. The association is deliberately fixed to
// the H3 reference expression; endpoint controls do not elide dead branches.
float exact_euler_value(float sample, float velocity,
                        float sigma_from_timestep, float ratio) noexcept;

class FlowScheduler {
 public:
  // `shift` is the exponential sigma shift: sigma' = s*sigma / (1 + (s-1)*sigma).
  explicit FlowScheduler(float shift = 12.0f);

  // Builds the schedule. `num_inference_steps` counts sigma grid points
  // *including* the terminal zero, so the model is evaluated
  // `timesteps().size()` times, which is one fewer. TaoMate3Step overrides
  // the count with four retained teacher states (three evaluations).
  void set_timesteps(int num_inference_steps, ScheduleKind schedule = ScheduleKind::kDefault);

  // Explicit decreasing sigma grid starting in (0,1] and ending at 0;
  // resets history. Distilled FastH3 starts just below sigma 1.
  void set_sigmas(const std::vector<float>& sigmas);

  // Sigma grid, strictly decreasing, ending at exactly 0.
  const std::vector<float>& sigmas() const { return sigmas_; }

  // t = 1 - sigma for every sigma except the terminal zero. One entry per
  // model evaluation.
  const std::vector<float>& timesteps() const { return timesteps_; }

  size_t num_steps() const { return timesteps_.size(); }
  float shift() const { return shift_; }

  // Selects the integrator. Clears any velocity history, so it is only
  // meaningful between trajectories. Default kEuler.
  void set_sampler(SamplerKind kind);
  SamplerKind sampler() const { return sampler_; }

  // Forgets the velocity history and the step cursor, so the next `step` may
  // carry any index and is treated as the first of a fresh trajectory.
  // `set_timesteps` and `set_sampler` both do this implicitly.
  void reset();

  // One step (eta = 0). `step_index` selects the sigma pair; `sample` is x_t
  // and `velocity` the transformer output. Both spans must be the same length.
  // Writes x_{t+1} into `out`, which may alias `sample`.
  //
  // Not const, and order-sensitive: kAb2 keeps v_{n-1} here, so a caller that
  // skips or repeats an index would silently extrapolate from the wrong
  // velocity. `step_index` must therefore be either 0 — which restarts the
  // trajectory, which is what makes one scheduler reusable across runs — or
  // exactly one past the previous call's. Anything else throws. `count` may
  // change between trajectories but not within one.
  void step(int step_index, const float* sample, const float* velocity, size_t count, float* out);

  // Rectified-flow forward process in H3's convention: x_t = t*x0 + (1-t)*noise.
  // Used to noise conditioning anchors, where t is a noise-augmentation level
  // rather than a schedule entry.
  static void scale_noise(const float* x0, const float* noise, float t, size_t count, float* out);

 private:
  void clear_history();

  float shift_;
  std::vector<float> sigmas_;
  std::vector<float> timesteps_;

  SamplerKind sampler_ = SamplerKind::kEuler;
  // v_{n-1}, kept only under kAb2. One buffer per scheduler, so the video and
  // audio instances that share a denoising loop never see each other's history
  // and may differ in length.
  std::vector<float> previous_velocity_;
  bool has_previous_ = false;
  // The only index `step` will accept next, or -1 for "fresh, any index".
  int expected_step_ = -1;
};

}  // namespace slopfab::sampler
