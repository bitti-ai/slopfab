#pragma once
#include "slopfab/sampler/noise.h"
#include "slopfab/sampler/scheduler.h"

namespace slopfab::sampler {
// Image models predict noise-ward velocity; the common flow solver uses H3's
// opposite convention. Keep the existing family-specific Euler updates intact.
inline void er_sde_image_step(FlowScheduler& scheduler, uint64_t seed, int step,
                              std::vector<float>& state, std::vector<float>& prediction) {
  std::vector<float> noise(state.size());
  if (step + 1 < int(scheduler.num_steps()))
    fill_renoise_normal(seed, step, NoiseStream::kVideoLatents, noise.data(), noise.size());
  for (auto& value : prediction)
    value = -value;
  scheduler.step(step, state.data(), prediction.data(), state.size(), state.data(), noise.data());
}
}
