#pragma once
#include <cstddef>
#include <functional>
#include <vector>

namespace slopfab {
// Host F32, normalized model space, target rows only. H3 video is
// [frame, y/2, x/2, channel, dy, dx]; audio is [channel, time, feature].
struct SamplingStep {
  int modality = 0;              // 0 video, 1 audio
  int step = 0, total_steps = 0; // this modality's update index/count
  int evaluation = 0;            // joint schedule index
  float sigma = 0, sigma_next = 0;
  int frames = 0, height = 0, width = 0; // latent geometry; zero for audio
  int row_width = 0;
  size_t count = 0;
  const float* latent = nullptr;
  float* denoised = nullptr;
};

using SamplingEffect = std::function<bool(const SamplingStep&)>; // false cancels

// Converts native velocity to x0, invokes the effect, then applies only the
// changed elements as velocity deltas. A read-only callback is bit-exact.
// The input velocity is never changed, including cached predictions.
bool apply_sampling_effect(const SamplingEffect& effect, SamplingStep step, const float* velocity,
                           std::vector<float>& modified, std::vector<float>& prediction);
} // namespace slopfab
