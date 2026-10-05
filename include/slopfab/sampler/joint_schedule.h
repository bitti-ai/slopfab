#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace slopfab::sampler {

// Merge update boundaries at (index + 1) / modality_steps. Integer comparisons
// avoid rounding mismatches. Each entry evaluates both modalities at their
// current noise levels, then advances only the indicated schedulers. Both
// terminal updates coincide; equal counts retain the original zipped loop.
struct JointStep {
  size_t video;
  size_t audio;
  bool advance_video;
  bool advance_audio;
};

inline size_t joint_step_count(size_t video, size_t audio) {
  return video + audio - std::gcd(video, audio);
}

inline std::vector<JointStep> joint_schedule(size_t video, size_t audio) {
  if (video == 0 || audio == 0)
    throw std::invalid_argument("joint schedule requires nonempty video and audio schedules");
  std::vector<JointStep> result;
  result.reserve(joint_step_count(video, audio));
  size_t v = 0, a = 0;
  while (v < video && a < audio) {
    const uint64_t video_boundary = uint64_t(v + 1) * audio;
    const uint64_t audio_boundary = uint64_t(a + 1) * video;
    const bool advance_video = video_boundary <= audio_boundary;
    const bool advance_audio = audio_boundary <= video_boundary;
    result.push_back({v, a, advance_video, advance_audio});
    v += advance_video;
    a += advance_audio;
  }
  return result;
}

} // namespace slopfab::sampler
