// Native adaptation of starsFriday's MiniMax H3 MotionCache (MIT).
// See third_party/motioncache/LICENSE and docs/motioncache.md.
#pragma once

#include <cstddef>
#include <vector>

#include "slopfab/dit/packing.h"

namespace slopfab::dit {

struct MotionCacheConfig {
  bool enabled = false;
  float reuse_threshold = 0.15f;
  float motion_strength = 1.0f;
  int warmup_steps = 4;
  int max_consecutive_skips = 2;
  float start_percent = 0.15f;
  float end_percent = 0.95f;
  int subsample_factor = 8;
  bool verbose = false;

  bool active() const { return enabled && reuse_threshold > 0.0f; }
  void validate() const;
};

// One instance per trajectory; pointers describe target rows only, excluding
// fixed reference anchors. Video is 2x2 patchified, audio is [stereo*time, C].
// Native velocities have the opposite sign to ComfyUI's model output, so the
// cached residual is v + x and reuse returns residual - current_x.
class MotionCache {
 public:
  MotionCache(const MotionCacheConfig& config, const SequenceLayout& layout,
              int video_dim, int audio_dim, int steps, float video_shift,
              bool pin_audio = false);
  bool enabled() const { return config_.active(); }
  bool should_compute(int step, float sigma, const float* video, const float* audio);
  void update(float sigma, const float* video, const float* audio,
              const float* video_velocity, const float* audio_velocity);
  void reuse(const float* video, const float* audio,
             float* video_velocity, float* audio_velocity) const;
  int computed() const { return computed_; }
  int skipped() const { return skipped_; }
  float score() const { return score_; }

 private:
  std::vector<float> sample(const float* data, const std::vector<size_t>& indices) const;
  MotionCacheConfig config_;
  int steps_, frames_, channels_;
  size_t spatial_ = 0, video_size_ = 0, audio_size_ = 0;
  float start_sigma_, end_sigma_;
  std::vector<size_t> video_indices_, audio_indices_;
  std::vector<float> previous_video_, previous_audio_, previous_vv_, previous_av_;
  std::vector<float> weights_, video_residual_, audio_residual_;
  float video_rate_ = -1, audio_rate_ = -1, video_norm_ = 1, audio_norm_ = 1;
  float accumulated_ = 0, score_ = 0;
  int consecutive_ = 0, computed_ = 0, skipped_ = 0;
};

}  // namespace slopfab::dit
