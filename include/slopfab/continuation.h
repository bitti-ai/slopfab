#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "slopfab/dit/ref2va.h"
#include "slopfab/model_geometry.h"

namespace slopfab {

// Normalized, completed H3 sampler output. Video is frame-major [V,96];
// audio is channel-major [2*A,32]. Frames describe the cumulative 24 fps clip.
struct LatentClip {
  int width = 0, height = 0, frames = 0;
  bool sampled = true;
  std::string transformer, video_vae, audio_vae;
  LatentGeometry geometry;
  std::vector<float> video_rows, audio_rows;

  dit::SequenceLayout layout() const;
  void validate() const;
  void save(const std::string& path) const;
  static std::shared_ptr<const LatentClip> load(const std::string& path);
};

struct ContinuationPlan {
  int overlap_frames = 0, extension_frames = 0, window_frames = 0;
  int output_frames = 0;
  int overlap_video_latents = 0, overlap_audio_latents = 0;
  int window_audio_latents = 0;
};

// Preserve a prefix in each channel-major group of target values. Video has
// one group of packed rows; stereo audio has two. Capture the initial target
// noise once per run, then use the same noise at every scheduler boundary.
struct LatentPrefixConstraint {
  size_t channels = 1;
  size_t target_values_per_channel = 0;
  // Original stores prefix followed by suffix within each channel.
  size_t suffix_values_per_channel = 0;
  std::vector<float> original;
  std::vector<float> noise;

  void capture_noise(const float* rows, size_t count);
  void apply(float* rows, size_t count, float sigma) const;
};

struct ContinuationConstraint {
  LatentPrefixConstraint video, audio;
};

ContinuationConstraint make_continuation_constraint(const LatentClip& source,
                                                    const ContinuationPlan& plan);

// Extension rounds up to a multiple of 17. Overlap is exactly 17*k+5,
// at least 5 frames. Audio boundaries are rounded on the cumulative timeline.
ContinuationPlan plan_continuation(const LatentClip& source, int overlap, int extension);
void append_continuation_guide(const LatentClip& source, const ContinuationPlan& plan,
                               uint64_t seed, std::vector<dit::ReferenceGeometry>& geometry,
                               std::vector<float>& video, std::vector<float>& audio);
LatentClip join_continuation(const LatentClip& source, const ContinuationPlan& plan,
                             const std::vector<float>& video, const std::vector<float>& audio);

struct LatentBridge {
  std::shared_ptr<const LatentClip> left, right;
  int left_margin_frames = 17, right_margin_frames = 17;
  int context_frames = 22;
};

struct BridgePlan {
  int gap_frames = 0, window_frames = 0, output_frames = 0;
  int context_video_latents = 0;
  int left_video_start = 0, left_video_keep = 0, right_video_drop = 0;
  int generated_video_latents = 0;
  int left_audio_start = 0, left_audio_keep = 0, right_audio_drop = 0;
  int left_context_audio = 0, right_context_audio = 0;
  int generated_audio_latents = 0, window_audio_latents = 0;
};

// Gap rounds up to 17*k+12; margins must be multiples of 17. This places
// both source clips and the sampling window on the same H3 temporal phase.
BridgePlan plan_bridge(const LatentBridge& bridge, int gap_frames);
ContinuationConstraint make_bridge_constraint(const LatentBridge& bridge, const BridgePlan& plan);
void append_bridge_guides(const LatentBridge& bridge, const BridgePlan& plan, uint64_t seed,
                          std::vector<dit::ReferenceGeometry>& geometry, std::vector<float>& video,
                          std::vector<float>& audio);
LatentClip join_bridge(const LatentBridge& bridge, const BridgePlan& plan,
                       const std::vector<float>& video, const std::vector<float>& audio);

} // namespace slopfab
