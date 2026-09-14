#pragma once
#include "slopfab/dit/ref2va.h"
#include "slopfab/reference_media.h"

namespace slopfab {
struct ReferenceConditionPlan {
  int width = 0, height = 0;
  int frames = 0;           // CFR frames presented to Qwen
  int encoding_frames = 0;  // snapped down for the VAE
  int audio_samples = 0;    // per stereo channel, before encoder padding
  dit::ReferenceGeometry geometry;
};
struct PreparedReference {
  ReferenceConditionPlan plan;
  std::vector<RGBImage> frames;
  std::vector<float>
      audio;  // [2, samples], native-rate input resampled to 32 kHz
};
ReferenceConditionPlan reference_condition_plan(const ReferenceMedia& reference,
                                                double target_seconds);
PreparedReference prepare_reference_condition(const ReferenceMedia& reference,
                                              double target_seconds);
// Normalized [24,T,H,W] to packed [T*H/2*W/2,96].
std::vector<float> patchify_reference_video(const float* latents, int frames,
                                            int height, int width);
}  // namespace slopfab
