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
                                              double target_seconds,
                                              bool prepare_vae = true);

// Host-only, seed-independent VAE outputs. A single request entry bounds cache
// retention; callers publish only after every encode completes successfully.
struct EncodedReferenceCondition {
  dit::ReferenceGeometry geometry;
  std::vector<float> video_rows, audio_rows;
};
class EncodedMediaCache {
 public:
  const std::vector<EncodedReferenceCondition>* find(const std::string& key) const {
    return valid_ && key == key_ ? &entries_ : nullptr;
  }
  void store(std::string key, std::vector<EncodedReferenceCondition> entries) {
    key_.swap(key);
    entries_.swap(entries);
    valid_ = true;
  }
  void clear() { valid_ = false; key_.clear(); entries_.clear(); }
 private:
  bool valid_ = false;
  std::string key_;
  std::vector<EncodedReferenceCondition> entries_;
};
void append_encoded_reference_condition(const EncodedReferenceCondition& encoded,
    uint64_t seed, size_t reference_index, std::vector<float>& video_rows,
    std::vector<float>& audio_rows);
// Normalized [24,T,H,W] to packed [T*H/2*W/2,96].
std::vector<float> patchify_reference_video(const float* latents, int frames,
                                            int height, int width);
}  // namespace slopfab
