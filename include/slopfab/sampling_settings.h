#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "slopfab/sampler/scheduler.h"

namespace slopfab {

// Unset fields inherit defaults. An explicit grid contains unshifted sigma
// points including terminal zero; its length determines the evaluation count.
struct SamplingSettings {
  std::optional<float> video_sigma_shift;
  std::optional<float> audio_sigma_shift;
  std::optional<std::vector<float>> base_sigmas;
};

// Strict, versioned JSON: {"version":1,"video_sigma_shift":12,
// "audio_sigma_shift":3,"base_sigmas":[1,0.5,0]}. Only version is required.
SamplingSettings parse_sampling_settings(std::string_view text);
void validate_sampling_settings(const SamplingSettings& settings);
void overlay_sampling_settings(SamplingSettings& destination,
                               const SamplingSettings& overrides);
SamplingSettings sampling_settings_from_metadata(
    const std::map<std::string, std::string>& metadata);

// Compatibility recipes. New checkpoints/adapters can supply the same data
// through metadata without adding another named scheduler implementation.
SamplingSettings sampling_schedule_defaults(sampler::ScheduleKind schedule);

}  // namespace slopfab
