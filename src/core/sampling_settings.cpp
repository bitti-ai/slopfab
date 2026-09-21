#include "slopfab/sampling_settings.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "slopfab/json.h"

namespace slopfab {
SamplingSettings parse_sampling_settings(std::string_view text) {
  const json::Value root = json::parse(text);
  if (!root.is_object())
    throw std::runtime_error("sampling settings: expected an object");
  const auto* version = root.find("version");
  if (!version || !version->is_number() || version->as_number() != 1.0)
    throw std::runtime_error("sampling settings: version must be 1");
  SamplingSettings settings;
  const auto scalar = [](const json::Value& entry) {
    const double value = entry.as_number();
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
      throw std::runtime_error(
          "sampling settings: number must be finite and representable as float32");
    return static_cast<float>(value);
  };
  for (const auto& [key, value] : root.as_object()) {
    if (key == "version")
      continue;
    if (key == "default_steps") {
      const double n = value.as_number();
      if (!std::isfinite(n) || n < 2 || n > 1000000 || std::floor(n) != n)
        throw std::invalid_argument(
            "sampling settings: default_steps must be an integer in [2,1000000]");
      settings.default_steps = static_cast<int>(n);
      continue;
    }
    if (key == "video_sigma_shift")
      settings.video_sigma_shift = scalar(value);
    else if (key == "audio_sigma_shift")
      settings.audio_sigma_shift = scalar(value);
    else if (key == "base_sigmas") {
      settings.base_sigmas.emplace();
      for (const auto& sigma : value.as_array())
        settings.base_sigmas->push_back(scalar(sigma));
    } else {
      throw std::runtime_error("sampling settings: unknown key " + key);
    }
  }
  validate_sampling_settings(settings);
  return settings;
}

void validate_sampling_settings(const SamplingSettings& settings) {
  if (settings.default_steps && (*settings.default_steps < 2 || *settings.default_steps > 1000000))
    throw std::invalid_argument("sampling settings: default_steps must be in [2,1000000]");
  for (const auto& shift : {settings.video_sigma_shift, settings.audio_sigma_shift}) {
    if (shift && (!std::isfinite(*shift) || *shift <= 0.0f))
      throw std::runtime_error("sampling settings: sigma shifts must be finite and positive");
  }
  if (settings.base_sigmas) {
    sampler::FlowScheduler validator(1.0f);
    validator.set_sigmas(*settings.base_sigmas);
  }
}

void overlay_sampling_settings(SamplingSettings& destination, const SamplingSettings& overrides) {
  validate_sampling_settings(overrides);
  if (overrides.default_steps)
    destination.default_steps = overrides.default_steps;
  if (overrides.video_sigma_shift)
    destination.video_sigma_shift = overrides.video_sigma_shift;
  if (overrides.audio_sigma_shift)
    destination.audio_sigma_shift = overrides.audio_sigma_shift;
  if (overrides.base_sigmas)
    destination.base_sigmas = overrides.base_sigmas;
}

SamplingSettings
sampling_settings_from_metadata(const std::map<std::string, std::string>& metadata) {
  const auto entry = metadata.find("slopfab.sampling");
  return entry == metadata.end() ? SamplingSettings{} : parse_sampling_settings(entry->second);
}

SamplingSettings sampling_schedule_defaults(sampler::ScheduleKind schedule) {
  SamplingSettings settings;
  switch (schedule) {
  case sampler::ScheduleKind::kDefault:
    return settings;
  case sampler::ScheduleKind::kTaoMate3Step:
    settings.base_sigmas.emplace();
    // Preserve the teacher-grid arithmetic, including float32 rounding.
    for (int i : {0, 16, 33, 49})
      settings.base_sigmas->push_back(static_cast<float>(49 - i) / 49.0f);
    return settings;
  case sampler::ScheduleKind::kFastH3V2:
    settings.video_sigma_shift = 10.0f;
    settings.audio_sigma_shift = 3.0f;
    settings.base_sigmas.emplace();
    for (int rung : {999, 874, 749, 624, 500, 375, 250, 125, 0})
      settings.base_sigmas->push_back(static_cast<float>(rung) / 1000.0f);
    return settings;
  }
  throw std::runtime_error("sampling settings: unknown schedule");
}

} // namespace slopfab
