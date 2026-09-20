#include "sampling_plan.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "slopfab/dit/checkpoint.h"
#include "slopfab/pipeline.h"

namespace slopfab {
namespace {

SamplingSettings file_settings(const std::string& path) {
  if (path.empty() || !std::filesystem::exists(path)) return {};
  SafeTensors checkpoint;
  checkpoint.open(path);
  try {
    return sampling_settings_from_metadata(checkpoint.metadata());
  } catch (const std::exception& e) {
    throw std::invalid_argument("sampling metadata in '" + path + "': " + e.what());
  }
}

bool has_settings(const SamplingSettings& settings) {
  return settings.default_steps || settings.video_sigma_shift || settings.audio_sigma_shift || settings.base_sigmas;
}

template <typename T>
void merge_adapter_field(std::optional<T>& target, const std::optional<T>& incoming,
                         const std::optional<T>& explicit_value, const char* field) {
  if (!incoming) return;
  if (target && *target != *incoming && !explicit_value)
    throw std::invalid_argument(std::string("conflicting LoRA sampling defaults for ") +
        field + "; supply an explicit sampling override");
  target = incoming;
}

}  // namespace

void resolve_sampling_plan(const GenerateRequest& request, GeneratePlan& plan) {
  SamplingSettings effective;
  effective.default_steps = request.animate ? 4 : 50;
  effective.video_sigma_shift = kVideoSigmaShift;
  effective.audio_sigma_shift = kAudioSigmaShift;
  plan.sampling_sources = {"H3 defaults"};
  SamplingSettings model_settings;
  if (!request.transformer_path.empty() && std::filesystem::exists(request.transformer_path)) {
    SafeTensors checkpoint;
    checkpoint.open(request.transformer_path);
    const auto architecture = plan.model.compatibility_architecture;
    plan.fasth3_v2 = architecture == dit::TransformerArchitecture::kFastH3V2PrunedTable;
    if (architecture == dit::TransformerArchitecture::kViggleAnimatePrunedTable)
      effective.video_sigma_shift = kViggleVideoSigmaShift;
    try {
      model_settings = sampling_settings_from_metadata(checkpoint.metadata());
    } catch (const std::exception& e) {
      throw std::invalid_argument("sampling metadata in '" + request.transformer_path + "': " + e.what());
    }
  }
  if (plan.fasth3_v2) {
    if (request.has_references() || request.continuation || request.animate ||
        request.schedule != sampler::ScheduleKind::kDefault)
      throw std::invalid_argument("FastH3 V2 supports text-to-video with its trained eight-step schedule; references, continuation and other schedules are incompatible");
    overlay_sampling_settings(effective, sampling_schedule_defaults(sampler::ScheduleKind::kFastH3V2));
    plan.sampling_sources.push_back("FastH3 V2 compatibility preset");
  }
  if (request.animate) {
    effective.video_sigma_shift = kViggleVideoSigmaShift;
    plan.sampling_sources.push_back("Animate compatibility preset");
  }
  overlay_sampling_settings(effective, model_settings);
  if (has_settings(model_settings)) plan.sampling_sources.push_back("model: " + request.transformer_path);

  // Named schedules remain compatibility aliases. Request fields win over the
  // alias, and both have higher precedence than adapter or model defaults.
  auto overrides = sampling_schedule_defaults(request.schedule);
  overlay_sampling_settings(overrides, request.sampling);
  SamplingSettings adapters;
  for (const auto& lora : request.loras) {
    if (lora.path.empty() || !std::isfinite(lora.strength))
      throw std::invalid_argument("LoRA path must be nonempty and strength finite");
    if (lora.strength == 0.0f) continue;
    const auto settings = file_settings(lora.path);
    merge_adapter_field(adapters.default_steps, settings.default_steps,
                        request.num_inference_steps != 0 ? std::optional<int>{request.num_inference_steps}
                                                         : overrides.default_steps, "default_steps");
    merge_adapter_field(adapters.video_sigma_shift, settings.video_sigma_shift,
                        overrides.video_sigma_shift, "video_sigma_shift");
    merge_adapter_field(adapters.audio_sigma_shift, settings.audio_sigma_shift,
                        overrides.audio_sigma_shift, "audio_sigma_shift");
    merge_adapter_field(adapters.base_sigmas, settings.base_sigmas,
                        overrides.base_sigmas, "base_sigmas");
    if (has_settings(settings)) plan.sampling_sources.push_back("LoRA: " + lora.path);
  }
  if (request.schedule == sampler::ScheduleKind::kTaoMate3Step &&
      std::none_of(request.loras.begin(), request.loras.end(),
                   [](const LoraSpec& lora) { return lora.strength != 0.0f; }))
    throw std::invalid_argument("taomate-3step requires an enabled TaoMate LoRA");
  overlay_sampling_settings(effective, adapters);
  overlay_sampling_settings(effective, overrides);
  if (has_settings(overrides)) plan.sampling_sources.push_back("request overrides");
  validate_sampling_settings(effective);

  if (plan.fasth3_v2) {
    const auto required = sampling_schedule_defaults(sampler::ScheduleKind::kFastH3V2);
    if (effective.video_sigma_shift != required.video_sigma_shift ||
        effective.audio_sigma_shift != required.audio_sigma_shift ||
        effective.base_sigmas != required.base_sigmas)
      throw std::invalid_argument("FastH3 V2 requires its trained sigma shifts and base grid");
  }
  plan.fixed_sampling_grid = effective.base_sigmas.has_value();
  if (plan.fixed_sampling_grid && (request.motion_cache.active() ||
      request.cache_threshold > 0 || request.skip_every > 0 || request.block_cache_span > 0))
    throw std::invalid_argument("fixed sampling grids require Euler without step, block or MotionCache reuse");
  plan.video_sigma_shift = *effective.video_sigma_shift;
  plan.audio_sigma_shift = *effective.audio_sigma_shift;
  plan.num_inference_steps = effective.base_sigmas
      ? static_cast<int>(effective.base_sigmas->size())
      : (request.num_inference_steps == 0 ? *effective.default_steps : request.num_inference_steps);
  sampler::FlowScheduler video(plan.video_sigma_shift), audio(plan.audio_sigma_shift);
  if (effective.base_sigmas) {
    video.set_base_sigmas(*effective.base_sigmas);
    audio.set_base_sigmas(*effective.base_sigmas);
  } else {
    video.set_timesteps(plan.num_inference_steps);
    audio.set_timesteps(plan.num_inference_steps);
    // Very small positive shifts can lose the denominator in float32. Reject
    // invalid generated grids here, before a runner reconstructs them.
    video.set_sigmas(video.sigmas());
    audio.set_sigmas(audio.sigmas());
  }
  plan.video_sigmas = video.sigmas();
  plan.audio_sigmas = audio.sigmas();
  plan.video_timesteps = video.timesteps();
  plan.audio_timesteps = audio.timesteps();
}

void validate_sampling_sampler(const GeneratePlan& plan, sampler::SamplerKind sampler) {
  if (plan.fixed_sampling_grid && sampler != sampler::SamplerKind::kEuler)
    throw std::invalid_argument("fixed sampling grids require the Euler sampler");
}

}  // namespace slopfab
