#include "slopfab/model_profile.h"
#include "slopfab/pipeline.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace slopfab {
const ModelProfile& model_profile(const std::string& family) {
  static const ModelProfile profiles[] = {
      {"h3", "MiniMax H3", "fl2va", "qwen3vl_32b", "video", false, 32, 5120},
      {"moss", "MOSS Image", "moss_image", "qwen3.5_9b", "flux2", true, 16, 4096}};
  for (const auto& profile : profiles)
    if (family == profile.family)
      return profile;
  throw std::invalid_argument("unsupported model family: " + family);
}

const ModelProfile& checkpoint_model_profile(const SafeTensors& checkpoint) {
  return model_profile(checkpoint.find("video_embedder.weight") &&
                               checkpoint.find("shared_block_modulation.weight")
                           ? "moss"
                           : "h3");
}

std::string request_model_family(const GenerateRequest& request) {
  if (!request.model_family.empty())
    model_profile(request.model_family);
  if (!request.transformer_path.empty() && std::filesystem::exists(request.transformer_path)) {
    SafeTensors checkpoint;
    checkpoint.open(request.transformer_path);
    const std::string detected = checkpoint_model_profile(checkpoint).family;
    if (!request.model_family.empty() && request.model_family != detected)
      throw std::invalid_argument("selected model family conflicts with transformer checkpoint");
    return detected;
  }
  return request.model_family.empty() ? "h3" : request.model_family;
}

std::vector<float> moss_sigma_schedule(int steps, float shift) {
  if (steps < 1 || steps > 10000 || !std::isfinite(shift) || shift <= 0)
    throw std::invalid_argument("MOSS requires 1..10000 steps and a positive finite shift");
  std::vector<float> sigmas;
  const float end = shift / (1000.f + shift - 1.f);
  // Shift, snap to the 1000-point training grid, remove consecutive duplicates.
  // Midpoint ties choose the lower training sigma, as in the upstream scheduler.
  for (int i = 0; i < steps; ++i) {
    const float t = steps == 1 ? 1.f : 1.f + (end - 1.f) * (float(i) / (steps - 1));
    const float shifted = shift * t / (1.f + (shift - 1.f) * t);
    const float scaled = shifted * 1000.f;
    const int lower = std::clamp(int(std::floor(scaled)), 1, 1000);
    const int upper = std::min(lower + 1, 1000);
    const float sigma = (scaled - lower <= upper - scaled ? lower : upper) / 1000.f;
    if (sigmas.empty() || sigma != sigmas.back())
      sigmas.push_back(sigma);
  }
  sigmas.push_back(0.f);
  return sigmas;
}

GeneratePlan resolve_moss_plan(const GenerateRequest& r) {
  if (r.has_references() || r.continuation || r.bridge || r.image_edit.image || r.animate ||
      r.video_transition || r.preserve_driving_audio || !r.loras.empty() ||
      r.cache_threshold != 0 || r.skip_every != 0 || r.block_cache_span != 0 ||
      r.motion_cache.active() || r.schedule != sampler::ScheduleKind::kDefault)
    throw std::invalid_argument(
        "MOSS supports text-to-image without H3 references, adapters or caches");
  validate_sampling_settings(r.sampling);
  const auto& c = r.conditioning;
  if (c.require_prompt_embedding || c.fixed_prompt_tokens || c.image_short_edge ||
      c.media_short_edge || c.media_max_pixels || c.references_at_target_canvas ||
      c.canvas_from_reference_video || c.include_reference_audio || c.video_first ||
      c.pin_target_audio || c.max_frames || c.require_euler || c.allow_caches)
    throw std::invalid_argument("MOSS does not use H3 conditioning settings");
  if (r.sampling.audio_steps || r.sampling.audio_sigma_shift || r.sampling.base_sigmas)
    throw std::invalid_argument(
        "MOSS uses its image training-grid schedule; audio/custom grids are unsupported");
  if (!std::isfinite(r.guidance_scale) || r.guidance_scale < 0)
    throw std::invalid_argument("MOSS guidance must be finite and nonnegative");
  for (int layer : r.uncond_skip_layers)
    if (layer < 0 || layer >= 42)
      throw std::invalid_argument("MOSS skip layers must be in [0,42)");
  GeneratePlan p;
  p.model.family = "moss";
  if (!r.transformer_path.empty() && std::filesystem::exists(r.transformer_path)) {
    SafeTensors checkpoint;
    checkpoint.open(r.transformer_path);
    p.model = dit::resolve_model_descriptor(checkpoint);
    p.checkpoint_available = true;
  }
  if ((r.canvas_width == 0) != (r.canvas_height == 0))
    throw std::invalid_argument("MOSS requires both canvas dimensions or neither");
  p.canvas_width = r.canvas_width ? r.canvas_width : 1024;
  p.canvas_height = r.canvas_height ? r.canvas_height : 1024;
  if (p.canvas_width < 16 || p.canvas_height < 16 || p.canvas_width % 16 || p.canvas_height % 16 ||
      int64_t(p.canvas_width) * p.canvas_height > 4096LL * 4096)
    throw std::invalid_argument(
        "MOSS dimensions must be positive multiples of 16, at most 16 megapixels");
  p.aligned_frames = p.sampling_frames = 1;
  p.geometry.canvas_multiple = model_profile("moss").canvas_multiple;
  p.geometry.video_channels = 128;
  p.geometry.patch_height = p.geometry.patch_width = 1;
  p.geometry.trained_max_pixels = 2048 * 2048;
  p.layout.num_latent_frames = 1;
  p.layout.latent_height = p.canvas_height / 16;
  p.layout.latent_width = p.canvas_width / 16;
  p.layout.num_video_rows = p.layout.latent_height * p.layout.latent_width;
  p.video_sigma_shift = r.sampling.video_sigma_shift.value_or(5.f);
  p.num_inference_steps =
      r.num_inference_steps ? r.num_inference_steps : r.sampling.default_steps.value_or(50);
  p.video_sigmas = moss_sigma_schedule(p.num_inference_steps, p.video_sigma_shift);
  p.video_timesteps.assign(p.video_sigmas.begin(), p.video_sigmas.end() - 1);
  for (auto& t : p.video_timesteps)
    t *= 1000.f;
  p.sampler = r.sampling.sampler.value_or(sampler::SamplerKind::kEuler);
  if (p.sampler != sampler::SamplerKind::kEuler)
    throw std::invalid_argument("MOSS currently supports Euler sampling");
  return p;
}
} // namespace slopfab
