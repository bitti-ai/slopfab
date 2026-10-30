#include "slopfab/krea.h"
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace slopfab {
float krea_guidance(const GenerateRequest& r) {
  return r.guidance_scale_set ? r.guidance_scale : 0.f;
}

std::vector<float> krea_sigma_schedule(int steps, float shift) {
  if (steps < 1 || steps > 1000 || !std::isfinite(shift) || shift <= 0)
    throw std::invalid_argument("Krea 2 requires 1..1000 Euler steps and a positive finite shift");
  std::vector<float> result;
  for (int i = 0; i <= steps; ++i) {
    const double t = 1. - double(i) / steps;
    result.push_back(float(shift * t / (1 + (shift - 1.) * t)));
  }
  return result;
}

GeneratePlan resolve_krea_plan(const GenerateRequest& r) {
  if (r.has_references() || r.image_edit.image || r.continuation || r.bridge || r.animate ||
      r.video_transition || r.preserve_driving_audio || r.keystone.source || r.end_anchor_latents ||
      !r.end_anchor_path.empty() || !r.loras.empty() || r.cache_threshold != 0 || r.skip_every ||
      r.block_cache_span || r.motion_cache.active() ||
      r.schedule != sampler::ScheduleKind::kDefault)
    throw std::invalid_argument(
        "Krea 2 supports text-to-image without references, H3 conditioning, LoRAs or caches");
  validate_sampling_settings(r.sampling);
  const auto& c = r.conditioning;
  if (c.require_prompt_embedding || c.fixed_prompt_tokens || c.image_short_edge ||
      c.media_short_edge || c.media_max_pixels || c.references_at_target_canvas ||
      c.canvas_from_reference_video || c.include_reference_audio || c.video_first ||
      c.pin_target_audio || c.max_frames || c.require_euler || c.allow_caches)
    throw std::invalid_argument("Krea 2 does not use H3 conditioning settings");
  if (r.sampling.audio_steps || r.sampling.audio_sigma_shift || r.sampling.base_sigmas ||
      r.sampling.sampler.value_or(sampler::SamplerKind::kEuler) != sampler::SamplerKind::kEuler)
    throw std::invalid_argument("Krea 2 requires Euler sampling without audio or custom grids");
  if (!std::isfinite(krea_guidance(r)) || krea_guidance(r) < 0)
    throw std::invalid_argument("Krea 2 guidance must be finite and nonnegative");
  if (r.uncond_skip_layers != std::vector<int>{22} && !r.uncond_skip_layers.empty())
    throw std::invalid_argument("Krea 2 does not support unconditional layer skipping");
  GeneratePlan p;
  p.model.family = "krea2";
  if (!r.transformer_path.empty() && std::filesystem::exists(r.transformer_path)) {
    SafeTensors st;
    st.open(r.transformer_path);
    p.model = dit::resolve_model_descriptor(st);
    p.checkpoint_available = true;
  }
  if ((r.canvas_width == 0) != (r.canvas_height == 0))
    throw std::invalid_argument("Krea 2 requires both canvas dimensions");
  p.canvas_width = r.canvas_width ? r.canvas_width : 1024;
  p.canvas_height = r.canvas_height ? r.canvas_height : 1024;
  if (p.canvas_width < 16 || p.canvas_height < 16 || p.canvas_width % 16 || p.canvas_height % 16 ||
      int64_t(p.canvas_width) * p.canvas_height > 4096LL * 4096)
    throw std::invalid_argument("Krea 2 dimensions must be multiples of 16, at most 16 megapixels");
  p.aligned_frames = p.sampling_frames = 1;
  p.geometry.canvas_multiple = 16;
  p.geometry.video_channels = 64;
  p.geometry.patch_height = p.geometry.patch_width = 1;
  p.geometry.trained_max_pixels = 2048 * 2048;
  p.layout.num_latent_frames = 1;
  p.layout.latent_width = p.canvas_width / 16;
  p.layout.latent_height = p.canvas_height / 16;
  p.layout.num_video_rows = p.layout.latent_width * p.layout.latent_height;
  p.video_sigma_shift = r.sampling.video_sigma_shift.value_or(float(std::exp(1.15)));
  p.num_inference_steps =
      r.num_inference_steps ? r.num_inference_steps : r.sampling.default_steps.value_or(8);
  p.video_sigmas = krea_sigma_schedule(p.num_inference_steps, p.video_sigma_shift);
  p.video_timesteps.assign(p.video_sigmas.begin(), p.video_sigmas.end() - 1);
  p.sampler = sampler::SamplerKind::kEuler;
  return p;
}
}
