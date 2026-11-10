#include "slopfab/ph3_image.h"
#include "slopfab/pipeline.h"
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace slopfab {
bool is_ph3_image_transformer(const SafeTensors& checkpoint) {
  const auto* table = checkpoint.find("adaln_t_table");
  return table && table->shape == std::vector<int64_t>{4097, 16} &&
         checkpoint.find("video_patch_proj.weight") &&
         checkpoint.find("condition_proj.weight") &&
         checkpoint.at("condition_proj.weight").shape == std::vector<int64_t>{5376, 5120} &&
         checkpoint.find("blocks.0.adaln_proj.linear.weight");
}

void validate_ph3_image_transformer(const SafeTensors& checkpoint) {
  // These projections remain unpacked in BF16, INT8 and W4A8 releases.
  for (const auto& item : std::vector<std::pair<std::string, std::vector<int64_t>>>{
           {"adaln_t_table", {4097, 16}},
           {"video_patch_proj.weight", {5376, 96}},
           {"audio_patch_proj.weight", {5376, 32}},
           {"condition_proj.weight", {5376, 5120}},
           {"blocks.0.adaln_proj.linear.weight", {96768, 16}},
           {"final_layer.adaln_proj.linear.weight", {10752, 16}},
           {"final_layer.video_out.weight", {96, 5376}},
           {"final_layer.audio_out.weight", {32, 5376}}}) {
    const auto* tensor = checkpoint.find(item.first);
    if (!tensor || tensor->shape != item.second)
      throw std::invalid_argument("P-H3 Image: incompatible tensor " + item.first);
  }
  if (checkpoint.find("blocks.0.attn.to_gate_compress.weight"))
    throw std::invalid_argument("P-H3 Image does not use compressed attention");
}

const std::vector<float>& ph3_image_sigmas() {
  // PrunaAI/P-H3-Image-Comfy workflows/p-h3-image_text_to_image.json,
  // ManualSigmas; ModelSamplingMiniMaxH3 shifts both modalities by 1.
  static const std::vector<float> sigmas = {
      1.f, .984753f, .967905f, .951688f, .925372f,
      .881923f, .810083f, .528718f, 0.f};
  return sigmas;
}

GeneratePlan resolve_ph3_image_plan(const GenerateRequest& r) {
  if (r.has_references() || r.image_edit.image || r.continuation || r.bridge ||
      r.animate || r.video_transition || r.preserve_driving_audio || !r.loras.empty() ||
      r.cache_threshold != 0 || r.skip_every != 0 || r.block_cache_span != 0 ||
      r.motion_cache.active())
    throw std::invalid_argument("P-H3 Image supports text-to-image only, without references, adapters or caches");
  if (!r.clip_projection_path.empty() || !r.video_decoder_path.empty() ||
      r.end_anchor_latents || !r.regenerate.mode.empty() || !r.remake_mode.empty())
    throw std::invalid_argument("P-H3 Image requires its own text encoder and image VAE without H3 editing modes");
  if (!r.negative_prompt.empty() || (r.guidance_scale_set && r.guidance_scale != 1.f))
    throw std::invalid_argument("P-H3 Image uses guidance 1 without a negative prompt");
  validate_sampling_settings(r.sampling);
  const auto& s = r.sampling;
  if (r.schedule != sampler::ScheduleKind::kDefault ||
      (r.num_inference_steps != 0 && r.num_inference_steps != 8) ||
      (s.default_steps && *s.default_steps != 8) ||
      (s.audio_steps && *s.audio_steps != 8) ||
      (s.video_sigma_shift && *s.video_sigma_shift != 1.f) ||
      (s.audio_sigma_shift && *s.audio_sigma_shift != 1.f) ||
      (s.base_sigmas && *s.base_sigmas != ph3_image_sigmas()) ||
      (s.sampler && *s.sampler != sampler::SamplerKind::kEuler) ||
      s.scheduler || s.beta_alpha || s.beta_beta)
    throw std::invalid_argument("P-H3 Image requires its fixed eight-step Euler grid and shifts 1/1");
  const auto& c = r.conditioning;
  if (c.require_prompt_embedding || c.fixed_prompt_tokens || c.image_short_edge ||
      c.media_short_edge || c.media_max_pixels || c.references_at_target_canvas ||
      c.canvas_from_reference_video || c.include_reference_audio || c.video_first ||
      c.pin_target_audio || c.max_frames || c.require_euler || c.allow_caches)
    throw std::invalid_argument("P-H3 Image does not accept H3 conditioning overrides");
  if ((r.canvas_width == 0) != (r.canvas_height == 0))
    throw std::invalid_argument("P-H3 Image requires both canvas dimensions or neither");

  GeneratePlan p;
  p.model.family = "ph3_image";
  p.model.compatibility_architecture = dit::TransformerArchitecture::kPH3Image;
  if (!r.transformer_path.empty() && std::filesystem::exists(r.transformer_path)) {
    SafeTensors checkpoint;
    checkpoint.open(r.transformer_path);
    validate_ph3_image_transformer(checkpoint);
    p.model = dit::resolve_model_descriptor(checkpoint);
    p.checkpoint_available = true;
  }
  p.canvas_width = r.canvas_width ? r.canvas_width : 2912;
  p.canvas_height = r.canvas_height ? r.canvas_height : 2912;
  if (p.canvas_width < 32 || p.canvas_height < 32 || p.canvas_width % 32 ||
      p.canvas_height % 32 || int64_t(p.canvas_width) * p.canvas_height > 4096LL * 4096)
    throw std::invalid_argument("P-H3 Image dimensions must be multiples of 32, at most 16 megapixels");
  p.geometry.trained_max_pixels = 2912 * 2912;
  p.aligned_frames = p.sampling_frames = p.layout.num_latent_frames = 1;
  p.duration_seconds = 1.0 / p.geometry.fps;
  p.layout.latent_height = p.canvas_height / 16;
  p.layout.latent_width = p.canvas_width / 16;
  p.layout.num_video_rows = p.layout.rows_per_frame();
  // Upstream PH3ImageConditioning uses _av_latent(width, height, 1, 2).
  // Audio is sampled internally even though delivery is a still image.
  p.layout.num_audio_latents = 2;
  p.layout.num_audio_rows = 2 * p.geometry.audio_channels;
  p.video_sigma_shift = p.audio_sigma_shift = 1.f;
  p.video_sigmas = p.audio_sigmas = ph3_image_sigmas();
  for (size_t i = 0; i + 1 < p.video_sigmas.size(); ++i)
    p.video_timesteps.push_back(1.f - p.video_sigmas[i]);
  p.audio_timesteps = p.video_timesteps;
  p.num_inference_steps = 8;
  p.fixed_sampling_grid = true;
  p.conditioning.require_euler = true;
  p.conditioning.allow_caches = false;
  p.sampling_sources = {"P-H3 Image published eight-step grid"};
  return p;
}
} // namespace slopfab
