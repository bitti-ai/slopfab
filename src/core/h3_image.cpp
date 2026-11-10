#include "slopfab/h3_image.h"
#include "slopfab/pipeline.h"
#include "slopfab/json.h"
#include "slopfab/dit/ref2va.h"
#include "slopfab/image.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace slopfab {
bool is_h3_image_transformer(const SafeTensors& checkpoint) {
  return checkpoint.find("adaln_t_table") && checkpoint.find("video_patch_proj.weight") &&
         checkpoint.find("condition_proj.weight") && !checkpoint.find("audio_patch_proj.weight");
}

void validate_h3_image_transformer(const SafeTensors& checkpoint) {
  for (const auto& item : std::vector<std::pair<std::string, std::vector<int64_t>>>{
           {"adaln_t_table", {1025, 8}}, {"video_patch_proj.weight", {5376, 96}},
           {"condition_proj.weight", {5376, 5120}},
           {"blocks.0.adaln_proj.linear.weight", {96768, 8}},
           {"final_layer.adaln_proj.linear.weight", {10752, 8}},
           {"final_layer.video_out.weight", {576, 5376}},
           {"final_layer.video_out.bias", {576}}}) {
    const auto* tensor = checkpoint.find(item.first);
    if (!tensor || tensor->shape != item.second)
      throw std::invalid_argument("MiniMax-H3-Image: incompatible tensor " + item.first);
  }
  if (checkpoint.find("audio_patch_proj.weight") || checkpoint.find("final_layer.audio_out.weight") ||
      checkpoint.find("blocks.0.attn.to_gate_compress.weight"))
    throw std::invalid_argument("MiniMax-H3-Image requires dense image-only attention");
  const auto metadata = checkpoint.metadata().find("config");
  if (metadata != checkpoint.metadata().end()) {
    const auto root = json::parse(metadata->second);
    if (const auto* config = root.find("transformer")) {
      if (const auto* steps = config->find("pdd_steps"))
        if (steps->as_number() != 6)
          throw std::invalid_argument("MiniMax-H3-Image requires six PDD heads");
      if (const auto* grid = config->find("pdd_sigmas")) {
        const auto& values = grid->as_array();
        const auto& expected = h3_image_sigmas();
        if (values.size() != expected.size())
          throw std::invalid_argument("MiniMax-H3-Image has an incompatible PDD grid");
        for (size_t i = 0; i < values.size(); ++i)
          if (std::abs(values[i].as_number() - expected[i]) > 1e-7)
            throw std::invalid_argument("MiniMax-H3-Image has an incompatible PDD grid");
      }
    }
  }
}

const std::vector<float>& h3_image_sigmas() {
  // MiniMaxAI/MiniMax-H3-Image/pdd_config.json, six trained PDD intervals.
  static const std::vector<float> sigmas = {
      1.f, .9675236940383911f, .8875485062599182f, .75f,
      .5327739119529724f, .23200887441635132f, 0.f};
  return sigmas;
}

vae::ViTConfig h3_image_vae_config() {
  vae::ViTConfig c;
  c.num_layers = 24;
  c.dim = 1024;
  c.heads = 16;
  c.ffn_inner = 4096;
  c.patch_t = 1;
  c.rope_dim = 32;
  c.rope_dims = 2;
  return c;
}

void h3_image_reference_size(int width, int height, int* out_height, int* out_width, bool vision) {
  const int short_edge = vision ? 1024 : std::clamp(std::min(width, height), 1024, 2048);
  dit::resolve_reference_image_size(width, height, out_height, out_width, short_edge);
}

std::string h3_image_picture_label(size_t index) {
  return "Picture " + std::to_string(index + 1) + ": ";
}

GeneratePlan resolve_h3_image_plan(const GenerateRequest& r) {
  r.image_edit.validate();
  const bool edit = bool(r.image_edit.image);
  const bool outpaint = r.image_edit.image && r.image_edit.invert_mask;
  if (edit && r.image_edit.strength != 1.f)
    throw std::invalid_argument("MiniMax-H3-Image masked editing requires strength 1 to retain six PDD evaluations");
  if (outpaint && r.image_edit.langevin_steps != 0)
    throw std::invalid_argument("MiniMax-H3-Image outpainting requires Langevin steps 0 to retain six PDD evaluations");
  if (!r.reference_media.empty() || r.has_refmods() || !r.end_anchor_path.empty() ||
      r.continuation || r.bridge ||
      r.animate || r.video_transition || r.preserve_driving_audio || !r.loras.empty() ||
      r.cache_threshold != 0 || r.skip_every != 0 || r.block_cache_span != 0 ||
      r.motion_cache.active())
    throw std::invalid_argument("MiniMax-H3-Image accepts image references; video/audio references, adapters and caches are unsupported");
  if (r.reference_image_paths.size() + size_t(edit) > 16)
    throw std::invalid_argument("MiniMax-H3-Image supports at most 16 reference images including the edit source");
  if (!r.clip_projection_path.empty() || !r.video_decoder_path.empty() ||
      r.end_anchor_latents || !r.regenerate.mode.empty() || !r.remake_mode.empty())
    throw std::invalid_argument("MiniMax-H3-Image requires the H3 text encoder and its image VAE without H3 editing modes");
  if (!r.negative_prompt.empty() || (r.guidance_scale_set && r.guidance_scale != 1.f))
    throw std::invalid_argument("MiniMax-H3-Image uses guidance 1 without a negative prompt");
  validate_sampling_settings(r.sampling);
  const auto& s = r.sampling;
  if (r.schedule != sampler::ScheduleKind::kDefault ||
      (r.num_inference_steps != 0 && r.num_inference_steps != 6) ||
      (s.default_steps && *s.default_steps != 6) ||
      (s.audio_steps && *s.audio_steps != 6) ||
      (s.video_sigma_shift && *s.video_sigma_shift != 1.f) ||
      (s.audio_sigma_shift && *s.audio_sigma_shift != 1.f) ||
      (s.base_sigmas && *s.base_sigmas != h3_image_sigmas()) ||
      (s.sampler && *s.sampler != sampler::SamplerKind::kEuler) ||
      s.scheduler || s.beta_alpha || s.beta_beta)
    throw std::invalid_argument("MiniMax-H3-Image requires its fixed six-step Euler grid and shifts 1/1");
  const auto& c = r.conditioning;
  if (c.require_prompt_embedding || c.fixed_prompt_tokens || c.image_short_edge ||
      c.media_short_edge || c.media_max_pixels || c.references_at_target_canvas ||
      c.canvas_from_reference_video || c.include_reference_audio || c.video_first ||
      c.pin_target_audio || c.max_frames || c.require_euler || c.allow_caches)
    throw std::invalid_argument("MiniMax-H3-Image does not accept H3 conditioning overrides");
  if ((r.canvas_width == 0) != (r.canvas_height == 0))
    throw std::invalid_argument("MiniMax-H3-Image requires both canvas dimensions or neither");

  GeneratePlan p;
  p.model.family = "h3_image";
  p.model.pdd_steps = 6;
  p.model.supports_references = true;
  p.model.compatibility_architecture = dit::TransformerArchitecture::kH3Image;
  if (!r.transformer_path.empty() && std::filesystem::exists(r.transformer_path)) {
    SafeTensors checkpoint;
    checkpoint.open(r.transformer_path);
    validate_h3_image_transformer(checkpoint);
    p.model = dit::resolve_model_descriptor(checkpoint);
    p.checkpoint_available = true;
  }
  p.canvas_width = r.canvas_width ? r.canvas_width : 4096;
  p.canvas_height = r.canvas_height ? r.canvas_height : 4096;
  if (edit) {
    p.canvas_width = (r.image_edit.image->width + 31) / 32 * 32;
    p.canvas_height = (r.image_edit.image->height + 31) / 32 * 32;
    if (r.has_explicit_canvas() && (r.canvas_width != p.canvas_width || r.canvas_height != p.canvas_height))
      throw std::invalid_argument("masked edit resolution must match the source canvas padded to 32 pixels");
  } else if (!r.has_explicit_canvas() && !r.reference_image_paths.empty()) {
    const auto image = load_reference_image(r.reference_image_paths.front());
    h3_image_reference_size(image.width, image.height, &p.canvas_height, &p.canvas_width);
  }
  if (p.canvas_width < 32 || p.canvas_height < 32 || p.canvas_width % 32 ||
      p.canvas_height % 32 || int64_t(p.canvas_width) * p.canvas_height > 20LL * 1024 * 1024)
    throw std::invalid_argument("MiniMax-H3-Image dimensions must be multiples of 32, at most 20 Mi-pixels (20971520 pixels)");
  p.geometry.trained_max_pixels = 4096 * 4096;
  p.aligned_frames = p.sampling_frames = p.layout.num_latent_frames = 1;
  p.duration_seconds = 1.0 / p.geometry.fps;
  p.layout.latent_height = p.canvas_height / 16;
  p.layout.latent_width = p.canvas_width / 16;
  p.layout.num_video_rows = p.layout.rows_per_frame();
  p.layout.num_audio_latents = p.layout.num_audio_rows = 0;
  p.video_sigma_shift = p.audio_sigma_shift = 1.f;
  p.video_sigmas = p.audio_sigmas = h3_image_sigmas();
  for (size_t i = 0; i + 1 < p.video_sigmas.size(); ++i)
    p.video_timesteps.push_back(1.f - p.video_sigmas[i]);
  p.audio_timesteps = p.video_timesteps;
  p.num_inference_steps = 6;
  p.fixed_sampling_grid = true;
  p.conditioning.require_euler = true;
  p.conditioning.allow_caches = false;
  p.sampling_sources = {"MiniMax-H3-Image published six-step grid"};
  return p;
}
} // namespace slopfab
