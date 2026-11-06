#include "slopfab/generate.h"
#include "slopfab/vae/sol_h3.h"
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace slopfab {
void validate_generation_options(const GenerateRequest& r, const GeneratePlan& p,
                                 const RunOptions& o) {
  const auto require = [](bool ok, const char* message) {
    if (!ok)
      throw std::invalid_argument(message);
  };
  require(r.clip_projection_path.empty() || o.prompt_embedding_path.empty(),
          "use either ClipProj or a precomputed prompt embedding, not both");
  if (!r.clip_projection_path.empty()) {
    require(o.attention_mode != AttentionMode::kExact, "ClipProj does not implement the exact conditioner arithmetic mode");
    require(o.inference_backend != DeviceBackend::kVulkan || o.vulkan_portable_arithmetic,
            "ClipProj Vulkan requires --vulkan-arithmetic portable");
  }
  require(r.video_decoder_path.empty() || p.model.family == "h3",
          "a separate video decoder is supported only for MiniMax H3");
  if (p.model.family == "h3" && !r.video_decode_path().empty() &&
      std::filesystem::is_regular_file(r.video_decode_path())) {
    SafeTensors decoder;
    decoder.open(r.video_decode_path());
    if (vae::is_sol_h3_vae(decoder)) {
      vae::validate_sol_h3_vae(decoder);
    }
  }
  if (p.model.family == "h3" && (!r.conditioning_image_paths().empty() ||
      !r.reference_media.empty() || r.image_edit.image) &&
      !r.video_vae_path.empty() && std::filesystem::is_regular_file(r.video_vae_path)) {
    SafeTensors encoder;
    encoder.open(r.video_vae_path);
    require(!vae::is_sol_h3_vae(encoder),
            "Sol-H3-VAE is decoder-only: keep the original VIDEO_VAE/--vae for reference encoding "
            "and select Sol with VIDEO_DECODER/--video-decoder");
  }
  if (p.model.family == "yue2" || p.model.family == "minimax_music3") {
    require(o.inference_backend == DeviceBackend::kCuda || o.inference_backend == DeviceBackend::kVulkan, "music requires CUDA or Vulkan");
    require(o.inference_backend != DeviceBackend::kVulkan || o.vulkan_portable_arithmetic, "Vulkan music requires portable arithmetic; use --vulkan-arithmetic portable or SLOPFAB_VULKAN_ARITHMETIC_PORTABLE");
    require(o.attention_mode == AttentionMode::kFlash2, "music uses fixed attention kernels; attention overrides are unsupported");
    require(o.source == LatentSource::kDenoise && o.init_latents_path.empty() && o.prompt_embedding_path.empty(), "music requires native conditioning and fresh noise");
    require(!o.sampler && !o.sampling_effect && !o.lcs_data && !o.lcs.active() && o.lcs_mask.empty(), "music does not support H3 sampler overrides, sampling effects or LCS");
    require(o.upscale_model_path.empty() && o.latent_upscale_model_path.empty() && o.save_latents_path.empty() && !o.on_latents && !o.attention_band && o.veda.predictor_path.empty(), "music does not support video archives, upscaling or attention overrides");
    return;
  }
  if (p.model.family == "krea2") {
    require(o.inference_backend == DeviceBackend::kCuda || o.inference_backend == DeviceBackend::kVulkan, "Krea 2 requires CUDA or Vulkan");
    require(o.inference_backend != DeviceBackend::kVulkan || o.vulkan_portable_arithmetic,
            "Krea 2 Vulkan requires portable arithmetic; use --vulkan-arithmetic portable");
    require(o.source == LatentSource::kDenoise && o.init_latents_path.empty(), "Krea 2 requires generation from fresh noise");
    require(resolve_sampling_sampler(p, o.sampler) == sampler::SamplerKind::kEuler ||
                resolve_sampling_sampler(p, o.sampler) == sampler::SamplerKind::kErSde,
            "Krea 2 requires Euler or ER-SDE sampling");
    require(o.attention_mode == AttentionMode::kFlash2, "Krea 2 supports flash2 attention");
    require(!o.sampling_effect && !o.lcs_data && !o.lcs.active() && o.lcs_mask.empty(), "Krea 2 does not support H3 sampling effects or LCS");
    require(o.latent_upscale_model_path.empty() && o.upscale_model_path.empty() && o.save_latents_path.empty() && !o.on_latents && o.attention_band == 0 && o.veda.predictor_path.empty(), "Krea 2 does not support H3 latent archives, upscaling or attention overrides");
    return;
  }
  if (p.model.family == "fibo2") {
    require(o.inference_backend == DeviceBackend::kCuda || o.inference_backend == DeviceBackend::kVulkan, "Fibo requires CUDA or Vulkan");
    require(o.inference_backend != DeviceBackend::kVulkan || o.vulkan_portable_arithmetic,
            "Fibo Vulkan requires portable arithmetic; use --vulkan-arithmetic portable");
    require(o.source == LatentSource::kDenoise && o.init_latents_path.empty(), "Fibo requires editing from fresh noise");
    require(resolve_sampling_sampler(p, o.sampler) == sampler::SamplerKind::kEuler ||
                resolve_sampling_sampler(p, o.sampler) == sampler::SamplerKind::kErSde,
            "Fibo requires Euler or ER-SDE sampling");
    require(o.attention_mode == AttentionMode::kFlash2, "Fibo supports flash2 attention");
    require(!o.sampling_effect && !o.lcs_data && !o.lcs.active() && o.lcs_mask.empty(), "Fibo does not support H3 sampling effects or LCS");
    require(o.latent_upscale_model_path.empty() && o.upscale_model_path.empty() && o.save_latents_path.empty() && !o.on_latents && o.attention_band == 0 && o.veda.predictor_path.empty(), "Fibo does not support H3 latent archives, upscaling or attention overrides");
    return;
  }
  if (p.model.family == "moss") {
    require(!o.sampling_effect && !o.lcs_data, "sampling effects and LCS currently require H3");
    require(o.inference_backend == DeviceBackend::kCuda || o.inference_backend == DeviceBackend::kVulkan, "MOSS requires CUDA or Vulkan");
    require(o.inference_backend != DeviceBackend::kVulkan || o.vulkan_portable_arithmetic,
            "MOSS Vulkan requires portable arithmetic; use --vulkan-arithmetic portable");
    require(resolve_sampling_sampler(p, o.sampler) == sampler::SamplerKind::kEuler ||
                resolve_sampling_sampler(p, o.sampler) == sampler::SamplerKind::kErSde,
            "MOSS requires Euler or ER-SDE sampling");
    require(o.attention_mode == AttentionMode::kFlash2, "MOSS currently supports --attention flash2");
    require(o.latent_upscale_model_path.empty() && o.upscale_model_path.empty() &&
                o.save_latents_path.empty() && !o.on_latents && o.attention_band == 0,
            "MOSS does not support H3 latent archives, upscaling or attention bands");
    return;
  }
  validate_sampling_sampler(p, o.sampler);
  require(!o.sampling_effect || o.source == LatentSource::kDenoise,
          "sampling effects require denoising");
  o.lcs.validate();
  require(!o.lcs.active() || o.lcs_data, "LCS requires calibration data");
  if (o.lcs_data) {
    o.lcs_data->validate();
    require(o.source == LatentSource::kDenoise, "LCS requires denoising");
    require(o.lcs.sharpness == 0 || !o.lcs_data->sharpness.empty(),
            "LCS calibration has no sharpness direction");
    require(o.lcs.drift_strength == 0 || !o.lcs_data->drift_direction.empty(),
            "LCS calibration has no drift direction");
    require(o.lcs.color_stability == 0 || !o.lcs_data->color_reference.empty(),
            "LCS calibration has no color reference");
    require(o.lcs.edge_stability == 0 || !o.lcs_data->edge_reference.empty(),
            "LCS calibration has no edge reference");
  }
  const size_t lcs_plane = size_t(p.layout.latent_height / 2) * size_t(p.layout.latent_width / 2);
  if (o.lcs.color_stability != 0 || o.lcs.edge_stability != 0) {
    require(!r.bridge && !r.end_anchor_latents, "output stability does not support bridges or latent ending anchors");
    require(o.lcs_mask.empty() || o.lcs_mask.size() == lcs_plane, "output stability needs a spatial mask");
  }
  require(!o.lcs.tone_after_denoise ||
              (r.end_anchor_latents && (o.lcs_mask.empty() || o.lcs_mask.size() == lcs_plane)),
          "output tone requires a latent ending anchor and a spatial mask");
  require(o.lcs_mask.empty() ||
              (o.lcs_data && (o.lcs_mask.size() == lcs_plane ||
                              o.lcs_mask.size() == size_t(p.layout.num_video_rows))),
          "LCS mask must match one spatial patch plane or the full sampling volume");
  for (float x : o.lcs_mask)
    require(std::isfinite(x) && x >= 0 && x <= 1, "LCS mask values must be in [0,1]");
  validate_latent_upscale_options(o.latent_upscale);
  if (!o.latent_upscale_model_path.empty()) {
    require(!r.image_edit.image,
            "latent upscaling cannot preserve the exact pixels outside an image edit box");
    const auto dims =
        latent_upscale_dimensions(p.canvas_height / 16, p.canvas_width / 16, o.latent_upscale);
    if (!o.upscale_model_path.empty())
      upscale_output_elements(p.aligned_frames, dims.first * 16, dims.second * 16, o.upscale_method,
                              o.upscale);
  }
  validate_upscale_options(o.upscale, o.upscale_method);
  upscale_scale_factor(o.upscale_method); // reject unknown identifiers even when disabled
  if (!o.upscale_model_path.empty())
    upscale_output_elements(p.aligned_frames, p.canvas_height, p.canvas_width, o.upscale_method, o.upscale);
  require(o.upscale_model_path.empty() || !r.image_edit.image,
          "upscaling cannot preserve the exact pixels outside an image edit box");
  const auto sampler_kind = resolve_sampling_sampler(p, o.sampler);
  r.motion_cache.validate();
  require(std::isfinite(r.cache_threshold) && r.cache_threshold >= 0 && r.skip_every >= 0 &&
              r.block_cache_span >= 0,
          "cache settings must be finite and nonnegative");
  const bool step_cache = r.cache_threshold > 0 || r.skip_every > 0;
  const bool caches = step_cache || r.block_cache_span > 0 || r.motion_cache.active();
  require(r.context_noise.strength == 0 ||
              (!caches && o.source == LatentSource::kDenoise && o.init_latents_path.empty()),
          "context noise requires denoising from fresh noise without approximate caches");
  require(!r.keystone.source ||
              (!caches && o.source == LatentSource::kDenoise && o.init_latents_path.empty()),
          "keystone requires denoising from fresh noise without approximate caches");
  require(!(r.continuation && r.keystone.source && r.keystone.capstone) ||
              o.attention_mode != AttentionMode::kVeda,
          "call-only Capstone requires attention without a fixed VEDA predictor layout");
  require(!r.bridge ||
              (!caches && o.source == LatentSource::kDenoise && o.init_latents_path.empty()),
          "latent bridge requires denoising from fresh noise without approximate caches");
  require(!r.end_anchor_latents ||
              (!caches && o.source == LatentSource::kDenoise && o.init_latents_path.empty()),
          "latent ending anchor requires denoising without imported initial latents or approximate caches");
  require(!r.continuation_lock_overlap || (r.continuation && !caches),
          "locked overlap requires continuation without approximate caches");
  require(!sampler::stochastic_sampler(sampler_kind) || !caches,
          "stochastic samplers do not support approximate caches");
  r.image_edit.validate();
  require(
      !r.image_edit.image ||
          (r.still_image && o.source == LatentSource::kDenoise && o.init_latents_path.empty() &&
           (sampler_kind == sampler::SamplerKind::kEuler ||
            sampler_kind == sampler::SamplerKind::kRenoise) &&
           !caches),
      "image editing requires still-image Euler or re-noising denoising without initial latents or caches");
  require(!r.image_edit.image || o.on_samples || r.out_path.empty() ||
              std::filesystem::path(r.out_path).extension() == ".ppm",
          "image editing file output requires a .ppm path");
  require(r.cache_threshold == 0 || r.skip_every == 0,
          "cache threshold and skip interval are alternatives");
  require(o.attention_band >= 0, "attention band must be nonnegative");
  if (o.attention_mode == AttentionMode::kVeda) {
    require(!o.veda.predictor_path.empty(), "VEDA requires a predictor bundle (--veda-predictor)");
    require(!p.model.compressed_attention, "VEDA cannot be combined with FastH3 VSA");
    if (o.veda.generated_sparsity != "trained")
      (void)dit::parse_veda_budget(o.veda.generated_sparsity);
  } else {
    require(o.veda.predictor_path.empty(), "a VEDA predictor requires --attention veda");
  }
  require(o.attention_band == 0 || o.attention_mode == AttentionMode::kFlash2 ||
              o.attention_mode == AttentionMode::kExact ||
              (o.inference_backend == DeviceBackend::kVulkan &&
               o.attention_mode == AttentionMode::kSage2),
          "frame banding requires flash2 or exact attention, or Vulkan sage2");
  const auto& sol = o.sol_schedule;
  require(std::isfinite(sol.beta) && std::isfinite(sol.error_k) && std::isfinite(sol.error_v) &&
              sol.error_k >= 0 && sol.error_v >= 0 && sol.step_every > 0 && sol.layer_every > 0 &&
              sol.step_begin >= 0 && sol.layer_begin >= 0 && sol.step_end >= sol.step_begin &&
              sol.layer_end >= sol.layer_begin,
          "invalid Sol beta/range/cadence");
  require(!step_cache || r.block_cache_span == 0, "step and block caches cannot be combined");
  require(r.block_cache_span == 0 || r.block_cache_interval >= 2,
          "block cache interval must be at least 2");
  require(!step_cache || sampler_kind == sampler::SamplerKind::kEuler,
          "step caching requires Euler");
  require(!p.fixed_sampling_grid || !caches,
          "fixed sampling grids do not support approximate caches");
  require(p.video_timesteps.size() == p.audio_timesteps.size() || !caches,
          "independent audio steps do not support approximate caches");
  require(p.conditioning.allow_caches || !caches,
          "conditioning recipe does not support approximate caches");
  require(!p.conditioning.require_euler || sampler_kind == sampler::SamplerKind::kEuler,
          "conditioning recipe requires Euler");
  require(!p.conditioning.require_prompt_embedding || !o.prompt_embedding_path.empty(),
          "conditioning recipe requires a prompt embedding");
  require(
      !r.motion_cache.active() ||
          (sampler_kind == sampler::SamplerKind::kEuler && !step_cache && r.block_cache_span == 0 &&
           !p.model.compressed_attention && !p.conditioning.pin_target_audio &&
           !p.conditioning.video_first && r.schedule == sampler::ScheduleKind::kDefault &&
           o.source == LatentSource::kDenoise),
      "MotionCache requires default-schedule Euler denoising without other caches or pinned/reference-first conditioning");
  require(!p.model.compressed_attention ||
              (sampler_kind == sampler::SamplerKind::kEuler && o.attention_band == 0 && !caches),
          "compressed attention requires Euler without frame banding or caches");
  require(!r.continuation || (o.source == LatentSource::kDenoise && o.init_latents_path.empty()),
          "continuation requires denoising without imported initial latents");
  require(r.end_anchor_path.empty() || o.source == LatentSource::kDenoise,
          "ending anchor requires denoising");
  require(!r.video_transition ||
              (o.source == LatentSource::kDenoise && o.init_latents_path.empty()),
          "Extend/Bridge requires denoising from fresh noise");
  require(!p.conditioning.pin_target_audio || o.init_latents_path.empty(),
          "pinned target audio is incompatible with initial latents");
  require(o.source == LatentSource::kDenoise ||
              (!r.has_references() && !r.continuation && !r.bridge && r.loras.empty() &&
               !p.conditioning.require_prompt_embedding && !p.conditioning.pin_target_audio),
          "references, adapters and fixed conditioning require denoising");
  require(generation_backend_supported(o.inference_backend, o.source, o.attention_mode),
          "selected attention mode is unsupported by the inference backend");
  require(o.inference_backend != DeviceBackend::kVulkan || o.source != LatentSource::kDenoise ||
              ((sampler_kind == sampler::SamplerKind::kEuler ||
                sampler_kind == sampler::SamplerKind::kRenoise ||
                sampler_kind == sampler::SamplerKind::kErSde) &&
               !step_cache && r.block_cache_span == 0),
          "Vulkan generation supports Euler, re-noising or ER-SDE without step or block caches");
}
} // namespace slopfab
