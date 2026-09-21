#include "slopfab/generate.h"
#include <cmath>
#include <stdexcept>

namespace slopfab {
void validate_generation_options(const GenerateRequest& r, const GeneratePlan& p,
                                 const RunOptions& o) {
  const auto require = [](bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
  };
  validate_sampling_sampler(p, o.sampler);
  r.motion_cache.validate();
  require(std::isfinite(r.cache_threshold) && r.cache_threshold >= 0 &&
          r.skip_every >= 0 && r.block_cache_span >= 0, "cache settings must be finite and nonnegative");
  const bool step_cache = r.cache_threshold > 0 || r.skip_every > 0;
  const bool caches = step_cache || r.block_cache_span > 0 || r.motion_cache.active();
  require(r.cache_threshold == 0 || r.skip_every == 0, "cache threshold and skip interval are alternatives");
  require(o.attention_band >= 0, "attention band must be nonnegative");
  require(o.attention_band == 0 || o.attention_mode == AttentionMode::kFlash2 ||
          o.attention_mode == AttentionMode::kExact ||
          (o.inference_backend == DeviceBackend::kVulkan && o.attention_mode == AttentionMode::kSage2),
          "frame banding requires flash2 or exact attention, or Vulkan sage2");
  const auto& sol = o.sol_schedule;
  require(std::isfinite(sol.beta) && std::isfinite(sol.error_k) && std::isfinite(sol.error_v) &&
          sol.error_k >= 0 && sol.error_v >= 0 && sol.step_every > 0 && sol.layer_every > 0 &&
          sol.step_begin >= 0 && sol.layer_begin >= 0 && sol.step_end >= sol.step_begin &&
          sol.layer_end >= sol.layer_begin, "invalid Sol beta/range/cadence");
  require(!step_cache || r.block_cache_span == 0, "step and block caches cannot be combined");
  require(r.block_cache_span == 0 || r.block_cache_interval >= 2, "block cache interval must be at least 2");
  require(!step_cache || o.sampler == sampler::SamplerKind::kEuler, "step caching requires Euler");
  require(!p.fixed_sampling_grid || !caches, "fixed sampling grids do not support approximate caches");
  require(p.conditioning.allow_caches || !caches, "conditioning recipe does not support approximate caches");
  require(!p.conditioning.require_euler || o.sampler == sampler::SamplerKind::kEuler,
          "conditioning recipe requires Euler");
  require(!p.conditioning.require_prompt_embedding || !o.prompt_embedding_path.empty(),
          "conditioning recipe requires a prompt embedding");
  require(!r.motion_cache.active() || (o.sampler == sampler::SamplerKind::kEuler &&
          !step_cache && r.block_cache_span == 0 && !p.model.compressed_attention &&
          !p.conditioning.pin_target_audio && !p.conditioning.video_first &&
          r.schedule == sampler::ScheduleKind::kDefault && o.source == LatentSource::kDenoise),
          "MotionCache requires default-schedule Euler denoising without other caches or pinned/reference-first conditioning");
  require(!p.model.compressed_attention || (o.sampler == sampler::SamplerKind::kEuler &&
          o.attention_band == 0 && !caches), "compressed attention requires Euler without frame banding or caches");
  require(!r.continuation || (o.source == LatentSource::kDenoise && o.init_latents_path.empty()),
          "continuation requires denoising from fresh noise");
  require(!r.video_transition || (o.source == LatentSource::kDenoise && o.init_latents_path.empty()),
          "Extend/Bridge requires denoising from fresh noise");
  require(!p.conditioning.pin_target_audio || o.init_latents_path.empty(),
          "pinned target audio is incompatible with initial latents");
  require(o.source == LatentSource::kDenoise ||
          (!r.has_references() && !r.continuation && r.loras.empty() &&
           !p.conditioning.require_prompt_embedding && !p.conditioning.pin_target_audio),
          "references, adapters and fixed conditioning require denoising");
  require(generation_backend_supported(o.inference_backend, o.source, o.attention_mode),
          "selected attention mode is unsupported by the inference backend");
  require(o.inference_backend != DeviceBackend::kVulkan || o.source != LatentSource::kDenoise ||
          (o.sampler == sampler::SamplerKind::kEuler && !step_cache && r.block_cache_span == 0),
          "Vulkan generation supports Euler without step or block caches");
}
}  // namespace slopfab
