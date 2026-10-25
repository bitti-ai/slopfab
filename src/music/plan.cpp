#include "slopfab/music.h"
#include "slopfab/pipeline.h"
#include "slopfab/json.h"
#include "slopfab/model_profile.h"
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace slopfab {
void validate_music_settings(const MusicSettings& s) {
  if (s.planning != "full" && s.planning != "melody" && s.planning != "off")
    throw std::invalid_argument("music planning must be full, melody or off");
  if (!s.abc.empty() && s.planning == "off")
    throw std::invalid_argument("ABC input requires full or melody planning");
  if (!std::isfinite(s.duration) || s.duration <= 0 || s.duration > 300)
    throw std::invalid_argument("music duration must be in (0,300] seconds");
  if (!std::isfinite(s.temperature) || s.temperature < 0 || !std::isfinite(s.top_p) ||
      s.top_p <= 0 || s.top_p > 1 || s.top_k < 0 || s.top_k > music::codec_size ||
      !std::isfinite(s.repetition_penalty) || s.repetition_penalty < 1 ||
      !std::isfinite(s.semantic_guidance) ||
      (s.semantic_guidance != 0 && s.semantic_guidance < 1) || s.abc_max_tokens < 32 ||
      s.abc_max_tokens > 4096)
    throw std::invalid_argument("invalid music sampling settings");
}

MusicSettings parse_music_settings(const std::string& source) {
  MusicSettings s;
  const auto root = json::parse(source);
  if (!root.is_object())
    throw std::invalid_argument("music settings must be an object");
  for (const auto& [key, value] : root.as_object()) {
    if (key == "lyrics")
      s.lyrics = value.as_string();
    else if (key == "planning")
      s.planning = value.as_string();
    else if (key == "abc")
      s.abc = value.as_string();
    else if (key == "duration")
      s.duration = value.as_number();
    else if (key == "temperature")
      s.temperature = float(value.as_number());
    else if (key == "top_p")
      s.top_p = float(value.as_number());
    else if (key == "repetition_penalty")
      s.repetition_penalty = float(value.as_number());
    else if (key == "semantic_guidance")
      s.semantic_guidance = float(value.as_number());
    else if (key == "top_k" || key == "abc_max_tokens") {
      const double n = value.as_number();
      if (!std::isfinite(n) || n != std::floor(n) || n < 0 || n > 32768)
        throw std::invalid_argument("music integer setting out of range: " + key);
      (key == "top_k" ? s.top_k : s.abc_max_tokens) = int(n);
    } else
      throw std::invalid_argument("unknown music setting: " + key);
  }
  validate_music_settings(s);
  return s;
}

GeneratePlan resolve_music_plan(const GenerateRequest& r) {
  validate_music_settings(r.music);
  if (r.has_references() || r.continuation || r.bridge || r.image_edit.image || r.animate ||
      r.video_transition || r.preserve_driving_audio || !r.loras.empty() || r.still_image ||
      r.canvas_width || r.canvas_height || r.cache_threshold || r.skip_every || r.block_cache_span ||
      r.motion_cache.active() || !r.negative_prompt.empty() ||
      r.schedule != sampler::ScheduleKind::kDefault)
    throw std::invalid_argument(
        "music generation does not support video/image controls, references, adapters or caches");
  const auto& c = r.conditioning;
  if (c.require_prompt_embedding || c.fixed_prompt_tokens || c.image_short_edge ||
      c.media_short_edge || c.media_max_pixels || c.references_at_target_canvas ||
      c.canvas_from_reference_video || c.include_reference_audio || c.video_first ||
      c.pin_target_audio || c.max_frames || c.require_euler || c.allow_caches)
    throw std::invalid_argument("music does not use H3 conditioning settings");
  if (r.sampling.audio_steps || r.sampling.audio_sigma_shift || r.sampling.video_sigma_shift ||
      r.sampling.base_sigmas || r.sampling.sampler)
    throw std::invalid_argument(
        "music uses its own flow solver; H3 sampling settings are unsupported");
  GeneratePlan p;
  p.model.family = request_model_family(r);
  p.model.origin = "music checkpoint";
  if (!r.transformer_path.empty() && std::filesystem::exists(r.transformer_path)) {
    SafeTensors checkpoint;
    checkpoint.open(r.transformer_path);
    p.model = dit::resolve_model_descriptor(checkpoint);
    p.checkpoint_available = true;
  }
  p.duration_seconds = std::ceil(r.music.duration * 25) / 25;
  p.layout.num_audio_rows = int(std::ceil(r.music.duration * 25));
  p.num_inference_steps = r.num_inference_steps ? r.num_inference_steps : r.sampling.default_steps.value_or(32);
  if (p.num_inference_steps < 1 || p.num_inference_steps > 1000)
    throw std::invalid_argument("music flow steps must be in [1,1000]");
  p.geometry.audio_features = 64;
  p.geometry.audio_channels = 2;
  p.geometry.audio_latents_per_second = 25;
  p.geometry.audio_sample_rate = 48000;
  return p;
}

std::string music::yue_instruction(const std::string& mode) {
  if (mode == "off")
    return "Generate music with codec tokens from the given conditions.";
  if (mode == "melody")
    return "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.";
  return "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.";
}
}
