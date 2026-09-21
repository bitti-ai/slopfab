#include "slopfab/conditioning_settings.h"
#include "slopfab/pipeline.h"
#include "slopfab/safetensors.h"
#include <filesystem>
#include <stdexcept>

namespace slopfab {
namespace {
ConditioningSettings read_settings(const std::string& path) {
  if (path.empty() || !std::filesystem::exists(path)) return {};
  SafeTensors st;
  st.open(path);
  const auto it = st.metadata().find("slopfab.conditioning");
  if (it == st.metadata().end()) return {};
  try { return parse_conditioning_settings(it->second); }
  catch (const std::exception& e) {
    throw std::invalid_argument("conditioning metadata in '" + path + "': " + e.what());
  }
}
template <typename T>
void merge(std::optional<T>& dest, const std::optional<T>& value,
           const std::optional<T>& override_value, const char* field) {
  if (!value) return;
  if (dest && *dest != *value && !override_value)
    throw std::invalid_argument(std::string("conflicting LoRA conditioning defaults for ") + field);
  dest = value;
}
#define CONDITION_FIELDS(F) \
  F(require_prompt_embedding) F(fixed_prompt_tokens) F(image_short_edge) \
  F(media_short_edge) F(media_max_pixels) F(references_at_target_canvas) \
  F(canvas_from_reference_video) F(include_reference_audio) F(video_first) \
  F(pin_target_audio) F(max_frames) F(require_euler) F(allow_caches)
void overlay(ResolvedConditioning& out, const ConditioningSettings& s) {
  validate_conditioning_settings(s);
#define FIELD(name) if (s.name) out.name = *s.name;
  CONDITION_FIELDS(FIELD)
#undef FIELD
}
}

ResolvedConditioning resolve_conditioning_settings(const GenerateRequest& request) {
  ResolvedConditioning out;
  if (request.animate) {
    out.require_prompt_embedding = true;
    out.fixed_prompt_tokens = 362;
    out.references_at_target_canvas = true;
    out.canvas_from_reference_video = true;
    out.include_reference_audio = false;
    out.video_first = true;
    out.max_frames = 360;
    out.require_euler = true;
    out.allow_caches = false;
  }
  overlay(out, read_settings(request.transformer_path));
  ConditioningSettings adapters;
  for (const auto& lora : request.loras) {
    if (lora.strength == 0) continue;
    const auto s = read_settings(lora.path);
#define FIELD(name) merge(adapters.name, s.name, request.conditioning.name, #name);
    CONDITION_FIELDS(FIELD)
#undef FIELD
  }
  overlay(out, adapters);
  if (request.preserve_driving_audio) out.pin_target_audio = true;
  overlay(out, request.conditioning);
  return out;
}

void validate_conditioning_request(const GenerateRequest& r, const ResolvedConditioning& s) {
  if (r.video_transition) {
    if (r.video_transition < 1 || r.video_transition > 2 || r.still_image || r.continuation ||
        r.animate || s.video_first || s.pin_target_audio || r.has_refmods() ||
        !r.reference_image_paths.empty() || r.reference_media.size() != size_t(r.video_transition))
      throw std::invalid_argument("Extend/Bridge requires exactly one/two source videos without other conditioning modes");
    for (const auto& media : r.reference_media)
      if (!media || !media->is_video() || media->duration_seconds() + 1e-9 < 22.0 / 24)
        throw std::invalid_argument("Extend/Bridge source videos need at least 22 frames at 24 fps");
  }
  if ((s.video_first || s.pin_target_audio) && s.include_reference_audio)
    throw std::invalid_argument("video_first and pin_target_audio require include_reference_audio=false");
  if (s.fixed_prompt_tokens > 0 && !s.require_prompt_embedding)
    throw std::invalid_argument("fixed_prompt_tokens requires require_prompt_embedding");
  if (s.video_first) {
    if (r.still_image || r.continuation || r.has_refmods() ||
        r.reference_image_paths.size() != 1 || r.reference_media.size() != 1 ||
        !r.reference_media.front()->is_video())
      throw std::invalid_argument("video_first requires one driving video and one image without continuation or refmods");
    if (!s.require_prompt_embedding)
      throw std::invalid_argument("video_first currently requires fixed prompt conditioning");
  }
  if (s.canvas_from_reference_video &&
      (r.reference_media.empty() || !r.reference_media.front()->is_video()))
    throw std::invalid_argument("canvas_from_reference_video requires a leading video reference");
  if (s.pin_target_audio && (r.still_image || r.continuation ||
      r.reference_media.size() != 1 || !r.reference_media.front()->is_video() ||
      !r.reference_media.front()->soundtrack()))
    throw std::invalid_argument("pin_target_audio requires one video with a soundtrack, without continuation or still-image mode");
}
#undef CONDITION_FIELDS
}  // namespace slopfab
