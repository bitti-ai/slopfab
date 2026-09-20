#include "slopfab/conditioning_settings.h"
#include "slopfab/json.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>

namespace slopfab {
ConditioningSettings parse_conditioning_settings(std::string_view text) {
  const auto root = json::parse(text);
  if (!root.is_object() || !root.find("version") ||
      root.find("version")->as_number() != 1)
    throw std::invalid_argument("conditioning settings: version must be 1");
  ConditioningSettings out;
  for (const auto& [key, value] : root.as_object()) {
    if (key == "version") continue;
#define BOOL_FIELD(name) if (key == #name) { out.name = value.as_bool(); continue; }
    BOOL_FIELD(require_prompt_embedding)
    BOOL_FIELD(references_at_target_canvas)
    BOOL_FIELD(canvas_from_reference_video)
    BOOL_FIELD(include_reference_audio)
    BOOL_FIELD(video_first)
    BOOL_FIELD(pin_target_audio)
    BOOL_FIELD(require_euler)
    BOOL_FIELD(allow_caches)
#undef BOOL_FIELD
    const auto integer = [&] {
      const double n = value.as_number();
      if (!std::isfinite(n) || std::floor(n) != n || n < 0 || n > INT_MAX)
        throw std::invalid_argument("conditioning settings: invalid integer " + key);
      return static_cast<int>(n);
    };
#define INT_FIELD(name) if (key == #name) { out.name = integer(); continue; }
    INT_FIELD(fixed_prompt_tokens)
    INT_FIELD(image_short_edge)
    INT_FIELD(media_short_edge)
    INT_FIELD(media_max_pixels)
    INT_FIELD(max_frames)
#undef INT_FIELD
    throw std::invalid_argument("conditioning settings: unknown field " + key);
  }
  validate_conditioning_settings(out);
  return out;
}

void validate_conditioning_settings(const ConditioningSettings& s) {
  for (auto n : {s.image_short_edge, s.media_short_edge, s.media_max_pixels})
    if (n && *n <= 0) throw std::invalid_argument("conditioning dimensions must be positive");
  for (auto n : {s.fixed_prompt_tokens, s.max_frames})
    if (n && *n < 0) throw std::invalid_argument("conditioning counts must be nonnegative");
}

ReferenceConditionOptions ResolvedConditioning::reference_options(int width, int height) const {
  ReferenceConditionOptions out;
  out.short_edge = references_at_target_canvas ? std::min(width, height) : media_short_edge;
  out.max_pixels = references_at_target_canvas ? INT_MAX : media_max_pixels;
  out.include_audio = include_reference_audio;
  return out;
}

std::string ResolvedConditioning::cache_identity() const {
  std::string key = "conditioning-v1";
  for (int n : {int(require_prompt_embedding), fixed_prompt_tokens, image_short_edge,
      media_short_edge, media_max_pixels, int(references_at_target_canvas),
      int(canvas_from_reference_video), int(include_reference_audio), int(video_first),
      int(pin_target_audio), max_frames}) key += ":" + std::to_string(n);
  return key;
}
}  // namespace slopfab
