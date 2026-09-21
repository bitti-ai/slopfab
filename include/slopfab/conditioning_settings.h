#pragma once

#include <optional>
#include <string>
#include <string_view>
#include "slopfab/reference_conditioning.h"

namespace slopfab {
struct GenerateRequest;

// Overrides are independent of model identity. Structural reference support
// is validated against the checkpoint separately.
struct ConditioningSettings {
  std::optional<bool> require_prompt_embedding;
  std::optional<int> fixed_prompt_tokens;
  std::optional<int> image_short_edge;
  std::optional<int> media_short_edge;
  std::optional<int> media_max_pixels;
  std::optional<bool> references_at_target_canvas;
  std::optional<bool> canvas_from_reference_video;
  std::optional<bool> include_reference_audio;
  std::optional<bool> video_first;
  std::optional<bool> pin_target_audio;
  std::optional<int> max_frames;
  std::optional<bool> require_euler;
  std::optional<bool> allow_caches;
};

struct ResolvedConditioning {
  bool require_prompt_embedding = false;
  int fixed_prompt_tokens = 0;
  int image_short_edge = 2048;
  int media_short_edge = 768;
  int media_max_pixels = 768 * 1344;
  bool references_at_target_canvas = false;
  bool canvas_from_reference_video = false;
  bool include_reference_audio = true;
  bool video_first = false;
  bool pin_target_audio = false;
  int max_frames = 0;
  bool require_euler = false;
  bool allow_caches = true;

  ReferenceConditionOptions reference_options(int width, int height) const;
  std::string cache_identity() const;
};

ConditioningSettings parse_conditioning_settings(std::string_view json);
void validate_conditioning_settings(const ConditioningSettings& settings);
ResolvedConditioning resolve_conditioning_settings(const GenerateRequest& request);
void validate_conditioning_request(const GenerateRequest& request,
                                   const ResolvedConditioning& settings);
} // namespace slopfab
