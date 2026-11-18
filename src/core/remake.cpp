#include "slopfab/remake.h"
#include "slopfab/pipeline.h"
#include "slopfab/model_profile.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace slopfab {
std::string remake_checkpoint_mode(const SafeTensors& checkpoint) {
  const auto& metadata = checkpoint.metadata();
  const auto source = metadata.find("source");
  if (source != metadata.end() && source->second != "Viggle/Viggle-Remake")
    return {};
  if (source != metadata.end()) {
    const auto mode = metadata.find("mode");
    if (mode != metadata.end() && (mode->second == "swap" || mode->second == "edit"))
      return mode->second;
  }
  std::string name = std::filesystem::path(checkpoint.path()).filename().string();
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  std::replace(name.begin(), name.end(), '-', '_');
  for (const char* mode : {"swap", "edit"})
    if (name.find(std::string("viggle_remake_") + mode + "_pruned_") != std::string::npos)
      return mode;
  return {};
}

SamplingSettings remake_sampling(const std::string& mode) {
  if (mode != "swap" && mode != "edit")
    throw std::invalid_argument("Remake mode must be swap or edit");
  const std::vector<float> video = mode == "swap"
      ? std::vector<float>{1, .990476f, .965243f, .923077f, .8f, .631579f, 0}
      : std::vector<float>{1, .983607f, .96f, .923077f, .857143f, .705882f, 0};
  SamplingSettings s;
  s.default_steps = 6;
  s.video_sigma_shift = 12;
  s.audio_sigma_shift = 3;
  s.sampler = sampler::SamplerKind::kEuler;
  s.base_sigmas = video;
  for (auto& sigma : *s.base_sigmas)
    sigma = sigma / (12 - 11 * sigma);
  return s;
}

void validate_remake_request(const GenerateRequest& r) {
  if (!std::isfinite(r.remake_keyframe_until) || r.remake_keyframe_until < 0 || r.remake_keyframe_until > 1)
    throw std::invalid_argument("Remake keyframe-until must be in [0,1]");
  if (r.remake_mode.empty()) {
    if (!r.remake_pins.empty() || r.remake_keyframe_until != 0.9f)
      throw std::invalid_argument("pin and keyframe-until require --remake");
    return;
  }
  (void)remake_sampling(r.remake_mode);
  if (request_model_family(r) != "h3" || r.animate || r.still_image || r.continuation ||
      r.bridge || r.end_anchor_latents || !r.end_anchor_path.empty() || r.video_transition ||
      r.image_edit.image || r.has_refmods() || r.context_noise.strength > 0 || r.keystone.source)
    throw std::invalid_argument("Remake requires H3 video without other editing/continuation modes");
  if (r.num_frames < 90 || r.num_frames > 345 || (r.num_frames - 5) % 17)
    throw std::invalid_argument("Remake requires 90..345 frames in the form 17*n+5 at 24 fps");
  if (!r.remake_pins.empty() && r.remake_pins.size() != r.reference_image_paths.size())
    throw std::invalid_argument("Remake requires one --pin per reference image, or no pins");
  for (int frame : r.remake_pins)
    if (frame < 0 || frame >= r.num_frames)
      throw std::invalid_argument("Remake pin must be a zero-based frame within the output clip");
  if (r.reference_image_paths.empty() || r.reference_image_paths.size() > 4 ||
      r.reference_media.size() != 1 || !r.reference_media.front() ||
      !r.reference_media.front()->is_video())
    throw std::invalid_argument("Remake requires one source video and one to four painted reference images");
  if (r.reference_media.front()->duration_seconds() + 1e-6 < double(r.num_frames) / 24)
    throw std::invalid_argument("Remake source video must cover the requested output duration");
  r.reference_media.front()->validate();
  if (r.prompt.empty())
    throw std::invalid_argument("Remake requires a six-section H3 prompt; the Animate frozen embedding is incompatible");
  if (!r.transformer_path.empty() && std::filesystem::exists(r.transformer_path)) {
    SafeTensors st;
    st.open(r.transformer_path);
    const auto model = dit::resolve_model_descriptor(st);
    const auto merged = remake_checkpoint_mode(st);
    if (model.family != "h3" || model.modulation != dit::ModulationImplementation::kTable ||
        model.compressed_attention || model.compatibility_architecture == dit::TransformerArchitecture::kViggleAnimatePrunedTable)
      throw std::invalid_argument("Remake needs its merged transformer or a pruned H3 FL2VA base plus Remake LoRA");
    if (!merged.empty()) {
      if (merged != r.remake_mode)
        throw std::invalid_argument("Remake mode does not match the merged checkpoint");
      if (std::any_of(r.loras.begin(), r.loras.end(), [](const auto& l) { return l.strength != 0; }))
        throw std::invalid_argument("Remake merged checkpoints already contain the mode LoRA; omit --lora");
    } else {
      if (model.supports_references)
        throw std::invalid_argument("Remake adapters require the FL2VA base, not Ref2VA");
      if (std::none_of(r.loras.begin(), r.loras.end(), [](const auto& l) { return l.strength != 0; }))
        throw std::invalid_argument("Remake on an unmerged FL2VA base requires the matching Remake LoRA");
    }
  }
}

RGBImage prepare_remake_keyframe(const RGBImage& image, int width, int height, bool first) {
  if (width <= 0 || height <= 0 || image.width <= 0 || image.height <= 0 ||
      image.pixels.size() != size_t(image.width) * image.height * 3)
    throw std::invalid_argument("invalid Remake keyframe image");
  if (first || (image.width == width && image.height == height))
    return resize_reference_lanczos(image, width, height);
  const double scale = std::max(double(width) / image.width, double(height) / image.height);
  const auto round_even = [](double v) {
    const double f = std::floor(v), d = v - f;
    return int(d < .5 ? f : d > .5 ? f + 1 : (std::fmod(f, 2.) == 0 ? f : f + 1));
  };
  const auto resized = resize_reference_lanczos(image, std::max(width, round_even(image.width * scale)),
                                               std::max(height, round_even(image.height * scale)));
  RGBImage out{width, height, std::vector<uint8_t>(size_t(width) * height * 3)};
  const int left = (resized.width - width) / 2, top = (resized.height - height) / 2;
  for (int y = 0; y < height; ++y)
    std::copy_n(resized.pixels.begin() + (size_t(y + top) * resized.width + left) * 3,
                size_t(width) * 3, out.pixels.begin() + size_t(y) * width * 3);
  return out;
}

std::vector<dit::ReferenceGeometry> remake_keyframe_geometry(
    const std::vector<int>& frames, const dit::SequenceLayout& layout) {
  std::vector<dit::ReferenceGeometry> result;
  for (int frame : frames) {
    dit::ReferenceGeometry g{dit::ReferenceKind::kImage, 1, layout.latent_height, layout.latent_width, 0};
    g.target_aligned = true;
    g.target_time_offset = (5.0 / 3.0) * frame;
    result.push_back(g);
  }
  return result;
}

std::vector<std::string> remake_prompt_segments(int frames, size_t images, const std::string& prompt) {
  if (frames <= 0 || frames > 345 || images == 0 || images > 4)
    throw std::invalid_argument("invalid Remake prompt geometry");
  std::vector<std::string> segments{"<Video 1>: "};
  const int sampled = (frames + 11) / 12;
  for (int pair = 0; pair < sampled; pair += 2) {
    const int scaled = (pair + std::min(pair + 1, sampled - 1)) * 5;
    const int tenths = scaled / 2 + ((scaled % 2) && ((scaled / 2) % 2));
    segments.push_back("<" + std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + " seconds>");
  }
  for (size_t i = 0; i < images; ++i)
    segments.push_back("<Picture " + std::to_string(i + 1) + ">: ");
  segments.push_back(prompt);
  return segments;
}
} // namespace slopfab
