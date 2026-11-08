#include "slopfab/remake.h"
#include "slopfab/pipeline.h"
#include "slopfab/model_profile.h"
#include <algorithm>
#include <cctype>
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
  if (r.remake_mode.empty())
    return;
  (void)remake_sampling(r.remake_mode);
  if (request_model_family(r) != "h3" || r.animate || r.still_image || r.continuation ||
      r.bridge || r.end_anchor_latents || !r.end_anchor_path.empty() || r.video_transition ||
      r.image_edit.image || r.has_refmods() || r.context_noise.strength > 0 || r.keystone.source)
    throw std::invalid_argument("Remake requires H3 video without other editing/continuation modes");
  if (r.num_frames < 90 || r.num_frames > 345 || (r.num_frames - 5) % 17)
    throw std::invalid_argument("Remake requires 90..345 frames in the form 17*n+5 at 24 fps");
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
