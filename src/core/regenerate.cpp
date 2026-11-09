#include "slopfab/regenerate.h"
#include "slopfab/pipeline.h"
#include "slopfab/model_profile.h"
#include "slopfab/remake.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace slopfab {
int regenerate_steps(const std::string& mode) {
  if (mode == "up3") return 3;
  if (mode == "up4") return 4;
  if (mode == "hf8") return 8;
  throw std::invalid_argument("Regenerate mode must be up3, up4 or hf8");
}

bool is_regenerate_lora(const SafeTensors& f) {
  const auto target = f.metadata().find("target");
  const auto step = f.metadata().find("training_step");
  const auto* a = f.find("transformer.token_refiner.refiner_blocks.0.attn.to_k.lora_A.weight");
  return target != f.metadata().end() && target->second == "MiniMaxAI/MiniMax-H3 transformer_ref (Ref2VA transformer)" &&
         step != f.metadata().end() && step->second == "6000" &&
         a && a->shape == std::vector<int64_t>{384, 5376} && f.tensors().size() == 624;
}

void validate_regenerate_request(const GenerateRequest& r) {
  if (r.regenerate.mode.empty()) {
    if (!r.regenerate.silence_path.empty() || !r.regenerate.upsampler_path.empty())
      throw std::invalid_argument("Regenerate companion options require --regenerate-2k");
    return;
  }
  const int steps = regenerate_steps(r.regenerate.mode);
  if (request_model_family(r) != "h3" || r.still_image || r.animate || !r.remake_mode.empty() ||
      r.continuation || r.bridge || r.end_anchor_latents || !r.end_anchor_path.empty() ||
      r.video_transition || r.image_edit.image || r.has_refmods() || r.keystone.source ||
      r.context_noise.strength > 0 || r.preserve_driving_audio)
    throw std::invalid_argument("Regenerate requires H3 Ref2VA video without other editing or continuation modes");
  if (r.num_frames != 124 || !r.reference_image_paths.empty() || r.reference_media.size() != 1 ||
      !r.reference_media.front() || !r.reference_media.front()->is_video())
    throw std::invalid_argument("Regenerate requires one draft video and 124 frames at 24 fps");
  r.reference_media.front()->validate();
  if (r.reference_media.front()->duration_seconds() + 1e-6 < 124.0 / 24)
    throw std::invalid_argument("Regenerate draft must cover 124 frames at 24 fps");
  const auto& frame = r.reference_media.front()->frames().front()->image;
  dit::validate_canvas_size(frame.height, frame.width);
  const int width = r.has_explicit_canvas() ? r.canvas_width : 2560;
  const int height = r.has_explicit_canvas() ? r.canvas_height : 1440;
  if (width < frame.width || height < frame.height || width > 2560 || height > 1440 ||
      double(width) / frame.width > 4 || double(height) / frame.height > 4)
    throw std::invalid_argument("Regenerate target must upscale the draft by at most 4x, up to 2560x1440");
  if ((r.num_inference_steps && r.num_inference_steps != steps) ||
      (r.sampling.default_steps && *r.sampling.default_steps != steps) ||
      (r.guidance_scale_set && r.guidance_scale != 1))
    throw std::invalid_argument("Regenerate requires CFG 1 and the selected mode's step count");
  if (!r.transformer_path.empty() && std::filesystem::exists(r.transformer_path)) {
    SafeTensors base;
    base.open(r.transformer_path);
    const auto model = dit::resolve_model_descriptor(base);
    if (model.family != "h3" ||
        model.compatibility_architecture != dit::TransformerArchitecture::kRef2VAPrunedTable ||
        !remake_checkpoint_mode(base).empty())
      throw std::invalid_argument("Regenerate requires the pruned H3 Ref2VA base");
  }
  bool found = false;
  for (const auto& spec : r.loras) {
    if (spec.strength == 0 || !std::filesystem::exists(spec.path)) continue;
    SafeTensors lora;
    lora.open(spec.path);
    if (lora.metadata().count("hyperflow") && spec.strength != 1)
      throw std::invalid_argument("Regenerate requires HyperFlow at strength 1");
    if (is_regenerate_lora(lora)) {
      if (found || spec.strength != 1)
        throw std::invalid_argument("Regenerate requires exactly one IC LoRA at strength 1");
      found = true;
    }
  }
  if (!found)
    throw std::invalid_argument("Regenerate requires open_h3_regenerate_2k_ic_lora.safetensors at strength 1");
}

std::string regenerate_prompt(const std::string& caption) {
  return "Enhance <Video 1> into a high-quality version of the same video. "
         "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
         "Restore fine details and remove degradation. Do not continue or change the scene."
         "\n\nScene description: " + caption;
}

std::vector<float> regenerate_silence(const std::string& path) {
  if (path.empty()) throw std::invalid_argument("Regenerate requires --regenerate-silence FILE");
  SafeTensors f;
  f.open(path);
  const auto& t = f.at("audio_latents");
  if (t.shape != std::vector<int64_t>{2, 32, 207})
    throw std::invalid_argument("Regenerate silence must contain audio_latents [2,32,207]");
  const auto data = to_f32(t);
  std::vector<float> rows(data.size());
  for (int ch = 0; ch < 2; ++ch)
    for (int a = 0; a < 207; ++a)
      for (int c = 0; c < 32; ++c) {
        const float v = data[(ch * 32 + c) * 207 + a];
        if (!std::isfinite(v)) throw std::invalid_argument("nonfinite Regenerate silence latent");
        rows[(ch * 207 + a) * 32 + c] = v;
      }
  return rows;
}
} // namespace slopfab
