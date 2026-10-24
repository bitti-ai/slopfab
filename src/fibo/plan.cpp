#include "slopfab/fibo.h"
#include "slopfab/model_profile.h"
#include "slopfab/pipeline.h"
#include "slopfab/json.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace slopfab {
std::vector<float> fibo_sigma_schedule(int steps, int tokens) {
  if (steps < 1 || steps > 1000 || tokens < 1)
    throw std::invalid_argument("Fibo: invalid sampling dimensions");
  const double mu = .5 + (tokens - 256.) * (.65 / (4096 - 256));
  const double shift = std::exp(mu);
  std::vector<float> out;
  for (int i = 0; i < steps; ++i) {
    const double t = 1 - double(i) / steps;
    out.push_back(float(shift * t / (1 + (shift - 1) * t)));
  }
  out.push_back(0);
  return out;
}

GeneratePlan resolve_fibo_plan(const GenerateRequest& r) {
  r.image_edit.validate();
  if (r.reference_image_paths.size() + bool(r.image_edit.image) > 4)
    throw std::invalid_argument("Fibo 2.0 Edit Turbo supports at most four input images");
  if (r.continuation || r.bridge || r.animate || r.video_transition || !r.reference_media.empty() ||
      r.has_refmods() || r.image_edit.invert_mask || !r.loras.empty() || r.cache_threshold != 0 ||
      r.skip_every || r.block_cache_span || r.motion_cache.active() ||
      r.schedule != sampler::ScheduleKind::kDefault)
    throw std::invalid_argument(
        "Fibo editing does not support H3 video conditioning, outpaint, LoRAs or caches");
  if (r.image_edit.image && !r.reference_image_paths.empty())
    throw std::invalid_argument("Fibo masked editing requires exactly one source image");
  if (r.sampling.audio_steps || r.sampling.audio_sigma_shift || r.sampling.base_sigmas ||
      r.sampling.video_sigma_shift ||
      r.sampling.sampler.value_or(sampler::SamplerKind::kEuler) != sampler::SamplerKind::kEuler)
    throw std::invalid_argument(
        "Fibo uses image-size-shifted Euler sampling without H3 schedule overrides");
  GeneratePlan p;
  p.model.family = "fibo2";
  if (!r.transformer_path.empty() && std::filesystem::exists(r.transformer_path)) {
    SafeTensors st;
    st.open(r.transformer_path);
    p.model = dit::resolve_model_descriptor(st);
    p.checkpoint_available = true;
  }
  if (r.image_edit.image) {
    p.canvas_width = (r.image_edit.image->width + 15) / 16 * 16;
    p.canvas_height = (r.image_edit.image->height + 15) / 16 * 16;
  } else {
    if ((r.canvas_width == 0) != (r.canvas_height == 0))
      throw std::invalid_argument("Fibo needs both canvas dimensions");
    p.canvas_width = r.canvas_width ? r.canvas_width : 1024;
    p.canvas_height = r.canvas_height ? r.canvas_height : 1024;
  }
  if (p.canvas_width < 16 || p.canvas_height < 16 || p.canvas_width % 16 || p.canvas_height % 16 ||
      int64_t(p.canvas_width) * p.canvas_height > 4096LL * 4096)
    throw std::invalid_argument("Fibo canvas must be multiples of 16, at most 16 megapixels");
  const int64_t image_rows = int64_t(p.canvas_width / 16) * (p.canvas_height / 16);
  const int64_t sources = r.image_edit.image ? 1 : int64_t(r.reference_image_paths.size());
  if ((image_rows * (1 + sources) + 512) * 19456 > std::numeric_limits<int>::max())
    throw std::invalid_argument("Fibo canvas and reference count exceed the tensor indexing limit");
  p.aligned_frames = p.sampling_frames = 1;
  p.geometry.canvas_multiple = 16;
  p.geometry.video_channels = 128;
  p.geometry.patch_height = p.geometry.patch_width = 1;
  p.layout.num_latent_frames = 1;
  p.layout.latent_width = p.canvas_width / 16;
  p.layout.latent_height = p.canvas_height / 16;
  p.layout.num_video_rows = p.layout.latent_width * p.layout.latent_height;
  const int steps =
      r.num_inference_steps ? r.num_inference_steps : r.sampling.default_steps.value_or(4);
  p.video_sigmas = fibo_sigma_schedule(steps, p.layout.num_video_rows);
  if (r.image_edit.image) {
    int keep = std::max(1, int(std::ceil(steps * r.image_edit.strength)));
    p.video_sigmas.erase(p.video_sigmas.begin(), p.video_sigmas.begin() + steps - keep);
  }
  p.video_timesteps.assign(p.video_sigmas.begin(), p.video_sigmas.end() - 1);
  p.num_inference_steps = int(p.video_timesteps.size());
  p.sampler = sampler::SamplerKind::kEuler;
  return p;
}

namespace fibo {
int gist_count(int tokens) {
  if (tokens < 1)
    throw std::invalid_argument("Fibo prompt is empty");
  return std::clamp(
      int(std::ceil((256 + std::clamp((tokens - 256.) / 1792, 0., 1.) * 256) / 64)) * 64, 256, 512);
}

namespace {
json::Value clean(const json::Value& v) {
  if (v.is_array()) {
    json::Array a;
    for (const auto& x : v.as_array()) {
      auto c = clean(x);
      if (!c.is_null())
        a.push_back(c);
    }
    return a.empty() ? json::Value() : json::Value(a);
  }
  if (v.is_object()) {
    json::Object o;
    for (const auto& kv : v.as_object()) {
      if (kv.first == "aesthetic_score" || kv.first == "preference_score")
        continue;
      auto c = clean(kv.second);
      if (!c.is_null())
        o.emplace(kv.first, c);
    }
    return o.empty() ? json::Value() : json::Value(o);
  }
  if (v.is_string() && v.as_string().empty())
    return {};
  return v;
}
}

std::string edit_prompt(const std::string& prompt, int images, std::vector<std::string>& rendered) {
  if (images < 0 || images > 4 || prompt.empty())
    throw std::invalid_argument("Fibo needs a prompt and zero to four images");
  json::Value caption;
  std::string instruction = prompt;
  try {
    auto value = json::parse(prompt);
    if (value.is_object() && value.find("edit_instruction")) {
      instruction = value.find("edit_instruction")->as_string();
      auto o = value.as_object();
      o.erase("edit_instruction");
      caption = clean(value.find("structured_caption") ? *value.find("structured_caption")
                                                       : json::Value(o));
    }
  } catch (const std::exception&) { /* Plain instructions are supported. */
  }
  const std::string marker = "<|vision_start|><|image_pad|><|vision_end|>";
  if (caption.is_object()) {
    auto o = caption.as_object();
    const auto it = o.find("text_render");
    if (it != o.end() && it->second.is_array()) {
      auto entries = it->second.as_array();
      for (auto& entry : entries)
        if (entry.is_object() && entry.find("text") && entry.find("text")->is_string()) {
          const auto text = entry.find("text")->as_string();
          if (!text.empty()) {
            rendered.push_back(text);
            auto e = entry.as_object();
            e["image_tokens"] = json::Value(marker);
            entry = json::Value(e);
          }
        }
      o["text_render"] = json::Value(entries);
    }
    caption = json::Value(o);
  }
  // Image fields precede the caption/instruction, matching the vision token order.
  std::string body = "{";
  for (int i = 0; i < images; ++i) {
    if (i)
      body += ',';
    body += json::stringify(json::Value(images == 1 ? std::string("image")
                                                    : "image_" + std::to_string(i + 1))) +
            ':' + json::stringify(json::Value(marker));
  }
  if (!caption.is_null()) {
    if (images) body += ',';
    body += "\"structured_caption\":" + json::stringify(caption);
  }
  if (images || !caption.is_null()) body += ',';
  body += "\"edit_instruction\":" + json::stringify(json::Value(instruction)) + '}';
  return "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n";
}

RGBImage resize_vision_image(const RGBImage& image, int width, int height) {
  if (width < 1 || height < 1 || image.width < 1 || image.height < 1 ||
      image.pixels.size() != size_t(image.width) * image.height * 3)
    throw std::invalid_argument("Fibo vision resize dimensions");
  if (width == image.width && height == image.height)
    return image;
  // Antialiased bicubic, half-pixel centers, a=-0.5 (Qwen's PIL/PyTorch
  // processor). Boundary taps are truncated and the remaining weights sum to 1.
  using Taps = std::vector<std::pair<int, double>>;
  const auto axis = [](int input, int output) {
    std::vector<Taps> result(output);
    const double scale = double(input) / output, support = std::max(1., scale);
    for (int o = 0; o < output; ++o) {
      const double center = (o + .5) * scale - .5;
      double sum = 0;
      for (int i = std::max(0, int(std::ceil(center - 2 * support)));
           i <= std::min(input - 1, int(std::floor(center + 2 * support))); ++i) {
        const double x = std::abs((i - center) / support);
        const double weight =
            x < 1 ? ((1.5 * x - 2.5) * x) * x + 1 : ((-.5 * x + 2.5) * x - 4) * x + 2;
        result[o].emplace_back(i, weight);
        sum += weight;
      }
      for (auto& tap : result[o])
        tap.second /= sum;
    }
    return result;
  };
  const auto horizontal = axis(image.width, width), vertical = axis(image.height, height);
  std::vector<double> temp(size_t(image.height) * width * 3);
  for (int y = 0; y < image.height; ++y)
    for (int x = 0; x < width; ++x)
      for (const auto& tap : horizontal[x])
        for (int c = 0; c < 3; ++c)
          temp[(size_t(y) * width + x) * 3 + c] +=
              tap.second * image.pixels[(size_t(y) * image.width + tap.first) * 3 + c];
  RGBImage out{width, height, std::vector<uint8_t>(size_t(width) * height * 3)};
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) {
        double sum = 0;
        for (const auto& tap : vertical[y])
          sum += tap.second * temp[(size_t(tap.first) * width + x) * 3 + c];
        out.pixels[(size_t(y) * width + x) * 3 + c] =
            uint8_t(std::clamp(std::nearbyint(sum), 0., 255.));
      }
  return out;
}

RGBImage crop_image(const RGBImage& image, int width, int height) {
  const double scale = std::max(double(width) / image.width, double(height) / image.height);
  auto resized = resize_reference_lanczos(image, int(std::ceil(image.width * scale)),
                                          int(std::ceil(image.height * scale)));
  RGBImage out{width, height, std::vector<uint8_t>(size_t(width) * height * 3)};
  int left = (resized.width - width) / 2, top = (resized.height - height) / 2;
  for (int y = 0; y < height; ++y)
    std::copy_n(resized.pixels.data() + (size_t(y + top) * resized.width + left) * 3,
                size_t(width) * 3, out.pixels.data() + size_t(y) * width * 3);
  return out;
}
}
}
