#include "slopfab/refmod.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

#include "slopfab/json.h"
#include "slopfab/reference_conditioning.h"
#include "slopfab/safetensors.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/sampler/noise.h"
#include "slopfab/sampler/scheduler.h"

namespace slopfab {
namespace {
void check_strength(float strength) {
  if (!std::isfinite(strength) || strength < 0 || strength > 1)
    throw std::invalid_argument("refmod strength must be finite and between 0 and 1");
}
void check_dimension(const json::Value& meta, const char* key, int actual) {
  if (const auto* value = meta.find(key)) {
    if (!value->is_number() || value->as_number() != actual)
      throw std::runtime_error(std::string("refmod metadata disagrees with latent shape: ") + key);
  }
}
// PyTorch adaptive_avg_pool2d followed by bilinear interpolate with
// align_corners=False. Video's time dimension stays intact; audio uses h=1.
void blur_plane(const float* input, float* output, int h, int w) {
  const int ph = std::max(1, h / 8), pw = std::max(1, w / 8);
  std::vector<float> pooled(static_cast<size_t>(ph) * pw);
  for (int y = 0; y < ph; ++y) for (int x = 0; x < pw; ++x) {
    const int y0 = static_cast<int>(int64_t(y) * h / ph);
    const int y1 = static_cast<int>((int64_t(y + 1) * h + ph - 1) / ph);
    const int x0 = static_cast<int>(int64_t(x) * w / pw);
    const int x1 = static_cast<int>((int64_t(x + 1) * w + pw - 1) / pw);
    float sum = 0;
    for (int yy = y0; yy < y1; ++yy) for (int xx = x0; xx < x1; ++xx)
      sum += input[static_cast<size_t>(yy) * w + xx];
    pooled[y * pw + x] = sum / ((y1 - y0) * (x1 - x0));
  }
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
    const float sy = std::max(0.0f, (y + .5f) * ph / h - .5f);
    const float sx = std::max(0.0f, (x + .5f) * pw / w - .5f);
    const int y0 = static_cast<int>(sy), x0 = static_cast<int>(sx);
    const int y1 = std::min(y0 + 1, ph - 1), x1 = std::min(x0 + 1, pw - 1);
    const float fy = sy - y0, fx = sx - x0;
    output[static_cast<size_t>(y) * w + x] =
        (1 - fy) * ((1 - fx) * pooled[y0 * pw + x0] + fx * pooled[y0 * pw + x1]) +
        fy * ((1 - fx) * pooled[y1 * pw + x0] + fx * pooled[y1 * pw + x1]);
  }
}
}  // namespace

std::shared_ptr<const RefMod> RefMod::load(const std::string& path) {
  SafeTensors file;
  file.open(path);
  json::Value meta;
  auto embedded = file.metadata().find("refmod_meta");
  if (embedded == file.metadata().end()) embedded = file.metadata().find("audio_refmod_meta");
  if (embedded != file.metadata().end()) {
    meta = json::parse(embedded->second);
  } else {
    auto sidecar = std::filesystem::u8path(path);
    sidecar.replace_extension(".json");
    std::ifstream in(sidecar, std::ios::binary);
    if (!in) throw std::runtime_error("refmod needs refmod_meta metadata or a .json sidecar: " + path);
    meta = json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
  }
  if (!meta.is_object()) throw std::runtime_error("refmod metadata must be an object");
  const std::string kind = meta.find("kind") ? meta.find("kind")->as_string() : "image";
  if (kind != "image" && kind != "video" && kind != "audio")
    throw std::runtime_error("refmod: unsupported kind '" + kind + "'; use a standalone image, video or audio refmod");
  const auto& tensor = file.at("latent");
  if (tensor.dtype != DType::kF32 && tensor.dtype != DType::kF16 && tensor.dtype != DType::kBF16)
    throw std::runtime_error("refmod latent must be F32, F16 or BF16");
  const auto& s = tensor.shape;
  // Keep all downstream packed-row and feature-count arithmetic in int range.
  int64_t elements = 1;
  for (int64_t d : s) {
    if (d <= 0 || d > std::numeric_limits<int>::max() / elements)
      throw std::runtime_error("refmod latent has invalid or excessive dimensions");
    elements *= d;
  }
  auto mod = std::shared_ptr<RefMod>(new RefMod);
  auto& g = mod->geometry_;
  if (kind == "audio") {
    if (s.size() != 4 || s[0] != 1 || s[1] != 32 || s[2] != 2)
      throw std::runtime_error("audio refmod latent must have shape [1,32,2,T]");
    g = {dit::ReferenceKind::kAudio, 0, 0, 0, static_cast<int>(s[3])};
    check_dimension(meta, "latent_t", g.num_audio_latents);
    check_dimension(meta, "sample_rate", 32000);
  } else {
    if (s.size() != 5 || s[0] != 1 || s[1] != 24 || s[3] % 2 || s[4] % 2 ||
        (kind == "image" && s[2] != 1))
      throw std::runtime_error("visual refmod latent must be [1,24,T,H,W], with even H/W and T=1 for images");
    g = {kind == "image" ? dit::ReferenceKind::kImage : dit::ReferenceKind::kVideo,
         static_cast<int>(s[2]), static_cast<int>(s[3]), static_cast<int>(s[4]), 0};
    check_dimension(meta, "latent_t", g.num_latent_frames);
    check_dimension(meta, "latent_h", g.latent_height);
    check_dimension(meta, "latent_w", g.latent_width);
  }
  mod->path_ = path;
  mod->name_ = meta.find("name") ? meta.find("name")->as_string() : std::filesystem::u8path(path).stem().u8string();
  mod->description_ = meta.find("description") ? meta.find("description")->as_string() : "";
  mod->dtype_ = tensor.dtype;
  mod->latent_ = to_f32(tensor);
  for (float value : mod->latent_) if (!std::isfinite(value))
    throw std::runtime_error("refmod latent contains NaN or infinity");
  return mod;
}

std::vector<float> RefMod::rows(float strength) const {
  check_strength(strength);
  if (strength == 0) return {};
  std::vector<float> z = latent_;
  const bool audio = geometry_.kind == dit::ReferenceKind::kAudio;
  if (strength < 1) {
    const int h = audio ? 1 : geometry_.latent_height;
    const int w = audio ? geometry_.num_audio_latents : geometry_.latent_width;
    const size_t plane = static_cast<size_t>(h) * w;
    std::vector<float> blurred(plane);
    const auto round = [&](float x) {
      return dtype_ == DType::kF16 ? f16_to_f32(f32_to_f16(x)) :
             dtype_ == DType::kBF16 ? bf16_to_f32(f32_to_bf16(x)) : x;
    };
    for (size_t offset = 0; offset < z.size(); offset += plane) {
      blur_plane(z.data() + offset, blurred.data(), h, w);
      for (size_t i = 0; i < plane; ++i)
        z[offset + i] = round(round(strength * z[offset + i]) +
                              round((1 - strength) * round(blurred[i])));
    }
  }
  if (!audio) return patchify_reference_video(z.data(), geometry_.num_latent_frames,
                                             geometry_.latent_height, geometry_.latent_width);
  const int t = geometry_.num_audio_latents;
  std::vector<float> packed(z.size());
  for (int ch = 0; ch < 2; ++ch) for (int i = 0; i < t; ++i) for (int c = 0; c < 32; ++c)
    packed[(static_cast<size_t>(ch) * t + i) * 32 + c] = z[(static_cast<size_t>(c) * 2 + ch) * t + i];
  return packed;
}

void validate_refmods(const std::vector<RefModReference>& refs) {
  int64_t tokens = 0;
  for (const auto& ref : refs) {
    if (!ref.mod) throw std::invalid_argument("refmod: null reference");
    check_strength(ref.strength);
    if (ref.copies < 1 || ref.copies > 10) throw std::invalid_argument("refmod copies must be between 1 and 10");
    if (ref.enabled()) tokens += static_cast<int64_t>(ref.mod->token_count()) * ref.copies;
    if (tokens > std::numeric_limits<int>::max() / 96)
      throw std::invalid_argument("refmods exceed the packed sequence size limit; reduce copies or references");
  }
}

void append_refmod_conditions(const std::vector<RefModReference>& refs,
                              uint64_t seed,
                              std::vector<dit::ReferenceGeometry>& geometry,
                              std::vector<float>& video_rows,
                              std::vector<float>& audio_rows) {
  validate_refmods(refs);
  for (const auto& ref : refs) {
    if (!ref.enabled()) continue;
    const auto& g = ref.mod->geometry();
    const auto clean = ref.mod->rows(ref.strength);
    for (int copy = 0; copy < ref.copies; ++copy) {
      auto rows = clean;
      if (g.kind != dit::ReferenceKind::kAudio) {
        const auto noise = sampler::video_noise(
            seed ^ (0x9e3779b97f4a7c15ULL * (geometry.size() + 1)),
            g.num_latent_frames, g.latent_height, g.latent_width);
        const auto packed = patchify_reference_video(noise.data(), g.num_latent_frames,
                                                     g.latent_height, g.latent_width);
        sampler::FlowScheduler::scale_noise(rows.data(), packed.data(), .999f, rows.size(), rows.data());
      }
      auto& destination = g.kind == dit::ReferenceKind::kAudio ? audio_rows : video_rows;
      destination.insert(destination.end(), rows.begin(), rows.end());
      geometry.push_back(g);
    }
  }
}

}  // namespace slopfab
