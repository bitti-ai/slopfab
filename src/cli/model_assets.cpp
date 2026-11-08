#include "slopfab/text/clip_projection.h"
// slopfab - MiniMax H3 video generation in C++/CUDA.

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Included directly rather than picked up from slopfab/generate.h, which is
// behind the CUDA guard below: the flag parsing that names an attention mode
// is not, so a build with SLOPFAB_ENABLE_CUDA=OFF could not see this type at
// all. It is a core header and costs a CPU-only build nothing.
#include "slopfab/attention_mode.h"
#include "slopfab/dtype.h"
#include "slopfab/json.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/dit/step_cache.h"
#include "slopfab/pipeline.h"
#include "slopfab/generate.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sampler/scheduler.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/tokenizer.h"
#include "reference_decode.h"

#include "slopfab/video/y4m.h"
#include "slopfab/video/y4m_compare.h"

#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/runtime.h"
#include "slopfab/vulkan/yuv_converter.h"
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#if SLOPFAB_WITH_CUDA
#include <chrono>

#include "slopfab/cuda/device.h"
#include "slopfab/cuda/cublas_dispatch.h"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/cuda/profile.h"
#include "slopfab/dit/transformer.h"
#include "slopfab/generate.h"
#include "slopfab/vae/vit_decoder.h"
#endif

#include "commands.h"
#include "agent.h"
#include "settings.h"
#include "slopfab/model_profile.h"

namespace slopfab::cli {
uint64_t random_seed() {
  std::random_device rd;
  return (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
}

std::string timestamped_output_path() {
  const auto settings = load_settings();
  const auto* output = settings.find("output_directory");
  return (std::filesystem::u8path(output ? output->as_string() : "output") / unique_id() /
          "video.y4m")
      .u8string();
}

std::string counted_output_path(const std::string& base, int index, int count) {
  if (count == 1)
    return base;
  const std::filesystem::path path(base);
  char suffix[24];
  std::snprintf(suffix, sizeof(suffix), "-%03d", index + 1);
  return (path.parent_path() / (path.stem().string() + suffix + path.extension().string()))
      .string();
}

namespace {
std::string find_weight(const std::vector<std::filesystem::path>& roots,
                        std::initializer_list<const char*> folders, std::string_view hint,
                        bool exact_name = false) {
  std::vector<std::filesystem::path> matches;
  auto match = [&](const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
      return;
    auto name = path.filename().u8string();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return char(std::tolower(c));
    });
    if (exact_name ? name != hint
                   : (path.extension() != ".safetensors" || name.find(hint) == std::string::npos))
      return;
    auto canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec)
      canonical = path;
    // The same file may be reachable through overlapping configured paths.
    for (const auto& old : matches) {
      if (std::filesystem::equivalent(old, canonical, ec) && !ec)
        return;
      ec.clear();
    }
    matches.push_back(std::move(canonical));
  };
  for (const auto& root : roots) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(root, ec)) {
      match(root);
      continue;
    }
    for (const auto* folder : folders) {
      const auto directory = *folder ? root / folder : root;
      if (exact_name) {
        match(directory / std::string(hint));
        continue;
      }
      ec.clear();
      for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end;
           it.increment(ec))
        match(it->path());
    }
  }
  std::sort(matches.begin(), matches.end());
  if (matches.size() > 1) {
    std::string detail;
    for (const auto& match : matches)
      detail += " " + match.u8string();
    throw CliError("MODEL_AMBIGUOUS",
                   "multiple matching checkpoints in weight_paths:" + detail +
                       "; configure a model alias or explicit paths",
                   3);
  }
  return matches.empty() ? std::string() : matches.front().u8string();
}
}

void discover_generate_checkpoints(slopfab::GenerateRequest& req, const char* executable,
                                   bool text_only) {
  (void)executable;
  const auto& profile = model_profile(request_model_family(req));
  if (profile.image_only)
    req.still_image = true;
  const auto weights = configured_weight_paths();
  if (weights.empty())
    return;
  if (std::string_view(profile.family) == "yue2") {
    if (req.transformer_path.empty())
      req.transformer_path = find_weight(weights, {"", "transformer"}, "yue2_3b");
    return;
  }
  if (std::string_view(profile.family) == "minimax_music3") {
    if (req.transformer_path.empty())
      req.transformer_path = find_weight(weights, {"", "transformer"}, profile.transformer_hint);
    if (req.text_encoder_path.empty())
      req.text_encoder_path = find_weight(weights, {"text_encoder", ""}, profile.conditioner_hint);
    if (req.audio_vae_path.empty())
      req.audio_vae_path = find_weight(weights, {"", "vae"}, profile.vae_hint);
    return;
  }
  if (req.text_encoder_path.empty()) {
    std::string hint = profile.conditioner_hint;
    if (!req.clip_projection_path.empty()) {
      text::ClipProjection projection(req.clip_projection_path);
      hint = projection.input_width() == 2560 ? "qwen3vl_4b" : "qwen3vl_8b";
    }
    req.text_encoder_path = find_weight(weights, {"text_encoder", "text_encoders", ""}, hint);
  }
  if (text_only)
    return;
  if (req.transformer_path.empty()) {
    const std::string architecture = !req.remake_mode.empty()
                                  ? "viggle_remake_" + req.remake_mode + "_pruned_"
                                  : (std::string_view(profile.family) == "h3" && req.has_references())
                                  ? "ref2va"
                                  : profile.transformer_hint;
    req.transformer_path = find_weight(weights, {"transformer", ""}, architecture);
  }
  if (req.video_vae_path.empty())
    req.video_vae_path = std::string_view(profile.family) == "krea2"
                             ? find_weight(weights, {"vae", "text_encoder", ""}, profile.vae_hint)
                             : find_weight(weights, {"vae", ""}, profile.vae_hint);
  if (std::string_view(profile.family) == "moss" && req.tokenizer_path.empty())
    req.tokenizer_path = find_weight(weights, {"text_encoder", ""}, "qwen3.5_tokenizer.json", true);
  if (!profile.image_only && req.audio_vae_path.empty())
    req.audio_vae_path = find_weight(weights, {"vae", ""}, "audio");
}

void ensure_generate_models(const slopfab::GenerateRequest& req, bool need_text_encoder) {
  if (request_model_family(req) == "minimax_music3") {
    if (req.transformer_path.empty() || req.text_encoder_path.empty() || req.audio_vae_path.empty())
      throw std::runtime_error(
          "MiniMax Music3 requires --transformer, --text-encoder and --audio-vae checkpoints");
    return;
  }
  if (request_model_family(req) == "yue2") {
    if (req.transformer_path.empty())
      throw std::runtime_error("missing YuE2 bundle: pass --transformer FILE");
    return;
  }
  const auto require_model = [](const std::string& path, const char* option, const char*) {
    if (path.empty())
      throw std::runtime_error(
          std::string(
              "missing model: set weight_paths with slopfab settings set weight_paths PATH, or pass ") +
          option + " <file>");
  };
  if (need_text_encoder)
    require_model(req.text_encoder_path, "--text-encoder", "text_encoder/");
  require_model(req.transformer_path, "--transformer", "transformer/");
  require_model(req.video_decode_path(), "--vae or --video-decoder", "vae/");
  if (!req.conditioning_image_paths().empty() || !req.reference_media.empty() ||
      req.image_edit.image)
    require_model(req.video_vae_path, "--vae (reference encoder)", "vae/");
  if (!req.still_image)
    require_model(req.audio_vae_path, "--audio-vae", "vae/");
}

int cmd_prepare_lora(int argc, char** argv) {
  std::string adapter;
  int width = 0;
  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg != "--adapter" && arg != "--width") || i + 1 == argc)
      throw std::invalid_argument("prepare-lora requires --adapter FILE --width N");
    const std::string value = argv[++i];
    if (arg == "--adapter")
      adapter = value;
    else {
      size_t used = 0;
      width = std::stoi(value, &used);
      if (used != value.size() || width <= 0)
        throw std::invalid_argument("prepare-lora: invalid width");
    }
  }
  if (adapter.empty() || width <= 0)
    throw std::invalid_argument("prepare-lora requires --adapter FILE --width N");
  slopfab::prepare_lora_grid(adapter, width);
  std::printf("prepared    %s\n", adapter.c_str());
  return 0;
}

}
