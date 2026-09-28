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

namespace slopfab::cli {
uint64_t random_seed() {
  std::random_device rd;
  return (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
}

std::string timestamped_output_path() {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
#if defined(_WIN32)
  localtime_s(&local, &now);
#else
  localtime_r(&now, &local);
#endif
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
  return (std::filesystem::path("output") / (std::string("video-") + stamp + ".mp4")).string();
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

std::filesystem::path find_weights_directory(const char* executable) {
  std::vector<std::filesystem::path> starts = {std::filesystem::current_path()};
  std::error_code ec;
  const std::filesystem::path exe = std::filesystem::absolute(executable, ec);
  if (!ec)
    starts.push_back(exe.parent_path());

  for (std::filesystem::path start : starts) {
    for (int level = 0; level < 4 && !start.empty(); ++level) {
      const std::filesystem::path candidate = start / "weights";
      if (std::filesystem::is_directory(candidate, ec))
        return candidate;
      start = start.parent_path();
    }
  }
  return {};
}

std::string find_checkpoint(const std::filesystem::path& directory,
                            std::string_view required_name_part) {
  std::vector<std::filesystem::path> matches;
  std::error_code ec;
  for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end;
       it.increment(ec)) {
    if (!it->is_regular_file(ec) || it->path().extension() != ".safetensors")
      continue;
    std::string name = it->path().filename().string();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (required_name_part.empty() || name.find(required_name_part) != std::string::npos) {
      matches.push_back(it->path());
    }
  }
  std::sort(matches.begin(), matches.end());
  return matches.empty() ? std::string() : matches.front().string();
}

void discover_generate_checkpoints(slopfab::GenerateRequest& req, const char* executable) {
  const std::filesystem::path weights = find_weights_directory(executable);
  if (weights.empty())
    return;
  if (req.text_encoder_path.empty())
    req.text_encoder_path = find_checkpoint(weights / "text_encoder", "");
  if (req.transformer_path.empty()) {
    const std::string_view architecture = req.has_references() ? "ref2va" : "fl2va";
    req.transformer_path = find_checkpoint(weights / "transformer", architecture);
  }
  if (req.video_vae_path.empty())
    req.video_vae_path = find_checkpoint(weights / "vae", "video");
  if (req.audio_vae_path.empty())
    req.audio_vae_path = find_checkpoint(weights / "vae", "audio");
}

void ensure_generate_models(const slopfab::GenerateRequest& req, bool need_text_encoder) {
  const auto require_model = [](const std::string& path, const char* option,
                                const char* directory) {
    if (path.empty())
      throw std::runtime_error(std::string("missing model: place a compatible checkpoint in weights/") +
                               directory + " or pass " + option + " <file>");
  };
  if (need_text_encoder)
    require_model(req.text_encoder_path, "--text-encoder", "text_encoder/");
  require_model(req.transformer_path, "--transformer", "transformer/");
  require_model(req.video_vae_path, "--vae", "vae/");
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
