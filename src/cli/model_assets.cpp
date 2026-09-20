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
#include <winhttp.h>
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
struct ModelDownload {
  const char* subdirectory;
  const char* filename;
  const wchar_t* url;
};

constexpr ModelDownload kRef2VATransformer = {
    "transformer", "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    L"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/"
    L"minimax_h3_ref2va_pruned_int8_convrot.safetensors?download=true"};
constexpr ModelDownload kFL2VATransformer = {
    "transformer", "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    L"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/"
    L"minimax_h3_fl2va_pruned_int8_convrot.safetensors?download=true"};
constexpr ModelDownload kVideoVAE = {
    "vae", "minimax_h3_video_vae_fp16.safetensors",
    L"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/"
    L"minimax_h3_video_vae_fp16.safetensors?download=true"};
constexpr ModelDownload kAudioVAE = {
    "vae", "minimax_h3_audio_vae_fp32.safetensors",
    L"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/"
    L"minimax_h3_audio_vae_fp32.safetensors?download=true"};
constexpr ModelDownload kTextEncoder = {
    "text_encoder", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    L"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/"
    L"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors?download=true"};

#if defined(_WIN32)
struct InternetHandle {
  HINTERNET value = nullptr;
  ~InternetHandle() { if (value != nullptr) WinHttpCloseHandle(value); }
};

void download_model(const ModelDownload& model, const std::filesystem::path& destination) {
  URL_COMPONENTS parts{};
  parts.dwStructSize = sizeof(parts);
  parts.dwSchemeLength = static_cast<DWORD>(-1);
  parts.dwHostNameLength = static_cast<DWORD>(-1);
  parts.dwUrlPathLength = static_cast<DWORD>(-1);
  parts.dwExtraInfoLength = static_cast<DWORD>(-1);
  if (!WinHttpCrackUrl(model.url, 0, 0, &parts)) {
    throw std::runtime_error("download: invalid embedded model URL");
  }
  const std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
  std::wstring path(parts.lpszUrlPath, parts.dwUrlPathLength);
  path.append(parts.lpszExtraInfo, parts.dwExtraInfoLength);

  InternetHandle session{WinHttpOpen(L"slopfab/0.1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                     WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0)};
  if (!session.value) throw std::runtime_error("download: cannot initialise WinHTTP");
  WinHttpSetTimeouts(session.value, 30000, 30000, 30000, 60000);
  InternetHandle connection{WinHttpConnect(session.value, host.c_str(), parts.nPort, 0)};
  if (!connection.value) throw std::runtime_error("download: cannot connect to Hugging Face");
  InternetHandle request{WinHttpOpenRequest(connection.value, L"GET", path.c_str(), nullptr,
                                            WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                            parts.nScheme == INTERNET_SCHEME_HTTPS
                                                ? WINHTTP_FLAG_SECURE : 0)};
  if (!request.value || !WinHttpSendRequest(request.value, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
      !WinHttpReceiveResponse(request.value, nullptr)) {
    throw std::runtime_error("download: request failed for " + std::string(model.filename));
  }
  DWORD status = 0;
  DWORD status_size = sizeof(status);
  WinHttpQueryHeaders(request.value, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                      WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size,
                      WINHTTP_NO_HEADER_INDEX);
  if (status != 200) {
    throw std::runtime_error("download: Hugging Face returned HTTP " + std::to_string(status) +
                             " for " + model.filename);
  }

  uint64_t total = 0;
  wchar_t length[64]{};
  DWORD length_size = sizeof(length);
  if (WinHttpQueryHeaders(request.value, WINHTTP_QUERY_CONTENT_LENGTH,
                          WINHTTP_HEADER_NAME_BY_INDEX, length, &length_size,
                          WINHTTP_NO_HEADER_INDEX)) {
    total = std::wcstoull(length, nullptr, 10);
  }

  const std::filesystem::path partial = destination.string() + ".part";
  std::ofstream out(partial, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("download: cannot create " + partial.string());
  std::vector<char> buffer(1 << 20);
  uint64_t received = 0;
  int last_percent = -1;
  for (;;) {
    DWORD count = 0;
    if (!WinHttpReadData(request.value, buffer.data(), static_cast<DWORD>(buffer.size()), &count)) {
      out.close();
      std::filesystem::remove(partial);
      throw std::runtime_error("download: connection interrupted for " +
                               std::string(model.filename));
    }
    if (count == 0) break;
    out.write(buffer.data(), count);
    if (!out) {
      out.close();
      std::filesystem::remove(partial);
      throw std::runtime_error("download: write failed for " + partial.string());
    }
    received += count;
    const int percent = total == 0 ? -1 : static_cast<int>(received * 100 / total);
    if (percent != last_percent && (percent < 0 || percent % 2 == 0)) {
      if (percent >= 0) std::printf("\rdownload    %-55s %3d%%", model.filename, percent);
      else std::printf("\rdownload    %-55s %.2f GB", model.filename, received / 1e9);
      std::fflush(stdout);
      last_percent = percent;
    }
  }
  out.close();
  std::printf("\rdownload    %-55s done (%.2f GB)\n", model.filename, received / 1e9);
  if (received < 1024 * 1024 || (total != 0 && received != total)) {
    std::filesystem::remove(partial);
    throw std::runtime_error("download: incomplete file for " + std::string(model.filename));
  }
  std::filesystem::rename(partial, destination);
}
#endif

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
  if (count == 1) return base;
  const std::filesystem::path path(base);
  char suffix[24];
  std::snprintf(suffix, sizeof(suffix), "-%03d", index + 1);
  return (path.parent_path() / (path.stem().string() + suffix + path.extension().string())).string();
}

std::filesystem::path find_weights_directory(const char* executable) {
  std::vector<std::filesystem::path> starts = {std::filesystem::current_path()};
  std::error_code ec;
  const std::filesystem::path exe = std::filesystem::absolute(executable, ec);
  if (!ec) starts.push_back(exe.parent_path());

  for (std::filesystem::path start : starts) {
    for (int level = 0; level < 4 && !start.empty(); ++level) {
      const std::filesystem::path candidate = start / "weights";
      if (std::filesystem::is_directory(candidate, ec)) return candidate;
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
    if (!it->is_regular_file(ec) || it->path().extension() != ".safetensors") continue;
    std::string name = it->path().filename().string();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (required_name_part.empty() || name.find(required_name_part) != std::string::npos) {
      matches.push_back(it->path());
    }
  }
  std::sort(matches.begin(), matches.end());
  return matches.empty() ? std::string() : matches.front().string();
}

void discover_generate_checkpoints(slopfab::GenerateRequest& req, const char* executable) {
  const std::filesystem::path weights = find_weights_directory(executable);
  if (weights.empty()) return;
  if (req.text_encoder_path.empty())
    req.text_encoder_path = find_checkpoint(weights / "text_encoder", "");
  if (req.transformer_path.empty()) {
    const std::string_view architecture =
        req.has_references() ? "ref2va" : "fl2va";
    req.transformer_path = find_checkpoint(weights / "transformer", architecture);
  }
  if (req.video_vae_path.empty())
    req.video_vae_path = find_checkpoint(weights / "vae", "video");
  if (req.audio_vae_path.empty())
    req.audio_vae_path = find_checkpoint(weights / "vae", "audio");
}

std::filesystem::path default_weights_directory(const char* executable) {
  if (const std::filesystem::path found = find_weights_directory(executable); !found.empty()) {
    return found;
  }
  std::error_code ec;
  const std::filesystem::path exe = std::filesystem::absolute(executable, ec);
  return (ec ? std::filesystem::current_path() : exe.parent_path()) / "weights";
}

void ensure_model(std::string& path, const ModelDownload& model,
                  const std::filesystem::path& weights) {
  if (!path.empty()) return;
  const std::filesystem::path directory = weights / model.subdirectory;
  std::filesystem::create_directories(directory);
  const std::filesystem::path destination = directory / model.filename;
  if (!std::filesystem::is_regular_file(destination)) {
#if defined(_WIN32)
    std::printf("model       %s is missing; downloading from Hugging Face\n", model.filename);
    download_model(model, destination);
#else
    throw std::runtime_error("model is missing and automatic download is only available on Windows: " +
                             destination.string());
#endif
  }
  path = destination.string();
}

void ensure_generate_models(slopfab::GenerateRequest& req, const char* executable, bool need_text_encoder) {
  const std::filesystem::path weights = default_weights_directory(executable);
  if (need_text_encoder) ensure_model(req.text_encoder_path, kTextEncoder, weights);
  ensure_model(req.transformer_path,
               req.has_references() ? kRef2VATransformer : kFL2VATransformer,
               weights);
  ensure_model(req.video_vae_path, kVideoVAE, weights);
  ensure_model(req.audio_vae_path, kAudioVAE, weights);
}

int cmd_prepare_lora(int argc, char** argv) {
  std::string adapter;
  int width = 0;
  bool download = false;
  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--download") { download = true; continue; }
    if ((arg != "--adapter" && arg != "--width") || i + 1 == argc)
      throw std::invalid_argument("prepare-lora requires --adapter FILE --width N [--download]");
    const std::string value = argv[++i];
    if (arg == "--adapter") adapter = value;
    else {
      size_t used = 0;
      width = std::stoi(value, &used);
      if (used != value.size() || width <= 0) throw std::invalid_argument("prepare-lora: invalid width");
    }
  }
  if (adapter.empty() || width <= 0)
    throw std::invalid_argument("prepare-lora requires --adapter FILE --width N");
  slopfab::prepare_lora_grid(adapter, width, download);
  std::printf("prepared    %s\n", adapter.c_str());
  return 0;
}

}
