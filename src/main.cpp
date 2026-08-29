// vidfab - MiniMax H3 video generation in C++/CUDA.

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

// Included directly rather than picked up from vidfab/generate.h, which is
// behind the CUDA guard below: the flag parsing that names an attention mode
// is not, so a build with VIDFAB_ENABLE_CUDA=OFF could not see this type at
// all. It is a core header and costs a CPU-only build nothing.
#include "vidfab/attention_mode.h"
#include "vidfab/dtype.h"
#include "vidfab/json.h"
#include "vidfab/dit/step_cache.h"
#include "vidfab/pipeline.h"
#include "vidfab/safetensors.h"
#include "vidfab/safetensors_write.h"
#include "vidfab/sampler/scheduler.h"
#include "vidfab/tensor_convert.h"
#include "vidfab/text/tokenizer.h"

#include "vidfab/video/y4m.h"
#include "vidfab/video/y4m_compare.h"

#if VIDFAB_WITH_VULKAN
#include "vidfab/vulkan/runtime.h"
#include "vidfab/vulkan/yuv_converter.h"
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#endif

#if VIDFAB_WITH_CUDA
#include <chrono>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/deterministic_attention.cuh"
#include "vidfab/cuda/profile.h"
#include "vidfab/dit/transformer.h"
#include "vidfab/generate.h"
#include "vidfab/vae/vit_decoder.h"
#endif

namespace {

constexpr const char* kVersion = "0.1.0";
constexpr const char* kRegionalLicenseUrl = "https://platform.minimax.io/h3-license";

std::string_view embedded_license() {
#if defined(_WIN32)
  HRSRC resource = FindResourceW(nullptr, MAKEINTRESOURCEW(102), MAKEINTRESOURCEW(10));
  if (resource == nullptr) throw std::runtime_error("license: embedded resource is missing");
  HGLOBAL loaded = LoadResource(nullptr, resource);
  const DWORD size = SizeofResource(nullptr, resource);
  const void* bytes = loaded == nullptr ? nullptr : LockResource(loaded);
  if (bytes == nullptr || size == 0) throw std::runtime_error("license: embedded text is empty");
  return {static_cast<const char*>(bytes), size};
#else
  throw std::runtime_error("license: this build has no embedded MiniMax H3 license");
#endif
}

std::string license_hash(std::string_view text) {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char byte : text) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  char out[17]{};
  std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(hash));
  return out;
}

std::string environment_value(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) return {};
  std::string result(value);
  std::free(value);
  return result;
#else
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
#endif
}

std::filesystem::path license_state_path() {
#if defined(_WIN32)
  std::string base = environment_value("LOCALAPPDATA");
  if (base.empty()) base = environment_value("APPDATA");
  if (base.empty())
    throw std::runtime_error("license: LOCALAPPDATA is not available");
  return std::filesystem::path(base) / "Vidfab" / "state.json";
#elif defined(__APPLE__)
  const std::string home = environment_value("HOME");
  if (home.empty()) throw std::runtime_error("license: HOME is not available");
  return std::filesystem::path(home) / "Library" / "Application Support" / "Vidfab" /
         "state.json";
#else
  const std::string state = environment_value("XDG_STATE_HOME");
  if (!state.empty()) return std::filesystem::path(state) / "vidfab" / "state.json";
  const std::string home = environment_value("HOME");
  if (home.empty()) throw std::runtime_error("license: HOME is not available");
  return std::filesystem::path(home) / ".local" / "state" / "vidfab" / "state.json";
#endif
}

bool license_is_accepted(const std::filesystem::path& path, const std::string& hash) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream contents;
  contents << in.rdbuf();
  try {
    const vidfab::json::Value state = vidfab::json::parse(contents.str());
    const vidfab::json::Value* accepted = state.find("accepted");
    const vidfab::json::Value* stored_hash = state.find("license_hash");
    return accepted != nullptr && accepted->as_bool() && stored_hash != nullptr &&
           stored_hash->as_string() == hash;
  } catch (const std::exception&) {
    return false;
  }
}

bool ask_yes_no(const char* question) {
  for (;;) {
    std::printf("%s [yes/no]: ", question);
    std::fflush(stdout);
    char answer[64]{};
    if (std::fgets(answer, sizeof(answer), stdin) == nullptr) return false;
    std::string value(answer);
    value.erase(std::remove_if(value.begin(), value.end(),
                               [](unsigned char c) { return std::isspace(c) != 0; }), value.end());
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "yes" || value == "y") return true;
    if (value == "no" || value == "n") return false;
    std::printf("Please answer yes or no.\n");
  }
}

void store_license_acceptance(const std::filesystem::path& path, const std::string& hash,
                              const char* route) {
  std::filesystem::create_directories(path.parent_path());
  const std::filesystem::path temporary = path.string() + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("license: cannot write " + temporary.string());
    out << "{\n  \"accepted\": true,\n  \"license_hash\": \"" << hash
        << "\",\n  \"route\": \"" << route << "\"\n}\n";
    if (!out) throw std::runtime_error("license: cannot finish writing state");
  }
#if defined(_WIN32)
  if (!MoveFileExW(temporary.c_str(), path.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    std::filesystem::remove(temporary);
    throw std::runtime_error("license: cannot install state file " + path.string());
  }
#else
  std::filesystem::rename(temporary, path);
#endif
}

bool ensure_license_acceptance() {
  const std::string_view license = embedded_license();
  const std::string hash = license_hash(license);
  const std::filesystem::path state = license_state_path();
  if (license_is_accepted(state, hash)) return true;

  std::printf("MiniMax H3 license check\n\n");
  const bool regional = ask_yes_no(
      "Are you from the European Union, South Korea, United Kingdom, or United States?");
  if (regional) {
    std::printf("\nYou need to apply for a MiniMax H3 license at:\n%s\n\n", kRegionalLicenseUrl);
    if (!ask_yes_no("Have you applied for a license?")) {
      std::printf("A license must be applied for before Vidfab can be used.\n");
      return false;
    }
    store_license_acceptance(state, hash, "regional-application");
    return true;
  }

  std::printf("\n%s\n", std::string(license).c_str());
  if (!ask_yes_no("Do you accept the MiniMax H3 license?")) {
    std::printf("The license was declined. Vidfab will now terminate.\n");
    return false;
  }
  store_license_acceptance(state, hash, "community-license");
  return true;
}

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

  InternetHandle session{WinHttpOpen(L"vidfab/0.1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
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

void discover_generate_checkpoints(vidfab::GenerateRequest& req, const char* executable) {
  const std::filesystem::path weights = find_weights_directory(executable);
  if (weights.empty()) return;
  if (req.text_encoder_path.empty())
    req.text_encoder_path = find_checkpoint(weights / "text_encoder", "");
  if (req.transformer_path.empty()) {
    const std::string_view architecture =
        req.reference_image_paths.empty() ? "fl2va" : "ref2va";
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

void ensure_generate_models(vidfab::GenerateRequest& req, const char* executable) {
  const std::filesystem::path weights = default_weights_directory(executable);
  ensure_model(req.text_encoder_path, kTextEncoder, weights);
  ensure_model(req.transformer_path,
               req.reference_image_paths.empty() ? kFL2VATransformer : kRef2VATransformer,
               weights);
  ensure_model(req.video_vae_path, kVideoVAE, weights);
  ensure_model(req.audio_vae_path, kAudioVAE, weights);
}

// Per-command help. Keeping the detail beside the summary in one table is what
// stops `vidfab <cmd> --help` from drifting out of step with the top-level
// usage, which is the usual way CLI help rots.
struct CommandHelp {
  const char* name;
  const char* usage;
  const char* summary;
  const char* detail;
};

const CommandHelp kCommands[] = {
    {"generate", "vidfab generate --prompt <text> [options]", "text to video and audio",
     "  --prompt <text>              the prompt (MiniMax Context-IR structure)\n"
     "  --prompt-file <file>         read that same prompt from a UTF-8 text file\n"
     "                               instead; a BOM and surrounding blank space are\n"
     "                               stripped. Cannot be combined with --prompt\n"
     "  --reference-image <file>     ordered Ref2VA image; repeat up to 9 times.\n"
#if VIDFAB_WITH_FFMPEG
     "                               Any still or video FFmpeg can decode (a video\n"
     "                               contributes its first frame), plus binary PPM.\n"
#else
     "                               PNG, JPEG, BMP, TIFF, GIF and binary PPM on\n"
     "                               Windows; binary PPM elsewhere (no FFmpeg in this\n"
     "                               build).\n"
#endif
     "                               Requires Ref2VA transformer weights. Files are\n"
     "                               read only when the run starts\n"
     "  --out <file>                 output path (default output/video-<timestamp>.mp4)\n"
     "  --aspect <W:H>               display aspect, 1:4 to 4:1 (overrides 864x480)\n"
     "  --resolution <WxH>           exact canvas instead of an aspect; both axes a\n"
     "                               multiple of 32, ratio 1:4 to 4:1. Not capped to the\n"
     "                               trained 1344x768 area — larger is allowed, warned\n"
     "                               about, and costs attention time quadratically\n"
     "  --frames <n>                 snapped up to 17k+5 (default 124, minimum 6)\n"
     "  --steps <n>                  sigma grid points, n-1 evaluations (default 15)\n"
     "  --sampler euler|ab2          integrator (default euler)\n"
     "  --seed <n>                   noise seed; negative or absent draws a random one\n"
     "  --count <n>                  generate n videos; explicit seeds increment by one,\n"
     "                               random ones are drawn afresh for each\n"
     "  --raw                        write .y4m + .wav instead of muxing MP4\n"
     "  --inference-backend cuda|vulkan\n"
     "                               neural model backend (default cuda); Vulkan\n"
     "                               inference is not implemented and is rejected\n"
     "  --output-accelerator cpu|vulkan\n"
     "                               RGB-to-YUV output conversion only (default cpu);\n"
     "                               model inference remains CUDA\n"
     "  --dry-run                    resolve and print the plan, touch no weights\n"
     "  --synthetic-latents          skip conditioning and denoising and decode seeded\n"
     "                               noise, to exercise the VAEs and the muxer\n"
     "  --attn-band <frames>         frame-banded attention: a video row attends to\n"
     "                               +/- this many latent frames instead of the whole\n"
     "                               sequence. 0 (default) is off, and saves more the\n"
     "                               longer the request.\n"
     "                               Lossy. At +/-9 on the default geometry: ~1.4x\n"
     "                               faster per step; video latents land 1.6-2.0x the\n"
     "                               sampler noise floor away from unbanded, measured\n"
     "                               at 6, 24 and 40 steps with no trend in between.\n"
     "                               The audio cost is NOT characterised -- one seed\n"
     "                               per point leaves its noise floor moving as much\n"
     "                               as the effect. Judge output before relying on it\n"
     "  --attention <backend>        none, flash2, sage2 (default), sol,\n"
     "                               sol-experimental, or exact. Vulkan neural\n"
     "                               inference accepts only exact attention, but\n"
     "                               its full model orchestrator is not complete.\n"
     "                               The experimental SM120-only\n"
     "                               path is lossy and fails rather than falling back.\n"
     "  --sol-beta <f>               routing threshold multiplier (default 1)\n"
     "  --sol-error-k/v <f>           experimental K-residual/V-dispersion weights\n"
     "  --sol-step-start <n>         first active denoise step (default 10)\n"
     "  --sol-step-end <n>           last active denoise step (default max)\n"
     "  --sol-step-every <n>         activate every nth step in that range\n"
     "  --sol-layer-start <n>        first active main layer (default 2)\n"
     "  --sol-layer-end <n>          last active main layer (default max)\n"
     "  --sol-layer-every <n>        activate every nth layer in that range\n"
     "                               unfused reference implementation. Sage2 is lossy\n"
     "                               INT8/FP8 attention and requires a supported GPU\n"
     "  --dump-latents <f>           the denoiser's own output as fp32 safetensors,\n"
     "                               before either VAE; the diff point for a change\n"
     "                               to the transformer\n"
     "  --init-latents <f>           start the loop from these latents instead of the\n"
     "                               seeded draw; with --synthetic-latents, decode\n"
     "                               them straight to video and audio. Same shape as\n"
     "                               --dump-latents writes. Off by default.\n"
     "\n"
     "step caching (all off by default; each one trades quality for time):\n"
     "  --cache-threshold <x>        reuse the previous step's velocity until the\n"
     "                               conditioning has moved by <x>. 0 = off (default).\n"
     "                               Units: accumulated relative L1 of the rank-8 AdaLN\n"
     "                               code c(t) over the pair (video t, audio t) --\n"
     "                               dimensionless, and 1.0 means the conditioning has\n"
     "                               moved, summed over the skipped steps, by as much as\n"
     "                               its own magnitude. Useful values are far below 1.\n"
     "                               LOSSY AND UNCHARACTERISED: no quality comparison\n"
     "                               has been run at any threshold. Skips are strongly\n"
     "                               front-loaded on this checkpoint, which is where\n"
     "                               errors have the most steps left to compound.\n"
     "  --cache-warmup <n>           first n steps always evaluated (default 3, floor 2)\n"
     "  --skip-every <n>             instead of the threshold, evaluate every n-th step.\n"
     "                               0 = off (default). A calibration-free baseline the\n"
     "                               threshold has to beat; the two cannot be combined.\n"
     "\n"
     "block caching (off by default; independent of the step cache above):\n"
     "  --block-cache-span <n>       reuse the combined residual of n consecutive\n"
     "                               transformer blocks instead of evaluating them.\n"
     "                               0 = off (default). Saves (n/50) x (reused steps /\n"
     "                               total steps); measured within 0.1 point of that.\n"
     "                               Costs one buffer the size of the residual stream\n"
     "                               while enabled -- 844 MB at the 248-frame geometry,\n"
     "                               88 MB at 22 frames.\n"
     "                               Cannot be combined with --cache-threshold or\n"
     "                               --skip-every: a step those skip is a step this one\n"
     "                               never sees, so its interval stops counting.\n"
     "                               LOSSY AND UNCHARACTERISED, like the step cache: the\n"
     "                               reused residual is added unscaled to a state that\n"
     "                               has moved since it was captured.\n"
     "  --block-cache-start <n>      first block of the span. Default -1, which centres\n"
     "                               it: the first and last blocks carry the fastest-\n"
     "                               changing residuals and are the worst to reuse.\n"
     "  --block-cache-interval <n>   evaluate the span every n-th step, reuse it on the\n"
     "                               others (default 2 = alternate, minimum 2).\n"
     "  --block-cache-warmup <n>     first n steps always evaluate the span (default 3,\n"
     "                               floor 1)\n"
     "\n"
     "The first steps and the last step are always evaluated whatever these say.\n"
     "The trajectory is most sensitive early, and at the terminal step the sigma\n"
     "ratio is zero, so a reused velocity there lands in the output undamped.\n"
     "Every run that reused anything prints how many of its evaluations it\n"
     "skipped.\n"
     "\n"
     "checkpoints (omitted weights are found under weights/ or downloaded there):\n"
     "  --tokenizer <f>              override the embedded tokenizer.json\n"
     "  --text-encoder <f>           Qwen3-VL conditioner, int8 ConvRot or nvfp4 AWQ\n"
     "  --transformer <f>            H3 omni transformer, fp8, nvfp4 or NF4\n"
     "  --vae <f>                    video VAE decoder\n"
     "  --audio-vae <f>              audio VAE decoder\n"
     "\n"
     "  --bench-load <n>             load --transformer n times and exit, timing each.\n"
     "                               The first pays for reading the file, the rest do\n"
     "                               not; the gap is the cold-start cost.\n"
     "\n"
     "The quantisation of each checkpoint is read out of the file, so there is\n"
     "no flag for it and the two need not match.\n"
     "\n"
     "--sampler ab2 is Adams-Bashforth 2: second order at the same one forward\n"
     "pass per step, so a step costs what it always did and the reason to use it\n"
     "is to lower --steps. Its first step has no velocity history and is Euler.\n"
     "\n"
     "The conditioner and the transformer are loaded and freed in sequence\n"
     "rather than together: at 23.1 GB and 19.3 GB the int8 and fp8 pair cannot\n"
     "co-exist on a 32 GB card. The nvfp4 pair would (13.1 GB and 12.5 GB), but\n"
     "the sequencing costs nothing measurable and is what makes every mixture of\n"
     "the four safe. Expect the first output well after the progress line starts\n"
     "moving.\n"},
    {"inspect", "vidfab inspect <file.safetensors> [options]",
     "summarise a checkpoint's tensors",
     "  --list                       print every tensor, not just a summary\n"
     "  --prefix <str>               only tensors whose name starts with <str>\n"
     "  --limit <n>                  cap listed tensors (default 40, 0 = all)\n"},
    {"compare", "vidfab compare <reference> <actual> [options]",
     "diff two checkpoints tensor by tensor",
     "  --abs-tol <x>                absolute tolerance (default 1e-3)\n"
     "  --rel-tol <x>                relative tolerance (default 1e-2)\n"
     "  --verbose                    report passing tensors too\n"
     "\n"
     "Pass/fail is elementwise: a tensor passes on either tolerance. Beside\n"
     "that, two whole-tensor quality metrics are reported, because \"is any\n"
     "element wrong\" and \"is this the same tensor\" are different questions and\n"
     "the second is the one a quality comparison asks.\n"
     "\n"
     "  rel_L2       ||reference - actual|| / ||reference||\n"
     "               = sqrt(sum (r-a)^2) / sqrt(sum r^2)\n"
     "               Normalised by the REFERENCE -- not by actual, and not by\n"
     "               the mean of the two norms. So it is asymmetric on purpose:\n"
     "               compare(a,b) and compare(b,a) ask different questions.\n"
     "               Zero reference gives 0 if the difference is zero, else inf.\n"
     "\n"
     "  correlation  Pearson over the flattened tensor, MEANS SUBTRACTED:\n"
     "               S_ra / sqrt(S_rr * S_aa), S_xy = sum (x-mean_x)(y-mean_y).\n"
     "               Not cosine similarity. Identical tensors give exactly +1,\n"
     "               a tensor against its own negation exactly -1, and a\n"
     "               constant tensor (zero variance) gives 0, undefined.\n"
     "\n"
     "Both are computed over the flattened tensor across element pairs where\n"
     "both sides are finite. Matching global statistics with falling\n"
     "correlation is this project's signature of a different sample rather\n"
     "than a degraded one -- see the README.\n"},
    {"compare-y4m", "vidfab compare-y4m <expected.y4m> <actual.y4m>",
     "byte-compare deterministic raw video outputs",
     "Reports both headers, sizes, and the first differing byte. The command\n"
     "streams its inputs and returns non-zero for any difference.\n"},
    {"decode", "vidfab decode --vae <f> [--latent <f>] [options]",
     "run the video VAE decoder",
     "  --vae <f>                    video VAE checkpoint\n"
     "  --latent <f>                 latent safetensors; omit for a synthetic one\n"
     "  --shape <T> <H> <W>          synthetic latent shape\n"
     "  --out <f>                    .y4m output\n"
     "  --ppm <f>                    also write frame 0 as a PPM\n"
     "  --dump <f>                   raw fp32 pixels as safetensors\n"},
    {"tokenize", "vidfab tokenize [--tokenizer <f>] <text>",
     "encode text and round-trip it",
     "  --tokenizer <f>              override the embedded tokenizer.json\n"
     "  --pieces                     also print the pre-tokenizer split\n"},
    {"devices", "vidfab devices", "list CUDA inference and Vulkan output devices", ""},
    {"version", "vidfab version", "print the version and exit", ""},
};

const CommandHelp* find_command(std::string_view name) {
  for (const CommandHelp& c : kCommands) {
    if (name == c.name) return &c;
  }
  return nullptr;
}

// True when the argument list asks for help. Checked before any other option,
// so `--help` works even alongside otherwise invalid arguments.
bool wants_help(int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--help" || a == "-h" || a == "help") return true;
  }
  return false;
}

int print_command_help(const CommandHelp& c) {
  std::printf("usage: %s\n\n%s\n", c.usage, c.summary);
  if (c.detail[0] != '\0') std::printf("\noptions:\n%s", c.detail);
  return 0;
}

void print_usage() {
  std::printf(
      "vidfab %s - MiniMax H3 video generation\n"
      "\n"
      "usage: vidfab <command> [options]\n"
      "       vidfab <command> --help\n"
      "\n"
      "commands:\n",
      kVersion);
  for (const CommandHelp& c : kCommands) {
    std::printf("  %-9s %s\n", c.name, c.summary);
  }
  std::printf("\nRun `vidfab <command> --help` for that command's options.\n");
}

std::string format_shape(const std::vector<int64_t>& shape) {
  if (shape.empty()) return "scalar";
  std::string out;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out += "x";
    out += std::to_string(shape[i]);
  }
  return out;
}

std::string format_bytes(uint64_t n) {
  constexpr uint64_t kKiB = 1024;
  constexpr uint64_t kMiB = kKiB * 1024;
  constexpr uint64_t kGiB = kMiB * 1024;
  char buf[64];
  if (n >= kGiB) {
    std::snprintf(buf, sizeof(buf), "%.2f GiB", static_cast<double>(n) / kGiB);
  } else if (n >= kMiB) {
    std::snprintf(buf, sizeof(buf), "%.2f MiB", static_cast<double>(n) / kMiB);
  } else if (n >= kKiB) {
    std::snprintf(buf, sizeof(buf), "%.2f KiB", static_cast<double>(n) / kKiB);
  } else {
    std::snprintf(buf, sizeof(buf), "%" PRIu64 " B", n);
  }
  return buf;
}

int cmd_inspect(int argc, char** argv) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("inspect"));

  std::string path;
  std::string prefix;
  bool list = false;
  size_t limit = 40;

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--list") {
      list = true;
    } else if (arg == "--prefix" && i + 1 < argc) {
      prefix = argv[++i];
      list = true;
    } else if (arg == "--limit" && i + 1 < argc) {
      limit = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (!arg.empty() && arg.front() == '-') {
      std::fprintf(stderr, "vidfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    } else if (path.empty()) {
      path = argv[i];
    } else {
      std::fprintf(stderr, "vidfab: unexpected argument '%s'\n", argv[i]);
      return 2;
    }
  }

  if (path.empty()) {
    std::fprintf(stderr, "vidfab: inspect needs a .safetensors path\n");
    return 2;
  }

  vidfab::SafeTensors st;
  st.open(path);

  std::printf("file       %s\n", st.path().c_str());
  std::printf("size       %s\n", format_bytes(st.file_size()).c_str());
  std::printf("tensors    %zu\n", st.tensor_count());

  if (!st.metadata().empty()) {
    std::printf("metadata\n");
    for (const auto& [k, v] : st.metadata()) {
      // Metadata values can be long JSON blobs; keep the summary readable.
      std::string shown = v.size() > 120 ? v.substr(0, 117) + "..." : v;
      std::printf("  %-20s %s\n", k.c_str(), shown.c_str());
    }
  }

  // Breakdown by dtype tells us at a glance which quantisation scheme a file
  // uses, which is the first thing we need when wiring up a new checkpoint.
  std::map<std::string, std::pair<uint64_t, uint64_t>> by_dtype;  // count, bytes
  uint64_t total_bytes = 0;
  for (const auto& [name, view] : st.tensors()) {
    auto& slot = by_dtype[vidfab::dtype_name(view.dtype)];
    slot.first += 1;
    slot.second += view.nbytes;
    total_bytes += view.nbytes;
  }

  std::printf("\ndtype breakdown\n");
  for (const auto& [name, stats] : by_dtype) {
    std::printf("  %-10s %6" PRIu64 " tensors  %12s\n", name.c_str(), stats.first,
                format_bytes(stats.second).c_str());
  }
  std::printf("  %-10s %6zu tensors  %12s\n", "total", st.tensor_count(),
              format_bytes(total_bytes).c_str());

  if (list) {
    std::printf("\ntensors\n");
    size_t shown = 0;
    size_t matched = 0;
    for (const auto& [name, view] : st.tensors()) {
      if (!prefix.empty() && name.rfind(prefix, 0) != 0) continue;
      ++matched;
      if (limit != 0 && shown >= limit) continue;
      std::printf("  %-58s %-8s %-20s %10s\n", name.c_str(), vidfab::dtype_name(view.dtype),
                  format_shape(view.shape).c_str(), format_bytes(view.nbytes).c_str());
      ++shown;
    }
    if (matched > shown) {
      std::printf("  ... %zu more (raise --limit to see them)\n", matched - shown);
    }
    if (matched == 0) {
      std::printf("  no tensors matched prefix '%s'\n", prefix.c_str());
    }
  }

  return 0;
}

// Compares every tensor shared by two checkpoints. This is how a ported stage
// is validated: dump reference activations from the Python implementation, run
// ours, and diff. Exit code is non-zero when any tensor exceeds tolerance, so
// it can be used directly as a test.
int cmd_compare(int argc, char** argv) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("compare"));

  std::string ref_path;
  std::string act_path;
  double abs_tol = 1e-3;
  double rel_tol = 1e-2;
  bool verbose = false;

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--abs-tol" && i + 1 < argc) {
      abs_tol = std::strtod(argv[++i], nullptr);
    } else if (arg == "--rel-tol" && i + 1 < argc) {
      rel_tol = std::strtod(argv[++i], nullptr);
    } else if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (!arg.empty() && arg.front() == '-') {
      std::fprintf(stderr, "vidfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    } else if (ref_path.empty()) {
      ref_path = argv[i];
    } else if (act_path.empty()) {
      act_path = argv[i];
    } else {
      std::fprintf(stderr, "vidfab: unexpected argument '%s'\n", argv[i]);
      return 2;
    }
  }

  if (ref_path.empty() || act_path.empty()) {
    std::fprintf(stderr, "vidfab: compare needs a reference and an actual .safetensors path\n");
    return 2;
  }

  vidfab::SafeTensors ref;
  vidfab::SafeTensors act;
  ref.open(ref_path);
  act.open(act_path);

  std::printf("reference  %s (%zu tensors)\n", ref.path().c_str(), ref.tensor_count());
  std::printf("actual     %s (%zu tensors)\n", act.path().c_str(), act.tensor_count());
  std::printf("tolerance  abs %g, rel %g  (a tensor passes on either)\n\n", abs_tol, rel_tol);

  size_t compared = 0;
  size_t failed = 0;
  size_t missing = 0;
  double worst_abs = 0.0;
  std::string worst_name;

  std::vector<float> ref_values;
  std::vector<float> act_values;

  for (const auto& [name, ref_view] : ref.tensors()) {
    const vidfab::TensorView* act_view = act.find(name);
    if (act_view == nullptr) {
      ++missing;
      if (verbose) std::printf("  MISSING  %s\n", name.c_str());
      continue;
    }

    vidfab::to_f32(ref_view, ref_values);
    vidfab::to_f32(*act_view, act_values);
    const vidfab::CompareStats stats = vidfab::compare(ref_values, act_values);
    ++compared;

    if (stats.max_abs_err > worst_abs) {
      worst_abs = stats.max_abs_err;
      worst_name = name;
    }

    const bool ok = stats.passes(abs_tol, rel_tol);
    if (!ok) ++failed;

    if (!ok || verbose) {
      std::printf("  %-7s %-52s max_abs %.3e  max_rel %.3e  rms %.3e\n", ok ? "ok" : "FAIL",
                  name.c_str(), stats.max_abs_err, stats.max_rel_err, stats.rms_err);
      // On a continuation line rather than widened into the one above: that
      // line is already 120 columns and anything reading it would break.
      // Suppressed for a shape mismatch, where neither metric was computed and
      // printing 0.000e+00 / +0.0000 would read as agreement.
      if (stats.shape_match) {
        std::printf("          rel_L2 %.4e  correlation %+.4f\n", stats.rel_l2,
                    stats.correlation);
      }
      if (!stats.shape_match) {
        std::printf("           shape/element-count mismatch: %lld vs %lld\n",
                    static_cast<long long>(ref_view.numel()),
                    static_cast<long long>(act_view->numel()));
      } else if (!ok && stats.argmax_abs >= 0) {
        std::printf("           worst at index %lld: reference %.6g, actual %.6g\n",
                    static_cast<long long>(stats.argmax_abs), stats.lhs_at_argmax,
                    stats.rhs_at_argmax);
      }
      if (stats.nan_mismatches != 0) {
        std::printf("           %lld non-finite mismatches\n",
                    static_cast<long long>(stats.nan_mismatches));
      }
    }
  }

  std::printf("\n%zu compared, %zu failed, %zu missing from actual\n", compared, failed, missing);
  if (!worst_name.empty()) {
    std::printf("worst absolute error %.3e in %s\n", worst_abs, worst_name.c_str());
  }
  return (failed == 0 && missing == 0) ? 0 : 1;
}

int cmd_compare_y4m(int argc, char** argv) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("compare-y4m"));
  if (argc != 2) {
    std::fprintf(stderr, "vidfab: compare-y4m needs expected and actual paths\n");
    return 2;
  }
  const vidfab::video::ExactY4mComparison result =
      vidfab::video::compare_y4m_exact(argv[0], argv[1]);
  std::printf("expected   %s (%llu bytes)\n", argv[0],
              static_cast<unsigned long long>(result.expected_size));
  std::printf("           %s\n", result.expected_header.c_str());
  std::printf("actual     %s (%llu bytes)\n", argv[1],
              static_cast<unsigned long long>(result.actual_size));
  std::printf("           %s\n", result.actual_header.c_str());
  if (result.equal()) {
    std::printf("exact      yes (all bytes identical)\n");
    return 0;
  }
  std::printf("exact      no; first difference at byte %llu: expected ",
              static_cast<unsigned long long>(result.first_difference));
  if (result.expected_byte < 0)
    std::printf("<EOF>");
  else
    std::printf("0x%02x", result.expected_byte);
  std::printf(", actual ");
  if (result.actual_byte < 0)
    std::printf("<EOF>\n");
  else
    std::printf("0x%02x\n", result.actual_byte);
  return 1;
}

#if VIDFAB_WITH_CUDA
// Reads latents_mean / latents_std from the checkpoint's __metadata__ JSON,
// which carries full fp32 text. The F16 tensors of the same name lose
// precision, and the FL2VA copy of config.json has corrupted digits.
bool latent_stats_from_metadata(const vidfab::SafeTensors& ckpt, std::vector<float>& mean,
                                std::vector<float>& std_dev) {
  auto it = ckpt.metadata().find("minimax_h3_video_vae");
  if (it == ckpt.metadata().end()) return false;
  try {
    const vidfab::json::Value meta = vidfab::json::parse(it->second);
    const vidfab::json::Value* m = meta.find("latents_mean");
    const vidfab::json::Value* s = meta.find("latents_std");
    if (m == nullptr || s == nullptr) return false;
    mean.clear();
    std_dev.clear();
    for (const auto& v : m->as_array()) mean.push_back(static_cast<float>(v.as_number()));
    for (const auto& v : s->as_array()) std_dev.push_back(static_cast<float>(v.as_number()));
    return mean.size() == 24 && std_dev.size() == 24;
  } catch (const std::exception&) {
    return false;
  }
}

// Deterministic pseudo-random latent, so a run without a real latent file
// still exercises the whole path reproducibly.
std::vector<float> synthetic_latent(int T, int H, int W, uint32_t seed) {
  std::vector<float> z(static_cast<size_t>(24) * T * H * W);
  uint32_t state = seed != 0 ? seed : 1u;
  for (float& v : z) {
    // xorshift32, then map to roughly standard normal via Box-Muller-free
    // approximation: sum of uniforms is adequate for a smoke test.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    const float u = static_cast<float>(state & 0xFFFFFFu) / static_cast<float>(0x1000000u);
    v = (u - 0.5f) * 2.0f;
  }
  return z;
}

int cmd_decode(int argc, char** argv) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("decode"));

  std::string vae_path;
  std::string latent_path;
  std::string out_path = "out.y4m";
  std::string ppm_path;
  std::string dump_path;
  int T = 7;
  int H = 16;
  int W = 16;
  uint32_t seed = 1234;
  bool no_tiling = false;
  bool bench_load = false;
  int fps = 24;
  int repeat = 1;

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--vae" && i + 1 < argc) {
      vae_path = argv[++i];
    } else if (arg == "--latent" && i + 1 < argc) {
      latent_path = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--ppm" && i + 1 < argc) {
      ppm_path = argv[++i];
    } else if (arg == "--dump" && i + 1 < argc) {
      dump_path = argv[++i];
    } else if (arg == "--shape" && i + 3 < argc) {
      T = std::atoi(argv[++i]);
      H = std::atoi(argv[++i]);
      W = std::atoi(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--fps" && i + 1 < argc) {
      fps = std::atoi(argv[++i]);
    } else if (arg == "--repeat" && i + 1 < argc) {
      repeat = std::max(1, std::atoi(argv[++i]));
    } else if (arg == "--no-tiling") {
      no_tiling = true;
    } else if (arg == "--bench-load") {
      bench_load = true;
    } else {
      std::fprintf(stderr, "vidfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    }
  }

  if (vae_path.empty()) {
    std::fprintf(stderr, "vidfab: decode needs --vae <video_vae.safetensors>\n");
    return 2;
  }

  vidfab::SafeTensors ckpt;
  ckpt.open(vae_path);

  std::vector<float> mean;
  std::vector<float> std_dev;
  if (!latent_stats_from_metadata(ckpt, mean, std_dev)) {
    mean = vidfab::vae::default_video_latents_mean();
    std_dev = vidfab::vae::default_video_latents_std();
  }

  std::vector<float> z;
  if (!latent_path.empty()) {
    vidfab::SafeTensors latent_file;
    latent_file.open(latent_path);
    const vidfab::TensorView& lv = latent_file.at("latent");
    if (lv.shape.size() != 4 || lv.shape[0] != 24) {
      std::fprintf(stderr, "vidfab: latent tensor must be [24, T, H, W]\n");
      return 1;
    }
    T = static_cast<int>(lv.shape[1]);
    H = static_cast<int>(lv.shape[2]);
    W = static_cast<int>(lv.shape[3]);
    z = vidfab::to_f32(lv);
  } else {
    std::printf("no --latent given; decoding a deterministic synthetic latent (seed %u)\n", seed);
    z = synthetic_latent(T, H, W, seed);
  }

  std::printf("latent     [24, %d, %d, %d] -> %d x %d px\n", T, H, W, W * 16, H * 16);

  // Loading twice in one process distinguishes two very different causes of a
  // slow load: if the second is much faster, the first was paying page faults
  // to pull the mapping in from storage; if they match, the cost is the host
  // memcpy and PCIe, and only then is parallelising or double-buffering it
  // worth building.
  if (bench_load) {
    auto time_load = [&](const char* label) {
      vidfab::vae::ViTDecoder probe;
      const auto s0 = std::chrono::steady_clock::now();
      probe.load(ckpt);
      const auto s1 = std::chrono::steady_clock::now();
      const double sec = std::chrono::duration<double>(s1 - s0).count();
      std::printf("load %-8s %s in %.3f s (%.2f GB/s of fp16 across PCIe)\n", label,
                  format_bytes(probe.weight_bytes()).c_str(), sec,
                  (static_cast<double>(probe.weight_bytes()) / 2.0) / sec / 1e9);
    };
    time_load("first");
    time_load("second");
    time_load("third");
    return 0;
  }

  vidfab::vae::ViTDecoder decoder;
  const auto load_start = std::chrono::steady_clock::now();
  decoder.load(ckpt);
  const auto load_end = std::chrono::steady_clock::now();
  std::printf("weights    %s on device in %.2f s\n",
              format_bytes(decoder.weight_bytes()).c_str(),
              std::chrono::duration<double>(load_end - load_start).count());

  vidfab::vae::DecodeSchedule schedule;
  schedule.tiling_enabled = !no_tiling;

  // The first decode pays one-time costs the steady state does not: scratch
  // allocation, the first RoPE build, and cuBLAS heuristic selection for each
  // GEMM shape. Reporting it as the decode time overstates the cost by a
  // noticeable margin, so timings are reported per run and `--repeat` exists to
  // expose the warm number.
  vidfab::vae::DecodedVideo video;
  for (int run = 0; run < repeat; ++run) {
    const auto t0 = std::chrono::steady_clock::now();
    video = decoder.decode(z.data(), T, H, W, mean, std_dev, schedule);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const char* label = (run == 0) ? "decoded   " : "  (warm)  ";
    std::printf("%s %d frames of %dx%d in %.3f s (%.2f fps)\n", label, video.frames, video.width,
                video.height, seconds,
                seconds > 0 ? static_cast<double>(video.frames) / seconds : 0.0);
    // The phase spans inside `decode` tile exactly this interval, so it is
    // their denominator. With --repeat both sides accumulate together.
    vidfab::cuda::PhaseProfiler::instance().add_total("video vae decode", seconds * 1000.0);
  }
  vidfab::cuda::PhaseProfiler::instance().report(stdout);

  // Report basic statistics: a decode that silently produced NaN or a constant
  // image should be visible here without opening the file.
  double sum = 0.0;
  float lo = 1e30f;
  float hi = -1e30f;
  size_t nonfinite = 0;
  for (float v : video.data) {
    if (!std::isfinite(v)) {
      ++nonfinite;
      continue;
    }
    sum += v;
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  std::printf("pixels     min %.4f  max %.4f  mean %.4f  non-finite %zu\n", lo, hi,
              sum / static_cast<double>(video.data.size()), nonfinite);
  if (nonfinite != 0) {
    std::fprintf(stderr, "vidfab: decode produced non-finite pixels\n");
    return 1;
  }

  if (!dump_path.empty()) {
    // Raw fp32 pixels, so two runs can be diffed with `vidfab compare` at
    // float precision rather than after 8-bit quantisation. The copy into the
    // writer's own vector type is what this diagnostic path already did.
    vidfab::write_safetensors(
        dump_path, {{"pixels",
                     {3, video.frames, video.height, video.width},
                     std::vector<float>(video.data.begin(), video.data.end())}});
    std::printf("wrote      %s\n", dump_path.c_str());
  }

  vidfab::video::write_y4m(out_path, video.data, video.frames, video.height, video.width,
                           {fps, 1});
  std::printf("wrote      %s\n", out_path.c_str());
  if (!ppm_path.empty()) {
    vidfab::video::write_ppm(ppm_path, video.data, video.frames, video.height, video.width, 0);
    std::printf("wrote      %s\n", ppm_path.c_str());
  }
  return 0;
}
#endif  // VIDFAB_WITH_CUDA

int cmd_tokenize(int argc, char** argv) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("tokenize"));

  std::string tok_path;
  std::string text;
  bool show_pieces = false;

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--tokenizer" && i + 1 < argc) {
      tok_path = argv[++i];
    } else if (arg == "--pieces") {
      show_pieces = true;
    } else if (!arg.empty() && arg.front() == '-') {
      std::fprintf(stderr, "vidfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    } else {
      if (!text.empty()) text += " ";
      text += argv[i];
    }
  }
  vidfab::text::Tokenizer tok;
  if (tok_path.empty()) tok.load_embedded();
  else tok.load(tok_path);
  std::printf("vocab      %zu tokens\n", tok.vocab_size());

  if (show_pieces) {
    std::printf("pieces     ");
    for (const std::string& p : tok.pre_tokenize(text)) std::printf("[%s]", p.c_str());
    std::printf("\n");
  }

  const std::vector<int32_t> ids = tok.encode(text);
  std::printf("ids (%zu)   ", ids.size());
  for (int32_t id : ids) std::printf("%d ", id);
  std::printf("\n");
  std::printf("tokens     ");
  for (int32_t id : ids) std::printf("[%s]", tok.id_to_token(id).c_str());
  std::printf("\n");

  const std::string round = tok.decode(ids);
  std::printf("decoded    %s\n", round.c_str());
  std::printf("round trip %s\n", round == text ? "OK" : "MISMATCH");
  return round == text ? 0 : 1;
}

// Resolves a request all the way to a plan, then runs it. The stages are added
// one at a time; until they all exist this reports precisely which one is
// missing rather than pretending. `--dry-run` stops after the plan, which
// costs no I/O and is the fastest way to check geometry and schedule.
int cmd_generate(int argc, char** argv, const char* executable) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("generate"));

  vidfab::GenerateRequest req;
  req.canvas_width = 864;
  req.canvas_height = 480;
  req.num_frames = 124;
  req.num_inference_steps = 15;
  req.seed = 0;
  bool dry_run = false;
  bool synthetic = false;
  vidfab::sampler::SamplerKind sampler_kind = vidfab::sampler::SamplerKind::kEuler;
  std::string dump_latents;
  std::string prompt_file;
  bool saw_prompt = false;
  int attn_band = 0;
  vidfab::AttentionMode attention_mode = vidfab::AttentionMode::kSage2;
  vidfab::SolSchedule sol_schedule;
  std::string init_latents;
  int bench_load = 0;
  bool saw_aspect = false;
  bool saw_resolution = false;
  bool saw_out = false;
  bool saw_seed = false;
  int count = 1;
  std::string inference_backend = "cuda";
  std::string output_accelerator = "cpu";

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("generate: ") + what + " needs a value");
      }
      return argv[++i];
    };
    if (arg == "--prompt") {
      req.prompt = next("--prompt");
      saw_prompt = true;
    } else if (arg == "--prompt-file") {
      prompt_file = next("--prompt-file");
    } else if (arg == "--out") {
      req.out_path = next("--out");
      saw_out = true;
    } else if (arg == "--frames") {
      req.num_frames = std::atoi(next("--frames"));
    } else if (arg == "--steps") {
      req.num_inference_steps = std::atoi(next("--steps"));
    } else if (arg == "--seed") {
      // A negative seed asks for a random one, the same as passing no --seed at
      // all. Checked on the text rather than on the parsed value because
      // `strtoull` silently wraps "-1" to 2^64-1, which would read as an
      // ordinary explicit seed. Testing the sign first also keeps the whole
      // unsigned range usable for seeds that really are meant to be large.
      const std::string_view value = next("--seed");
      const size_t first = value.find_first_not_of(" \t");
      if (first != std::string_view::npos && value[first] == '-') {
        saw_seed = false;
      } else {
        req.seed = std::strtoull(value.data(), nullptr, 10);
        saw_seed = true;
      }
    } else if (arg == "--count") {
      count = std::atoi(next("--count"));
    } else if (arg == "--sampler") {
      const std::string v = next("--sampler");
      if (v == "euler") {
        sampler_kind = vidfab::sampler::SamplerKind::kEuler;
      } else if (v == "ab2") {
        sampler_kind = vidfab::sampler::SamplerKind::kAb2;
      } else {
        std::fprintf(stderr, "vidfab: --sampler wants euler or ab2, got '%s'\n", v.c_str());
        return 2;
      }
    } else if (arg == "--aspect") {
      const std::string v = next("--aspect");
      const size_t colon = v.find(':');
      if (colon == std::string::npos) {
        std::fprintf(stderr, "vidfab: --aspect wants W:H, e.g. 16:9\n");
        return 2;
      }
      req.aspect_w = std::atoi(v.substr(0, colon).c_str());
      req.aspect_h = std::atoi(v.substr(colon + 1).c_str());
      req.canvas_width = 0;
      req.canvas_height = 0;
      saw_aspect = true;
    } else if (arg == "--resolution") {
      const std::string v = next("--resolution");
      const size_t x = v.find_first_of("xX");
      if (x == std::string::npos) {
        std::fprintf(stderr, "vidfab: --resolution wants WxH, e.g. 1344x768\n");
        return 2;
      }
      req.canvas_width = std::atoi(v.substr(0, x).c_str());
      req.canvas_height = std::atoi(v.substr(x + 1).c_str());
      if (req.canvas_width <= 0 || req.canvas_height <= 0) {
        std::fprintf(stderr, "vidfab: --resolution wants two positive numbers, got '%s'\n",
                     v.c_str());
        return 2;
      }
      saw_resolution = true;
    } else if (arg == "--transformer") {
      req.transformer_path = next("--transformer");
    } else if (arg == "--text-encoder") {
      req.text_encoder_path = next("--text-encoder");
    } else if (arg == "--tokenizer") {
      req.tokenizer_path = next("--tokenizer");
    } else if (arg == "--vae") {
      req.video_vae_path = next("--vae");
    } else if (arg == "--audio-vae") {
      req.audio_vae_path = next("--audio-vae");
    } else if (arg == "--reference-image") {
      req.reference_image_paths.emplace_back(next("--reference-image"));
    } else if (arg == "--raw") {
      req.raw_output = true;
    } else if (arg == "--inference-backend") {
      inference_backend = next("--inference-backend");
      if (inference_backend != "cuda" && inference_backend != "vulkan") {
        std::fprintf(stderr,
                     "vidfab: --inference-backend wants cuda or vulkan, got '%s'\n",
                     inference_backend.c_str());
        return 2;
      }
    } else if (arg == "--output-accelerator") {
      output_accelerator = next("--output-accelerator");
      if (output_accelerator != "cpu" && output_accelerator != "vulkan") {
        std::fprintf(stderr,
                     "vidfab: --output-accelerator wants cpu or vulkan, got '%s'\n",
                     output_accelerator.c_str());
        return 2;
      }
    } else if (arg == "--cache-threshold") {
      req.cache_threshold = static_cast<float>(std::strtod(next("--cache-threshold"), nullptr));
    } else if (arg == "--cache-warmup") {
      req.cache_warmup = std::atoi(next("--cache-warmup"));
    } else if (arg == "--skip-every") {
      req.skip_every = std::atoi(next("--skip-every"));
    } else if (arg == "--block-cache-span") {
      req.block_cache_span = std::atoi(next("--block-cache-span"));
    } else if (arg == "--block-cache-start") {
      req.block_cache_start = std::atoi(next("--block-cache-start"));
    } else if (arg == "--block-cache-interval") {
      req.block_cache_interval = std::atoi(next("--block-cache-interval"));
    } else if (arg == "--block-cache-warmup") {
      req.block_cache_warmup = std::atoi(next("--block-cache-warmup"));
    } else if (arg == "--dry-run") {
      dry_run = true;
    } else if (arg == "--synthetic-latents") {
      synthetic = true;
    } else if (arg == "--dump-latents") {
      dump_latents = next("--dump-latents");
    } else if (arg == "--attn-band") {
      attn_band = std::atoi(next("--attn-band"));
    } else if (arg == "--attention") {
      const std::string v = next("--attention");
      if (!vidfab::parse_attention_mode(v, &attention_mode)) {
        std::fprintf(stderr,
                     "vidfab: --attention wants none, flash2, sage2, sol, "
                     "sol-experimental, or exact, got '%s'\n", v.c_str());
        return 2;
      }
    } else if (arg == "--sol-beta") {
      sol_schedule.beta = std::strtof(next("--sol-beta"), nullptr);
    } else if (arg == "--sol-error-k") {
      sol_schedule.error_k=std::strtof(next("--sol-error-k"),nullptr);
    } else if (arg == "--sol-error-v") {
      sol_schedule.error_v=std::strtof(next("--sol-error-v"),nullptr);
    } else if (arg == "--sol-step-start") {
      sol_schedule.step_begin = std::atoi(next("--sol-step-start"));
    } else if (arg == "--sol-step-end") {
      sol_schedule.step_end = std::atoi(next("--sol-step-end"));
    } else if (arg == "--sol-step-every") {
      sol_schedule.step_every = std::atoi(next("--sol-step-every"));
    } else if (arg == "--sol-layer-start") {
      sol_schedule.layer_begin = std::atoi(next("--sol-layer-start"));
    } else if (arg == "--sol-layer-end") {
      sol_schedule.layer_end = std::atoi(next("--sol-layer-end"));
    } else if (arg == "--sol-layer-every") {
      sol_schedule.layer_every = std::atoi(next("--sol-layer-every"));
    } else if (arg == "--init-latents") {
      init_latents = next("--init-latents");
    } else if (arg == "--bench-load") {
      bench_load = std::atoi(next("--bench-load"));
    } else {
      std::fprintf(stderr, "vidfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    }
  }

  if (inference_backend == "vulkan") {
    if (!vidfab::attention_mode_supported(vidfab::DeviceBackend::kVulkan,
                                          attention_mode)) {
      std::fprintf(stderr,
                   "vidfab: Vulkan inference supports only --attention exact; "
                   "mode '%s' is unavailable and will not be remapped\n",
                   vidfab::attention_mode_name(attention_mode));
      return 1;
    }
    std::fprintf(stderr,
                 "vidfab: --attention exact is the only defined Vulkan attention choice, "
                 "but Vulkan neural inference orchestration is not implemented; "
                 "no attention pipeline ran and no CUDA fallback was used\n");
    return 1;
  }

  // Both write the same field, so accepting both would mean silently honouring
  // one of them and dropping the other.
  if (saw_prompt && !prompt_file.empty()) {
    std::fprintf(stderr, "vidfab: --prompt and --prompt-file cannot be combined\n");
    return 2;
  }
  if (!prompt_file.empty()) {
    // Read here rather than at run time: the prompt shapes the plan, so
    // `--dry-run` has to see it, and a bad path should fail before any weights
    // are touched. The text is used verbatim apart from a UTF-8 BOM, CRLF line
    // endings and surrounding blank space -- all things an editor adds and no
    // prompt wants in its token stream.
    std::ifstream in(prompt_file, std::ios::binary);
    if (!in) {
      std::fprintf(stderr, "vidfab: cannot read --prompt-file '%s'\n", prompt_file.c_str());
      return 2;
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    std::string text = contents.str();
    if (text.rfind("\xEF\xBB\xBF", 0) == 0) text.erase(0, 3);
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    const size_t first = text.find_first_not_of(" \t\n");
    if (first == std::string::npos) {
      std::fprintf(stderr, "vidfab: --prompt-file '%s' has no prompt in it\n",
                   prompt_file.c_str());
      return 2;
    }
    const size_t last = text.find_last_not_of(" \t\n");
    req.prompt = text.substr(first, last - first + 1);
  }

  // `--synthetic-latents --init-latents <f>` is the decode-an-existing-latent
  // path and needs no prompt; the seeded-noise form still does not either.
  if (req.prompt.empty() && !dry_run && !synthetic) {
    std::fprintf(stderr, "vidfab: generate needs --prompt \"...\" or --prompt-file <file>\n");
    return 2;
  }
  if (!std::isfinite(sol_schedule.beta)||!std::isfinite(sol_schedule.error_k)||
      !std::isfinite(sol_schedule.error_v)||sol_schedule.error_k<0||sol_schedule.error_v<0||
      sol_schedule.step_every <= 0 ||
      sol_schedule.layer_every <= 0 || sol_schedule.step_begin < 0 ||
      sol_schedule.layer_begin < 0 || sol_schedule.step_end < sol_schedule.step_begin ||
      sol_schedule.layer_end < sol_schedule.layer_begin) {
    std::fprintf(stderr,"vidfab: invalid Sol beta/range/cadence\n");
    return 2;
  }
  if (synthetic && !req.reference_image_paths.empty()) {
    std::fprintf(stderr,
                 "vidfab: --reference-image needs denoising and cannot be combined with "
                 "--synthetic-latents\n");
    return 2;
  }
  if (req.cache_threshold < 0.0f) {
    std::fprintf(stderr, "vidfab: --cache-threshold cannot be negative (0 disables it)\n");
    return 2;
  }
  if (req.skip_every < 0) {
    std::fprintf(stderr, "vidfab: --skip-every cannot be negative (0 disables it)\n");
    return 2;
  }
  if (req.block_cache_span < 0) {
    std::fprintf(stderr, "vidfab: --block-cache-span cannot be negative (0 disables it)\n");
    return 2;
  }
  // Rejected rather than clamped up to 2. An interval of 1 evaluates the span
  // every step and still pays the snapshot and the subtract, so it is not a
  // slower setting but a strictly pointless one, and a user who typed it meant
  // something else.
  if (req.block_cache_span > 0 && req.block_cache_interval < 2) {
    std::fprintf(stderr,
                 "vidfab: --block-cache-interval must be at least 2 (1 would evaluate every "
                 "step and cache for nothing)\n");
    return 2;
  }
  // Refused, like `--sampler ab2` with step caching above, and for a related
  // reason: the two caches do not compose the way their flags suggest.
  //
  // The step cache skips a step by not calling `forward` at all, and the block
  // cache's decision lives *inside* `forward`. So a step the step cache skips
  // is a step the block cache never sees: its interval stops counting real
  // steps, and a delta it believed was one step old becomes three or more. The
  // two lossy approximations then compound on the same step, and nothing
  // records that they did — the printed "N of M spans reused" denominator
  // silently shrinks to the steps that actually ran, so the log understates the
  // staleness rather than revealing it.
  //
  // That is a quality question nobody has measured, on the two features whose
  // entire failure mode is drift. Refusing costs a combination nobody has shown
  // to be useful; allowing it costs a number that looks like a measurement and
  // is not one. If the combination is ever wanted, the fix is for the denoise
  // loop to tell the block cache about skipped steps so the phase and the
  // counters stay defined — not to delete this check.
  if (req.block_cache_span > 0 && (req.cache_threshold > 0.0f || req.skip_every > 0)) {
    std::fprintf(stderr,
                 "vidfab: --block-cache-span cannot be combined with --cache-threshold or "
                 "--skip-every; a skipped step hides the block cache's schedule from it\n");
    return 2;
  }
  if (attn_band > 0 && attention_mode != vidfab::AttentionMode::kFlash2 &&
      attention_mode != vidfab::AttentionMode::kExact) {
    std::fprintf(stderr,
                 "vidfab: --attn-band currently requires --attention flash2 or exact\n");
    return 2;
  }
  // Rejected rather than silently resolved. A fixed interval and an adaptive
  // threshold ORed together is neither policy, and the fixed one exists
  // precisely so the adaptive one has something to be measured against.
  if (req.cache_threshold > 0.0f && req.skip_every > 0) {
    std::fprintf(stderr,
                 "vidfab: --cache-threshold and --skip-every are alternatives; pass one\n");
    return 2;
  }
  // Refused, not warned about. AB2 extrapolates from `v_{n-1}`, and with step
  // caching on that is a *reused* velocity — a point the model never visited at
  // that timestep — so the two-point extrapolation is extrapolating a constant
  // across the skipped interval. The composition is wrong by construction, not
  // merely unmeasured, and the failure mode is a plausible-looking number
  // rather than a crash: whoever ran it would report a combined speedup that
  // multiplies two savings which do not multiply. See the reasoning recorded in
  // src/sampler/scheduler.cpp.
  //
  // The test is `StepCacheConfig::enabled()` rather than a second copy of its
  // condition spelled out here, so that "caching is on" has exactly one
  // definition. A duplicated predicate is how `--sampler ab2` alone — the floor
  // control, which must stay legal — would eventually start being refused by a
  // guard that had drifted from the library it is guarding.
  vidfab::dit::StepCacheConfig cache_cfg;
  cache_cfg.threshold = req.cache_threshold;
  cache_cfg.warmup = req.cache_warmup;
  cache_cfg.skip_every = req.skip_every;
  if (cache_cfg.enabled() && sampler_kind == vidfab::sampler::SamplerKind::kAb2) {
    std::fprintf(stderr,
                 "vidfab: --sampler ab2 does not compose with step caching: ab2 extrapolates\n"
                 "        from the previous velocity, which caching makes a reused one. Run\n"
                 "        them separately; their savings are not multiplicative.\n");
    return 2;
  }

  // Rejected rather than ranked. Silently letting one win would mean a run
  // whose canvas is not the one half the command line asked for, and the two
  // flags are close enough in intent that a user passing both has made a
  // mistake worth telling them about.
  if (saw_aspect && saw_resolution) {
    std::fprintf(stderr,
                 "vidfab: --aspect and --resolution set the same thing; pass one or the other\n");
    return 2;
  }
  if (count <= 0) {
    std::fprintf(stderr, "vidfab: --count must be a positive integer\n");
    return 2;
  }
  if (!saw_out) req.out_path = timestamped_output_path();
  const std::string base_out_path = req.out_path;
  const uint64_t base_seed = req.seed;
  const vidfab::GeneratePlan plan = vidfab::resolve_plan(req);

  // After `resolve_plan`, so a canvas that is going to be rejected outright is
  // not first warned about — an invalid request should produce one message
  // about what is wrong with it, not a size advisory followed by a refusal.
  if (saw_resolution && vidfab::dit::canvas_exceeds_trained_area(req.canvas_height,
                                                                req.canvas_width)) {
    // A warning, not a refusal: the caller named this canvas. But packed rows
    // grow with area and attention with their square, so an innocent-looking
    // doubling is roughly four times the attention cost.
    std::fprintf(stderr,
                 "vidfab: %dx%d is %.2fx the 1344x768 area the model was trained at; "
                 "attention cost grows with the square of that, and quality outside the "
                 "trained range is uncharacterised\n",
                 req.canvas_width, req.canvas_height,
                 static_cast<double>(req.canvas_width) * req.canvas_height / (1344.0 * 768.0));
  }
  if (dry_run) {
    for (int generation = 0; generation < count; ++generation) {
      req.seed = saw_seed ? base_seed + static_cast<uint64_t>(generation) : random_seed();
      req.out_path = counted_output_path(base_out_path, generation, count);
      if (generation > 0) std::printf("\n");
      std::fputs(vidfab::describe_plan(req, plan).c_str(), stdout);
    }
    return 0;
  }

#if VIDFAB_WITH_CUDA
  // Dry-run above is deliberately device-free. Synthetic latents skip the
  // transformer, so only a real denoise run needs the pinned exact tuple.
  if (!synthetic && attention_mode == vidfab::AttentionMode::kExact &&
      !vidfab::cuda::deterministic_h3_attention_available()) {
    std::fprintf(stderr,
                 "vidfab: --attention exact is unavailable on this CUDA device/runtime tuple\n");
    return 1;
  }
#endif

#if !VIDFAB_WITH_CUDA
  (void)executable;
  std::fprintf(stderr,
               "vidfab: built without CUDA support; model inference requires CUDA. "
               "Vulkan accelerates output conversion only\n");
  return 1;
#else

#if VIDFAB_WITH_VULKAN
  std::unique_ptr<vidfab::vulkan::Yuv420Converter> output_converter;
  if (output_accelerator == "vulkan") {
    try {
      output_converter = std::make_unique<vidfab::vulkan::Yuv420Converter>();
      std::printf("output      Vulkan RGB-to-YUV on %s (model inference remains CUDA)\n",
                  output_converter->device_name());
    } catch (const std::exception& error) {
      std::fprintf(stderr, "vidfab: Vulkan output accelerator unavailable: %s\n", error.what());
      return 1;
    }
  }
#else
  if (output_accelerator == "vulkan") {
    std::fprintf(stderr,
                 "vidfab: Vulkan output accelerator requested, but this build disabled Vulkan\n");
    return 1;
  }
#endif

  if (!saw_out) std::filesystem::create_directories(std::filesystem::path(req.out_path).parent_path());

  discover_generate_checkpoints(req, executable);
  ensure_generate_models(req, executable);

#if VIDFAB_WITH_CUDA
  // Times the transformer load on its own, the same way `decode --bench-load`
  // times the VAE's and for the same reason: the first load in a process pays
  // for pulling the mapping in from storage and the later ones do not, and the
  // gap between them is the entire subject of cold-start work. Loading through
  // `generate` proper would first stream 25 GB of conditioner, which both costs
  // a minute and evicts the very file being measured.
  //
  // Nothing here evicts the cache, so the first number is only a *cold* number
  // if the caller made it one.
  if (bench_load > 0) {
    if (req.transformer_path.empty()) {
      std::fprintf(stderr, "vidfab: --bench-load needs --transformer <f>\n");
      return 2;
    }
    vidfab::SafeTensors ckpt;
    ckpt.open(req.transformer_path);
    std::printf("\n%s\n%.3f GB on disk, %zu tensors\n", req.transformer_path.c_str(),
                ckpt.file_size() / 1e9, ckpt.tensor_count());
    for (int i = 0; i < bench_load; ++i) {
      vidfab::dit::Transformer probe;
      const auto s0 = std::chrono::steady_clock::now();
      probe.load(ckpt);
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
      std::printf("load %d: %s on device in %6.3f s  (%.2f GB/s off disk)\n", i + 1,
                  format_bytes(probe.weight_bytes()).c_str(), sec,
                  static_cast<double>(ckpt.file_size()) / sec / 1e9);
      std::fflush(stdout);
    }
    return 0;
  }
#else
  if (bench_load > 0) {
    std::fprintf(stderr, "vidfab: --bench-load needs a GPU build\n");
    return 1;
  }
#endif

#if !VIDFAB_WITH_CUDA
  (void)sampler_kind;
  (void)dump_latents;
  (void)init_latents;
  (void)attn_band;
  std::fprintf(stderr, "vidfab: built without CUDA support; generate needs a GPU\n");
  return 1;
#else
  vidfab::RunOptions options;
  options.source =
      synthetic ? vidfab::LatentSource::kSyntheticNoise : vidfab::LatentSource::kDenoise;
  options.sampler = sampler_kind;
  options.dump_latents_path = dump_latents;
  options.attention_band = attn_band;
  options.attention_mode = attention_mode;
  options.sol_schedule = sol_schedule;
  options.init_latents_path = init_latents;
#if VIDFAB_WITH_VULKAN
  options.output_frame_converter = output_converter.get();
#endif

  for (int generation = 0; generation < count; ++generation) {
    req.seed = saw_seed ? base_seed + static_cast<uint64_t>(generation) : random_seed();
    req.out_path = counted_output_path(base_out_path, generation, count);
    options.reuse_models = count > 1;
    options.release_reused_models = generation + 1 == count;
    if (generation > 0) std::printf("\n");
    std::fputs(vidfab::describe_plan(req, plan).c_str(), stdout);
    std::printf("\n");
    const vidfab::RunResult run = vidfab::run_generate(req, plan, options);
    if (!run.ok) {
      std::fprintf(stderr, "\nvidfab: generation %d of %d: %s\n", generation + 1, count,
                   run.message.c_str());
      return 1;
    }
    std::printf("\ndone in %.2f s\n", run.seconds_denoise + run.seconds_video_decode +
                                          run.seconds_audio_decode + run.seconds_output);
  }
  return 0;
#endif
#endif  // VIDFAB_WITH_CUDA
}

int cmd_devices() {
#if !VIDFAB_WITH_CUDA
  std::printf("CUDA inference       unavailable in this build\n");
#else
  const int count = vidfab::cuda::device_count();
  if (count == 0) {
    std::printf("CUDA inference       no visible device\n");
  }
  for (int i = 0; i < count; ++i) {
    const vidfab::cuda::DeviceInfo d = vidfab::cuda::query_device(i);
    std::printf("device %d  %s\n", d.index, d.name.c_str());
    std::printf("  compute capability  %d.%d\n", d.major, d.minor);
    std::printf("  memory              %s free of %s\n", format_bytes(d.free_memory).c_str(),
                format_bytes(d.total_memory).c_str());
    std::printf("  multiprocessors     %d\n", d.multiprocessors);
    std::printf("  shared mem / block  %s\n", format_bytes(d.shared_memory_per_block).c_str());
    std::printf("  numeric support     bf16=%s fp8=%s fp4=%s\n", d.supports_bf16 ? "yes" : "no",
                d.supports_fp8 ? "yes" : "no", d.supports_fp4 ? "yes" : "no");
  }
#endif
#if VIDFAB_WITH_VULKAN
  std::string diagnostic;
  if (!vidfab::vulkan::Instance::available(&diagnostic)) {
    std::printf("Vulkan output        unavailable: %s\n", diagnostic.c_str());
  } else {
    try {
      auto instance = vidfab::vulkan::Instance::create();
      const auto devices = instance.enumerate_devices();
      if (devices.empty()) std::printf("Vulkan output        no compute device\n");
      for (size_t i = 0; i < devices.size(); ++i) {
        const auto& d = devices[i].info();
        std::printf("Vulkan output %zu     %s (timeline=%s)\n", i, d.name.c_str(),
                    d.timeline_semaphore ? "yes" : "no");
      }
    } catch (const std::exception& error) {
      std::printf("Vulkan output        unavailable: %s\n", error.what());
    }
  }
#else
  std::printf("Vulkan output        disabled in this build\n");
#endif
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  try {
    if (!ensure_license_acceptance()) return 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vidfab: %s\n", e.what());
    return 1;
  }

  if (argc < 2) {
    print_usage();
    return 2;
  }

  const std::string_view command = argv[1];

  // `--help` is honoured for every command here rather than inside each one,
  // so a command that takes no arguments at all still answers it. Checked
  // before dispatch, so it never runs the command by accident.
  if (const CommandHelp* c = find_command(command); c != nullptr && wants_help(argc - 2, argv + 2)) {
    return print_command_help(*c);
  }

  try {
    if (command == "generate") return cmd_generate(argc - 2, argv + 2, argv[0]);
    if (command == "inspect") return cmd_inspect(argc - 2, argv + 2);
    if (command == "compare") return cmd_compare(argc - 2, argv + 2);
    if (command == "compare-y4m") return cmd_compare_y4m(argc - 2, argv + 2);
    if (command == "devices") return cmd_devices();
    if (command == "tokenize") return cmd_tokenize(argc - 2, argv + 2);
#if VIDFAB_WITH_CUDA
    if (command == "decode") return cmd_decode(argc - 2, argv + 2);
#endif
    if (command == "version") {
      std::printf("vidfab %s\n", kVersion);
      return 0;
    }
    if (command == "help" || command == "--help" || command == "-h") {
      if (argc > 2) {
        const CommandHelp* c = find_command(argv[2]);
        if (c != nullptr) return print_command_help(*c);
        std::fprintf(stderr, "vidfab: unknown command '%s'\n\n", argv[2]);
        print_usage();
        return 2;
      }
      print_usage();
      return 0;
    }
    std::fprintf(stderr, "vidfab: unknown command '%s'\n\n", argv[1]);
    print_usage();
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vidfab: %s\n", e.what());
    return 1;
  }
}


