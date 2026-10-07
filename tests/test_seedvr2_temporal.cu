#include "harness.h"
#include "slopfab/seedvr2.h"
#include "slopfab/cuda/device.h"
#include "slopfab/safetensors.h"
#include "slopfab/tensor_convert.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>

namespace {
using namespace slopfab;
using namespace slopfab::seedvr2;

struct Environment {
  std::string name, previous;
  bool present;
  Environment(const char* key, const char* value) : name(key), present(std::getenv(key) != nullptr) {
    if (present)
      previous = std::getenv(key);
    set(value);
  }
  void set(const char* value) const {
#ifdef _WIN32
    _putenv_s(name.c_str(), value ? value : "");
#else
    if (value)
      setenv(name.c_str(), value, 1);
    else
      unsetenv(name.c_str());
#endif
  }
  ~Environment() { set(present ? previous.c_str() : nullptr); }
};

struct CaptureDirectory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("slopfab_seedvr2_conditioning_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  ~CaptureDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

bool real_options(Options& options) {
  const char* dit = std::getenv("SLOPFAB_SEEDVR2_DIT");
  const char* vae = std::getenv("SLOPFAB_SEEDVR2_VAE");
  if (!dit || !vae || !std::filesystem::is_regular_file(std::filesystem::u8path(dit)) ||
      !std::filesystem::is_regular_file(std::filesystem::u8path(vae))) {
    SKIP_MISSING_FIXTURE("Set SLOPFAB_SEEDVR2_DIT and SLOPFAB_SEEDVR2_VAE to real checkpoints");
    return false;
  }
  if (cuda::device_count() == 0) {
    SKIP_UNSUPPORTED_HARDWARE("CUDA required");
    return false;
  }
  options.transformer = dit;
  options.vae = vae;
  options.width = options.height = 128;
  options.segment_frames = 5;
  options.vae_tile = 0;
  options.color_match = false;
  cuda::set_device(options.device);
  size_t available = 0, total = 0;
  SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&available, &total));
  if (available < (size_t(2) << 30)) {
    SKIP_INSUFFICIENT_VRAM("2 GiB free required for real-checkpoint restoration");
    return false;
  }
  return true;
}

std::vector<Frame> static_input(const Options& options) {
  Frame frame(size_t(options.height) * options.width * 3);
  for (size_t i = 0; i < frame.size(); ++i)
    frame[i] = float((i * 17) % 251) / 250.0f;
  return std::vector<Frame>(options.segment_frames, frame);
}
}

SLOPFAB_TEST_CATEGORY(seedvr2_conditioning_uses_posterior_mode, "integration") {
  Options options;
  if (!real_options(options))
    return;
  Environment cache("SLOPFAB_SEEDVR2_CACHE_MIB", "0");
  CaptureDirectory directory;
  const auto path = directory.path.u8string();
  Environment capture("SLOPFAB_SEEDVR2_CAPTURE_DIR", path.c_str());
  struct Captured {};
  {
    Restorer model(options);
    model.progress = [](const std::string& message) {
      if (message == "DiT block 1/32")
        throw Captured{};
    };
    bool captured = false;
    try {
      model.restore(static_input(options), 37);
    } catch (const Captured&) {
      captured = true;
    }
    CHECK(captured);
  }
  SafeTensors moment_file, patch_file;
  moment_file.open((directory.path / "vae_moments.safetensors").u8string());
  patch_file.open((directory.path / "dit_patches.safetensors").u8string());
  const auto& moment_view = moment_file.at("vae_moments");
  const auto& patch_view = patch_file.at("dit_patches");
  CHECK(moment_view.shape == std::vector<int64_t>({2, 16, 16, 32}));
  CHECK(patch_view.shape == std::vector<int64_t>({2, 8, 8, 132}));
  const auto moments = to_f32(moment_view), patches = to_f32(patch_view);
  if (moments.size() != size_t(2 * 16 * 16 * 32) || patches.size() != size_t(2 * 8 * 8 * 132))
    return;
  // Independent specification: Gaussian posterior.mode() is its mean. The
  // variance channels must never perturb conditioning or consume noise draws.
  // Reading actual model captures also verifies the 2x2 patch channel layout.
  std::vector<float> expected_condition, actual_condition, expected_noise, actual_noise;
  std::mt19937_64 rng(options.seed);
  std::normal_distribution<float> gaussian;
  float min_logvar = moments[16], max_logvar = moments[16];
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) {
        const size_t in = (size_t(z) * 16 + y) * 16 + x;
        const size_t patch = (((size_t(z) * 8 + y / 2) * 8 + x / 2) * 4 +
                              (y % 2) * 2 + x % 2) * 33;
        for (int c = 0; c < 16; ++c) {
          expected_condition.push_back(moments[in * 32 + c] * 0.9152f);
          actual_condition.push_back(patches[patch + 16 + c]);
          expected_noise.push_back(gaussian(rng));
          actual_noise.push_back(patches[patch + c]);
          min_logvar = std::min(min_logvar, moments[in * 32 + 16 + c]);
          max_logvar = std::max(max_logvar, moments[in * 32 + 16 + c]);
        }
        CHECK(patches[patch + 32] == 1.0f);
      }
  CHECK(min_logvar < max_logvar);
  CHECK_CLOSE(expected_condition, actual_condition, 0, "conditioning is posterior mean times 0.9152");
  CHECK_CLOSE(expected_noise, actual_noise, 0, "diffusion noise is independent of posterior and position");
}

SLOPFAB_TEST_CATEGORY(seedvr2_identical_segments_ignore_stream_position, "integration") {
  Options options;
  if (!real_options(options))
    return;
  Environment cache("SLOPFAB_SEEDVR2_CACHE_MIB", "0");
  Environment capture("SLOPFAB_SEEDVR2_CAPTURE_DIR", nullptr);
  Restorer model(options);
  const auto input = static_input(options);
  const auto first = model.restore(input, 0);
  const auto later = model.restore(input, 4);
  CHECK(first.size() == later.size());
  if (first.size() != later.size())
    return;
  for (size_t frame = 0; frame < first.size(); ++frame) {
    CHECK(first[frame].size() == later[frame].size());
    if (first[frame].size() == later[frame].size())
      CHECK(std::memcmp(first[frame].data(), later[frame].data(),
                        first[frame].size() * sizeof(float)) == 0);
  }
}
