#include "harness.h"
#include "slopfab/seedvr2.h"
#include "slopfab/cuda/device.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace {
struct Environment {
  std::string name, previous;
  bool present;

  Environment(const char* key, const char* value)
      : name(key), present(std::getenv(key) != nullptr) {
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

  ~Environment() {
    set(present ? previous.c_str() : nullptr);
  }
};
} // namespace

SLOPFAB_TEST_CATEGORY(seedvr2_restore_after_cancel_matches_fresh_instance, "integration") {
  using namespace slopfab::seedvr2;
  const char* dit = std::getenv("SLOPFAB_SEEDVR2_DIT");
  const char* vae = std::getenv("SLOPFAB_SEEDVR2_VAE");
  if (!dit || !vae || !std::filesystem::is_regular_file(std::filesystem::u8path(dit)) ||
      !std::filesystem::is_regular_file(std::filesystem::u8path(vae))) {
    SKIP_MISSING_FIXTURE("Set SLOPFAB_SEEDVR2_DIT and SLOPFAB_SEEDVR2_VAE to real checkpoints");
    return;
  }
  if (slopfab::cuda::device_count() == 0) {
    SKIP_UNSUPPORTED_HARDWARE("CUDA required");
    return;
  }
  Options options;
  options.transformer = dit;
  options.vae = vae;
  options.width = options.height = 128;
  options.segment_frames = 5;
  options.vae_tile = 0;
  options.color_match = false;
  slopfab::cuda::set_device(options.device);
  size_t free = 0, total = 0;
  SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free, &total));
  if (free < (size_t(2) << 30)) {
    SKIP_INSUFFICIENT_VRAM("2 GiB free required for real-checkpoint restoration");
    return;
  }
  Environment cache("SLOPFAB_SEEDVR2_CACHE_MIB", "0");
  Environment pool("SLOPFAB_SEEDVR2_POOL", "1");
  Environment capture("SLOPFAB_SEEDVR2_CAPTURE_DIR", nullptr);
  std::vector<Frame> input(5, Frame(size_t(128) * 128 * 3));
  for (size_t frame = 0; frame < input.size(); ++frame)
    for (size_t i = 0; i < input[frame].size(); ++i)
      input[frame][i] = float((i * 17 + frame * 31) % 251) / 250.0f;

  std::vector<Frame> recovered;
  {
    Restorer model(options);
    bool cancel_requested = false, saw_second_block = false, cancelled = false;
    model.cancelled = [&] {
      return cancel_requested;
    };
    model.progress = [&](const std::string& message) {
      if (message == "DiT block 2/32") {
        saw_second_block = true;
        cancel_requested = true;
      }
    };
    try {
      model.restore(input, 0);
    } catch (const std::exception& error) {
      cancelled = std::string(error.what()) == "SeedVR2: cancelled";
      slopfab::test::check_printf(cancelled, __FILE__, __LINE__,
                                  "unexpected restoration exception: %s", error.what());
    }
    CHECK(saw_second_block);
    CHECK(cancelled);
    // Cancellation is observed at the next progress checkpoint, after block 2
    // has issued work. Reusing pooled storage must remain safe on retry.
    cancel_requested = false;
    model.progress = {};
    recovered = model.restore(input, 0);
  }

  Restorer fresh(options);
  const auto expected = fresh.restore(input, 0);
  CHECK(recovered.size() == expected.size());
  if (recovered.size() != expected.size())
    return;
  for (size_t frame = 0; frame < expected.size(); ++frame) {
    CHECK(recovered[frame].size() == expected[frame].size());
    if (recovered[frame].size() == expected[frame].size())
      CHECK(std::memcmp(recovered[frame].data(), expected[frame].data(),
                        expected[frame].size() * sizeof(float)) == 0);
  }
}
