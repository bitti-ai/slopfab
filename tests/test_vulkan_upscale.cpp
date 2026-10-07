#include "harness.h"
#include "../src/upscale/backend.h"
#include "slopfab/vulkan/runtime.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace {
using namespace slopfab;
using namespace slopfab::upscale_detail;

struct BatchEnvironment {
  bool present;
  std::string previous;

  explicit BatchEnvironment(const char* value)
      : present(std::getenv("SLOPFAB_REALESRGAN_VULKAN_BATCH") != nullptr) {
    if (present)
      previous = std::getenv("SLOPFAB_REALESRGAN_VULKAN_BATCH");
    set(value);
  }

  static void set(const char* value) {
#ifdef _WIN32
    _putenv_s("SLOPFAB_REALESRGAN_VULKAN_BATCH", value ? value : "");
#else
    if (value)
      setenv("SLOPFAB_REALESRGAN_VULKAN_BATCH", value, 1);
    else
      unsetenv("SLOPFAB_REALESRGAN_VULKAN_BATCH");
#endif
  }

  ~BatchEnvironment() {
    set(present ? previous.c_str() : nullptr);
  }
};

bool have_vulkan() {
  if (vulkan::Instance::available() && !vulkan::Instance::create().enumerate_devices().empty())
    return true;
  SKIP_UNSUPPORTED_HARDWARE("Vulkan compute device required");
  return false;
}

std::vector<float> chain(const char* batch) {
  BatchEnvironment environment(batch);
  auto backend = make_vulkan_backend();
  constexpr int h = 3, w = 5, channels = 3, count = h * w * channels;
  auto x = backend->upload(test::make_data(count, 51, 0.5f));
  auto residual = backend->upload(test::make_data(count, 73, 0.125f));
  auto weights = backend->upload(test::make_data(channels * channels * 9, 97, 0.05f));
  auto bias = backend->upload(test::make_data(channels, 101, 0.01f));
  std::vector<float> snapshots;
  for (int step = 0; step < 49; ++step) {
    auto out = backend->allocate(count);
    if (step % 2)
      backend->run({kResidual, h, w, channels, channels, count, 0, 0.25f}, x, residual, residual,
                   out);
    else
      backend->run({kConv, h, w, channels, channels, count, 1, 0}, x, weights, bias, out);
    // The previous activation may be referenced by an unsubmitted command.
    x = std::move(out);
    if (step == 20)
      residual = backend->upload(test::make_data(count, 113, 0.125f));
    if (step % 11 == 10) {
      const auto values = backend->download(x, count);
      snapshots.insert(snapshots.end(), values.begin(), values.end());
    }
  }
  // Drop all input owners before the final partial batch is submitted.
  residual.reset();
  weights.reset();
  bias.reset();
  const auto values = backend->download(x, count);
  CHECK(backend->download(x, count) == values);
  snapshots.insert(snapshots.end(), values.begin(), values.end());
  return snapshots;
}
} // namespace

SLOPFAB_TEST(vulkan_upscale_batched_dependencies_match_synchronous) {
  if (!have_vulkan())
    return;
  const auto expected = chain("1");
  CHECK(chain("3") == expected);
  CHECK(chain("16") == expected);
  CHECK(chain("64") == expected);
}

SLOPFAB_TEST(vulkan_upscale_batch_retention_budget) {
  if (!have_vulkan())
    return;
  BatchEnvironment environment("64");
  auto backend = make_vulkan_backend();
  constexpr uint32_t count = 1 << 20;
  const std::vector<float> ones(count, 1);
  auto x = backend->upload(ones), residual = backend->upload(ones);
  // Forty 4-MiB outputs cross the 64-MiB resource limit several times while
  // staying below the dispatch-count limit. Old activation owners are dropped.
  for (int step = 0; step < 40; ++step) {
    auto out = backend->allocate(count);
    backend->run({kResidual, 1, count, 1, 1, count, 0, 0.25f}, x, residual, residual, out);
    x = std::move(out);
  }
  const auto values = backend->download(x, count);
  CHECK(std::all_of(values.begin(), values.end(), [](float v) {
    return v == 11;
  }));
}

SLOPFAB_TEST(vulkan_upscale_rejects_invalid_batch_limits) {
  for (const char* value : {"0", "65", "-1", "2junk", "invalid"}) {
    BatchEnvironment environment(value);
    bool rejected = false;
    try {
      make_vulkan_backend();
    } catch (const std::exception&) {
      rejected = true;
    }
    CHECK(rejected);
  }
}
