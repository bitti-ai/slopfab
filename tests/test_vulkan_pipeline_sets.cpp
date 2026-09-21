#include "harness.h"
#include "slopfab/vulkan/tensor.h"
#include <stdexcept>

SLOPFAB_TEST(vulkan_pipeline_sets_cache_and_recording_boundary) {
  using namespace slopfab::vulkan;
  std::string diagnostic;
  if (!Instance::available(&diagnostic)) {
    SKIP_UNSUPPORTED_HARDWARE("Vulkan loader unavailable: %s", diagnostic.c_str());
    return;
  }
  auto instance = Instance::create();
  auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().fp32_signed_zero_inf_nan_preserve ||
      !physical.front().info().fp32_rounding_rte) {
    SKIP_UNSUPPORTED_HARDWARE("Vulkan tensor prerequisites unavailable");
    return;
  }
  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  auto device = physical.front().create_device(device_options);
  TensorContextOptions options;
  options.pipeline_sets = TensorPipelineSet::kCore;
  TensorContext first(device, options);
  CHECK(first.prepared_pipeline_sets() == TensorPipelineSet::kCore);
  const uint64_t initial_hits = device.pipeline_cache_hits();
  TensorContext second(device, options);
  CHECK(device.pipeline_cache_hits() > initial_hits);
  const uint64_t extent = 4;
  auto layout = slopfab::TensorLayout::contiguous(&extent, 1);
  auto a = first.allocate(layout), b = first.allocate(layout), output = first.allocate(layout);
  const float inputs[] = {1, 2, 3, 4};
  first.upload(a, inputs, 4);
  first.upload(b, inputs, 4);
  {
    auto batch = first.begin_batch();
    bool rejected = false;
    try {
      first.prepare_pipeline_sets(device, TensorPipelineSet::kAudio);
    } catch (const std::logic_error&) {
      rejected = true;
    }
    CHECK(rejected);
    batch.add(a, b, output);
    batch.submit().wait();
  }
  float actual[4] = {};
  first.download(output, actual, 4);
  for (int i = 0; i < 4; ++i)
    CHECK(actual[i] == 2 * inputs[i]);
  first.prepare_pipeline_sets(device, TensorPipelineSet::kAudio);
  CHECK(first.prepared_pipeline_sets() == (TensorPipelineSet::kCore | TensorPipelineSet::kAudio));
  const auto hits = device.pipeline_cache_hits();
  first.prepare_pipeline_sets(device, TensorPipelineSet::kAudio);
  CHECK(device.pipeline_cache_hits() == hits); // no recompilation/relookup
  bool rejected = false;
  try {
    first.prepare_pipeline_sets(device, static_cast<TensorPipelineSet>(1u << 31));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}
