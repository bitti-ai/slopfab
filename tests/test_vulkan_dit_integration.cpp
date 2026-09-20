#include "detail/vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(vulkan_h3_int8_still_full_stack_memory, "integration") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const auto path = std::filesystem::path(SLOPFAB_TEST_SOURCE_DIR) /
      "weights/transformer/minimax_h3_fl2va_fasth3_dense_pruned_int8_convrot.safetensors";
  if (!std::filesystem::exists(path)) {
    SKIP_MISSING_FIXTURE("INT8 H3 checkpoint unavailable");
    return;
  }
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty()");
    return;
  }
  uint64_t device_heap = 0;
  for (const auto& heap : physical.front().info().memory_heaps)
    if (heap.device_local) device_heap = std::max(device_heap, heap.bytes);
  if (device_heap < (28ull << 30)) {
    SKIP_INSUFFICIENT_VRAM("full INT8 still regression needs a 28-GiB device heap");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 2048;
  TensorContext context(device, context_options);
  if (!context.exact_h3_attention()) {
    SKIP_UNSUPPORTED_HARDWARE("exact H3 attention unavailable");
    return;
  }
  ExactH3DenoiseConfig config;
  config.layout.num_text = 65;
  config.layout.num_latent_frames = 1;
  config.layout.latent_height = config.layout.latent_width = 48; // 768x768 still.
  config.layout.num_video_rows = config.layout.rows_per_frame();
  config.indices = dit::build_indices(config.layout);
  config.position_ids = dit::build_position_ids(config.layout);
  config.transformer.text_rows = config.layout.num_text;
  config.transformer.video_rows = config.layout.num_video_rows;
  config.transformer.main.layers = 50;
  config.transformer.main.block.sequence = config.layout.total_rows();
  config.transformer.main.block.timesteps = 2;
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  auto model = ExactH3Denoiser::create(context, config);
  model.load(checkpoint);
  CHECK(model.loaded());
  CHECK(context.reserved_bytes() < (24ull << 30));
  std::vector<float> prompt(size_t(config.layout.num_text) * 5120);
  std::vector<float> video(size_t(config.layout.num_video_rows) * 96);
  for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = float(int(i % 127) - 63) / 64;
  for (size_t i = 0; i < video.size(); ++i) video[i] = float(int(i % 61) - 30) / 32;
  model.prepare(prompt.data(), prompt.size(), video.data(), video.size(), nullptr, 0);
  sampler::FlowScheduler video_schedule(12.0f), audio_schedule(3.0f);
  video_schedule.set_timesteps(2);
  audio_schedule.set_timesteps(2);
  const auto output = model.run(video_schedule, audio_schedule);
  CHECK(output.steps_completed == 1 && !output.cancelled);
  CHECK(output.video_rows.size() == video.size() && output.audio_rows.empty());
  CHECK(std::all_of(output.video_rows.begin(), output.video_rows.end(),
                    [](float value) { return std::isfinite(value); }));
  CHECK(context.reserved_bytes() < (24ull << 30));
  std::printf("  INT8 50-block 768x768 still: persistent %.3f GiB, peak %.3f GiB, pool %.3f GiB\n",
      double(model.persistent_bytes()) / (1ull << 30),
      double(model.peak_device_bytes()) / (1ull << 30),
      double(context.reserved_bytes()) / (1ull << 30));
  model.unload();
  CHECK(!model.loaded() && model.persistent_bytes() == 0);
}
