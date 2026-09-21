#include "detail/vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(vulkan_exact_dit_euler_matches_host_scheduler, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(device_options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 4;
  TensorContext context(device, context_options);
  if (!context.exact_vae_pointwise()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !context.exact_vae_pointwise()");
    return;
  }

  constexpr uint64_t count = 257;
  const TensorLayout layout = TensorLayout::contiguous(&count, 1);
  DeviceTensor sample = context.allocate(layout);
  DeviceTensor velocity = context.allocate(layout);
  std::vector<float> host_sample(count), host_velocity(count), got(count);
  for (uint64_t i = 0; i < count; ++i) {
    host_sample[i] = float(int((i * 37) % 257) - 128) / 64.0f;
    host_velocity[i] = float(int((i * 53) % 193) - 96) / 128.0f;
  }
  context.upload(sample, host_sample.data(), count);
  context.upload(velocity, host_velocity.data(), count);

  sampler::FlowScheduler schedule(12.0f);
  schedule.set_timesteps(6);
  for (size_t step = 0; step < schedule.num_steps(); ++step) {
    schedule.step(static_cast<int>(step), host_sample.data(), host_velocity.data(), count,
                  host_sample.data());
    const float sigma_from_timestep = 1.0f - schedule.timesteps()[step];
    const float ratio = schedule.sigmas()[step + 1] / schedule.sigmas()[step];
    TensorBatch batch = context.begin_batch();
    batch.dit_euler_step_f32(sample, velocity, sigma_from_timestep, ratio);
    CHECK(batch.remaining_operator_capacity() == 3u);
    batch.submit().wait();
    context.download(sample, got.data(), count);
    CHECK(std::memcmp(host_sample.data(), got.data(), count * sizeof(float)) == 0);
  }

  // Validation is transactional and the same batch remains usable.
  TensorBatch transactional = context.begin_batch();
  bool alias_rejected = false;
  try {
    transactional.dit_euler_step_f32(sample, sample, 0.5f, 0.5f);
  } catch (const std::invalid_argument&) {
    alias_rejected = true;
  }
  CHECK(alias_rejected && transactional.remaining_operator_capacity() == 4u);
  transactional.dit_euler_step_f32(sample, velocity, 0.5f, 0.0f);
  transactional.submit().wait();

  // Total-domain parity, including a one-element dispatch and a 64-thread
  // tail. Exceptional source values are canonicalized on device, never read
  // back for graph-owned validation.
  const std::vector<float> exceptional_samples{0.0f,
                                               -0.0f,
                                               float_from_bits(0x00000001u),
                                               float_from_bits(0x80000001u),
                                               float_from_bits(0x7fc12345u),
                                               float_from_bits(0xffdabcdeu),
                                               float_from_bits(0x7f800000u),
                                               float_from_bits(0xff800000u),
                                               std::numeric_limits<float>::max(),
                                               -std::numeric_limits<float>::max(),
                                               1.0f,
                                               -1.0f};
  auto run_exceptional = [&](uint64_t elements, float sigma, float ratio) {
    const TensorLayout test_layout = TensorLayout::contiguous(&elements, 1);
    DeviceTensor test_sample = context.allocate(test_layout);
    DeviceTensor test_velocity = context.allocate(test_layout);
    std::vector<float> x(elements), v(elements), expected(elements), actual(elements);
    for (uint64_t i = 0; i < elements; ++i) {
      x[i] = exceptional_samples[i % exceptional_samples.size()];
      v[i] = exceptional_samples[(i * 5u + 1u) % exceptional_samples.size()];
      expected[i] = sampler::exact_euler_value(x[i], v[i], sigma, ratio);
    }
    context.upload(test_sample, x.data(), elements);
    context.upload(test_velocity, v.data(), elements);
    TensorBatch batch = context.begin_batch();
    batch.dit_euler_step_f32(test_sample, test_velocity, sigma, ratio);
    batch.submit().wait();
    context.download(test_sample, actual.data(), elements);
    CHECK(std::memcmp(expected.data(), actual.data(), elements * sizeof(float)) == 0);
  };
  run_exceptional(1u, 1.0f, 1.0f);
  for (const auto controls : {std::array<float, 2>{0.0f, 0.0f}, std::array<float, 2>{1.0f, 0.0f},
                              std::array<float, 2>{0.0f, 1.0f}, std::array<float, 2>{1.0f, 1.0f},
                              std::array<float, 2>{float_from_bits(1u), float_from_bits(1u)},
                              std::array<float, 2>{0.5f, 0.5f}})
    run_exceptional(65u, controls[0], controls[1]);
}

SLOPFAB_TEST_CATEGORY(vulkan_exact_h3_attention_single_key, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
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
  const DeviceInfo& info = physical.front().info();
  DeviceOptions disabled_options;
  disabled_options.enable_timeline_semaphore = info.timeline_semaphore;
  disabled_options.enable_shader_int64 = info.shader_int64;
  disabled_options.enable_shader_float16 = info.shader_float16;
  disabled_options.enable_storage_buffer_16bit = info.storage_buffer_16bit;
  Device disabled_device = physical.front().create_device(disabled_options);
  TensorContext disabled(disabled_device);
  CHECK(!disabled.exact_h3_attention());
  bool unavailable_rejected = false;
  try {
    (void)H3AttentionPlan::create(disabled, {1, 1, 64, exact_attention_scale(64)});
  } catch (const std::runtime_error&) {
    unavailable_rejected = true;
  }
  CHECK(unavailable_rejected);
  if (!info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 ||
      !info.storage_buffer_16bit || !info.cooperative_matrix) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 || !info.storage_buffer_16bit || !info.cooperative_matrix");
    return;
  }
  DeviceOptions options = disabled_options;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  if (!context.exact_h3_attention()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !context.exact_h3_attention()");
    return;
  }
  constexpr uint32_t dim = 64;
  const uint64_t shape[] = {1, 1, dim};
  const TensorLayout layout = TensorLayout::contiguous(shape, 3);
  DeviceTensor q = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor k = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor v = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor out = context.allocate(layout, ScalarType::kBFloat16);
  std::vector<uint16_t> zeros(dim, 0), values(dim);
  for (uint32_t i = 0; i < dim; ++i)
    values[i] = reference_bf16(float(int(i % 15) - 7) / 8.0f);
  context.upload_bytes(q, zeros.data(), zeros.size() * 2);
  context.upload_bytes(k, zeros.data(), zeros.size() * 2);
  context.upload_bytes(v, values.data(), values.size() * 2);
  H3AttentionPlan plan = H3AttentionPlan::create(context, {1, 1, dim, exact_attention_scale(dim)});
  TensorBatch batch = context.begin_batch();
  plan.record(batch, q, k, v, out);
  batch.submit().wait();
  std::vector<uint16_t> actual(dim);
  context.download_bytes(out, actual.data(), actual.size() * 2);
  CHECK(actual == values);
}
