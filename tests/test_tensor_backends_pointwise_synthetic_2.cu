#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_tensor_exact_vae_fused_rope, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore || !physical.front().info().shader_int64");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_normalization()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_normalization()");
    return;
  }

  constexpr uint32_t sequence = 5, heads = 3, head_dim = 64, rope_dim = 48, num_patches = 3;
  constexpr float epsilon = 1.0e-6f;
  const size_t qkv_count = static_cast<size_t>(sequence) * heads * 3 * head_dim;
  const size_t bias_count = static_cast<size_t>(heads) * 3 * head_dim;
  const size_t output_count = static_cast<size_t>(sequence) * heads * head_dim;
  std::vector<float> input = test::make_data(qkv_count, 0x13579u, 1.25f);
  std::vector<float> bias_values = test::make_data(bias_count, 0x24680u, 0.3f);
  std::vector<float> cosine(static_cast<size_t>(sequence) * rope_dim);
  std::vector<float> sine(cosine.size());
  for (uint32_t row = 0; row < sequence; ++row) {
    for (uint32_t d = 0; d < rope_dim / 2; ++d) {
      const float angle = static_cast<float>((row + 1) * (d + 3)) / 41.0f;
      const float c = std::cos(angle), s = std::sin(angle);
      cosine[static_cast<size_t>(row) * rope_dim + d] =
          cosine[static_cast<size_t>(row) * rope_dim + d + rope_dim / 2] = c;
      sine[static_cast<size_t>(row) * rope_dim + d] =
          sine[static_cast<size_t>(row) * rope_dim + d + rope_dim / 2] = s;
    }
  }

  cuda::DeviceBuffer<float> c_input(qkv_count), c_bias(bias_count), c_cos(cosine.size()),
      c_sin(sine.size()), c_q(output_count), c_k(output_count), c_v(output_count);
  c_input.copy_from_host(input.data(), input.size());
  c_bias.copy_from_host(bias_values.data(), bias_values.size());
  c_cos.copy_from_host(cosine.data(), cosine.size());
  c_sin.copy_from_host(sine.data(), sine.size());
  cuda::launch_split_qkv_norm_rope(c_input.get(), c_bias.get(), c_cos.get(), c_sin.get(), c_q.get(),
                                   c_k.get(), c_v.get(), sequence, heads, head_dim, rope_dim,
                                   num_patches, epsilon, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::array<std::vector<float>, 3> expected{std::vector<float>(output_count),
                                             std::vector<float>(output_count),
                                             std::vector<float>(output_count)};
  c_q.copy_to_host(expected[0].data(), output_count);
  c_k.copy_to_host(expected[1].data(), output_count);
  c_v.copy_to_host(expected[2].data(), output_count);

  const uint64_t qkv_shape[] = {sequence, heads, 3 * head_dim};
  const uint64_t bias_shape[] = {heads, 3 * head_dim};
  const uint64_t table_shape[] = {sequence, rope_dim};
  const uint64_t output_shape[] = {heads, sequence, head_dim};
  DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(qkv_shape, 3));
  DeviceTensor v_bias = vk.allocate(TensorLayout::contiguous(bias_shape, 2));
  DeviceTensor v_cos = vk.allocate(TensorLayout::contiguous(table_shape, 2));
  DeviceTensor v_sin = vk.allocate(TensorLayout::contiguous(table_shape, 2));
  std::array<DeviceTensor, 3> outputs{vk.allocate(TensorLayout::contiguous(output_shape, 3)),
                                      vk.allocate(TensorLayout::contiguous(output_shape, 3)),
                                      vk.allocate(TensorLayout::contiguous(output_shape, 3))};
  vk.upload(v_input, input.data(), input.size());
  vk.upload(v_bias, bias_values.data(), bias_values.size());
  vk.upload(v_cos, cosine.data(), cosine.size());
  vk.upload(v_sin, sine.data(), sine.size());
  TensorBatch batch = vk.begin_batch();
  batch.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin, outputs[0], outputs[1], outputs[2],
                                num_patches, epsilon);
  batch.submit().wait();
  std::array<std::vector<float>, 3> actual_outputs{std::vector<float>(output_count),
                                                   std::vector<float>(output_count),
                                                   std::vector<float>(output_count)};
  for (size_t output = 0; output < outputs.size(); ++output) {
    auto& actual = actual_outputs[output];
    vk.download(outputs[output], actual.data(), actual.size());
    size_t mismatch = output_count;
    for (size_t i = 0; i < output_count; ++i) {
      uint32_t want = 0, got = 0;
      std::memcpy(&want, &expected[output][i], 4);
      std::memcpy(&got, &actual[i], 4);
      if (want != got) {
        mismatch = i;
        break;
      }
    }
    CHECK_MSG(mismatch == output_count,
              "CUDA/Vulkan fused VAE RoPE output%zu mismatch at %zu: %08x != %08x",
              output, mismatch,
              mismatch == output_count ? 0u : [&] { uint32_t x; std::memcpy(&x, &expected[output][mismatch], 4); return x; }(),
              mismatch == output_count ? 0u : [&] { uint32_t x; std::memcpy(&x, &actual[mismatch], 4); return x; }());
  }
  // Independent head-major V layout reference: V is bias-only and bypasses
  // both normalization and rotation.
  for (uint32_t token = 0; token < sequence; ++token) {
    for (uint32_t head = 0; head < heads; ++head) {
      for (uint32_t d = 0; d < head_dim; ++d) {
        const size_t source =
            (static_cast<size_t>(token) * heads + head) * (3 * head_dim) + 2 * head_dim + d;
        const size_t destination = (static_cast<size_t>(head) * sequence + token) * head_dim + d;
        const float reference =
            input[source] +
            bias_values[static_cast<size_t>(head) * 3 * head_dim + 2 * head_dim + d];
        uint32_t want = 0, got = 0;
        std::memcpy(&want, &reference, 4);
        std::memcpy(&got, &actual_outputs[2][destination], 4);
        CHECK_MSG(want == got, "fused V layout mismatch token%u head%u dim%u", token, head, d);
      }
    }
  }

  std::array<DeviceTensor, 3> second_outputs{
      vk.allocate(TensorLayout::contiguous(output_shape, 3)),
      vk.allocate(TensorLayout::contiguous(output_shape, 3)),
      vk.allocate(TensorLayout::contiguous(output_shape, 3))};
  auto submit = [&](std::array<DeviceTensor, 3>& selected) {
    TensorBatch next = vk.begin_batch();
    next.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin, selected[0], selected[1],
                                 selected[2], num_patches, epsilon);
    return next.submit();
  };
  std::vector<float> suffix_cosine = cosine, suffix_sine = sine;
  for (uint32_t token = num_patches; token < sequence; ++token) {
    for (uint32_t d = 0; d < rope_dim; ++d) {
      suffix_cosine[static_cast<size_t>(token) * rope_dim + d] = static_cast<float>(17 + token + d);
      suffix_sine[static_cast<size_t>(token) * rope_dim + d] =
          static_cast<float>(-31 - static_cast<int>(token) - static_cast<int>(d));
    }
  }
  vk.upload(v_cos, suffix_cosine.data(), suffix_cosine.size());
  vk.upload(v_sin, suffix_sine.data(), suffix_sine.size());
  submit(second_outputs).wait();
  for (size_t output = 0; output < 2; ++output) {
    std::vector<float> suffix_actual(output_count);
    vk.download(second_outputs[output], suffix_actual.data(), suffix_actual.size());
    for (uint32_t head = 0; head < heads; ++head) {
      for (uint32_t token = num_patches; token < sequence; ++token) {
        const size_t begin = (static_cast<size_t>(head) * sequence + token) * head_dim;
        CHECK(std::memcmp(suffix_actual.data() + begin, actual_outputs[output].data() + begin,
                          head_dim * sizeof(float)) == 0);
      }
    }
  }
  vk.upload(v_cos, cosine.data(), cosine.size());
  vk.upload(v_sin, sine.data(), sine.size());
  Submission warm_a = submit(outputs), warm_b = submit(second_outputs);
  warm_a.wait();
  warm_b.wait();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 4; ++repeat) {
    Submission a = submit(outputs), b = submit(second_outputs), c = submit(outputs);
    CHECK(b.value() > a.value() && c.value() > b.value());
    a.wait();
    b.wait();
    c.wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  }

  const uint64_t short_table_shape[] = {sequence, rope_dim - 1};
  DeviceTensor short_table = vk.allocate(TensorLayout::contiguous(short_table_shape, 2));
  DeviceTensor wrong_type =
      vk.allocate(TensorLayout::contiguous(table_shape, 2), ScalarType::kBFloat16);
  auto valid_after_rejection = [&](auto&& invalid) {
    TensorBatch next = vk.begin_batch();
    bool rejected = false;
    try {
      invalid(next);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    CHECK(rejected);
    next.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin, outputs[0], outputs[1], outputs[2],
                                 num_patches, epsilon);
    next.submit().wait();
  };
  valid_after_rejection([&](TensorBatch& next) {
    next.split_qkv_norm_rope_f32(v_input, v_bias, short_table, v_sin, outputs[0], outputs[1],
                                 outputs[2], num_patches, epsilon);
  });
  valid_after_rejection([&](TensorBatch& next) {
    next.split_qkv_norm_rope_f32(v_input, v_bias, wrong_type, v_sin, outputs[0], outputs[1],
                                 outputs[2], num_patches, epsilon);
  });
  valid_after_rejection([&](TensorBatch& next) {
    next.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin, outputs[0], outputs[0], outputs[2],
                                 num_patches, epsilon);
  });
  valid_after_rejection([&](TensorBatch& next) {
    next.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin, outputs[0], outputs[1], outputs[2],
                                 sequence + 1, epsilon);
  });
  valid_after_rejection([&](TensorBatch& next) {
    next.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin, outputs[0], outputs[1], outputs[2],
                                 num_patches, 0.0f);
  });

  // A submitted job owns all seven buffers after every public wrapper drops.
  // Once the exact token completes and the slot is collected, no hidden
  // descriptor/scratch reference may keep any of them alive.
  const uint64_t used_before_drop = vk.pooled_used_bytes();
  Submission retained;
  {
    const uint64_t tiny_qkv_shape[] = {2, 1, 192};
    const uint64_t tiny_bias_shape[] = {1, 192};
    const uint64_t tiny_table_shape[] = {2, 48};
    const uint64_t tiny_output_shape[] = {1, 2, 64};
    DeviceTensor in = vk.allocate(TensorLayout::contiguous(tiny_qkv_shape, 3));
    DeviceTensor bv = vk.allocate(TensorLayout::contiguous(tiny_bias_shape, 2));
    DeviceTensor cv = vk.allocate(TensorLayout::contiguous(tiny_table_shape, 2));
    DeviceTensor sv = vk.allocate(TensorLayout::contiguous(tiny_table_shape, 2));
    DeviceTensor qv = vk.allocate(TensorLayout::contiguous(tiny_output_shape, 3));
    DeviceTensor kv = vk.allocate(TensorLayout::contiguous(tiny_output_shape, 3));
    DeviceTensor vv = vk.allocate(TensorLayout::contiguous(tiny_output_shape, 3));
    TensorBatch next = vk.begin_batch();
    next.split_qkv_norm_rope_f32(in, bv, cv, sv, qv, kv, vv, 1, epsilon);
    retained = next.submit();
  }
  CHECK(vk.pooled_used_bytes() > used_before_drop);
  retained.wait();
  retained = Submission{};
  {
    TensorBatch collect_completed_slot = vk.begin_batch();
  }
  CHECK(vk.pooled_used_bytes() == used_before_drop);
}
