#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(reference_conditioning_aggregate_fails_before_model_load, "synthetic") {
  using namespace slopfab;
  const std::string unique = std::to_string(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  const std::filesystem::path image_path =
      std::filesystem::temp_directory_path() /
      ("slopfab-reference-preflight-" + unique + ".ppm");
  {
    std::ofstream ppm(image_path, std::ios::binary);
    const std::array<uint8_t, 3> pixel{0x17, 0x83, 0xd1};
    ppm << "P6\n1 1\n255\n";
    ppm.write(reinterpret_cast<const char*>(pixel.data()), pixel.size());
    CHECK(static_cast<bool>(ppm));
  }

  GenerateRequest request;
  request.prompt = "two references must fail before any model opens";
  request.canvas_width = 256;
  request.canvas_height = 256;
  request.num_frames = 22;
  request.num_inference_steps = 4;
  request.reference_image_paths = {image_path.string(), image_path.string()};
  request.tokenizer_path =
      (std::filesystem::path(SLOPFAB_TEST_SOURCE_DIR) /
       "ref/text_encoder/tokenizer.json").string();
  request.text_encoder_path = "missing-text-encoder.safetensors";
  request.transformer_path = "missing-transformer.safetensors";
  request.video_vae_path = "missing-video-vae.safetensors";
  request.audio_vae_path = "missing-audio-vae.safetensors";
  const GeneratePlan plan = resolve_plan(request);
  RunOptions options;
  options.inference_backend = DeviceBackend::kVulkan;
  options.attention_mode = AttentionMode::kExact;
  options.verbose = false;
  const RunResult result = run_generate(request, plan, options);
  CHECK(!result.ok && !result.cancelled);
  CHECK(result.message.find("conditioning exceeds max prompt tokens") !=
        std::string::npos);
  CHECK(result.seconds_conditioning == 0.0 && result.steps_computed == 0);

  std::error_code ignored;
  std::filesystem::remove(image_path, ignored);
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_keyframe_conv3d_exact, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_vae_pointwise()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_vae_pointwise()");
    return;
  }

  auto run_case = [&](uint32_t cin, uint32_t cout, uint32_t height,
                      uint32_t width, uint32_t kernel, uint32_t stride,
                      bool reflect, bool asymmetric) {
    const uint32_t oh = stride == 2 ? height / 2 : height;
    const uint32_t ow = stride == 2 ? width / 2 : width;
    const size_t input_count = static_cast<size_t>(cin) * height * width;
    const size_t weight_count = static_cast<size_t>(cout) * cin * kernel * kernel * kernel;
    const size_t output_count = static_cast<size_t>(cout) * oh * ow;
    std::vector<float> input(input_count);
    std::vector<__half> weight(weight_count), bias(cout);
    std::vector<uint16_t> weight_bits(weight_count), bias_bits(cout);
    for (size_t i = 0; i < input_count; ++i)
      input[i] = static_cast<float>(static_cast<int>((i * 17) % 71) - 35) / 32.0f;
    for (size_t i = 0; i < weight_count; ++i) {
      weight[i] = __float2half(
          static_cast<float>(static_cast<int>((i * 13) % 37) - 18) / 128.0f);
      std::memcpy(&weight_bits[i], &weight[i], 2);
    }
    for (uint32_t i = 0; i < cout; ++i) {
      bias[i] = __float2half(static_cast<float>(static_cast<int>(i) - 2) / 16.0f);
      std::memcpy(&bias_bits[i], &bias[i], 2);
    }

    cuda::DeviceBuffer<float> c_input(input_count), c_output(output_count);
    cuda::DeviceBuffer<__half> c_weight(weight_count), c_bias(cout);
    c_input.copy_from_host(input.data(), input_count);
    c_weight.copy_from_host(weight.data(), weight_count);
    c_bias.copy_from_host(bias.data(), cout);
    cuda::launch_keyframe_conv3d(
        c_input.get(), c_weight.get(), c_bias.get(), c_output.get(),
        static_cast<int>(cin), static_cast<int>(cout), static_cast<int>(height),
        static_cast<int>(width), static_cast<int>(kernel), static_cast<int>(stride),
        reflect, asymmetric, nullptr);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> expected(output_count), actual(output_count);
    c_output.copy_to_host(expected.data(), output_count);

    const uint64_t input_shape[] = {cin, height, width};
    const uint64_t weight_shape[] = {cout, cin, kernel, kernel, kernel};
    const uint64_t bias_shape[] = {cout};
    const uint64_t output_shape[] = {cout, oh, ow};
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(input_shape, 3));
    DeviceTensor v_weight = vk.allocate(
        TensorLayout::contiguous(weight_shape, 5), ScalarType::kFloat16);
    DeviceTensor v_bias = vk.allocate(
        TensorLayout::contiguous(bias_shape, 1), ScalarType::kFloat16);
    DeviceTensor v_output = vk.allocate(TensorLayout::contiguous(output_shape, 3));
    vk.upload(v_input, input.data(), input_count);
    vk.upload_bytes(v_weight, weight_bits.data(), weight_bits.size() * 2);
    vk.upload_bytes(v_bias, bias_bits.data(), bias_bits.size() * 2);
    TensorBatch batch = vk.begin_batch();
    batch.keyframe_conv3d_f16(v_input, v_weight, v_bias, v_output,
                              cin, cout, height, width, kernel, stride,
                              reflect, asymmetric);
    batch.submit().wait();
    vk.download(v_output, actual.data(), actual.size());
    size_t mismatch = output_count;
    for (size_t i = 0; i < output_count; ++i) {
      if (std::memcmp(&expected[i], &actual[i], 4) != 0) {
        mismatch = i;
        break;
      }
    }
    uint32_t eb = 0, ab = 0;
    if (mismatch != output_count) {
      std::memcpy(&eb, &expected[mismatch], 4);
      std::memcpy(&ab, &actual[mismatch], 4);
    }
    CHECK_MSG(mismatch == output_count,
              "keyframe Conv3D C%u->%u %ux%u k%u/s%u mismatch at %zu: %08x != %08x",
              cin, cout, height, width, kernel, stride, mismatch, eb, ab);
  };

  run_case(3, 5, 5, 7, 3, 1, true, false);    // 175-element tail.
  run_case(5, 7, 6, 10, 3, 2, false, true);   // constant right/bottom pad.
  run_case(7, 3, 3, 5, 1, 1, true, false);    // checkpoint 1x1 shortcut.

  // Reject contradictory asymmetric+reflect semantics without consuming the
  // batch; the same batch remains recordable and exact afterward.
  const uint64_t input_shape[] = {1, 4, 6};
  const uint64_t weight_shape[] = {1, 1, 3, 3, 3};
  const uint64_t bias_shape[] = {1};
  const uint64_t output_shape[] = {1, 2, 3};
  DeviceTensor input = vk.allocate(TensorLayout::contiguous(input_shape, 3));
  DeviceTensor weight = vk.allocate(
      TensorLayout::contiguous(weight_shape, 5), ScalarType::kFloat16);
  DeviceTensor bias = vk.allocate(
      TensorLayout::contiguous(bias_shape, 1), ScalarType::kFloat16);
  DeviceTensor output = vk.allocate(TensorLayout::contiguous(output_shape, 3));
  TensorBatch batch = vk.begin_batch();
  bool rejected = false;
  try {
    batch.keyframe_conv3d_f16(input, weight, bias, output, 1, 1, 4, 6,
                              3, 2, true, true);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  batch.keyframe_conv3d_f16(input, weight, bias, output, 1, 1, 4, 6,
                            3, 2, false, true);
  batch.submit().wait();
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_keyframe_groupnorm_large_divisor, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_fp32_vae_normalization()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_fp32_vae_normalization()");
    return;
  }

  constexpr uint32_t channels = 4, height = 4097, width = 1024, groups = 1;
  constexpr size_t count = size_t(channels) * height * width;
  static_assert(count > (size_t{1} << 24));
  std::vector<float> values(count), expected(count), actual(count);
  for (size_t i = 0; i < count; ++i)
    values[i] = float(int((i * 23) % 257) - 128) / 64.0f;
  std::vector<__half> affine(channels, __float2half(1.0f));
  std::vector<__half> offsets(channels, __float2half(0.0f));
  std::vector<uint16_t> affine_bits(channels), offset_bits(channels);
  std::memcpy(affine_bits.data(), affine.data(), channels * 2);
  std::memcpy(offset_bits.data(), offsets.data(), channels * 2);
  cuda::DeviceBuffer<float> c_input(count), c_output(count);
  cuda::DeviceBuffer<__half> c_weight(channels), c_bias(channels);
  c_input.copy_from_host(values.data(), count);
  c_weight.copy_from_host(affine.data(), channels);
  c_bias.copy_from_host(offsets.data(), channels);
  cuda::launch_keyframe_groupnorm_silu(
      c_input.get(), c_weight.get(), c_bias.get(), c_output.get(), channels,
      height, width, groups, 1.0e-6f, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  c_output.copy_to_host(expected.data(), count);

  const uint64_t flat = count, feature = channels;
  DeviceTensor input = vk.allocate(TensorLayout::contiguous(&flat, 1));
  DeviceTensor output = vk.allocate(TensorLayout::contiguous(&flat, 1));
  DeviceTensor weight = vk.allocate(
      TensorLayout::contiguous(&feature, 1), ScalarType::kFloat16);
  DeviceTensor bias = vk.allocate(
      TensorLayout::contiguous(&feature, 1), ScalarType::kFloat16);
  vk.upload_transient(input, values.data(), values.size());
  vk.upload_transient_bytes(weight, affine_bits.data(), channels * 2);
  vk.upload_transient_bytes(bias, offset_bits.data(), channels * 2);
  TensorBatch batch = vk.begin_batch();
  batch.keyframe_group_norm_silu_f16_affine(
      input, weight, bias, output, channels, height, width, groups, 1.0e-6f);
  batch.submit().wait();
  vk.download(output, actual.data(), actual.size());
  size_t mismatch = count;
  for (size_t i = 0; i < count; ++i) {
    if (std::memcmp(&expected[i], &actual[i], 4) != 0) {
      mismatch = i;
      break;
    }
  }
  CHECK_MSG(mismatch == count,
            "large keyframe GroupNorm mismatch at %zu of %zu", mismatch, count);
}
