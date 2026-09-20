#include "detail/vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(vulkan_tensor_exact_vae_norms, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  const DeviceInfo& info = physical.front().info();
  CHECK(!detail::norm_dispatch_fits(0, info.max_compute_workgroup_count[0]));
  CHECK(detail::norm_dispatch_fits(info.max_compute_workgroup_count[0],
                                   info.max_compute_workgroup_count[0]));
  CHECK(!detail::norm_dispatch_fits(
      static_cast<uint64_t>(info.max_compute_workgroup_count[0]) + 1,
      info.max_compute_workgroup_count[0]));
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  Device disabled_device = physical.front().create_device(options);
  TensorContext disabled_tensors(disabled_device);
  CHECK(!disabled_tensors.exact_normalization());
  CHECK(!disabled_tensors.exact_vae_pointwise());
  CHECK(disabled_tensors.exact_fp32_vae_normalization() ==
        disabled_tensors.exact_normalization());
  bool disabled_rejected = false;
  try { disabled_tensors.require_exact_normalization(); }
  catch (const std::runtime_error&) { disabled_rejected = true; }
  CHECK(disabled_rejected);
  options.enable_shader_int64 = info.shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext tensors(device);
  const bool expected_capability = detail::known_exact_vae_norm_device(
                                       info.vendor_id, info.device_id,
                                       info.driver_version) &&
                                   info.fp32_signed_zero_inf_nan_preserve &&
                                   info.shader_int64;
  CHECK(!detail::known_exact_vae_norm_device(0x10deu, 0x2b85u, 0x98960001u));
  CHECK(!detail::known_exact_vae_norm_device(0x10deu, 0x2b86u, 0x98960000u));
  const bool expected_pointwise = detail::known_exact_vae_pointwise_device(
                                      info.vendor_id, info.device_id,
                                      info.driver_version) &&
                                  info.fp32_signed_zero_inf_nan_preserve &&
                                  info.fp32_rounding_rte && info.shader_int64;
  CHECK(!detail::known_exact_vae_pointwise_device(
      0x10deu, 0x2b85u, 0x98960001u));
  CHECK(!detail::known_exact_vae_pointwise_device(
      0x10deu, 0x2b86u, 0x98960000u));
  CHECK(tensors.exact_vae_pointwise() == expected_pointwise);
  CHECK(tensors.exact_normalization() == expected_capability);
  CHECK(tensors.exact_fp32_vae_normalization() == tensors.exact_normalization());
  if (!expected_capability) {
    bool rejected = false;
    try { tensors.require_exact_normalization(); }
    catch (const std::runtime_error&) { rejected = true; }
    CHECK(rejected);
    return;
  }
  tensors.require_exact_normalization();
  tensors.require_exact_fp32_vae_normalization();

  const uint64_t extents[] = {2, 8};
  const uint64_t features = 8;
  const TensorLayout matrix = TensorLayout::contiguous(extents, 2);
  const TensorLayout vector = TensorLayout::contiguous(&features, 1);
  DeviceTensor input = tensors.allocate(matrix);
  DeviceTensor weight = tensors.allocate(vector);
  DeviceTensor bias = tensors.allocate(vector);
  DeviceTensor rms = tensors.allocate(matrix);
  DeviceTensor layer = tensors.allocate(matrix);
  std::vector<float> values(16), weights(8, 1.0f), biases(8, 0.0f);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = (i & 1) ? -0.5f : 0.5f;
  tensors.upload(input, values.data(), values.size());
  tensors.upload(weight, weights.data(), weights.size());
  tensors.upload(bias, biases.data(), biases.size());
  TensorBatch exact = tensors.begin_batch();
  // mean(x^2)+eps = .25+.75 = 1 and mean(x)=0, so both expected
  // results are exactly the input without relying on host sqrt behavior.
  exact.rms_norm(input, weight, rms, 0.75f);
  exact.layer_norm(input, weight, bias, layer, 0.75f);
  exact.submit().wait();
  std::vector<float> rms_host(values.size()), layer_host(values.size());
  tensors.download(rms, rms_host.data(), rms_host.size());
  tensors.download(layer, layer_host.data(), layer_host.size());
  CHECK(std::memcmp(rms_host.data(), values.data(), values.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(layer_host.data(), values.data(), values.size() * sizeof(float)) == 0);

  // RMSNorm preserves signed zero when sqrt(eps) is exactly one.
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = (i & 1) ? -0.0f : 0.0f;
  tensors.upload(input, values.data(), values.size());
  TensorBatch signed_zero = tensors.begin_batch();
  signed_zero.rms_norm(input, weight, rms, 1.0f);
  signed_zero.submit().wait();
  tensors.download(rms, rms_host.data(), rms_host.size());
  CHECK(std::memcmp(rms_host.data(), values.data(), values.size() * sizeof(float)) == 0);

  // A zero-variance LayerNorm row maps exactly to its affine bias.
  std::fill(values.begin(), values.end(), 2.0f);
  for (size_t i = 0; i < biases.size(); ++i)
    biases[i] = static_cast<float>(static_cast<int>(i) - 4) * 0.25f;
  tensors.upload(input, values.data(), values.size());
  tensors.upload(bias, biases.data(), biases.size());
  TensorBatch constant = tensors.begin_batch();
  constant.layer_norm(input, weight, bias, layer, 1.0e-6f);
  constant.submit().wait();
  tensors.download(layer, layer_host.data(), layer_host.size());
  for (size_t row = 0; row < 2; ++row) {
    CHECK(std::memcmp(layer_host.data() + row * 8, biases.data(),
                      biases.size() * sizeof(float)) == 0);
  }

  bool subnormal_epsilon_rejected = false;
  try {
    TensorBatch invalid = tensors.begin_batch();
    invalid.rms_norm(input, weight, rms, std::numeric_limits<float>::denorm_min());
  } catch (const std::invalid_argument&) {
    subnormal_epsilon_rejected = true;
  }
  CHECK(subnormal_epsilon_rejected);
}

SLOPFAB_TEST_CATEGORY(vulkan_yuv420_output, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance probe = Instance::create();
  const auto devices = probe.enumerate_devices();
  if (devices.empty() || !devices.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: devices.empty() || !devices.front().info().timeline_semaphore");
    return;
  }

  Yuv420Converter converter;
  bool unavailable_device_rejected = false;
  try {
    Yuv420Converter unavailable(std::numeric_limits<uint32_t>::max());
  } catch (const std::runtime_error& error) {
    unavailable_device_rejected = std::string(error.what()).find("unavailable") != std::string::npos;
  }
  CHECK(unavailable_device_rejected);
  struct Extent { int width; int height; };
  const Extent extents[] = {{2, 2}, {10, 6}, {128, 66}, {1280, 768}};
  uint64_t previous_high_water = 0;
  for (const Extent extent : extents) {
    const size_t pixels = static_cast<size_t>(extent.width) * extent.height;
    std::vector<float> r(pixels), g(pixels), b(pixels);
    for (size_t i = 0; i < pixels; ++i) {
      r[i] = static_cast<float>((i * 17) % 113) / 97.0f - 0.08f;
      g[i] = static_cast<float>((i * 29 + 3) % 127) / 109.0f;
      b[i] = static_cast<float>((i * 43 + 11) % 139) / 101.0f - 0.12f;
    }
    size_t boundary_block = 0;
    for (int by = 0; by < extent.height && boundary_block < 96; by += 2) {
      for (int bx = 0; bx < extent.width && boundary_block < 96; bx += 2, ++boundary_block) {
        const float boundary =
            (static_cast<float>(96 + (boundary_block % 64)) + 0.5f - 128.0f) / 112.0f;
        const int direction = static_cast<int>(boundary_block % 3) - 1;
        const float value = direction < 0 ? std::nextafter(boundary, -INFINITY)
                            : direction > 0 ? std::nextafter(boundary, INFINITY)
                                            : boundary;
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            const size_t i = static_cast<size_t>(by + dy) * extent.width + bx + dx;
            r[i] = (boundary_block & 1) != 0 ? value : 0.0f;
            g[i] = 0.0f;
            b[i] = (boundary_block & 1) == 0 ? value : 0.0f;
          }
        }
      }
    }
    // Keep luma half-step fixtures at the opposite end of larger frames so
    // the chroma-boundary blocks above cannot overwrite them.
    if (pixels >= 512) {
      for (size_t fixture = 0; fixture < 192; ++fixture) {
        const size_t i = pixels - 192 + fixture;
        const float boundary =
            (static_cast<float>(16 + (fixture % 220)) + 0.5f - 16.0f) / 219.0f;
        const int direction = static_cast<int>(fixture % 3) - 1;
        const float value = direction < 0 ? std::nextafter(boundary, -INFINITY)
                            : direction > 0 ? std::nextafter(boundary, INFINITY)
                                            : boundary;
        r[i] = value;
        g[i] = value;
        b[i] = value;
      }
    }
    const int ys = extent.width + 13;
    const int cs = extent.width / 2 + 7;
    std::vector<uint8_t> cpu_y(static_cast<size_t>(ys) * extent.height, 0xa5);
    std::vector<uint8_t> cpu_u(static_cast<size_t>(cs) * (extent.height / 2), 0xa5);
    std::vector<uint8_t> cpu_v(cpu_u.size(), 0xa5);
    std::vector<uint8_t> vk_y(cpu_y.size(), 0xa5), vk_u(cpu_u.size(), 0xa5), vk_v(cpu_v.size(), 0xa5);
    video::rgb_frame_to_yuv420(r.data(), g.data(), b.data(), extent.height, extent.width,
                               cpu_y.data(), ys, cpu_u.data(), cs, cpu_v.data(), cs);
    converter.convert(r.data(), g.data(), b.data(), extent.height, extent.width,
                      vk_y.data(), ys, vk_u.data(), cs, vk_v.data(), cs);

    int differences = 0;
    int max_difference = 0;
    auto compare_plane = [&](const std::vector<uint8_t>& expected,
                             const std::vector<uint8_t>& actual, int rows, int columns,
                             int stride) {
      for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < columns; ++x) {
          const int delta = std::abs(static_cast<int>(expected[static_cast<size_t>(y) * stride + x]) -
                                     static_cast<int>(actual[static_cast<size_t>(y) * stride + x]));
          if (delta != 0) ++differences;
          max_difference = std::max(max_difference, delta);
        }
        for (int x = columns; x < stride; ++x) {
          CHECK(actual[static_cast<size_t>(y) * stride + x] == 0xa5);
        }
      }
    };
    compare_plane(cpu_y, vk_y, extent.height, extent.width, ys);
    compare_plane(cpu_u, vk_u, extent.height / 2, extent.width / 2, cs);
    compare_plane(cpu_v, vk_v, extent.height / 2, extent.width / 2, cs);
    CHECK_MSG(differences == 0,
              "Vulkan YUV differs by %d (changed samples %d) at %dx%d",
              max_difference, differences, extent.width, extent.height);
    CHECK(converter.capacity_pixels() >= pixels);
    CHECK(converter.high_water_bytes() >= previous_high_water);
    previous_high_water = converter.high_water_bytes();

    const uint64_t reserved = converter.reserved_bytes();
    converter.convert(r.data(), g.data(), b.data(), extent.height, extent.width,
                      vk_y.data(), ys, vk_u.data(), cs, vk_v.data(), cs);
    CHECK(converter.reserved_bytes() == reserved);
  }
  CHECK_MSG(converter.high_water_bytes() <= (48ull << 20),
            "Vulkan YUV persistent high-water is %llu bytes",
            static_cast<unsigned long long>(converter.high_water_bytes()));

  const uint64_t capacity_before_reject = converter.capacity_pixels();
  const uint64_t reserved_before_reject = converter.reserved_bytes();
  bool index_overflow_rejected = false;
  uint8_t reject_byte = 0;
  float reject_sample = 0.0f;
  try {
    converter.convert(&reject_sample, &reject_sample, &reject_sample, 32768, 65536,
                      &reject_byte, 65536, &reject_byte, 32768, &reject_byte, 32768);
  } catch (const std::overflow_error&) {
    index_overflow_rejected = true;
  }
  CHECK(index_overflow_rejected);
  CHECK(converter.capacity_pixels() == capacity_before_reject);
  CHECK(converter.reserved_bytes() == reserved_before_reject);

  bool odd_rejected = false;
  uint8_t byte = 0;
  float sample = 0;
  try {
    converter.convert(&sample, &sample, &sample, 2, 3, &byte, 3, &byte, 1, &byte, 1);
  } catch (const std::invalid_argument&) {
    odd_rejected = true;
  }
  CHECK(odd_rejected);

  bool null_rejected = false;
  try {
    converter.convert(nullptr, &sample, &sample, 2, 2, &byte, 2, &byte, 1, &byte, 1);
  } catch (const std::invalid_argument&) {
    null_rejected = true;
  }
  CHECK(null_rejected);

  bool stride_rejected = false;
  try {
    converter.convert(&sample, &sample, &sample, 2, 2, &byte, 1, &byte, 1, &byte, 1);
  } catch (const std::invalid_argument&) {
    stride_rejected = true;
  }
  CHECK(stride_rejected);

  const int frames = 3;
  const int y4m_width = 10;
  const int y4m_height = 6;
  const size_t frame_pixels = static_cast<size_t>(y4m_width) * y4m_height;
  PixelBuffer clip(static_cast<size_t>(3) * frames * frame_pixels);
  const size_t plane = static_cast<size_t>(frames) * frame_pixels;
  for (int frame = 0; frame < frames; ++frame) {
    for (size_t i = 0; i < frame_pixels; ++i) {
      const size_t offset = static_cast<size_t>(frame) * frame_pixels + i;
      clip[offset] = static_cast<float>((i * 11 + frame * 7) % 101) / 100.0f;
      clip[plane + offset] = static_cast<float>((i * 23 + frame * 5) % 103) / 102.0f;
      clip[2 * plane + offset] = static_cast<float>((i * 37 + frame * 3) % 107) / 106.0f;
    }
  }
  const auto cpu_path = std::filesystem::temp_directory_path() / "slopfab_vulkan_cpu.y4m";
  const auto vk_path = std::filesystem::temp_directory_path() / "slopfab_vulkan_output.y4m";
  std::filesystem::remove(cpu_path);
  std::filesystem::remove(vk_path);
  video::write_y4m(cpu_path.string(), clip, frames, y4m_height, y4m_width);
  const uint64_t before_y4m = converter.reserved_bytes();
  video::write_y4m(vk_path.string(), clip, frames, y4m_height, y4m_width, {}, &converter);
  CHECK(read_bytes(cpu_path) == read_bytes(vk_path));
  CHECK(converter.reserved_bytes() == before_y4m);
  std::filesystem::remove(cpu_path);
  std::filesystem::remove(vk_path);
}

SLOPFAB_TEST_CATEGORY(vulkan_video_vae_decoder_contract, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
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

  vae::ViTConfig shipped;
  bool shipped_rejected = false;
  try {
    (void)VideoVaeDecoder::create(device, shipped);
  } catch (const std::invalid_argument&) {
    shipped_rejected = true;
  }
  CHECK(shipped_rejected);

  vae::ViTConfig too_many;
  too_many.transformer_mode = vae::ViTTransformerMode::kExact;
  too_many.num_layers = 205;  // 15 + 20*205 > the 4096-op transaction.
  bool capacity_rejected = false;
  try {
    (void)VideoVaeDecoder::create(device, too_many);
  } catch (const std::invalid_argument&) {
    capacity_rejected = true;
  }
  CHECK(capacity_rejected);

  vae::ViTConfig exact;
  exact.transformer_mode = vae::ViTTransformerMode::kExact;
  VideoVaeDecoder decoder;
  try {
    decoder = VideoVaeDecoder::create(device, exact);
  } catch (const std::runtime_error&) {
    // The exact arithmetic allow-list is intentionally narrower than Vulkan
    // availability. The two validation checks above are device-independent.
    SKIP_UNSUPPORTED_HARDWARE("device does not support exact codec arithmetic");
    return;
  }
  CHECK(decoder.operators_per_document() == 735);

  std::vector<float> normalized(size_t(exact.in_channels) * 7, 0.0f);
  bool unloaded_decode_rejected = false;
  try {
    (void)decoder.decode(normalized.data(), 7, 1, 1,
                         vae::default_video_latents_mean(),
                         vae::default_video_latents_std());
  } catch (const std::logic_error&) {
    unloaded_decode_rejected = true;
  }
  CHECK(unloaded_decode_rejected);

  vae::ViTConfig one_layer = exact;
  one_layer.num_layers = 1;
  VideoVaeDecoder minimum = VideoVaeDecoder::create(device, one_layer);
  CHECK(minimum.operators_per_document() == 35);
  vae::ViTConfig boundary = exact;
  boundary.num_layers = 204;
  VideoVaeDecoder maximum = VideoVaeDecoder::create(device, boundary);
  CHECK(maximum.operators_per_document() == 4095);
}

SLOPFAB_TEST_CATEGORY(vulkan_audio_vae_decoder_cuda_off_contract, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore || !physical.front().info().shader_int64");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);

  AudioDecoder decoder = AudioDecoder::create(device);
  CHECK(decoder.recorded_operators() == 497u);
  CHECK(decoder.weight_bytes() == 0u);
  CHECK(decoder.peak_device_bytes() == 0u);
  std::vector<float> latent(64, 0.0f);
  bool unloaded_decode_rejected = false;
  try {
    (void)decoder.decode(latent.data(), 1);
  } catch (const std::logic_error&) {
    unloaded_decode_rejected = true;
  }
  CHECK(unloaded_decode_rejected);
  decoder.unload();
  CHECK(decoder.weight_bytes() == 0u);

  if (!std::getenv("SLOPFAB_AUDIO_DECODER_REAL")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_AUDIO_DECODER_REAL\")");
    return;
  }
  const char* configured_path = std::getenv("SLOPFAB_AUDIO_VAE_PATH");
  const std::filesystem::path checkpoint_path = configured_path != nullptr
      ? configured_path
      : "weights/vae/minimax_h3_audio_vae_fp32.safetensors";
  if (!std::filesystem::exists(checkpoint_path)) {
    SKIP_MISSING_FIXTURE("unavailable prerequisite: !std::filesystem::exists(checkpoint_path)");
    return;
  }
  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  CHECK(checkpoint.file_size() == 605254808u);
  CHECK(checkpoint.tensor_count() == 917u);
  decoder.load(checkpoint);
  CHECK(decoder.recorded_operators() == 497u);
  CHECK(decoder.weight_bytes() == 259672032u);
  constexpr int latent_length = 3;
  std::vector<float> real_latent(size_t(2) * 32 * latent_length);
  for (size_t i = 0; i < real_latent.size(); ++i)
    real_latent[i] = float(int((i * 67) % 607) - 303) / 128.0f;
  const vae::DecodedAudio first = decoder.decode(real_latent.data(),
                                                  latent_length);
  const uint64_t first_digest = fnv64_floats(first.samples);
  CHECK(first_digest == 0x528f17a83d5ef7eeull);
  const uint64_t stable_reserved = decoder.allocator_reserved_bytes();
  const uint64_t stable_descriptors = decoder.descriptor_set_allocations();
  const vae::DecodedAudio repeat = decoder.decode(real_latent.data(),
                                                   latent_length);
  CHECK(first.samples == repeat.samples);
  CHECK(fnv64_floats(repeat.samples) == first_digest);
  CHECK(decoder.allocator_reserved_bytes() == stable_reserved);
  CHECK(decoder.descriptor_set_allocations() == stable_descriptors);
  std::printf("  CUDA-off real Vulkan audio A3 FNV64 %016llx\n",
              static_cast<unsigned long long>(first_digest));
  decoder.unload();
  CHECK(decoder.weight_bytes() == 0u);
  CHECK(decoder.peak_device_bytes() == 0u);
  unloaded_decode_rejected = false;
  try {
    (void)decoder.decode(real_latent.data(), latent_length);
  } catch (const std::logic_error&) {
    unloaded_decode_rejected = true;
  }
  CHECK(unloaded_decode_rejected);
}
