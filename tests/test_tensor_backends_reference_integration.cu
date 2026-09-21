#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_keyframe_encoder_real_graph, "integration") {
  using namespace slopfab;
  if (!std::getenv("SLOPFAB_KEYFRAME_ENCODER_REAL")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_KEYFRAME_ENCODER_REAL\")");
    return;
  }
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !vulkan::Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !vulkan::Instance::available()");
    return;
  }
  const std::filesystem::path path = "weights/vae/minimax_h3_video_vae_fp16.safetensors";
  if (!std::filesystem::exists(path)) {
    SKIP_MISSING_FIXTURE("unavailable prerequisite: !std::filesystem::exists(path)");
    return;
  }
  const Sha256Digest expected_sha{0x7c, 0x1f, 0x13, 0x14, 0x92, 0xe7, 0xed, 0xda, 0xca, 0xac, 0x90,
                                  0x69, 0xa6, 0x1b, 0x81, 0xbd, 0xd3, 0x9d, 0xe5, 0xcc, 0x96, 0x56,
                                  0x1e, 0x67, 0x7c, 0x5e, 0xab, 0x1c, 0xdc, 0xe5, 0xe5, 0x22};
  CHECK(sha256_file(path.string()) == expected_sha);
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  const vae::EncoderWeightSummary weight_summary =
      vae::validate_keyframe_encoder_weights(checkpoint);
  CHECK(weight_summary.tensors == 118);

  vulkan::Instance instance = vulkan::Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  vulkan::DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  vulkan::Device device = physical.front().create_device(options);
  vulkan::KeyframeEncoder vk = vulkan::KeyframeEncoder::create(device);
  const auto vk_load_begin = std::chrono::steady_clock::now();
  vk.load(checkpoint);
  const double vk_load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - vk_load_begin).count();
  vae::KeyframeEncoder cu(checkpoint);

  constexpr int height = 64;
  constexpr int width = 96;
  std::vector<float> pixels(size_t(3) * height * width);
  for (size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = float(int((i * 31) % 509) - 254) / 128.0f;
  const auto cu_begin = std::chrono::steady_clock::now();
  const std::vector<float> expected = cu.encode_moments(pixels.data(), height, width);
  const double cu_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - cu_begin).count();
  const auto vk_begin = std::chrono::steady_clock::now();
  const std::vector<float> actual = vk.encode_moments(pixels.data(), height, width);
  const double vk_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - vk_begin).count();
  CHECK(expected.size() == actual.size());
  size_t mismatch = expected.size();
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::memcmp(&expected[i], &actual[i], 4) != 0) {
      mismatch = i;
      break;
    }
  }
  uint32_t cb = 0, vb = 0;
  if (mismatch != expected.size()) {
    std::memcpy(&cb, &expected[mismatch], 4);
    std::memcpy(&vb, &actual[mismatch], 4);
  }
  CHECK_MSG(mismatch == expected.size(),
            "real keyframe graph mismatch at %zu: CUDA=%08x Vulkan=%08x", mismatch, cb, vb);
  uint64_t hash = 1469598103934665603ull;
  for (float value : actual) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= (bits >> (8 * byte)) & 0xffu;
      hash *= 1099511628211ull;
    }
  }
  const vulkan::KeyframeEncoderStats stats = vk.stats();
  CHECK(hash == 0xcfd864f091297976ull);
  CHECK(stats.operators == 72);
  CHECK(stats.persistent_bytes == weight_summary.bytes);
  const uint64_t stable_reserved = stats.allocator_reserved_bytes;
  const uint64_t stable_descriptors = stats.descriptor_set_allocations;
  const std::vector<float> repeat = vk.encode_moments(pixels.data(), height, width);
  CHECK(repeat == actual);
  CHECK(vk.stats().allocator_reserved_bytes == stable_reserved);
  CHECK(vk.stats().descriptor_set_allocations == stable_descriptors);
  std::printf("  keyframe real 64x96 fnv=%016llx load=%.3fs CUDA=%.3fs Vulkan=%.3fs "
              "persistent=%.1fMiB activation=%.1fMiB used=%.1fMiB reserved=%.1fMiB desc=%llu\n",
              static_cast<unsigned long long>(hash), vk_load_seconds, cu_seconds, vk_seconds,
              stats.persistent_bytes / 1048576.0, stats.activation_bytes / 1048576.0,
              stats.allocator_used_bytes / 1048576.0, stats.allocator_reserved_bytes / 1048576.0,
              static_cast<unsigned long long>(stats.descriptor_set_allocations));
  if (const char* max_shape = std::getenv("SLOPFAB_KEYFRAME_ENCODER_MAX");
      max_shape && max_shape[0] == '1') {
    constexpr int max_height = 2048;
    constexpr int max_width = 2048;
    std::vector<float> max_pixels(size_t(3) * max_height * max_width);
    for (size_t i = 0; i < max_pixels.size(); ++i)
      max_pixels[i] = float(int((i * 29) % 509) - 254) / 128.0f;
    const auto max_cuda_begin = std::chrono::steady_clock::now();
    const std::vector<float> max_expected =
        cu.encode_moments(max_pixels.data(), max_height, max_width);
    const double max_cuda_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - max_cuda_begin).count();
    const auto max_vk_begin = std::chrono::steady_clock::now();
    const std::vector<float> max_actual =
        vk.encode_moments(max_pixels.data(), max_height, max_width);
    const double max_vk_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - max_vk_begin).count();
    CHECK(max_actual == max_expected);
    uint64_t max_hash = 1469598103934665603ull;
    const auto* max_bytes = reinterpret_cast<const uint8_t*>(max_actual.data());
    for (size_t i = 0; i < max_actual.size() * sizeof(float); ++i) {
      max_hash ^= max_bytes[i];
      max_hash *= 1099511628211ull;
    }
    CHECK(max_hash == 0x0afec53595c5874dull);
    std::printf(
        "  keyframe public 2048x2048 CUDA/Vulkan %.3f/%.3fs fnv=%016llx activation/used/reserved %.1f/%.1f/%.1fMiB\n",
        max_cuda_seconds, max_vk_seconds, static_cast<unsigned long long>(max_hash),
        vk.stats().activation_bytes / 1048576.0, vk.stats().allocator_used_bytes / 1048576.0,
        vk.stats().allocator_reserved_bytes / 1048576.0);
  }
  vk.unload();
  CHECK(!vk.loaded());
}
