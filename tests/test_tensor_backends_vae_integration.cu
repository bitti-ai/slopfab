#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_exact_vae_vit_decoder_integration, "integration") {
  using namespace slopfab;
  vae::ViTConfig default_config;
  CHECK(default_config.transformer_mode == vae::ViTTransformerMode::kShipped);
  if (!std::getenv("SLOPFAB_VAE_VIT_DECODER_REAL")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_VAE_VIT_DECODER_REAL\")");
    return;
  }
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0");
    return;
  }
  const std::filesystem::path checkpoint_path =
      "weights/vae/minimax_h3_video_vae_fp16.safetensors";
  if (!std::filesystem::exists(checkpoint_path)) {
    SKIP_MISSING_FIXTURE("unavailable prerequisite: !std::filesystem::exists(checkpoint_path)");
    return;
  }
  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_checkpoint_sha{
      0x7c, 0x1f, 0x13, 0x14, 0x92, 0xe7, 0xed, 0xda,
      0xca, 0xac, 0x90, 0x69, 0xa6, 0x1b, 0x81, 0xbd,
      0xd3, 0x9d, 0xe5, 0xcc, 0x96, 0x56, 0x1e, 0x67,
      0x7c, 0x5e, 0xab, 0x1c, 0xdc, 0xe5, 0xe5, 0x22};
  CHECK(sha256_mapping(checkpoint.mapping_base(), checkpoint.file_size()) ==
        expected_checkpoint_sha);
#endif
  vae::ViTConfig config;
  config.transformer_mode = vae::ViTTransformerMode::kExact;
  vae::ViTDecoder decoder;
  const auto load_begin = std::chrono::steady_clock::now();
  decoder.load(checkpoint, config);
  const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();
  CHECK(decoder.config().transformer_mode == vae::ViTTransformerMode::kExact);
  std::vector<float> latent(size_t(config.in_channels) * 7 * 16 * 16);
  for (size_t i = 0; i < latent.size(); ++i)
    latent[i] = float(int((i * 37) % 509) - 254) / 512.0f;
  std::vector<float> first, repeat;
  const auto forward_begin = std::chrono::steady_clock::now();
  decoder.forward_window(latent.data(), 7, 16, 16, first);
  const double forward_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - forward_begin).count();
  decoder.forward_window(latent.data(), 7, 16, 16, repeat);
  CHECK(first == repeat);
  const size_t loaded_weight_bytes = decoder.weight_bytes();
  std::vector<float> ragged_latent(size_t(config.in_channels) * 7 * 8 * 16);
  for (size_t i = 0; i < ragged_latent.size(); ++i)
    ragged_latent[i] = float(int((i * 41) % 509) - 254) / 512.0f;
  std::vector<float> ragged, first_after_ragged;
  decoder.forward_window(ragged_latent.data(), 7, 8, 16, ragged);
  decoder.forward_window(latent.data(), 7, 16, 16, first_after_ragged);
  CHECK(first == first_after_ragged);
  CHECK(decoder.weight_bytes() == loaded_weight_bytes);
  uint64_t digest = 1469598103934665603ull;
  for (float value : first) {
    uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits));
    for (int byte = 0; byte < 4; ++byte) {
      digest ^= (bits >> (8 * byte)) & 0xffu;
      digest *= 1099511628211ull;
    }
  }
  CHECK(first.size() == 5505024u);
  CHECK(ragged.size() == 2752512u);
  CHECK(digest == 0x4d84e832e07db0a8ull);
  if (vulkan::Instance::available()) {
    vulkan::Instance instance = vulkan::Instance::create();
    const auto physical = instance.enumerate_devices();
    if (!physical.empty() && physical.front().info().timeline_semaphore) {
      vulkan::DeviceOptions options;
      options.enable_timeline_semaphore = true;
      options.enable_shader_int64 = physical.front().info().shader_int64;
      vulkan::Device device = physical.front().create_device(options);
      vulkan::VideoVaeDecoder vk_decoder =
          vulkan::VideoVaeDecoder::create(device, config);
      const auto vk_load_begin = std::chrono::steady_clock::now();
      vk_decoder.load(checkpoint);
      const double vk_load_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - vk_load_begin).count();
      const uint64_t loaded_accounted = vk_decoder.peak_device_bytes();
      const uint64_t loaded_used = vk_decoder.allocator_used_bytes();
      std::vector<float> vk_first, vk_ragged, vk_first_again;
      const auto vk_forward_begin = std::chrono::steady_clock::now();
      vk_decoder.forward_window(latent.data(), 7, 16, 16, vk_first);
      const double vk_forward_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - vk_forward_begin).count();
      const uint64_t one_shape_accounted = vk_decoder.peak_device_bytes();
      const uint64_t one_shape_used = vk_decoder.allocator_used_bytes();
      vk_decoder.forward_window(ragged_latent.data(), 7, 8, 16, vk_ragged);
      const uint64_t two_shape_accounted = vk_decoder.peak_device_bytes();
      const uint64_t two_shape_used = vk_decoder.allocator_used_bytes();
      const uint64_t one_accounted_delta = one_shape_accounted - loaded_accounted;
      const uint64_t one_used_delta = one_shape_used - loaded_used;
      const uint64_t two_accounted_delta = two_shape_accounted - loaded_accounted;
      const uint64_t two_used_delta = two_shape_used - loaded_used;
      CHECK(one_accounted_delta <= one_used_delta);
      CHECK(two_accounted_delta <= two_used_delta);
      CHECK_MSG(one_used_delta <= one_accounted_delta + (32ull << 20),
                "one-shape allocator delta exceeds accounting by %.1f MiB",
                double(one_used_delta) / 1048576.0 -
                    double(one_accounted_delta) / 1048576.0);
      CHECK_MSG(two_used_delta <= two_accounted_delta + (32ull << 20),
                "two-shape allocator delta exceeds accounting by %.1f MiB",
                double(two_used_delta) / 1048576.0 -
                    double(two_accounted_delta) / 1048576.0);
      vk_decoder.forward_window(latent.data(), 7, 16, 16, vk_first_again);
      size_t window_mismatch = first.size();
      for (size_t i = 0; i < first.size(); ++i) {
        if (std::memcmp(&first[i], &vk_first[i], sizeof(float)) != 0) {
          window_mismatch = i;
          break;
        }
      }
      if (window_mismatch != first.size()) {
        uint32_t cb = 0, vb = 0;
        std::memcpy(&cb, &first[window_mismatch], 4);
        std::memcpy(&vb, &vk_first[window_mismatch], 4);
        std::printf("  window mismatch %zu CUDA=%08x Vulkan=%08x\n",
                    window_mismatch, cb, vb);
      }
      CHECK(std::memcmp(first.data(), vk_first.data(), first.size() * 4) == 0);
      CHECK(std::memcmp(ragged.data(), vk_ragged.data(), ragged.size() * 4) == 0);
      CHECK(vk_first == vk_first_again);
      CHECK(vk_decoder.cached_shapes() == 2);
      const uint64_t stable_reserved = vk_decoder.allocator_reserved_bytes();
      const uint64_t stable_descriptors =
          vk_decoder.descriptor_set_allocations();
      vk_decoder.forward_window(latent.data(), 7, 16, 16, vk_first_again);
      CHECK(vk_decoder.allocator_reserved_bytes() == stable_reserved);
      CHECK(vk_decoder.descriptor_set_allocations() == stable_descriptors);

      // Six equal-shape documents exceed one 4096-op transaction (735 ops per
      // document), so this exercises the production 5+1 split and nontrivial
      // output-slot mapping against the exact CUDA implementation.
      constexpr int kSplitBatch = 6;
      std::vector<float> split_latent(size_t(kSplitBatch) * config.in_channels * 7);
      for (size_t i = 0; i < split_latent.size(); ++i)
        split_latent[i] = float(int((i * 53) % 251) - 125) / 256.0f;
      const std::array<size_t, kSplitBatch> split_slots{5, 0, 4, 1, 3, 2};
      std::vector<std::vector<float>> cuda_split(kSplitBatch), vk_split(kSplitBatch);
      const uint64_t before_bad_peak = vk_decoder.peak_device_bytes();
      const uint64_t before_bad_reserved = vk_decoder.allocator_reserved_bytes();
      const uint64_t before_bad_descriptors =
          vk_decoder.descriptor_set_allocations();
      const uint32_t before_bad_shapes = vk_decoder.cached_shapes();
      const std::array<size_t, 2> bad_slots{0, kSplitBatch};
      bool rejected_bad_slots = false;
      try {
        vk_decoder.forward_windows(split_latent.data(), 2, 7, 2, 2,
                                   vk_split, bad_slots.data());
      } catch (const std::out_of_range&) {
        rejected_bad_slots = true;
      }
      CHECK(rejected_bad_slots);
      CHECK(vk_decoder.peak_device_bytes() == before_bad_peak);
      CHECK(vk_decoder.allocator_reserved_bytes() == before_bad_reserved);
      CHECK(vk_decoder.descriptor_set_allocations() == before_bad_descriptors);
      CHECK(vk_decoder.cached_shapes() == before_bad_shapes);
      decoder.forward_windows(split_latent.data(), kSplitBatch, 7, 1, 1,
                              cuda_split, split_slots.data());
      decoder.release_host_registrations();
      vk_decoder.forward_windows(split_latent.data(), kSplitBatch, 7, 1, 1,
                                 vk_split, split_slots.data());
      for (size_t slot = 0; slot < kSplitBatch; ++slot) {
        CHECK(cuda_split[slot].size() == vk_split[slot].size());
        CHECK(std::memcmp(cuda_split[slot].data(), vk_split[slot].data(),
                          cuda_split[slot].size() * sizeof(float)) == 0);
      }
      CHECK(vk_decoder.cached_shapes() == 2);

      // Exercise the shared temporal scheduler, device latent de-normalize,
      // stitch/pixel de-normalize, and the exact Y4M output boundary.
      std::vector<float> normalized(size_t(config.in_channels) * 7);
      for (size_t i = 0; i < normalized.size(); ++i)
        normalized[i] = float(int((i * 29) % 127) - 63) / 128.0f;
      vae::DecodeSchedule decode_schedule;
      decode_schedule.tiling_enabled = false;
      const vae::DecodedVideo cuda_video = decoder.decode(
          normalized.data(), 7, 1, 1, vae::default_video_latents_mean(),
          vae::default_video_latents_std(), decode_schedule);
      const vae::DecodedVideo vk_video = vk_decoder.decode(
          normalized.data(), 7, 1, 1, vae::default_video_latents_mean(),
          vae::default_video_latents_std(), decode_schedule);
      CHECK(cuda_video.channels == vk_video.channels);
      CHECK(cuda_video.frames == vk_video.frames);
      CHECK(cuda_video.height == vk_video.height);
      CHECK(cuda_video.width == vk_video.width);
      CHECK(cuda_video.data.size() == vk_video.data.size());
      CHECK(std::memcmp(cuda_video.data.data(), vk_video.data.data(),
                        cuda_video.data.size() * sizeof(float)) == 0);
      uint64_t pixel_digest = 1469598103934665603ull;
      for (float value : vk_video.data) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
          pixel_digest ^= (bits >> (8 * byte)) & 0xffu;
          pixel_digest *= 1099511628211ull;
        }
      }
      CHECK(pixel_digest == 0xf455f77e718d9c21ull);
      const std::filesystem::path cuda_y4m =
          std::filesystem::temp_directory_path() / "slopfab_exact_vae_cuda.y4m";
      const std::filesystem::path vk_y4m =
          std::filesystem::temp_directory_path() / "slopfab_exact_vae_vulkan.y4m";
      std::filesystem::remove(cuda_y4m);
      std::filesystem::remove(vk_y4m);
      video::write_y4m(cuda_y4m.string(), cuda_video.data, cuda_video.frames,
                       cuda_video.height, cuda_video.width);
      vulkan::Yuv420Converter yuv_converter;
      video::write_y4m(vk_y4m.string(), vk_video.data, vk_video.frames,
                       vk_video.height, vk_video.width, {}, &yuv_converter);
      const video::ExactY4mComparison y4m_comparison =
          video::compare_y4m_exact(cuda_y4m.string(), vk_y4m.string());
      CHECK(y4m_comparison.equal());
      std::filesystem::remove(cuda_y4m);
      std::filesystem::remove(vk_y4m);
      const uint64_t final_peak = vk_decoder.peak_device_bytes();
      const uint64_t final_reserved = vk_decoder.allocator_reserved_bytes();
      const uint64_t final_descriptors = vk_decoder.descriptor_set_allocations();
      CHECK(vk_decoder.cached_shapes() == 2);
      CHECK(final_peak <= 5700ull * 1024 * 1024);
      std::printf(
          "  exact Vulkan ViTDecoder: load %.1f ms, forward %.1f ms, persistent/peak %.1f/%.1f MiB, pool used/reserved %.1f/%.1f MiB, descriptors %llu, final %dx%dx%d FNV64 %016llx and Y4M exact\n",
          vk_load_ms, vk_forward_ms,
          double(vk_decoder.persistent_bytes()) / 1048576.0,
          double(final_peak) / 1048576.0,
          double(vk_decoder.allocator_used_bytes()) / 1048576.0,
          double(final_reserved) / 1048576.0,
          static_cast<unsigned long long>(final_descriptors),
          vk_video.frames, vk_video.height, vk_video.width,
          static_cast<unsigned long long>(pixel_digest));
    }
  }
  std::printf(
      "  exact ViTDecoder: load %.1f ms, forward %.1f ms, weights %.1f MiB, output %zu words, ragged output %zu words, FNV64 %016llx\n",
      load_ms, forward_ms, double(decoder.weight_bytes()) / 1048576.0,
      first.size(), ragged.size(), static_cast<unsigned long long>(digest));
}
