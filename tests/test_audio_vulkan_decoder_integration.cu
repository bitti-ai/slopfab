#include "detail/audio_vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_exact_audio_decoder_graph, "integration") {
  using namespace slopfab;
  if (!std::getenv("SLOPFAB_AUDIO_DECODER_REAL")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_AUDIO_DECODER_REAL\")");
    return;
  }
  const std::filesystem::path checkpoint_path =
      "weights/vae/minimax_h3_audio_vae_fp32.safetensors";
  if (!std::filesystem::exists(checkpoint_path)) {
    SKIP_MISSING_FIXTURE("unavailable prerequisite: !std::filesystem::exists(checkpoint_path)");
    return;
  }
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !vulkan::Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !vulkan::Instance::available()");
    return;
  }
  vulkan::Instance instance = vulkan::Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore || !physical.front().info().shader_int64");
    return;
  }
  vulkan::DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  vulkan::Device device = physical.front().create_device(options);

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  vae::AudioDecoder cuda_decoder;
  cuda_decoder.load(checkpoint);
  vulkan::AudioDecoder vk_decoder = vulkan::AudioDecoder::create(device);
  const uint64_t empty_decoder_used = vk_decoder.allocator_used_bytes();
  const auto load_begin = std::chrono::steady_clock::now();
  vk_decoder.load(checkpoint);
  const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();
  CHECK(cuda_decoder.weight_bytes() == vk_decoder.weight_bytes());
  CHECK(vk_decoder.weight_bytes() == 259672032u);
  CHECK(vk_decoder.recorded_operators() == 497u);
  const uint64_t largest_host_tensor =
      (uint64_t(1024) * 2048 * 7 + 1024) * sizeof(float);
  CHECK(vk_decoder.host_loader_peak_bytes() == largest_host_tensor);
  CHECK(vk_decoder.host_loader_peak_bytes() < vk_decoder.weight_bytes() / 4);
  CHECK(vk_decoder.staging_capacity_bytes() ==
        uint64_t(1024) * 2048 * 7 * sizeof(float));
  CHECK(vk_decoder.load_pool_high_water_bytes() >= vk_decoder.weight_bytes());
  CHECK(vk_decoder.load_pool_high_water_bytes() <=
        vk_decoder.weight_bytes() + 2 * vk_decoder.staging_capacity_bytes() +
            (4ull << 20));
  CHECK(cuda_decoder.latents_mean() == vk_decoder.latents_mean());
  CHECK(cuda_decoder.latents_std() == vk_decoder.latents_std());

  constexpr int latent_length = 3;
  std::vector<float> latent = values(size_t(2) * 32 * latent_length,
                                     67, 607, 1.0f / 128.0f);
  vae::AudioDecodeTrace cuda_trace;
  const vae::DecodedAudio cuda_audio = cuda_decoder.decode(
      latent.data(), latent_length, &cuda_trace);
  const auto forward_begin = std::chrono::steady_clock::now();
  const vae::DecodedAudio vk_audio = vk_decoder.decode(
      latent.data(), latent_length);
  const double forward_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - forward_begin).count();
  CHECK(cuda_audio.channels == vk_audio.channels);
  CHECK(cuda_audio.sample_rate == vk_audio.sample_rate);
  check_exact(cuda_audio.samples, vk_audio.samples, "complete audio decoder");

  const auto cuda_wav = std::filesystem::temp_directory_path() /
      "slopfab_cuda_audio_exact.wav";
  const auto vk_wav = std::filesystem::temp_directory_path() /
      "slopfab_vulkan_audio_exact.wav";
  audio::write_wav(cuda_wav.string(), cuda_audio.samples, cuda_audio.channels,
                   cuda_audio.sample_rate, audio::SampleFormat::kPcm16);
  audio::write_wav(vk_wav.string(), vk_audio.samples, vk_audio.channels,
                   vk_audio.sample_rate, audio::SampleFormat::kPcm16);
  auto read_file = [](const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), {});
  };
  CHECK(read_file(cuda_wav) == read_file(vk_wav));
  std::filesystem::remove(cuda_wav);
  std::filesystem::remove(vk_wav);

  // A mid-graph reload failure must preserve the complete old graph. This
  // archive gets through both input convolutions and then fails at stage 0.
  const std::filesystem::path partial_path =
      std::filesystem::temp_directory_path() /
      "slopfab_vulkan_audio_partial_reload.safetensors";
  write_safetensors(
      partial_path.string(),
      {{"dec_in_proj.weight", {2048, 32, 1},
        to_f32(checkpoint.at("dec_in_proj.weight"))},
       {"dec_in_proj.bias", {2048},
        to_f32(checkpoint.at("dec_in_proj.bias"))},
       {"decoder.conv_pre.weight", {1024, 2048, 7},
        to_f32(checkpoint.at("decoder.conv_pre.weight"))},
       {"decoder.conv_pre.bias", {1024},
        to_f32(checkpoint.at("decoder.conv_pre.bias"))}});
  const uint64_t before_failed_reload_used =
      vk_decoder.allocator_used_bytes();
  const uint64_t before_failed_reload_weights = vk_decoder.weight_bytes();
  bool partial_reload_rejected = false;
  {
    SafeTensors partial;
    partial.open(partial_path.string());
    try {
      vk_decoder.load(partial);
    } catch (const std::exception&) {
      partial_reload_rejected = true;
    }
  }
  CHECK(partial_reload_rejected);
  CHECK(vk_decoder.weight_bytes() == before_failed_reload_weights);
  CHECK(vk_decoder.allocator_used_bytes() == before_failed_reload_used);
  check_exact(cuda_audio.samples,
              vk_decoder.decode(latent.data(), latent_length).samples,
              "A3 after transactional reload failure");
  std::filesystem::remove(partial_path);

  constexpr int production_length = 405;
  std::vector<float> production = values(
      size_t(2) * 32 * production_length, 71, 613, 1.0f / 256.0f);
  const auto cuda_begin = std::chrono::steady_clock::now();
  const vae::DecodedAudio cuda_production = cuda_decoder.decode(
      production.data(), production_length);
  const double cuda_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cuda_begin).count();
  const vae::DecodedAudio vk_warm = vk_decoder.decode(
      production.data(), production_length);
  check_exact(cuda_production.samples, vk_warm.samples,
              "complete A405 audio decoder");
  const uint64_t stable_reserved = vk_decoder.allocator_reserved_bytes();
  const uint64_t stable_descriptors = vk_decoder.descriptor_set_allocations();
  const auto vk_begin = std::chrono::steady_clock::now();
  const vae::DecodedAudio vk_production = vk_decoder.decode(
      production.data(), production_length);
  const double vk_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - vk_begin).count();
  check_exact(cuda_production.samples, vk_production.samples,
              "repeated A405 audio decoder");
  CHECK(vk_decoder.allocator_reserved_bytes() == stable_reserved);
  CHECK(vk_decoder.descriptor_set_allocations() == stable_descriptors);
  const vae::DecodedAudio vk_small_after_production = vk_decoder.decode(
      latent.data(), latent_length);
  check_exact(cuda_audio.samples, vk_small_after_production.samples,
              "A3 audio decoder after A405");
  CHECK(vk_decoder.allocator_reserved_bytes() == stable_reserved);
  CHECK(vk_decoder.descriptor_set_allocations() == stable_descriptors);
  CHECK(vk_decoder.allocator_used_bytes() >= vk_decoder.peak_device_bytes());
  CHECK(vk_decoder.allocator_used_bytes() <=
        vk_decoder.peak_device_bytes() + (128ull << 20));
  CHECK(vk_decoder.decode_pool_high_water_bytes() >=
        vk_decoder.allocator_used_bytes());
  CHECK(vk_decoder.decode_pool_high_water_bytes() <=
        vk_decoder.peak_device_bytes() +
            2 * vk_decoder.staging_capacity_bytes() + (4ull << 20));
  const uint64_t production_digest = fnv64({vk_production.samples});
  CHECK(production_digest == 0x0b9084d3f1c6355aull);
  std::printf(
      "  exact Vulkan audio decoder: load %.1f ms, A3 forward %.1f ms, A405 CUDA/Vulkan %.1f/%.1f ms, FNV64 %016llx, weights/logical %.1f/%.1f MiB, pool used/reserved/decode-HWM %.1f/%.1f/%.1f MiB, staging-pair %.1f MiB, host-loader peak %.1f MiB, descriptors %llu\n",
      load_ms, forward_ms, cuda_ms, vk_ms,
      static_cast<unsigned long long>(production_digest),
      double(vk_decoder.weight_bytes()) / 1048576.0,
      double(vk_decoder.peak_device_bytes()) / 1048576.0,
      double(vk_decoder.allocator_used_bytes()) / 1048576.0,
      double(vk_decoder.allocator_reserved_bytes()) / 1048576.0,
      double(vk_decoder.decode_pool_high_water_bytes()) / 1048576.0,
      double(2 * vk_decoder.staging_capacity_bytes()) / 1048576.0,
      double(vk_decoder.host_loader_peak_bytes()) / 1048576.0,
      static_cast<unsigned long long>(vk_decoder.descriptor_set_allocations()));
  const uint64_t reserved_before_unload = vk_decoder.allocator_reserved_bytes();
  const uint64_t used_before_unload = vk_decoder.allocator_used_bytes();
  vk_decoder.unload();
  CHECK(vk_decoder.weight_bytes() == 0u);
  CHECK(vk_decoder.peak_device_bytes() == 0u);
  CHECK(vk_decoder.allocator_used_bytes() < used_before_unload);
  CHECK(vk_decoder.allocator_used_bytes() <= empty_decoder_used + (128ull << 20));
  CHECK(vk_decoder.allocator_reserved_bytes() == reserved_before_unload);
  bool unloaded_decode_rejected = false;
  try {
    (void)vk_decoder.decode(latent.data(), latent_length);
  } catch (const std::logic_error&) {
    unloaded_decode_rejected = true;
  }
  CHECK(unloaded_decode_rejected);

  // Reload the same object and use the extra in-batch diagnostic copies to
  // compare every meaningful graph boundary. Production above stayed the
  // original 497-operator transaction; diagnostics add 13 copies but remain
  // one 510-operator transaction under the same 512 cap.
  vk_decoder.load(checkpoint);
  CHECK(vk_decoder.weight_bytes() == before_failed_reload_weights);
  vae::AudioDecodeTrace vk_trace;
  const vae::DecodedAudio vk_reloaded = vk_decoder.decode(
      latent.data(), latent_length, &vk_trace);
  check_exact(cuda_audio.samples, vk_reloaded.samples,
              "A3 after unload and reload");
  const std::array<const char*, 13> boundary_names{
      "dec_in_proj", "conv_pre", "stage0 average", "stage1 average",
      "stage2 average", "stage3 average", "stage4 average",
      "stage5 average", "stage6 average", "activation_post", "conv_post",
      "clamp", "interleave"};
  CHECK(cuda_trace.boundaries.size() == boundary_names.size());
  CHECK(vk_trace.boundaries.size() == boundary_names.size());
  for (size_t index = 0; index < boundary_names.size(); ++index)
    check_exact(cuda_trace.boundaries[index], vk_trace.boundaries[index],
                boundary_names[index]);
  CHECK(cuda_trace.boundaries.back() == cuda_audio.samples);
  CHECK(vk_trace.boundaries.back() == vk_reloaded.samples);
  const uint64_t reload_reserved = vk_decoder.allocator_reserved_bytes();
  const uint64_t reload_descriptors = vk_decoder.descriptor_set_allocations();
  check_exact(cuda_audio.samples,
              vk_decoder.decode(latent.data(), latent_length).samples,
              "repeated A3 after reload");
  CHECK(vk_decoder.allocator_reserved_bytes() == reload_reserved);
  CHECK(vk_decoder.descriptor_set_allocations() == reload_descriptors);
  vk_decoder.unload();
}
