#include "detail/audio_vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_exact_generate_vertical_slice, "integration") {
  using namespace slopfab;
  if (!std::getenv("SLOPFAB_GENERATE_VULKAN_REAL")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_GENERATE_VULKAN_REAL\")");
    return;
  }
  const std::filesystem::path video_checkpoint =
      "weights/vae/minimax_h3_video_vae_fp16.safetensors";
  const std::filesystem::path audio_checkpoint =
      "weights/vae/minimax_h3_audio_vae_fp32.safetensors";
  if (!std::filesystem::exists(video_checkpoint) ||
      !std::filesystem::exists(audio_checkpoint)) {
    SKIP_MISSING_FIXTURE("unavailable prerequisite: !std::filesystem::exists(video_checkpoint) || !std::filesystem::exists(audio_checkpoint)");
    return;
  }
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !vulkan::Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !vulkan::Instance::available()");
    return;
  }

#ifdef _WIN32
  SafeTensors video_provenance;
  video_provenance.open(video_checkpoint.string());
  const std::array<uint8_t, 32> expected_video_sha{
      0x7c, 0x1f, 0x13, 0x14, 0x92, 0xe7, 0xed, 0xda,
      0xca, 0xac, 0x90, 0x69, 0xa6, 0x1b, 0x81, 0xbd,
      0xd3, 0x9d, 0xe5, 0xcc, 0x96, 0x56, 0x1e, 0x67,
      0x7c, 0x5e, 0xab, 0x1c, 0xdc, 0xe5, 0xe5, 0x22};
  CHECK(mapping_sha256(video_provenance.mapping_base(),
                       video_provenance.file_size()) == expected_video_sha);
  SafeTensors audio_provenance;
  audio_provenance.open(audio_checkpoint.string());
  const std::array<uint8_t, 32> expected_audio_sha{
      0x8e, 0x50, 0x5d, 0x95, 0xdd, 0x15, 0x61, 0xd4,
      0x7a, 0xbd, 0x43, 0xd4, 0x23, 0x8f, 0xd4, 0x0d,
      0x9b, 0xb1, 0xae, 0x9e, 0x14, 0x7e, 0xd0, 0xa4,
      0xcb, 0xa7, 0x78, 0xd7, 0x6a, 0xe4, 0xdb, 0x48};
  CHECK(mapping_sha256(audio_provenance.mapping_base(),
                       audio_provenance.file_size()) == expected_audio_sha);
#endif

  GenerateRequest request;
  request.canvas_width = 32;
  request.canvas_height = 32;
  request.num_frames = 6;  // Aligns to the minimum 22-frame / 7-latent window.
  request.num_inference_steps = 2;
  request.seed = 0x564b414e45584143ull;
  request.video_vae_path = video_checkpoint.string();
  request.audio_vae_path = audio_checkpoint.string();
  const GeneratePlan plan = resolve_plan(request);

  // A durable backend-neutral latent archive makes both runs consume the same
  // bytes, independently of RNG implementation and without conditioning or a
  // denoiser entering the comparison.
  const std::vector<float> video_rows = values(
      size_t(plan.layout.num_video_rows) * 96, 83, 1021, 1.0f / 512.0f);
  const std::vector<float> audio_rows = values(
      size_t(plan.layout.num_audio_rows) * 32, 89, 1013, 1.0f / 512.0f);
  const std::filesystem::path base = std::filesystem::temp_directory_path();
  const std::filesystem::path latent_path =
      base / "slopfab_cuda_vulkan_exact_latents.safetensors";
  write_safetensors(
      latent_path.string(),
      {{"video_rows", {plan.layout.num_video_rows, 96}, video_rows},
       {"audio_rows", {plan.layout.num_audio_rows, 32}, audio_rows}});

  CapturedGeneration cuda_capture, vk_capture;
  RunOptions cuda_options;
  cuda_options.source = LatentSource::kSyntheticNoise;
  cuda_options.inference_backend = DeviceBackend::kCuda;
  cuda_options.attention_mode = AttentionMode::kExact;
  cuda_options.init_latents_path = latent_path.string();
  cuda_options.verbose = false;
  cuda_options.on_samples = &capture_generation;
  cuda_options.hook_userdata = &cuda_capture;
  const RunResult cuda_result = run_generate(request, plan, cuda_options);
  CHECK_MSG(cuda_result.ok, "CUDA exact generation failed: %s",
            cuda_result.message.c_str());

  RunOptions vk_options = cuda_options;
  vk_options.inference_backend = DeviceBackend::kVulkan;
  vk_options.hook_userdata = &vk_capture;
  const RunResult vk_result = run_generate(request, plan, vk_options);
  CHECK_MSG(vk_result.ok, "Vulkan exact generation failed: %s",
            vk_result.message.c_str());

  CHECK(cuda_capture.channels == vk_capture.channels);
  CHECK(cuda_capture.frames == vk_capture.frames);
  CHECK(cuda_capture.height == vk_capture.height);
  CHECK(cuda_capture.width == vk_capture.width);
  CHECK(cuda_capture.audio_channels == vk_capture.audio_channels);
  CHECK(cuda_capture.audio_sample_rate == vk_capture.audio_sample_rate);
  CHECK(cuda_capture.video == vk_capture.video);
  check_exact(cuda_capture.audio, vk_capture.audio,
              "run_generate interleaved PCM");

  const std::filesystem::path cuda_y4m = base / "slopfab_generate_cuda_exact.y4m";
  const std::filesystem::path vk_y4m = base / "slopfab_generate_vulkan_exact.y4m";
  const std::filesystem::path cuda_wav = base / "slopfab_generate_cuda_exact.wav";
  const std::filesystem::path vk_wav = base / "slopfab_generate_vulkan_exact.wav";
  video::write_y4m(cuda_y4m.string(), cuda_capture.video,
                   cuda_capture.frames, cuda_capture.height,
                   cuda_capture.width);
  video::write_y4m(vk_y4m.string(), vk_capture.video, vk_capture.frames,
                   vk_capture.height, vk_capture.width);
  audio::write_wav(cuda_wav.string(), cuda_capture.audio,
                   cuda_capture.audio_channels, cuda_capture.audio_sample_rate,
                   audio::SampleFormat::kPcm16);
  audio::write_wav(vk_wav.string(), vk_capture.audio,
                   vk_capture.audio_channels, vk_capture.audio_sample_rate,
                   audio::SampleFormat::kPcm16);
  const std::vector<uint8_t> cuda_y4m_bytes = read_bytes(cuda_y4m);
  const std::vector<uint8_t> vk_y4m_bytes = read_bytes(vk_y4m);
  const std::vector<uint8_t> cuda_wav_bytes = read_bytes(cuda_wav);
  const std::vector<uint8_t> vk_wav_bytes = read_bytes(vk_wav);
  CHECK(cuda_y4m_bytes == vk_y4m_bytes);
  CHECK(cuda_wav_bytes == vk_wav_bytes);

  const uint64_t pixel_digest = fnv64_floats(
      cuda_capture.video.data(), cuda_capture.video.size());
  const uint64_t pcm_digest = fnv64_floats(
      cuda_capture.audio.data(), cuda_capture.audio.size());
  const uint64_t y4m_digest = fnv64_bytes(cuda_y4m_bytes);
  const uint64_t wav_digest = fnv64_bytes(cuda_wav_bytes);
  CHECK(pixel_digest == 0xe2ca5273e36e9ed7ull);
  CHECK(pcm_digest == 0x5933499108ce9c79ull);
  CHECK(y4m_digest == 0xd55dbd1d534b8787ull);
  CHECK(wav_digest == 0x6b066c7cf430117dull);

  std::printf(
      "  exact run_generate slice: latent/pixels/PCM/Y4M/WAV FNV64 %016llx/%016llx/%016llx/%016llx/%016llx, %zu pixels, %zu interleaved PCM samples\n",
      static_cast<unsigned long long>(fnv64({video_rows, audio_rows})),
      static_cast<unsigned long long>(pixel_digest),
      static_cast<unsigned long long>(pcm_digest),
      static_cast<unsigned long long>(y4m_digest),
      static_cast<unsigned long long>(wav_digest),
      cuda_capture.video.size(), cuda_capture.audio.size());
  std::error_code ignored;
  std::filesystem::remove(latent_path, ignored);
  std::filesystem::remove(cuda_y4m, ignored);
  std::filesystem::remove(vk_y4m, ignored);
  std::filesystem::remove(cuda_wav, ignored);
  std::filesystem::remove(vk_wav, ignored);
}
