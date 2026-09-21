#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_dit_real_transformer_capture_replay, "integration") {
  using namespace slopfab;
  using namespace slopfab::dit;
  using namespace slopfab::vulkan;
  const std::filesystem::path checkpoint_path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  const std::filesystem::path capture_path = "tests/data/h3_transformer_step0_seed424242_256.vfh3f";
  if (!std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(capture_path) ||
      !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(capture_path) || !Instance::available()");
    return;
  }
  std::ifstream stream(capture_path, std::ios::binary | std::ios::ate);
  const std::streamsize capture_bytes = stream.tellg();
  stream.seekg(0);
  std::vector<uint8_t> capture(static_cast<size_t>(capture_bytes));
  CHECK(static_cast<bool>(stream.read(reinterpret_cast<char*>(capture.data()), capture_bytes)));
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_capture_sha{
      0x3e, 0x34, 0x76, 0xe3, 0x97, 0xfc, 0xee, 0x20, 0x37, 0x37, 0x33,
      0x2d, 0x17, 0x1d, 0x46, 0x50, 0xf5, 0x54, 0x33, 0x47, 0x12, 0x31,
      0xa7, 0xfd, 0x4f, 0xf2, 0x4f, 0x8e, 0x0f, 0x84, 0xf8, 0xe7};
  CHECK(sha256_mapping(capture.data(), capture.size()) == expected_capture_sha);
#endif
  if (capture.size() < sizeof(H3TransformerCaptureHeader))
    throw std::runtime_error("truncated H3 transformer capture");
  H3TransformerCaptureHeader header{};
  std::memcpy(&header, capture.data(), sizeof(header));
  CHECK(std::memcmp(header.magic, "VFH3FWD\0", 8) == 0);
  CHECK(header.version == 1 && header.header_bytes == sizeof(header));
  CHECK(header.sequence == 526 && header.hidden == 5376 && header.heads == 56 &&
        header.head_dim == 128 && header.ffn == 14336);
  CHECK(header.timesteps == 1 && header.modalities == 3 && header.adaln_rank == 8 &&
        header.layers == 50 && header.text_rows == 4 && header.video_rows == 448 &&
        header.audio_rows == 74 && header.text_dim == 5120 && header.video_dim == 96 &&
        header.audio_dim == 32 && header.refiner_layers == 2 && header.range_values == 20 &&
        header.denoise_step == 0);
  const std::array<uint64_t, 6> expected_text_hashes{0x6d891a14ee2a38bdull, 0xcc91d0e61a56bd7bull,
                                                     0xd5ddff8656b1d581ull, 0xaf5733d914839cf2ull,
                                                     0xd510022f4c0e9032ull, 0x1e4af4a0c48fffc7ull};
  for (uint32_t stage = 0; stage < expected_text_hashes.size(); ++stage)
    CHECK(header.text_boundary_fnv64[stage] == expected_text_hashes[stage]);
  CHECK(header.packed_input_fnv64 == 0x54c4e5ce3af6d0deull);
  CHECK(header.main_final_fnv64 == 0xda1038eb60eea19full);
  CHECK(header.video_output_fnv64 == 0x7d7af2480929ae03ull);
  CHECK(header.audio_output_fnv64 == 0x58197cbc23da3ab3ull);

  size_t cursor = sizeof(header);
  auto take = [&](auto& values, size_t count) {
    using Value = typename std::decay_t<decltype(values)>::value_type;
    if (count > (capture.size() - cursor) / sizeof(Value))
      throw std::runtime_error("truncated H3 transformer capture payload");
    values.resize(count);
    std::memcpy(values.data(), capture.data() + cursor, count * sizeof(Value));
    cursor += count * sizeof(Value);
  };
  std::vector<float> prompt, video, audio, code, cosine, sine, expected_video, expected_audio;
  std::vector<int32_t> selectors, ranges_data, video_ts, audio_ts;
  std::vector<uint16_t> expected_text, expected_packed, expected_main;
  take(prompt, header.prompt_elements);
  take(video, header.video_elements);
  take(audio, header.audio_elements);
  take(selectors, header.sequence);
  take(code, size_t(header.timesteps) * header.adaln_rank);
  take(cosine, header.rope_elements);
  take(sine, header.rope_elements);
  take(ranges_data, header.range_values);
  take(video_ts, header.video_rows);
  take(audio_ts, header.audio_rows);
  take(expected_text, header.text_elements);
  take(expected_packed, header.packed_elements);
  take(expected_main, header.packed_elements);
  take(expected_video, header.video_elements);
  take(expected_audio, header.audio_elements);
  CHECK(cursor == capture.size());
  CHECK(std::any_of(cosine.begin(), cosine.end(), [](float x) {
    return x != 1.0f;
  }));
  CHECK(std::any_of(sine.begin(), sine.end(), [](float x) {
    return x != 0.0f;
  }));
  auto fnv_bytes = [](const void* data, size_t bytes) {
    uint64_t hash = 1469598103934665603ull;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) {
      hash ^= p[i];
      hash *= 1099511628211ull;
    }
    return hash;
  };
  CHECK(fnv_bytes(expected_text.data(), expected_text.size() * 2) == header.text_boundary_fnv64[5]);
  CHECK(fnv_bytes(expected_packed.data(), expected_packed.size() * 2) == header.packed_input_fnv64);
  CHECK(fnv_bytes(expected_main.data(), expected_main.size() * 2) == header.main_final_fnv64);
  CHECK(fnv_bytes(expected_video.data(), expected_video.size() * 4) == header.video_output_fnv64);
  CHECK(fnv_bytes(expected_audio.data(), expected_audio.size() * 4) == header.audio_output_fnv64);

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_checkpoint_sha{
      0x6a, 0xb7, 0xf0, 0xc4, 0x81, 0x41, 0xe7, 0x91, 0x9b, 0x32, 0xf9,
      0x25, 0xca, 0x3d, 0xef, 0x22, 0xe0, 0x6a, 0x6a, 0xeb, 0xeb, 0x9e,
      0x0b, 0x6f, 0x5a, 0x0b, 0xe0, 0xfe, 0x84, 0x09, 0x97, 0x6f};
  CHECK(sha256_mapping(checkpoint.mapping_base(), checkpoint.file_size()) ==
        expected_checkpoint_sha);
#endif
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty()");
    return;
  }
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 ||
      !info.storage_buffer_16bit || !info.cooperative_matrix_bf16_f32_16x16x16) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 || !info.storage_buffer_16bit || !info.cooperative_matrix_bf16_f32_16x16x16");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 2048;
  TensorContext vk(device, context_options);
  if (!vk.exact_h3_attention()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_h3_attention()");
    return;
  }
  ExactH3TransformerConfig config;
  config.main.layers = header.layers;
  config.main.block.sequence = header.sequence;
  config.main.block.hidden = header.hidden;
  config.main.block.heads = header.heads;
  config.main.block.head_dim = header.head_dim;
  config.main.block.ffn = header.ffn;
  config.main.block.timesteps = header.timesteps;
  config.main.block.modalities = header.modalities;
  config.main.block.adaln_rank = header.adaln_rank;
  config.text_rows = header.text_rows;
  config.video_rows = header.video_rows;
  config.audio_rows = header.audio_rows;
  config.text_dim = header.text_dim;
  config.video_dim = header.video_dim;
  config.audio_dim = header.audio_dim;
  config.refiner_layers = header.refiner_layers;

  auto tensor2 = [&](uint64_t rows, uint64_t columns, ScalarType type = ScalarType::kFloat32) {
    const uint64_t shape[] = {rows, columns};
    return vk.allocate(TensorLayout::contiguous(shape, 2), type);
  };
  auto tensor1 = [&](uint64_t rows, ScalarType type) {
    const uint64_t shape[] = {rows};
    return vk.allocate(TensorLayout::contiguous(shape, 1), type);
  };
  DeviceTensor prompt_tensor = tensor2(header.text_rows, header.text_dim);
  DeviceTensor video_tensor = tensor2(header.video_rows, header.video_dim);
  DeviceTensor audio_tensor = tensor2(header.audio_rows, header.audio_dim);
  DeviceTensor selector_tensor = tensor1(header.sequence, ScalarType::kInt32);
  DeviceTensor code_tensor = tensor2(header.timesteps, header.adaln_rank);
  DeviceTensor cosine_tensor = tensor2(header.sequence, 96);
  DeviceTensor sine_tensor = tensor2(header.sequence, 96);
  DeviceTensor video_ts_tensor = tensor1(header.video_rows, ScalarType::kInt32);
  DeviceTensor audio_ts_tensor = tensor1(header.audio_rows, ScalarType::kInt32);
  DeviceTensor video_output = tensor2(header.video_rows, header.video_dim);
  DeviceTensor audio_output = tensor2(header.audio_rows, header.audio_dim);
  vk.upload(prompt_tensor, prompt.data(), prompt.size());
  vk.upload(video_tensor, video.data(), video.size());
  vk.upload(audio_tensor, audio.data(), audio.size());
  vk.upload_bytes(selector_tensor, selectors.data(), selectors.size() * 4);
  vk.upload(code_tensor, code.data(), code.size());
  vk.upload(cosine_tensor, cosine.data(), cosine.size());
  vk.upload(sine_tensor, sine.data(), sine.size());
  vk.upload_bytes(video_ts_tensor, video_ts.data(), video_ts.size() * 4);
  vk.upload_bytes(audio_ts_tensor, audio_ts.data(), audio_ts.size() * 4);
  H3AttentionRanges ranges =
      H3AttentionRanges::create(vk, header.sequence, ranges_data.data(), header.range_values);

  std::vector<DeviceTensor> text_boundaries;
  for (uint32_t i = 0; i < 6; ++i)
    text_boundaries.push_back(tensor2(header.text_rows, header.hidden, ScalarType::kBFloat16));
  DeviceTensor packed_tap = tensor2(header.sequence, header.hidden, ScalarType::kBFloat16);
  DeviceTensor main_tap = tensor2(header.sequence, header.hidden, ScalarType::kBFloat16);
  H3TransformerTextReplayTaps text_taps{text_boundaries.data(), 6};
  H3TransformerForwardReplayTaps forward_taps{&packed_tap, &main_tap, nullptr};
  const auto load_begin = std::chrono::steady_clock::now();
  ExactH3Transformer transformer = ExactH3Transformer::create(vk, config);
  transformer.load(checkpoint);
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_begin)
          .count();
  const uint32_t text_ops = transformer.required_prepare_text_operators(&text_taps);
  CHECK_MSG(text_ops == 43u, "real H3 transformer text ops %u != 43", text_ops);
  const auto text_begin = std::chrono::steady_clock::now();
  transformer.prepare_text(prompt_tensor, &text_taps);
  const double text_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - text_begin)
          .count();
  for (uint32_t stage = 0; stage < 6; ++stage) {
    std::vector<uint16_t> actual(header.text_elements);
    vk.download_bytes(text_boundaries[stage], actual.data(), actual.size() * 2);
    CHECK_MSG(fnv_bytes(actual.data(), actual.size() * 2) == header.text_boundary_fnv64[stage],
              "real H3 transformer text boundary %u mismatch: %016llx != %016llx", stage,
              static_cast<unsigned long long>(fnv_bytes(actual.data(), actual.size() * 2)),
              static_cast<unsigned long long>(header.text_boundary_fnv64[stage]));
    if (stage == 5)
      CHECK(actual == expected_text);
  }
  CHECK(transformer.required_forward_operators(&forward_taps) == 1469u);
  const auto first_begin = std::chrono::steady_clock::now();
  TensorBatch first = vk.begin_batch();
  transformer.record_forward(first, video_tensor, audio_tensor, selector_tensor, code_tensor,
                             cosine_tensor, sine_tensor, video_ts_tensor, audio_ts_tensor,
                             video_output, audio_output, &ranges, &forward_taps);
  CHECK(first.remaining_operator_capacity() == 579u);
  first.submit().wait();
  const double first_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - first_begin)
          .count();
  std::vector<uint16_t> actual_packed(header.packed_elements), actual_main(header.packed_elements);
  std::vector<float> actual_video(header.video_elements), actual_audio(header.audio_elements);
  vk.download_bytes(packed_tap, actual_packed.data(), actual_packed.size() * 2);
  vk.download_bytes(main_tap, actual_main.data(), actual_main.size() * 2);
  vk.download(video_output, actual_video.data(), actual_video.size());
  vk.download(audio_output, actual_audio.data(), actual_audio.size());
  CHECK(actual_packed == expected_packed);
  CHECK(actual_main == expected_main);
  CHECK(std::memcmp(actual_video.data(), expected_video.data(), actual_video.size() * 4) == 0);
  CHECK(std::memcmp(actual_audio.data(), expected_audio.data(), actual_audio.size() * 4) == 0);
  const uint64_t stable_used = vk.pooled_used_bytes();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  const auto repeat_begin = std::chrono::steady_clock::now();
  TensorBatch repeat = vk.begin_batch();
  transformer.record_forward(repeat, video_tensor, audio_tensor, selector_tensor, code_tensor,
                             cosine_tensor, sine_tensor, video_ts_tensor, audio_ts_tensor,
                             video_output, audio_output, &ranges);
  repeat.submit().wait();
  const double repeat_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - repeat_begin)
          .count();
  vk.download(video_output, actual_video.data(), actual_video.size());
  vk.download(audio_output, actual_audio.data(), actual_audio.size());
  CHECK(std::memcmp(actual_video.data(), expected_video.data(), actual_video.size() * 4) == 0);
  CHECK(std::memcmp(actual_audio.data(), expected_audio.data(), actual_audio.size() * 4) == 0);
  CHECK(vk.pooled_used_bytes() == stable_used);
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  std::printf(
      "  real H3 transformer S%u: load/text %.3f/%.3f ms, Vulkan taps/repeat %.3f/%.3f ms, packed/main/video/audio %016llx/%016llx/%016llx/%016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, pool %.2f/%.2f MiB, descriptors %llu\n",
      header.sequence, load_ms, text_ms, first_ms, repeat_ms,
      static_cast<unsigned long long>(header.packed_input_fnv64),
      static_cast<unsigned long long>(header.main_final_fnv64),
      static_cast<unsigned long long>(header.video_output_fnv64),
      static_cast<unsigned long long>(header.audio_output_fnv64),
      double(transformer.persistent_bytes()) / 1048576.0,
      double(transformer.scratch_bytes()) / 1048576.0,
      double(transformer.peak_device_bytes()) / 1048576.0,
      double(vk.pooled_used_bytes()) / 1048576.0, double(vk.reserved_bytes()) / 1048576.0,
      static_cast<unsigned long long>(vk.descriptor_set_allocations()));

  const uint64_t loaded_used = vk.pooled_used_bytes();
  transformer.unload();
  CHECK(!transformer.loaded() && transformer.persistent_bytes() == 0u &&
        transformer.scratch_bytes() == 0u && transformer.peak_device_bytes() == 0u);
  CHECK(vk.pooled_used_bytes() < loaded_used);
  transformer.load(checkpoint);
  transformer.prepare_text(prompt_tensor);
  TensorBatch reloaded = vk.begin_batch();
  transformer.record_forward(reloaded, video_tensor, audio_tensor, selector_tensor, code_tensor,
                             cosine_tensor, sine_tensor, video_ts_tensor, audio_ts_tensor,
                             video_output, audio_output, &ranges);
  reloaded.submit().wait();
  vk.download(video_output, actual_video.data(), actual_video.size());
  vk.download(audio_output, actual_audio.data(), actual_audio.size());
  CHECK(std::memcmp(actual_video.data(), expected_video.data(), actual_video.size() * 4) == 0);
  CHECK(std::memcmp(actual_audio.data(), expected_audio.data(), actual_audio.size() * 4) == 0);
  transformer.unload();

  if (const char* real_denoise = std::getenv("SLOPFAB_DIT_DENOISE_REAL");
      real_denoise && real_denoise[0] == '1') {
    SequenceLayout denoise_layout;
    denoise_layout.num_text = static_cast<int>(header.text_rows);
    denoise_layout.num_audio_rows = static_cast<int>(header.audio_rows);
    denoise_layout.num_video_rows = static_cast<int>(header.video_rows);
    denoise_layout.num_audio_latents = static_cast<int>(header.audio_rows / 2);
    denoise_layout.num_latent_frames = 7;
    denoise_layout.latent_height = 16;
    denoise_layout.latent_width = 16;
    CHECK(denoise_layout.total_rows() == static_cast<int>(header.sequence) &&
          denoise_layout.num_latent_frames * denoise_layout.rows_per_frame() ==
              static_cast<int>(header.video_rows));
    const PackedIndices denoise_indices = build_indices(denoise_layout);
    const std::vector<double> denoise_positions = build_position_ids(denoise_layout);

    auto joined_hash = [&](const std::vector<float>& video_rows,
                           const std::vector<float>& audio_rows) {
      uint64_t hash = 1469598103934665603ull;
      auto append = [&](const std::vector<float>& rows) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(rows.data());
        for (size_t i = 0; i < rows.size() * sizeof(float); ++i) {
          hash ^= bytes[i];
          hash *= 1099511628211ull;
        }
      };
      append(video_rows);
      append(audio_rows);
      return hash;
    };

    struct Boundary {
      std::vector<float> video;
      std::vector<float> audio;
    };

    std::vector<Boundary> cuda_boundaries;

    dit::Transformer cuda_model;
    const auto cuda_load_begin = std::chrono::steady_clock::now();
    cuda_model.load(checkpoint);
    cuda_model.set_attention_mode(AttentionMode::kExact);
    cuda_model.set_attention_band(0);
    cuda_model.prepare_text(prompt.data(), static_cast<int>(header.text_rows));
    cuda_model.prepare_sequence(denoise_layout, denoise_indices, denoise_positions);
    const double cuda_load_prepare_ms = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - cuda_load_begin)
                                            .count();
    sampler::FlowScheduler cuda_video_schedule(12.0f);
    sampler::FlowScheduler cuda_audio_schedule(3.0f);
    cuda_video_schedule.set_timesteps(4);
    cuda_audio_schedule.set_timesteps(4);
    DenoiseInputs cuda_inputs;
    cuda_inputs.layout = &denoise_layout;
    cuda_inputs.indices = &denoise_indices;
    cuda_inputs.video_timesteps = &cuda_video_schedule.timesteps();
    cuda_inputs.audio_timesteps = &cuda_audio_schedule.timesteps();
    cuda_inputs.video_scheduler = &cuda_video_schedule;
    cuda_inputs.audio_scheduler = &cuda_audio_schedule;
    cuda_inputs.init_video_rows = &video;
    cuda_inputs.init_audio_rows = &audio;
    cuda_inputs.boundary = [&](int, const std::vector<float>& video_rows,
                               const std::vector<float>& audio_rows) {
      cuda_boundaries.push_back({video_rows, audio_rows});
    };
    const auto cuda_run_begin = std::chrono::steady_clock::now();
    const DenoiseOutputs cuda_result = denoise(cuda_model, cuda_inputs);
    const double cuda_run_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_run_begin)
            .count();
    CHECK(cuda_result.steps_computed == 3 && cuda_result.steps_skipped == 0 &&
          cuda_boundaries.size() == 3u);
    cuda_model.unload();
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

    ExactH3DenoiseConfig denoise_config;
    denoise_config.transformer = config;
    denoise_config.transformer.main.block.timesteps = 2;
    denoise_config.layout = denoise_layout;
    denoise_config.indices = denoise_indices;
    denoise_config.position_ids = denoise_positions;
    denoise_config.attention_ranges = ranges_data;
    ExactH3Denoiser vk_denoiser = ExactH3Denoiser::create(vk, denoise_config);
    const auto vk_load_begin = std::chrono::steady_clock::now();
    vk_denoiser.load(checkpoint);
    vk_denoiser.prepare(prompt.data(), prompt.size(), video.data(), video.size(), audio.data(),
                        audio.size());
    const double vk_load_prepare_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vk_load_begin)
            .count();
    sampler::FlowScheduler vk_video_schedule(12.0f);
    sampler::FlowScheduler vk_audio_schedule(3.0f);
    vk_video_schedule.set_timesteps(4);
    vk_audio_schedule.set_timesteps(4);
    std::vector<uint64_t> boundary_hashes;
    size_t boundary_index = 0;
    const auto vk_run_begin = std::chrono::steady_clock::now();
    const ExactH3DenoiseResult vk_result =
        vk_denoiser.run(vk_video_schedule, vk_audio_schedule, {},
                        [&](uint32_t step, const std::vector<float>& video_rows,
                            const std::vector<float>& audio_rows) {
                          CHECK(step == boundary_index && boundary_index < cuda_boundaries.size());
                          if (boundary_index < cuda_boundaries.size()) {
                            CHECK(video_rows == cuda_boundaries[boundary_index].video);
                            CHECK(audio_rows == cuda_boundaries[boundary_index].audio);
                          }
                          boundary_hashes.push_back(joined_hash(video_rows, audio_rows));
                          ++boundary_index;
                        });
    const double vk_run_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vk_run_begin)
            .count();
    CHECK(!vk_result.cancelled && vk_result.steps_completed == 3u &&
          boundary_index == cuda_boundaries.size());
    CHECK(vk_result.video_rows == cuda_result.video_rows &&
          vk_result.audio_rows == cuda_result.audio_rows);
    CHECK(boundary_hashes.size() == 3u);
    const std::array<uint64_t, 3> expected_denoise_hashes{
        0x2472491d7573a692ull, 0x3343aa4828944315ull, 0xbfc3aec499e7b836ull};
    CHECK(std::equal(boundary_hashes.begin(), boundary_hashes.end(),
                     expected_denoise_hashes.begin()));
    std::printf(
        "  real exact denoise S%u x3: CUDA load+prep/run %.3f/%.3f ms, Vulkan %.3f/%.3f ms, boundaries %016llx/%016llx/%016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB\n",
        header.sequence, cuda_load_prepare_ms, cuda_run_ms, vk_load_prepare_ms, vk_run_ms,
        static_cast<unsigned long long>(boundary_hashes[0]),
        static_cast<unsigned long long>(boundary_hashes[1]),
        static_cast<unsigned long long>(boundary_hashes[2]),
        double(vk_denoiser.persistent_bytes()) / 1048576.0,
        double(vk_denoiser.scratch_bytes()) / 1048576.0,
        double(vk_denoiser.peak_device_bytes()) / 1048576.0);
    vk_denoiser.unload();

    const char* ref2va_real = std::getenv("SLOPFAB_REF2VA_DENOISE_REAL");
    const char* ref2va_production = std::getenv("SLOPFAB_REF2VA_DENOISE_PRODUCTION_REAL");
    if ((ref2va_real && ref2va_real[0] == '1') ||
        (ref2va_production && ref2va_production[0] == '1')) {
      const bool production_shape = ref2va_production && ref2va_production[0] == '1';
      const char* ref_graph_capture_env =
          production_shape ? std::getenv("SLOPFAB_REF2VA_GRAPH_CAPTURE") : nullptr;
      const std::string ref_graph_capture = ref_graph_capture_env ? ref_graph_capture_env : "";
      const char* ref_graph_reuse_env = std::getenv("SLOPFAB_REF2VA_GRAPH_CAPTURE_REUSE");
      const bool reuse_ref_graph =
          !ref_graph_capture.empty() && ref_graph_reuse_env && ref_graph_reuse_env[0] == '1';
      if (reuse_ref_graph)
        CHECK(std::filesystem::exists(ref_graph_capture));
      const std::filesystem::path ref_checkpoint_path =
          "weights/transformer/minimax_h3_ref2va_pruned_nvfp4.safetensors";
      const std::filesystem::path video_vae_path =
          "weights/vae/minimax_h3_video_vae_fp16.safetensors";
      CHECK(std::filesystem::exists(ref_checkpoint_path) &&
            std::filesystem::exists(video_vae_path));
      const Sha256Digest ref_checkpoint_sha{0x8e, 0xea, 0x02, 0xf4, 0x39, 0x02, 0xe6, 0x99,
                                            0x04, 0x99, 0x0c, 0x44, 0x05, 0x96, 0x8d, 0x01,
                                            0xa1, 0x3c, 0x66, 0x56, 0xaa, 0x39, 0x2d, 0x37,
                                            0xab, 0x80, 0x33, 0x1e, 0x79, 0xb5, 0xdf, 0x2f};
      CHECK(sha256_file(ref_checkpoint_path.string()) == ref_checkpoint_sha);

      SafeTensors video_vae;
      video_vae.open(video_vae_path.string());
      const std::vector<float> latent_mean = to_f32(video_vae.at("latents_mean"));
      const std::vector<float> latent_std = to_f32(video_vae.at("latents_std"));
      CHECK(latent_mean.size() == 24u && latent_std.size() == 24u);
      RGBImage reference;
      reference.width = production_shape ? 2048 : 96;
      reference.height = production_shape ? 2048 : 64;
      reference.pixels.resize(size_t(reference.width) * reference.height * 3u);
      for (size_t i = 0; i < reference.pixels.size(); ++i)
        reference.pixels[i] = static_cast<uint8_t>((i * 37u + 19u) % 251u);
      std::vector<float> cuda_condition;
      if (!reuse_ref_graph) {
        vae::KeyframeEncoder cuda_keyframe(video_vae);
        cuda_condition = cuda_keyframe.encode_reference_image(reference, latent_mean, latent_std);
      }
      vulkan::KeyframeEncoder vk_keyframe = vulkan::KeyframeEncoder::create(device);
      vk_keyframe.load(video_vae);
      const std::vector<float> vk_condition =
          vk_keyframe.encode_reference_image(reference, latent_mean, latent_std);
      if (reuse_ref_graph)
        cuda_condition = vk_condition;
      CHECK(vk_condition == cuda_condition);
      const uint32_t condition_rows = production_shape ? 4096u : 6u;
      CHECK(cuda_condition.size() == uint64_t(condition_rows) * header.video_dim);
      vk_keyframe.unload();

      const uint32_t ref_text_rows = production_shape ? 4100u : header.text_rows;
      std::vector<float> ref_prompt(uint64_t(ref_text_rows) * header.text_dim);
      for (size_t i = 0; i < ref_prompt.size(); ++i)
        ref_prompt[i] = prompt[i % prompt.size()];
      std::vector<int32_t> ref_text_tags(ref_text_rows, kTagText);
      if (production_shape)
        std::fill(ref_text_tags.begin() + 2, ref_text_tags.begin() + 4098, kTagVideo);
      const std::vector<ReferenceGeometry> ref_geometries{
          {ReferenceKind::kImage, 1, production_shape ? 128 : 4, production_shape ? 128 : 6, 0}};
      const Ref2VAPackedSequence ref_packed = build_ref2va_packed_sequence(
          ref_text_tags, ref_geometries, 7, 16, 16, static_cast<int>(header.audio_rows / 2));
      CHECK(ref_packed.layout.num_condition_video == static_cast<int>(condition_rows) &&
            ref_packed.layout.num_condition_audio == 0 &&
            ref_packed.layout.num_video_rows == static_cast<int>(header.video_rows) &&
            ref_packed.layout.num_audio_rows == static_cast<int>(header.audio_rows) &&
            ref_packed.layout.total_rows() ==
                static_cast<int>(ref_text_rows + condition_rows + header.audio_rows +
                                 header.video_rows));

      SafeTensors ref_checkpoint;
      ref_checkpoint.open(ref_checkpoint_path.string());
      double ref_cuda_load_ms = 0.0;
      double ref_cuda_run_ms = 0.0;
      const int ref_schedule_points = production_shape ? 2 : 4;
      const size_t ref_expected_steps = static_cast<size_t>(ref_schedule_points - 1);
      std::vector<Boundary> ref_cuda_boundaries;
      DenoiseOutputs ref_cuda_result;
      if (!reuse_ref_graph) {
        if (!ref_graph_capture.empty()) {
#ifdef _WIN32
          _putenv_s("SLOPFAB_H3_GRAPH_CAPTURE", ref_graph_capture.c_str());
          _putenv_s("SLOPFAB_H3_GRAPH_CAPTURE_STEP", "0");
#else
          setenv("SLOPFAB_H3_GRAPH_CAPTURE", ref_graph_capture.c_str(), 1);
          setenv("SLOPFAB_H3_GRAPH_CAPTURE_STEP", "0", 1);
#endif
        }
        dit::Transformer ref_cuda_model;
        const auto ref_cuda_load_begin = std::chrono::steady_clock::now();
        ref_cuda_model.load(ref_checkpoint);
        if (!ref_graph_capture.empty()) {
#ifdef _WIN32
          _putenv_s("SLOPFAB_H3_GRAPH_CAPTURE", "");
          _putenv_s("SLOPFAB_H3_GRAPH_CAPTURE_STEP", "");
#else
          unsetenv("SLOPFAB_H3_GRAPH_CAPTURE");
          unsetenv("SLOPFAB_H3_GRAPH_CAPTURE_STEP");
#endif
        }
        ref_cuda_model.set_attention_mode(AttentionMode::kExact);
        ref_cuda_model.set_attention_band(0);
        ref_cuda_model.prepare_text(ref_prompt.data(), static_cast<int>(ref_text_rows));
        ref_cuda_model.prepare_sequence(ref_packed.layout, ref_packed.indices,
                                        ref_packed.position_ids);
        ref_cuda_load_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - ref_cuda_load_begin)
                               .count();
        sampler::FlowScheduler ref_cuda_video(12.0f), ref_cuda_audio(3.0f);
        ref_cuda_video.set_timesteps(ref_schedule_points);
        ref_cuda_audio.set_timesteps(ref_schedule_points);
        DenoiseInputs ref_cuda_inputs;
        ref_cuda_inputs.layout = &ref_packed.layout;
        ref_cuda_inputs.indices = &ref_packed.indices;
        ref_cuda_inputs.video_timesteps = &ref_cuda_video.timesteps();
        ref_cuda_inputs.audio_timesteps = &ref_cuda_audio.timesteps();
        ref_cuda_inputs.video_scheduler = &ref_cuda_video;
        ref_cuda_inputs.audio_scheduler = &ref_cuda_audio;
        ref_cuda_inputs.condition_video_rows = &cuda_condition;
        ref_cuda_inputs.init_video_rows = &video;
        ref_cuda_inputs.init_audio_rows = &audio;
        ref_cuda_inputs.boundary = [&](int, const std::vector<float>& video_rows,
                                       const std::vector<float>& audio_rows) {
          ref_cuda_boundaries.push_back({video_rows, audio_rows});
        };
        const auto ref_cuda_run_begin = std::chrono::steady_clock::now();
        ref_cuda_result = denoise(ref_cuda_model, ref_cuda_inputs);
        ref_cuda_run_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - ref_cuda_run_begin)
                              .count();
        CHECK(ref_cuda_result.steps_computed == static_cast<int>(ref_expected_steps) &&
              ref_cuda_result.steps_skipped == 0 &&
              ref_cuda_boundaries.size() == ref_expected_steps);
        ref_cuda_model.unload();
        SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      }

      ExactH3DenoiseConfig ref_vk_config;
      ref_vk_config.transformer = config;
      ref_vk_config.transformer.main.block.sequence =
          static_cast<uint32_t>(ref_packed.layout.total_rows());
      ref_vk_config.transformer.main.block.timesteps = 4;
      ref_vk_config.transformer.text_rows = ref_text_rows;
      ref_vk_config.transformer.video_rows = static_cast<uint32_t>(ref_packed.indices.video.size());
      ref_vk_config.transformer.audio_rows = static_cast<uint32_t>(ref_packed.indices.audio.size());
      ref_vk_config.transformer.video_output_rows =
          static_cast<uint32_t>(ref_packed.layout.num_video_rows);
      ref_vk_config.transformer.audio_output_rows =
          static_cast<uint32_t>(ref_packed.layout.num_audio_rows);
      ref_vk_config.transformer.video_output_start =
          static_cast<uint32_t>(ref_packed.layout.video_start());
      ref_vk_config.transformer.audio_output_start =
          static_cast<uint32_t>(ref_packed.layout.audio_start());
      ref_vk_config.layout = ref_packed.layout;
      ref_vk_config.indices = ref_packed.indices;
      ref_vk_config.position_ids = ref_packed.position_ids;
      ExactH3Denoiser ref_vk = ExactH3Denoiser::create(vk, ref_vk_config);
      const auto ref_vk_load_begin = std::chrono::steady_clock::now();
      ref_vk.load(ref_checkpoint);
      std::vector<float> ref_video = cuda_condition;
      ref_video.insert(ref_video.end(), video.begin(), video.end());
      std::vector<DeviceTensor> ref_text_tap_tensors;
      H3TransformerTextReplayTaps ref_text_taps{};
      const H3TransformerTextReplayTaps* ref_text_taps_ptr = nullptr;
      if (!ref_graph_capture.empty()) {
        const uint64_t text_tap_shape[]{ref_text_rows,
                                        static_cast<uint64_t>(config.main.block.hidden)};
        const TensorLayout text_tap_layout = TensorLayout::contiguous(text_tap_shape, 2);
        ref_text_tap_tensors.reserve(6);
        for (uint32_t stage = 0; stage < 6; ++stage)
          ref_text_tap_tensors.push_back(vk.allocate(text_tap_layout, ScalarType::kBFloat16));
        ref_text_taps = {ref_text_tap_tensors.data(), 6};
        ref_text_taps_ptr = &ref_text_taps;
      }
      ref_vk.prepare(ref_prompt.data(), ref_prompt.size(), ref_video.data(), ref_video.size(),
                     audio.data(), audio.size(), ref_text_taps_ptr);
      if (ref_text_taps_ptr) {
        const std::array<uint64_t, 6> expected_ref_text_hashes{
            0x969fa5eb28da8046ull, 0xb0175bdeb4cb4ba7ull, 0x0ccd4df69b2a2ee0ull,
            0x81a6a4bbc794f16dull, 0xc12674b992f4bf38ull, 0xad26e13685a40043ull};
        std::printf("    Ref2VA Vulkan text L4100 stage FNV64:");
        for (uint32_t stage = 0; stage < 6; ++stage) {
          std::vector<uint16_t> values(uint64_t(ref_text_rows) * config.main.block.hidden);
          vk.download_bytes(ref_text_tap_tensors[stage], values.data(),
                            values.size() * sizeof(uint16_t));
          uint64_t hash = 1469598103934665603ull;
          for (uint16_t value : values) {
            hash ^= value & 0xffu;
            hash *= 1099511628211ull;
            hash ^= value >> 8;
            hash *= 1099511628211ull;
          }
          std::printf(" %016llx", static_cast<unsigned long long>(hash));
          CHECK(hash == expected_ref_text_hashes[stage]);
        }
        std::printf("\n");
        std::fflush(stdout);
      }
      const double ref_vk_load_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - ref_vk_load_begin)
                                        .count();
      sampler::FlowScheduler ref_vk_video(12.0f), ref_vk_audio(3.0f);
      ref_vk_video.set_timesteps(ref_schedule_points);
      ref_vk_audio.set_timesteps(ref_schedule_points);
      std::vector<uint64_t> ref_hashes;
      size_t ref_boundary = 0;
      DeviceTensor ref_packed_tap;
      DeviceTensor ref_main_tap;
      H3TransformerForwardReplayTaps ref_forward_taps{};
      const H3TransformerForwardReplayTaps* ref_forward_taps_ptr = nullptr;
      if (!ref_graph_capture.empty()) {
        const uint64_t tap_shape[]{static_cast<uint64_t>(ref_packed.layout.total_rows()),
                                   static_cast<uint64_t>(config.main.block.hidden)};
        const TensorLayout tap_layout = TensorLayout::contiguous(tap_shape, 2);
        ref_packed_tap = vk.allocate(tap_layout, ScalarType::kBFloat16);
        ref_main_tap = vk.allocate(tap_layout, ScalarType::kBFloat16);
        ref_forward_taps.packed_input = &ref_packed_tap;
        ref_forward_taps.main_final = &ref_main_tap;
        ref_forward_taps_ptr = &ref_forward_taps;
        CHECK(ref_vk.required_step_operators(ref_forward_taps_ptr) ==
              ref_vk.required_step_operators() + 2u);
      }
      const auto ref_vk_run_begin = std::chrono::steady_clock::now();
      const ExactH3DenoiseResult ref_vk_result = ref_vk.run(
          ref_vk_video, ref_vk_audio, {},
          [&](uint32_t step, const std::vector<float>& video_rows,
              const std::vector<float>& audio_rows) {
            CHECK(step == ref_boundary);
            if (!reuse_ref_graph)
              CHECK(ref_boundary < ref_cuda_boundaries.size());
            if (!reuse_ref_graph && ref_boundary < ref_cuda_boundaries.size()) {
              if (production_shape) {
                const Boundary& cuda_boundary = ref_cuda_boundaries[ref_boundary];
                auto report_first_difference = [](const char* name, const std::vector<float>& a,
                                                  const std::vector<float>& b, size_t row_width) {
                  auto bits = [](float value) {
                    uint32_t result = 0;
                    std::memcpy(&result, &value, sizeof(result));
                    return result;
                  };
                  const size_t common = std::min(a.size(), b.size());
                  size_t index = 0;
                  while (index < common && bits(a[index]) == bits(b[index]))
                    ++index;
                  if (index != common || a.size() != b.size()) {
                    std::printf("    Ref2VA %s first difference element %zu row %zu "
                                "column %zu CUDA %.9g (0x%08x) Vulkan %.9g (0x%08x); "
                                "sizes %zu/%zu\n",
                                name, index, index / row_width, index % row_width,
                                index < a.size() ? a[index] : 0.0f,
                                index < a.size() ? bits(a[index]) : 0,
                                index < b.size() ? b[index] : 0.0f,
                                index < b.size() ? bits(b[index]) : 0, a.size(), b.size());
                  }
                };
                std::printf("    Ref2VA boundary %zu CUDA/Vulkan FNV64 "
                            "%016llx/%016llx\n",
                            ref_boundary,
                            static_cast<unsigned long long>(
                                joined_hash(cuda_boundary.video, cuda_boundary.audio)),
                            static_cast<unsigned long long>(joined_hash(video_rows, audio_rows)));
                report_first_difference("video", cuda_boundary.video, video_rows, 96);
                report_first_difference("audio", cuda_boundary.audio, audio_rows, 32);
              }
              CHECK(video_rows == ref_cuda_boundaries[ref_boundary].video);
              CHECK(audio_rows == ref_cuda_boundaries[ref_boundary].audio);
            }
            ref_hashes.push_back(joined_hash(video_rows, audio_rows));
            ++ref_boundary;
          },
          ref_forward_taps_ptr);
      const double ref_vk_run_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - ref_vk_run_begin)
                                       .count();
      CHECK(ref_boundary == ref_expected_steps && ref_hashes.size() == ref_expected_steps);
      if (!reuse_ref_graph)
        CHECK(ref_vk_result.video_rows == ref_cuda_result.video_rows &&
              ref_vk_result.audio_rows == ref_cuda_result.audio_rows);
      if (ref_forward_taps_ptr) {
        H3MainGraphCaptureHeader captured{};
        std::ifstream capture_stream(ref_graph_capture, std::ios::binary);
        CHECK(static_cast<bool>(
            capture_stream.read(reinterpret_cast<char*>(&captured), sizeof(captured))));
        CHECK(std::memcmp(captured.magic, "VFH3GRF\0", 8) == 0 && captured.version == 1 &&
              captured.sequence == static_cast<uint32_t>(ref_packed.layout.total_rows()) &&
              captured.residual_elements ==
                  uint64_t(ref_packed.layout.total_rows()) * config.main.block.hidden);
        std::vector<uint16_t> native_packed(captured.residual_elements);
        std::vector<uint16_t> native_main(captured.residual_elements);
        vk.download_bytes(ref_packed_tap, native_packed.data(),
                          native_packed.size() * sizeof(uint16_t));
        vk.download_bytes(ref_main_tap, native_main.data(), native_main.size() * sizeof(uint16_t));
        auto bf16_hash = [](const std::vector<uint16_t>& values) {
          uint64_t hash = 1469598103934665603ull;
          for (uint16_t value : values) {
            hash ^= value & 0xffu;
            hash *= 1099511628211ull;
            hash ^= value >> 8;
            hash *= 1099511628211ull;
          }
          return hash;
        };
        const uint64_t packed_hash = bf16_hash(native_packed);
        const uint64_t main_hash = bf16_hash(native_main);
        std::printf("    Ref2VA native packed/main CUDA/Vulkan FNV64 "
                    "%016llx/%016llx %016llx/%016llx\n",
                    static_cast<unsigned long long>(captured.input_fnv64),
                    static_cast<unsigned long long>(packed_hash),
                    static_cast<unsigned long long>(captured.final_fnv64),
                    static_cast<unsigned long long>(main_hash));
        if (packed_hash != captured.input_fnv64) {
          std::vector<uint16_t> cuda_packed(captured.residual_elements);
          CHECK(static_cast<bool>(capture_stream.read(
              reinterpret_cast<char*>(cuda_packed.data()),
              static_cast<std::streamsize>(cuda_packed.size() * sizeof(uint16_t)))));
          size_t element = 0;
          while (element < cuda_packed.size() && cuda_packed[element] == native_packed[element])
            ++element;
          const uint32_t row = static_cast<uint32_t>(element / config.main.block.hidden);
          const uint32_t column = static_cast<uint32_t>(element % config.main.block.hidden);
          const char* category = "uncovered";
          int32_t tag = -1;
          for (size_t i = 0; i < ref_packed.indices.text.size(); ++i)
            if (ref_packed.indices.text[i] == static_cast<int32_t>(row)) {
              category = "text";
              tag = ref_text_tags[i];
              break;
            }
          if (tag == -1 &&
              std::find(ref_packed.indices.audio.begin(), ref_packed.indices.audio.end(),
                        static_cast<int32_t>(row)) != ref_packed.indices.audio.end())
            category = "audio";
          if (tag == -1 &&
              std::find(ref_packed.indices.video.begin(), ref_packed.indices.video.end(),
                        static_cast<int32_t>(row)) != ref_packed.indices.video.end())
            category = "video";
          std::printf("    Ref2VA packed first difference row %u column %u "
                      "category %s tag %d CUDA/Vulkan 0x%04x/0x%04x\n",
                      row, column, category, tag,
                      element < cuda_packed.size() ? cuda_packed[element] : 0,
                      element < native_packed.size() ? native_packed[element] : 0);
        }
        CHECK(packed_hash == captured.input_fnv64);
        CHECK(main_hash == captured.final_fnv64);
      }
      const std::array<uint64_t, 3> expected_ref_hashes{
          0x74f3fe411d378300ull, 0x2703fe32812c4f7dull, 0xdd3cc78b26122404ull};
      if (!production_shape)
        CHECK(std::equal(ref_hashes.begin(), ref_hashes.end(), expected_ref_hashes.begin()));
      const uint64_t ref_used = vk.pooled_used_bytes();
      const uint64_t ref_reserved = vk.reserved_bytes();
      const uint64_t ref_descriptors = vk.descriptor_set_allocations();
      if (!production_shape) {
        ref_vk.prepare(ref_prompt.data(), ref_prompt.size(), ref_video.data(), ref_video.size(),
                       audio.data(), audio.size());
        const ExactH3DenoiseResult ref_repeat = ref_vk.run(ref_vk_video, ref_vk_audio);
        CHECK(ref_repeat.video_rows == ref_vk_result.video_rows &&
              ref_repeat.audio_rows == ref_vk_result.audio_rows);
        CHECK(vk.pooled_used_bytes() == ref_used && vk.reserved_bytes() == ref_reserved &&
              vk.descriptor_set_allocations() == ref_descriptors);
      }
      std::printf(
          "  real Ref2VA exact S%u x%zu: CUDA load/run %.3f/%.3f ms, Vulkan %.3f/%.3f ms, first %016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, desc %llu; text/condition/target-a/target-v %u/%u/%u/%u\n",
          ref_vk_config.transformer.main.block.sequence, ref_expected_steps, ref_cuda_load_ms,
          ref_cuda_run_ms, ref_vk_load_ms, ref_vk_run_ms,
          static_cast<unsigned long long>(ref_hashes[0]),
          double(ref_vk.persistent_bytes()) / 1048576.0, double(ref_vk.scratch_bytes()) / 1048576.0,
          double(ref_vk.peak_device_bytes()) / 1048576.0,
          static_cast<unsigned long long>(ref_descriptors), ref_text_rows, condition_rows,
          header.audio_rows, header.video_rows);
      ref_vk.unload();
      CHECK(!ref_vk.loaded() && !ref_vk.prepared() && ref_vk.persistent_bytes() == 0u &&
            ref_vk.scratch_bytes() == 0u && ref_vk.peak_device_bytes() == 0u &&
            vk.pooled_used_bytes() < ref_used);
    }
  }

  const char* captured_vertical = std::getenv("SLOPFAB_DIT_VERTICAL_REAL");
  const char* qwen_vertical = std::getenv("SLOPFAB_QWEN_VERTICAL_REAL");
  const char* ref2va_vertical = std::getenv("SLOPFAB_REF2VA_VERTICAL_REAL");
  const char* ref2va_nonsquare_vertical = std::getenv("SLOPFAB_REF2VA_NONSQUARE_VERTICAL_REAL");
  if ((captured_vertical && captured_vertical[0] == '1') ||
      (qwen_vertical && qwen_vertical[0] == '1') ||
      (ref2va_vertical && ref2va_vertical[0] == '1') ||
      (ref2va_nonsquare_vertical && ref2va_nonsquare_vertical[0] == '1')) {
    const bool nonsquare_reference =
        ref2va_nonsquare_vertical && ref2va_nonsquare_vertical[0] == '1';
    const bool reference_prompt =
        (ref2va_vertical && ref2va_vertical[0] == '1') || nonsquare_reference;
    const bool normal_prompt = reference_prompt || (qwen_vertical && qwen_vertical[0] == '1');
    const std::filesystem::path video_vae_path =
        "weights/vae/minimax_h3_video_vae_fp16.safetensors";
    const std::filesystem::path audio_vae_path =
        "weights/vae/minimax_h3_audio_vae_fp32.safetensors";
    CHECK(std::filesystem::exists(video_vae_path) && std::filesystem::exists(audio_vae_path));
    const std::string unique =
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const std::filesystem::path temp = std::filesystem::temp_directory_path();
    const std::filesystem::path prompt_path =
        temp / ("slopfab-g7d-prompt-" + unique + ".safetensors");
    const std::filesystem::path init_path = temp / ("slopfab-g7d-init-" + unique + ".safetensors");
    const std::filesystem::path cuda_out = temp / ("slopfab-g7d-cuda-" + unique + ".raw");
    const std::filesystem::path vulkan_out = temp / ("slopfab-g7d-vulkan-" + unique + ".raw");
    const std::filesystem::path reference_path = temp / ("slopfab-g9-reference-" + unique + ".ppm");
    write_safetensors(prompt_path.string(), {{"prompt_embedding",
                                              {static_cast<int64_t>(header.text_rows),
                                               static_cast<int64_t>(header.text_dim)},
                                              prompt}});
    write_safetensors(
        init_path.string(),
        {{"video_rows",
          {static_cast<int64_t>(header.video_rows), static_cast<int64_t>(header.video_dim)},
          video},
         {"audio_rows",
          {static_cast<int64_t>(header.audio_rows), static_cast<int64_t>(header.audio_dim)},
          audio}});
    GenerateRequest request;
    request.canvas_width = 256;
    request.canvas_height = 256;
    request.num_frames = 22;
    request.num_inference_steps = 4;
    request.seed = 424242;
    if (normal_prompt) {
      request.prompt = "A copper airship glides above a snowy forest at sunrise.";
      request.text_encoder_path = "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
      request.tokenizer_path = "ref/text_encoder/tokenizer.json";
    }
    request.transformer_path =
        reference_prompt ? "weights/transformer/minimax_h3_ref2va_pruned_nvfp4.safetensors"
                         : checkpoint_path.string();
    request.video_vae_path = video_vae_path.string();
    request.audio_vae_path = audio_vae_path.string();
    request.raw_output = true;
    if (reference_prompt) {
      std::ofstream ppm(reference_path, std::ios::binary);
      if (nonsquare_reference) {
        constexpr int source_width = 16;
        constexpr int source_height = 9;
        std::vector<uint8_t> pixels(source_width * source_height * 3);
        for (int y = 0; y < source_height; ++y) {
          for (int x = 0; x < source_width; ++x) {
            const size_t offset = size_t(y * source_width + x) * 3;
            pixels[offset + 0] = static_cast<uint8_t>(17 * x + 11 * y + 3);
            pixels[offset + 1] = static_cast<uint8_t>(7 * x + 23 * y + 41);
            pixels[offset + 2] = static_cast<uint8_t>(29 * x + 5 * y + 97);
          }
        }
        ppm << "P6\n" << source_width << " " << source_height << "\n255\n";
        ppm.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
        int keyframe_h = 0, keyframe_w = 0;
        resolve_reference_image_size(source_width, source_height, &keyframe_h, &keyframe_w);
        CHECK(keyframe_w == 3648 && keyframe_h == 2048);
        const auto qwen_grid = text::qwen3vl_conditioning_grid(keyframe_w, keyframe_h);
        CHECK(qwen_grid.width == 170 && qwen_grid.height == 94 && qwen_grid.patch_count() == 15980);
      } else {
        const std::array<uint8_t, 3> pixel{0x31, 0x97, 0xe3};
        ppm << "P6\n1 1\n255\n";
        ppm.write(reinterpret_cast<const char*>(pixel.data()), pixel.size());
      }
      CHECK(static_cast<bool>(ppm));
      request.reference_image_paths.push_back(reference_path.string());
    }
    const GeneratePlan vertical_plan = resolve_plan(request);
    CHECK(vertical_plan.layout.num_video_rows == static_cast<int>(header.video_rows) &&
          vertical_plan.layout.num_audio_rows == static_cast<int>(header.audio_rows) &&
          vertical_plan.num_model_evaluations() == 3);
    if (normal_prompt && !reference_prompt) {
      // A conditioner cache hit is allowed only within one explicit execution
      // authority. Cancel at the next stage so this exercises the public
      // run_generate cache without loading the transformer or either VAE.
      auto stop_after_conditioning = +[](RunStage stage, int, int, void*) {
        return stage != RunStage::kTransformerLoad;
      };
      GenerateRequest cache_request = request;
      cache_request.prompt += " cache-authority-" + unique;
      const GeneratePlan cache_plan = resolve_plan(cache_request);
      auto conditioning_only = [&](DeviceBackend backend, AttentionMode arithmetic, bool release,
                                   bool captured = false) {
        RunOptions options;
        options.inference_backend = backend;
        options.attention_mode = arithmetic;
        options.reuse_models = true;
        options.release_reused_models = release;
        if (captured)
          options.prompt_embedding_path = prompt_path.string();
        options.verbose = false;
        options.on_progress = stop_after_conditioning;
        const RunResult result = run_generate(cache_request, cache_plan, options);
        CHECK(result.cancelled && !result.ok);
        return result.conditioner_executed;
      };
      CHECK(!conditioning_only(DeviceBackend::kVulkan, AttentionMode::kExact, false, true));
      CHECK(conditioning_only(DeviceBackend::kVulkan, AttentionMode::kExact, false));
      CHECK(!conditioning_only(DeviceBackend::kVulkan, AttentionMode::kExact, true));

      CHECK(conditioning_only(DeviceBackend::kCuda, AttentionMode::kFlash2, false));
      CHECK(conditioning_only(DeviceBackend::kVulkan, AttentionMode::kExact, false));
      CHECK(!conditioning_only(DeviceBackend::kVulkan, AttentionMode::kExact, true));

      CHECK(conditioning_only(DeviceBackend::kCuda, AttentionMode::kExact, false));
      CHECK(conditioning_only(DeviceBackend::kVulkan, AttentionMode::kExact, false));
      CHECK(!conditioning_only(DeviceBackend::kVulkan, AttentionMode::kExact, true));
    }

    struct CapturedSamples {
      PixelBuffer video;
      std::vector<float> audio;
      int channels = 0;
      int frames = 0;
      int height = 0;
      int width = 0;
      int audio_channels = 0;
      int sample_rate = 0;
    } cuda_samples, vulkan_samples;

    auto capture_samples = +[](RunSamples& samples, void* userdata) {
      auto* captured = static_cast<CapturedSamples*>(userdata);
      captured->channels = samples.channels;
      captured->frames = samples.frames;
      captured->height = samples.height;
      captured->width = samples.width;
      captured->audio_channels = samples.audio_channels;
      captured->sample_rate = samples.audio_sample_rate;
      if (samples.video)
        captured->video = *samples.video;
      if (samples.audio)
        captured->audio = *samples.audio;
      return false;
    };
    auto run_vertical = [&](DeviceBackend backend, const std::filesystem::path& output,
                            CapturedSamples& samples) {
      request.out_path = output.string();
      RunOptions options;
      options.inference_backend = backend;
      options.attention_mode = AttentionMode::kExact;
      if (!normal_prompt)
        options.prompt_embedding_path = prompt_path.string();
      options.init_latents_path = init_path.string();
      options.verbose = nonsquare_reference;
      options.on_samples = capture_samples;
      options.hook_userdata = &samples;
      const auto begin = std::chrono::steady_clock::now();
      const RunResult result = run_generate(request, vertical_plan, options);
      const double elapsed =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
              .count();
      CHECK_MSG(result.ok, "real vertical %s failed: %s",
                backend == DeviceBackend::kCuda ? "CUDA" : "Vulkan", result.message.c_str());
      CHECK(result.conditioner_executed == normal_prompt);
      CHECK(result.steps_computed == 3 && result.steps_skipped == 0 && result.outputs.size() == 2u);
      return std::pair<RunResult, double>{result, elapsed};
    };
    const auto cuda_vertical = run_vertical(DeviceBackend::kCuda, cuda_out, cuda_samples);
    const auto vulkan_vertical = run_vertical(DeviceBackend::kVulkan, vulkan_out, vulkan_samples);
    CHECK(cuda_samples.channels == vulkan_samples.channels &&
          cuda_samples.frames == vulkan_samples.frames &&
          cuda_samples.height == vulkan_samples.height &&
          cuda_samples.width == vulkan_samples.width &&
          cuda_samples.audio_channels == vulkan_samples.audio_channels &&
          cuda_samples.sample_rate == vulkan_samples.sample_rate);
    CHECK(cuda_samples.video == vulkan_samples.video);
    CHECK(cuda_samples.audio == vulkan_samples.audio);
    auto read_file = [](const std::string& path) {
      std::ifstream input(path, std::ios::binary | std::ios::ate);
      if (!input)
        throw std::runtime_error("cannot open vertical output " + path);
      const std::streamsize bytes = input.tellg();
      input.seekg(0);
      std::vector<uint8_t> result(static_cast<size_t>(bytes));
      if (!input.read(reinterpret_cast<char*>(result.data()), bytes))
        throw std::runtime_error("cannot read vertical output " + path);
      return result;
    };
    const std::vector<uint8_t> cuda_y4m = read_file(cuda_vertical.first.outputs[0]);
    const std::vector<uint8_t> cuda_wav = read_file(cuda_vertical.first.outputs[1]);
    const std::vector<uint8_t> vulkan_y4m = read_file(vulkan_vertical.first.outputs[0]);
    const std::vector<uint8_t> vulkan_wav = read_file(vulkan_vertical.first.outputs[1]);
    CHECK(cuda_y4m == vulkan_y4m && cuda_wav == vulkan_wav);
    const uint64_t pixel_hash =
        fnv_bytes(cuda_samples.video.data(), cuda_samples.video.size() * sizeof(float));
    const uint64_t pcm_hash =
        fnv_bytes(cuda_samples.audio.data(), cuda_samples.audio.size() * sizeof(float));
    const uint64_t y4m_hash = fnv_bytes(cuda_y4m.data(), cuda_y4m.size());
    const uint64_t wav_hash = fnv_bytes(cuda_wav.data(), cuda_wav.size());
    if (!normal_prompt) {
      CHECK(pixel_hash == 0x714a67162495817eull);
      CHECK(pcm_hash == 0x671f1519e5d0cea1ull);
      CHECK(y4m_hash == 0xbb480c4fd04ac34aull);
      CHECK(wav_hash == 0xa447eb6d02620637ull);
    } else if (!reference_prompt) {
      CHECK(pixel_hash == 0x52f6148fa46959f8ull);
      CHECK(pcm_hash == 0xb2e09a49fc952e5eull);
      CHECK(y4m_hash == 0x2f595da467a8ac60ull);
      CHECK(wav_hash == 0xe0d84106a3018c29ull);
    } else if (!nonsquare_reference) {
      CHECK(pixel_hash == 0x821ea69c8412d682ull);
      CHECK(pcm_hash == 0xb92888172863f265ull);
      CHECK(y4m_hash == 0x6b6a71322a7033cfull);
      CHECK(wav_hash == 0xce580a62a54051d2ull);
    } else {
      CHECK(pixel_hash == 0xb47a2b3e91e9c744ull);
      CHECK(pcm_hash == 0x334e7829e92a479full);
      CHECK(y4m_hash == 0xdc958cbd7468dd84ull);
      CHECK(wav_hash == 0xec54a7c6ac251e5full);
    }
    std::printf(
        "  real exact %s vertical x3: CUDA/Vulkan %.3f/%.3f ms, pixels/pcm/y4m/wav %016llx/%016llx/%016llx/%016llx\n",
        nonsquare_reference
            ? "nonsquare-reference-prompt"
            : (reference_prompt ? "reference-prompt"
                                : (normal_prompt ? "normal-prompt" : "captured-prompt")),
        cuda_vertical.second, vulkan_vertical.second, static_cast<unsigned long long>(pixel_hash),
        static_cast<unsigned long long>(pcm_hash), static_cast<unsigned long long>(y4m_hash),
        static_cast<unsigned long long>(wav_hash));
    std::error_code ignored;
    for (const std::filesystem::path& path :
         {prompt_path, init_path, reference_path,
          std::filesystem::path(cuda_vertical.first.outputs[0]),
          std::filesystem::path(cuda_vertical.first.outputs[1]),
          std::filesystem::path(vulkan_vertical.first.outputs[0]),
          std::filesystem::path(vulkan_vertical.first.outputs[1])})
      std::filesystem::remove(path, ignored);
  }

  if (const char* production = std::getenv("SLOPFAB_DIT_TRANSFORMER_PRODUCTION");
      production && production[0] == '1') {
    constexpr uint32_t prod_sequence = 9864;
    constexpr uint32_t prod_audio_rows = 74;
    constexpr uint32_t prod_video_rows = prod_sequence - 4 - prod_audio_rows;
    ExactH3TransformerConfig prod_config = config;
    prod_config.main.block.sequence = prod_sequence;
    prod_config.video_rows = prod_video_rows;
    prod_config.audio_rows = prod_audio_rows;
    ExactH3Transformer prod = ExactH3Transformer::create(vk, prod_config);
    const auto prod_load_begin = std::chrono::steady_clock::now();
    prod.load(checkpoint);
    const double prod_load_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - prod_load_begin)
                                    .count();
    DeviceTensor prod_prompt = tensor2(4, 5120);
    DeviceTensor prod_video = tensor2(prod_video_rows, 96);
    DeviceTensor prod_audio = tensor2(prod_audio_rows, 32);
    DeviceTensor prod_selectors = tensor1(prod_sequence, ScalarType::kInt32);
    DeviceTensor prod_code = tensor2(1, 8);
    DeviceTensor prod_cosine = tensor2(prod_sequence, 96);
    DeviceTensor prod_sine = tensor2(prod_sequence, 96);
    DeviceTensor prod_video_ts = tensor1(prod_video_rows, ScalarType::kInt32);
    DeviceTensor prod_audio_ts = tensor1(prod_audio_rows, ScalarType::kInt32);
    DeviceTensor prod_video_out = tensor2(prod_video_rows, 96);
    DeviceTensor prod_audio_out = tensor2(prod_audio_rows, 32);
    std::vector<float> prod_video_values(size_t(prod_video_rows) * 96);
    std::vector<float> prod_audio_values(size_t(prod_audio_rows) * 32);
    for (size_t i = 0; i < prod_video_values.size(); ++i)
      prod_video_values[i] = float(int(i % 251) - 125) / 128.0f;
    for (size_t i = 0; i < prod_audio_values.size(); ++i)
      prod_audio_values[i] = float(int(i % 127) - 63) / 64.0f;
    std::vector<int32_t> prod_selector_values(prod_sequence, 0);
    std::vector<int32_t> prod_video_ts_values(prod_video_rows, 0);
    std::vector<int32_t> prod_audio_ts_values(prod_audio_rows, 0);
    std::vector<float> prod_cos_values(size_t(prod_sequence) * 96, 1.0f);
    std::vector<float> prod_sin_values(prod_cos_values.size(), 0.0f);
    vk.upload(prod_prompt, prompt.data(), prompt.size());
    vk.upload(prod_video, prod_video_values.data(), prod_video_values.size());
    vk.upload(prod_audio, prod_audio_values.data(), prod_audio_values.size());
    vk.upload_bytes(prod_selectors, prod_selector_values.data(), prod_selector_values.size() * 4);
    vk.upload(prod_code, code.data(), code.size());
    vk.upload(prod_cosine, prod_cos_values.data(), prod_cos_values.size());
    vk.upload(prod_sine, prod_sin_values.data(), prod_sin_values.size());
    vk.upload_bytes(prod_video_ts, prod_video_ts_values.data(), prod_video_ts_values.size() * 4);
    vk.upload_bytes(prod_audio_ts, prod_audio_ts_values.data(), prod_audio_ts_values.size() * 4);
    prod.prepare_text(prod_prompt);
    auto run_prod = [&] {
      const auto begin = std::chrono::steady_clock::now();
      TensorBatch batch = vk.begin_batch();
      prod.record_forward(batch, prod_video, prod_audio, prod_selectors, prod_code, prod_cosine,
                          prod_sine, prod_video_ts, prod_audio_ts, prod_video_out, prod_audio_out);
      CHECK(batch.remaining_operator_capacity() == 581u);
      batch.submit().wait();
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
          .count();
    };
    const double prod_first_ms = run_prod();
    std::vector<float> prod_video_result(prod_video_values.size());
    std::vector<float> prod_audio_result(prod_audio_values.size());
    vk.download(prod_video_out, prod_video_result.data(), prod_video_result.size());
    vk.download(prod_audio_out, prod_audio_result.data(), prod_audio_result.size());
    uint64_t prod_hash = 1469598103934665603ull;
    auto append_hash = [&](const void* data, size_t bytes) {
      const auto* p = static_cast<const uint8_t*>(data);
      for (size_t i = 0; i < bytes; ++i) {
        prod_hash ^= p[i];
        prod_hash *= 1099511628211ull;
      }
    };
    append_hash(prod_video_result.data(), prod_video_result.size() * 4);
    append_hash(prod_audio_result.data(), prod_audio_result.size() * 4);
    CHECK(prod_hash == 0xa998bb5ff7a03383ull);
    const uint64_t prod_used = vk.pooled_used_bytes();
    const uint64_t prod_reserved = vk.reserved_bytes();
    const uint64_t prod_descriptors = vk.descriptor_set_allocations();
    const double prod_repeat_ms = run_prod();
    std::vector<float> repeated_video(prod_video_values.size());
    std::vector<float> repeated_audio(prod_audio_values.size());
    vk.download(prod_video_out, repeated_video.data(), repeated_video.size());
    vk.download(prod_audio_out, repeated_audio.data(), repeated_audio.size());
    CHECK(repeated_video == prod_video_result && repeated_audio == prod_audio_result);
    CHECK(vk.pooled_used_bytes() == prod_used && vk.reserved_bytes() == prod_reserved &&
          vk.descriptor_set_allocations() == prod_descriptors);
    std::printf(
        "  production H3 transformer S9864: load %.3f ms, Vulkan first/repeat %.3f/%.3f ms, final %016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, pool %.2f/%.2f MiB, descriptors %llu\n",
        prod_load_ms, prod_first_ms, prod_repeat_ms, static_cast<unsigned long long>(prod_hash),
        double(prod.persistent_bytes()) / 1048576.0, double(prod.scratch_bytes()) / 1048576.0,
        double(prod.peak_device_bytes()) / 1048576.0, double(vk.pooled_used_bytes()) / 1048576.0,
        double(vk.reserved_bytes()) / 1048576.0,
        static_cast<unsigned long long>(vk.descriptor_set_allocations()));
  }
}
