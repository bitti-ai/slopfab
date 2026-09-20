#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_dit_real_main50_capture_replay, "integration") {
  using namespace slopfab;
  using namespace slopfab::dit;
  using namespace slopfab::vulkan;
  const char* replay_capture =
      std::getenv("SLOPFAB_REF2VA_GRAPH_REPLAY_CAPTURE");
  const bool dynamic_capture = replay_capture && replay_capture[0] != '\0';
  const std::filesystem::path checkpoint_path = dynamic_capture
      ? std::filesystem::path(
            "weights/transformer/minimax_h3_ref2va_pruned_nvfp4.safetensors")
      : std::filesystem::path(
            "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors");
  const std::filesystem::path capture_path = dynamic_capture
      ? std::filesystem::path(replay_capture)
      : std::filesystem::path(
            "tests/data/h3_main50_step0_seed424242_256.vfh3g");
  if (!std::filesystem::exists(checkpoint_path) ||
      !std::filesystem::exists(capture_path) || !Instance::available()) {
    SKIP_MISSING_FIXTURE("unavailable prerequisite: !std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(capture_path) || !Instance::available()");
    return;
  }
  std::ifstream stream(capture_path, std::ios::binary | std::ios::ate);
  const std::streamsize capture_bytes = stream.tellg();
  stream.seekg(0);
  std::vector<uint8_t> capture(static_cast<size_t>(capture_bytes));
  CHECK(static_cast<bool>(stream.read(
      reinterpret_cast<char*>(capture.data()), capture_bytes)));
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_capture_sha{
      0xbb,0x15,0x68,0x92,0x83,0x66,0x94,0x96,0xc5,0x61,0x4e,0xc3,0x0f,0xaa,0xf0,0x88,
      0x12,0x9d,0x8c,0xb8,0x97,0xcb,0x34,0x0a,0xe5,0x36,0x37,0x3a,0xd4,0x2c,0xb3,0x62};
  if (!dynamic_capture)
    CHECK(sha256_mapping(capture.data(), capture.size()) ==
          expected_capture_sha);
#endif
  if (capture.size() < sizeof(H3MainGraphCaptureHeader))
    throw std::runtime_error("truncated H3 main graph capture");
  H3MainGraphCaptureHeader header{};
  std::memcpy(&header, capture.data(), sizeof(header));
  CHECK(std::memcmp(header.magic, "VFH3GRF\0", 8) == 0);
  CHECK(header.version == 1 && header.header_bytes == sizeof(header));
  CHECK(header.sequence > 0 && header.hidden == 5376 &&
        header.heads == 56 && header.head_dim == 128 && header.ffn == 14336);
  CHECK(header.timesteps > 0 && header.timesteps <= 4 &&
        header.modalities == 3 && header.adaln_rank == 8 &&
        header.layers == 50 && header.denoise_step == 0);
  if (!dynamic_capture) {
    CHECK(header.sequence == 526 && header.timesteps == 1 &&
          header.range_values == 20);
    CHECK(header.input_fnv64 == 0x8e130a074619290full);
    CHECK(header.final_fnv64 == 0x94d7dfcfcef6f4ceull);
  }
  const std::array<uint64_t, 50> expected_boundaries{
      0xf7d651766756f015ull,0x09afbb821887020bull,0x779869da69c31d75ull,
      0x8b76a7261ab91253ull,0x1df1afd29841879cull,0x61db215dee7be558ull,
      0x93bdf2ff13a3ba62ull,0xa8a94eae981cddb7ull,0x23e0dd2de693a165ull,
      0xdb0e0f8aa5a49972ull,0x1ee2ee965c134dc1ull,0x4bc93b42a31b9a63ull,
      0xed1f848054d1213cull,0xcd0c9af21969802cull,0x1c2dbc71eddb6bbaull,
      0x4e5096ae2df1ef5full,0xddca1935d085d1d1ull,0xaa8ed6a379181141ull,
      0xa1ae4a76ac4cf58aull,0xddfb16645d9e3ddaull,0x8d7e09064314ab3cull,
      0x50f0367cff5988b9ull,0x28be37c299d8d34dull,0xef16ad91e90c119aull,
      0x47edfc041ca17670ull,0x0d42e4191058937dull,0x53ffb9a8f3a2809eull,
      0x6bf28610fbf92805ull,0x71b964652480244dull,0xbd860e69ea9f1bfeull,
      0x3f1fe901496195cfull,0x5b04fa6007303ccfull,0xb97828cd5a0c0ddfull,
      0x9be29ebd17324ad8ull,0x9593307ba6dca3faull,0x25b1d163a816d621ull,
      0xff23adde8d9a9e50ull,0xc779a185f7d8aa8aull,0xf1951f02921680a6ull,
      0x18b7758650a17769ull,0x7ffd3c6465091afbull,0x9b99975035dcf76dull,
      0x2977798f27e1a904ull,0xdace59bec11d235aull,0x6a30ef6d8415272full,
      0xbdb70b87dcc70527ull,0xabd55e9b1b59d6caull,0x2220fd508ca8a60cull,
      0xd8f6dc81376879f7ull,0x94d7dfcfcef6f4ceull};
  if (!dynamic_capture)
    for (uint32_t layer = 0; layer < 50; ++layer)
      CHECK(header.boundary_fnv64[layer] == expected_boundaries[layer]);

  size_t cursor = sizeof(header);
  auto take = [&](auto& values, size_t count) {
    using Value = typename std::decay_t<decltype(values)>::value_type;
    if (count > (capture.size() - cursor) / sizeof(Value))
      throw std::runtime_error("truncated H3 graph capture payload");
    values.resize(count);
    std::memcpy(values.data(), capture.data() + cursor, count * sizeof(Value));
    cursor += count * sizeof(Value);
  };
  std::vector<uint16_t> input, expected_final;
  std::vector<int32_t> selectors, range_values;
  std::vector<float> code, cosine, sine;
  take(input, static_cast<size_t>(header.residual_elements));
  take(selectors, header.sequence);
  take(code, size_t(header.timesteps) * header.adaln_rank);
  take(cosine, static_cast<size_t>(header.rope_elements));
  take(sine, static_cast<size_t>(header.rope_elements));
  take(range_values, header.range_values);
  take(expected_final, static_cast<size_t>(header.residual_elements));
  CHECK(cursor == capture.size());
  CHECK(std::any_of(cosine.begin(), cosine.end(), [](float x) { return x != 1.0f; }));
  CHECK(std::any_of(sine.begin(), sine.end(), [](float x) { return x != 0.0f; }));

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_fl_checkpoint_sha{
      0x6a,0xb7,0xf0,0xc4,0x81,0x41,0xe7,0x91,0x9b,0x32,0xf9,0x25,0xca,0x3d,0xef,0x22,
      0xe0,0x6a,0x6a,0xeb,0xeb,0x9e,0x0b,0x6f,0x5a,0x0b,0xe0,0xfe,0x84,0x09,0x97,0x6f};
  const std::array<uint8_t, 32> expected_ref_checkpoint_sha{
      0x8e,0xea,0x02,0xf4,0x39,0x02,0xe6,0x99,0x04,0x99,0x0c,0x44,0x05,0x96,0x8d,0x01,
      0xa1,0x3c,0x66,0x56,0xaa,0x39,0x2d,0x37,0xab,0x80,0x33,0x1e,0x79,0xb5,0xdf,0x2f};
  CHECK(sha256_mapping(checkpoint.mapping_base(), checkpoint.file_size()) ==
        (dynamic_capture ? expected_ref_checkpoint_sha
                         : expected_fl_checkpoint_sha));
#endif
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty()");
    return;
  }
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 ||
      !info.storage_buffer_16bit ||
      !info.cooperative_matrix_bf16_f32_16x16x16) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 || !info.storage_buffer_16bit || !info.cooperative_matrix_bf16_f32_16x16x16");
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
  H3MainGraphConfig graph_config;
  graph_config.layers = header.layers;
  graph_config.block.sequence = header.sequence;
  graph_config.block.hidden = header.hidden;
  graph_config.block.heads = header.heads;
  graph_config.block.head_dim = header.head_dim;
  graph_config.block.ffn = header.ffn;
  graph_config.block.timesteps = header.timesteps;
  graph_config.block.modalities = header.modalities;
  graph_config.block.adaln_rank = header.adaln_rank;
  const auto load_begin = std::chrono::steady_clock::now();
  ExactH3MainGraph graph = ExactH3MainGraph::create(vk, graph_config);
  graph.load(checkpoint);
  const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();
  CHECK(graph.layers() == 50u && graph.loaded());
  CHECK(graph.required_operators() == 1450u);
  const uint64_t residual_shape[] = {header.sequence, header.hidden};
  const uint64_t selector_shape[] = {header.sequence};
  const uint64_t code_shape[] = {header.timesteps, header.adaln_rank};
  const uint64_t rope_shape[] = {header.sequence, 96};
  auto bf = [&](const uint64_t* shape, uint32_t rank) {
    return vk.allocate(TensorLayout::contiguous(shape, rank), ScalarType::kBFloat16);
  };
  DeviceTensor tokens = bf(residual_shape, 2);
  DeviceTensor selector_tensor = vk.allocate(
      TensorLayout::contiguous(selector_shape, 1), ScalarType::kInt32);
  DeviceTensor code_tensor = vk.allocate(TensorLayout::contiguous(code_shape, 2));
  DeviceTensor cosine_tensor = vk.allocate(TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor sine_tensor = vk.allocate(TensorLayout::contiguous(rope_shape, 2));
  vk.upload_bytes(selector_tensor, selectors.data(), selectors.size() * 4);
  vk.upload(code_tensor, code.data(), code.size());
  vk.upload(cosine_tensor, cosine.data(), cosine.size());
  vk.upload(sine_tensor, sine.data(), sine.size());
  H3AttentionRanges ranges;
  if (header.range_values != 0)
    ranges = H3AttentionRanges::create(
        vk, header.sequence, range_values.data(), header.range_values);
  H3AttentionRanges* range_ptr = ranges ? &ranges : nullptr;
  std::vector<DeviceTensor> boundary_tensors;
  boundary_tensors.reserve(header.layers);
  for (uint32_t layer = 0; layer < header.layers; ++layer)
    boundary_tensors.push_back(bf(residual_shape, 2));
  H3MainGraphReplayTaps taps{boundary_tensors.data(), header.layers};
  CHECK(graph.required_operators(&taps) == 1500u);
  vk.upload_bytes(tokens, input.data(), input.size() * 2);
  const auto first_begin = std::chrono::steady_clock::now();
  TensorBatch batch = vk.begin_batch();
  graph.record(batch, tokens, selector_tensor, code_tensor, cosine_tensor,
               sine_tensor, range_ptr, &taps);
  CHECK(batch.remaining_operator_capacity() == 548u);
  batch.submit().wait();
  const double first_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - first_begin).count();
  auto fnv = [](const std::vector<uint16_t>& values) {
    uint64_t hash = 1469598103934665603ull;
    for (uint16_t bits : values) {
      hash ^= bits & 0xffu; hash *= 1099511628211ull;
      hash ^= bits >> 8; hash *= 1099511628211ull;
    }
    return hash;
  };
  for (uint32_t layer = 0; layer < header.layers; ++layer) {
    std::vector<uint16_t> boundary(header.residual_elements);
    vk.download_bytes(boundary_tensors[layer], boundary.data(), boundary.size() * 2);
    const uint64_t actual_hash = fnv(boundary);
    if (dynamic_capture && actual_hash != header.boundary_fnv64[layer])
      std::printf("  Ref2VA S%u first main mismatch layer %u CUDA/Vulkan "
                  "%016llx/%016llx\n", header.sequence, layer,
                  static_cast<unsigned long long>(
                      header.boundary_fnv64[layer]),
                  static_cast<unsigned long long>(actual_hash));
    CHECK(actual_hash == header.boundary_fnv64[layer]);
  }
  std::vector<uint16_t> final(header.residual_elements);
  vk.download_bytes(tokens, final.data(), final.size() * 2);
  CHECK(final == expected_final);
  CHECK(fnv(final) == header.final_fnv64);
  const uint64_t stable_used = vk.pooled_used_bytes();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  auto repeat = [&] {
    vk.upload_bytes(tokens, input.data(), input.size() * 2);
    const auto begin = std::chrono::steady_clock::now();
    TensorBatch next = vk.begin_batch();
    graph.record(next, tokens, selector_tensor, code_tensor, cosine_tensor,
                 sine_tensor, range_ptr);
    next.submit().wait();
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    std::vector<uint16_t> result(header.residual_elements);
    vk.download_bytes(tokens, result.data(), result.size() * 2);
    CHECK(result == expected_final);
    return elapsed;
  };
  const double repeat_ms = repeat();
  CHECK(vk.pooled_used_bytes() == stable_used);
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  (void)repeat();
  CHECK(vk.pooled_used_bytes() == stable_used);
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  std::printf(
      "  real H3 main50 S%u: load %.3f ms, Vulkan taps/repeat %.3f/%.3f ms, final %016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, pool %.2f/%.2f MiB, descriptors %llu\n",
      header.sequence, load_ms, first_ms, repeat_ms,
      static_cast<unsigned long long>(header.final_fnv64),
      double(graph.persistent_bytes()) / 1048576.0,
      double(graph.scratch_bytes()) / 1048576.0,
      double(graph.peak_device_bytes()) / 1048576.0,
      double(vk.pooled_used_bytes()) / 1048576.0,
      double(vk.reserved_bytes()) / 1048576.0,
      static_cast<unsigned long long>(vk.descriptor_set_allocations()));

}
