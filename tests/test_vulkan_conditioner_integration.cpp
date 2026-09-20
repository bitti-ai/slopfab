#include "detail/vulkan_fixture.h"

#if !defined(SLOPFAB_WITH_CUDA) || !SLOPFAB_WITH_CUDA
SLOPFAB_TEST_CATEGORY(vulkan_qwen_full50_real_l132_replay, "integration") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const std::filesystem::path source(SLOPFAB_TEST_SOURCE_DIR);
  const std::filesystem::path checkpoint_path = source /
      "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
  const std::filesystem::path capture_path = source /
      "tests/data/qwen_layer0_l132.vfqw";
  if (!std::filesystem::exists(checkpoint_path) ||
      !std::filesystem::exists(capture_path) || !Instance::available()) {
    SKIP_MISSING_FIXTURE("unavailable prerequisite: !std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(capture_path) || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty()");
    return;
  }
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 ||
      !info.shader_float16 || !info.storage_buffer_16bit ||
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
  context_options.max_batch_operators =
      std::getenv("SLOPFAB_QWEN_MULTIMODAL_REAL") ? 128 : 37;
  TensorContext context(device, context_options);
  const uint64_t cold_baseline = context.pooled_used_bytes();

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  const text::QwenLayerCapture capture =
      text::read_qwen_layer_capture(capture_path.string());
  ExactQwenTextEncoder encoder = ExactQwenTextEncoder::create(context);
  encoder.load(checkpoint);
  text::EncoderTrace trace;
  const auto begin = std::chrono::steady_clock::now();
  const text::PromptEmbedding output = encoder.encode(capture.token_ids, &trace);
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  constexpr std::array<uint64_t, 50> expected{
      0xfb3966de636ac098ull,0x16b0e53a0265959dull,
      0x7e2377403c49f851ull,0xdacca747d7db7ccaull,
      0xb3de1cdb9248573cull,0xcb7ef7852ef4ce79ull,
      0xfe09924dff391f53ull,0x0ac372b850d058d8ull,
      0xe818f084c10d99d5ull,0xeed04e74ca44abcfull,
      0xe3e63048102b3e14ull,0x6e5647916a5d3586ull,
      0x1b9a96cd7cb703c2ull,0xe60df16cc8c0390dull,
      0xcf624f216f397d84ull,0x2e680a56b7030b0aull,
      0x8b8e97be1f24e88eull,0x61e2e50339318a1eull,
      0x410a6b1ab0a9c0e4ull,0xaf70d3523ce3a525ull,
      0xfb6b26f66fc1eb31ull,0x732c1c3ed4d7e6a6ull,
      0x1b81b2dd36a55f5aull,0x01f09dd599e38e09ull,
      0xc93c6edb4d56f691ull,0xcb9bd628a2587064ull,
      0x30c8bbdab894f8cfull,0xd0c82e71b41c4e74ull,
      0xef88fb29b4c38602ull,0x3b865ed94b23284eull,
      0x160d4d0750485e84ull,0xa3128750a0466a21ull,
      0xc1c6ba2884daa0e1ull,0xbf2a42d54c6d336eull,
      0x7ae8060855d02ab2ull,0x2fe95298685c12c3ull,
      0x17610f06aabb0ce5ull,0xd09a61dcf57388c9ull,
      0xc363bc14f4fabe3bull,0xc857cd797d07a823ull,
      0xb5ba41c7ec13df0bull,0x153186eecfcb34e2ull,
      0x89792cd842ae3215ull,0x03cbb8ab112884f9ull,
      0x0d9c0669be4b1818ull,0xc45536c76bfb5268ull,
      0x465a47fdc0f38a3bull,0xc6c70427de251f9dull,
      0x2082d9a03b0f2c88ull,0x141e4954a3b02693ull};
  std::array<uint64_t, 50> actual{};
  const size_t layer_elements = size_t(132) * 5120;
  CHECK(trace.layer_residual_bf16.size() == 50 * layer_elements);
  for (size_t layer = 0; layer < actual.size(); ++layer)
    actual[layer] = fnv64_bytes(
        trace.layer_residual_bf16.data() + layer * layer_elements,
        layer_elements * sizeof(uint16_t));
  CHECK(actual == expected);
  CHECK(fnv64_floats(output.data) == 0x579170f52abfc8dbull);
  const ExactQwenTextEncoderStats stats = encoder.stats();
  CHECK(stats.peak_device_bytes < 900ull * 1024 * 1024);
  CHECK(stats.allocator_peak_nonstaging_bytes >=
        stats.allocator_baseline_bytes);
  CHECK(stats.peak_device_bytes >=
        stats.allocator_peak_nonstaging_bytes -
            stats.allocator_baseline_bytes);
  CHECK(stats.allocator_peak_used_bytes <=
        stats.allocator_peak_nonstaging_bytes +
            2 * context.staging_capacity_bytes());
  CHECK(stats.descriptor_set_allocations == 36);
  if (std::getenv("SLOPFAB_QWEN_MULTIMODAL_REAL")) {
    std::vector<uint8_t> rgb(size_t(256) * 256 * 3);
    for (size_t i = 0; i < rgb.size(); ++i)
      rgb[i] = static_cast<uint8_t>((i * 37 + i / 17 + 23) & 255u);
    const text::QwenPixelValues image =
        text::qwen3vl_patchify_resized_rgb(rgb, 256, 256);
    std::vector<int32_t> ids = {7, 151652};
    ids.insert(ids.end(), 64, 151655);
    ids.push_back(151653); ids.push_back(8);
    text::EncoderTrace multimodal_trace;
    const auto multimodal_begin = std::chrono::steady_clock::now();
    const text::PromptEmbedding multimodal =
        encoder.encode(ids, {image}, &multimodal_trace);
    const double multimodal_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - multimodal_begin).count();
    constexpr std::array<uint64_t, 50> multimodal_expected{
        0x9bc6bf41395e5988ull,0xc4d3d076719a2e86ull,
        0xe4289d1b1a484315ull,0x1de297a17774b2daull,
        0xaa8e33366da795cdull,0xe617ee114012d5e7ull,
        0x3b78f59733b170d7ull,0x081ab2230db6d492ull,
        0xdde573a762aa566bull,0x62e6cc60bb283460ull,
        0xc294e7e6e0ad7c8dull,0x89bb7aa5fbce690eull,
        0x7119d1c8da174f22ull,0x63d787697ec85a40ull,
        0x8399915f68486301ull,0xfa1d8de1d422f176ull,
        0x58b8abc47facc8c8ull,0x6144a5b72b70cf05ull,
        0xdc368514a4f73d2bull,0x1c0bddc1b4974754ull,
        0x6163c26eaf590ab0ull,0xd5ffede93b76c96full,
        0x5c6162ad0e7c2d26ull,0xc115be698ebd82caull,
        0xd0ec1146b028af92ull,0x36ddcdf97dcb35dbull,
        0x906344b8641944b7ull,0xccb1ef009b800e31ull,
        0xa5cae2f1510da93aull,0xe4acfa2f20f0b6dcull,
        0x0073b8158369f0a9ull,0xae3c24faa60587feull,
        0xf96720c37d8c20a6ull,0x00c978b6fd38ec76ull,
        0x3284573098f8768bull,0x91a355c911609dd1ull,
        0x13024b4727939ddbull,0x61012e1835da4b67ull,
        0x35571f785e52f5d7ull,0x647e2a033f951255ull,
        0xb32089da85204802ull,0x57a50f36bca3fcc1ull,
        0xd1aaa9ff352d0ed8ull,0x13b2b0a3ea604db5ull,
        0x319b4e0aef0e459eull,0x548da104d1d5adbcull,
        0x264608fbf678e286ull,0x2101297dde74a635ull,
        0x1886e6cedc626cd1ull,0x3021ad7836c55dcdull};
    std::array<uint64_t, 50> multimodal_actual{};
    const size_t multimodal_layer_elements = size_t(68) * 5120;
    for (size_t layer = 0; layer < multimodal_actual.size(); ++layer)
      multimodal_actual[layer] = fnv64_bytes(
          multimodal_trace.layer_residual_bf16.data() +
              layer * multimodal_layer_elements,
          multimodal_layer_elements * sizeof(uint16_t));
    CHECK(multimodal_actual == multimodal_expected);
    CHECK(fnv64_floats(multimodal.data) == 0xa875c128aa7a0e9dull);
    CHECK(multimodal.modality_tags.front() == 1 &&
          multimodal.modality_tags.back() == 1);
    for (size_t row = 1; row <= 66; ++row)
      CHECK(multimodal.modality_tags[row] == 0);
    const ExactQwenTextEncoderStats multimodal_stats = encoder.stats();
    CHECK(multimodal_stats.peak_device_bytes < 900ull * 1024 * 1024);
    CHECK(multimodal_stats.allocator_peak_nonstaging_bytes >=
          multimodal_stats.allocator_baseline_bytes);
    CHECK(multimodal_stats.peak_device_bytes >=
          multimodal_stats.allocator_peak_nonstaging_bytes -
              multimodal_stats.allocator_baseline_bytes);
    CHECK(multimodal_stats.allocator_peak_used_bytes <=
          multimodal_stats.allocator_peak_nonstaging_bytes +
              2 * context.staging_capacity_bytes());
    std::printf("  CUDA-off multimodal exact Qwen full50 L68 %.2f s final %016llx\n",
                multimodal_seconds,
                static_cast<unsigned long long>(fnv64_floats(multimodal.data)));
  }
  encoder.unload();
  context.collect();
  CHECK(context.pooled_used_bytes() >= cold_baseline);
  CHECK(context.pooled_used_bytes() <=
        cold_baseline + 2 * context.staging_capacity_bytes());
  std::printf(
      "  CUDA-off real exact Qwen full50 L132 %.2f s final %016llx peak/reserved %.1f/%.1f MiB descriptors %llu\n",
      seconds,
      static_cast<unsigned long long>(fnv64_floats(output.data)),
      double(stats.peak_device_bytes) / 1048576.0,
      double(stats.allocator_reserved_bytes) / 1048576.0,
      static_cast<unsigned long long>(stats.descriptor_set_allocations));
}
#endif

