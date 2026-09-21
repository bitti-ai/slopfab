#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_qwen_vision_real_tower, "integration") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!std::getenv("SLOPFAB_QWEN_VISION_TOWER_REAL")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_QWEN_VISION_TOWER_REAL\")");
    return;
  }
  const std::filesystem::path checkpoint_path =
      std::filesystem::path(SLOPFAB_TEST_SOURCE_DIR) /
      "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
  int cuda_devices = 0;
  if (!std::filesystem::exists(checkpoint_path) ||
      cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(checkpoint_path) || cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
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
  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = true;
  device_options.enable_shader_float16 = true;
  device_options.enable_storage_buffer_16bit = true;
  device_options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(device_options);

  // 256x256 is the processor's minimum production image. Its 16x16 patch grid
  // exercises S256 and produces 64 decoder tokens.
  std::vector<uint8_t> rgb(size_t(256) * 256 * 3);
  for (size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = static_cast<uint8_t>((i * 37 + i / 17 + 23) & 255u);
  const text::QwenPixelValues image = text::qwen3vl_patchify_resized_rgb(rgb, 256, 256);
  SafeTensors archive;
  archive.open(checkpoint_path.string());

  text::QwenVisionEncoder cuda_encoder;
  cuda_encoder.load(archive);
  text::QwenVisionTrace cuda_trace;
  const auto cuda_begin = std::chrono::steady_clock::now();
  const text::QwenVisionEmbedding cuda_output = cuda_encoder.encode_exact({image}, &cuda_trace);
  const double cuda_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - cuda_begin).count();
  cuda_encoder.unload();
  CHECK(cuda_trace.tokens == 256 && cuda_trace.hidden == 1152);
  CHECK(cuda_trace.block_residuals.size() == size_t(27) * 256 * 1152);
  CHECK(cuda_output.tokens == 64 && cuda_output.hidden == 5120);

  TensorContextOptions options;
  options.max_batch_operators = 128;
  TensorContext vk(device, options);
  ExactQwenVisionEncoder vk_encoder = ExactQwenVisionEncoder::create(vk);
  const uint64_t unloaded_used = vk.pooled_used_bytes();
  vk_encoder.load(archive);
  text::QwenVisionTrace vk_trace;
  const auto vk_begin = std::chrono::steady_clock::now();
  vk_encoder.encode(image, &vk_trace);
  const double vk_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - vk_begin).count();
  std::vector<uint16_t> vk_main(cuda_output.main.size());
  std::array<std::vector<uint16_t>, 3> vk_deep;
  vk.download_bytes(vk_encoder.main_output(), vk_main.data(), vk_main.size() * 2);
  for (uint32_t slot = 0; slot < 3; ++slot) {
    vk_deep[slot].resize(cuda_output.deepstack[slot].size());
    vk.download_bytes(vk_encoder.deepstack_output(slot), vk_deep[slot].data(),
                      vk_deep[slot].size() * 2);
  }
  CHECK(vk_trace.tokens == cuda_trace.tokens && vk_trace.hidden == cuda_trace.hidden);
  CHECK(vk_trace.block_residuals == cuda_trace.block_residuals);
  CHECK(vk_main == cuda_output.main);
  for (uint32_t slot = 0; slot < 3; ++slot)
    CHECK(vk_deep[slot] == cuda_output.deepstack[slot]);
  const ExactQwenVisionStats stats = vk_encoder.stats();
  CHECK(stats.patch_rows == 256 && stats.max_streamed_weight_bytes < 96ull * 1024 * 1024);
  CHECK(stats.descriptor_set_allocations <= 128);
  const uint64_t valid_used = vk.pooled_used_bytes();
  const uint64_t valid_reserved = vk.reserved_bytes();
  const uint64_t valid_descriptors = vk.descriptor_set_allocations();
  const std::array<text::QwenImageGrid, 5> invalid_grids{
      {{0, 16, 16},
       {1, -2, 16},
       {1, 15, 16},
       {std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max()},
       {1, 16, 16}}};
  for (size_t invalid_index = 0; invalid_index < invalid_grids.size(); ++invalid_index) {
    text::QwenPixelValues invalid;
    invalid.grid = invalid_grids[invalid_index];
    if (invalid_index == invalid_grids.size() - 1)
      invalid.rows.resize(1);
    bool rejected = false;
    try {
      vk_encoder.encode(invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    CHECK(rejected);
    CHECK(vk_encoder.output_tokens() == 64);
    CHECK(vk.pooled_used_bytes() == valid_used);
    CHECK(vk.reserved_bytes() == valid_reserved);
    CHECK(vk.descriptor_set_allocations() == valid_descriptors);
  }
  std::vector<uint16_t> preserved_main(vk_main.size());
  vk.download_bytes(vk_encoder.main_output(), preserved_main.data(),
                    preserved_main.size() * sizeof(uint16_t));
  CHECK(preserved_main == vk_main);
  {
    TensorBatch capacity = vk.begin_batch();
    CHECK(capacity.remaining_operator_capacity() == 128);
  }
  vk_encoder.unload();
  vk.collect();
  const uint64_t warmed_used = vk.pooled_used_bytes();
  CHECK(unloaded_used == 0);
  CHECK(warmed_used == 2 * vk.staging_capacity_bytes());
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  vk_encoder.load(archive);
  text::QwenVisionTrace repeat_trace;
  vk_encoder.encode(image, &repeat_trace);
  CHECK(repeat_trace.block_residuals == cuda_trace.block_residuals);
  vk_encoder.unload();
  vk.collect();
  CHECK(vk.pooled_used_bytes() == warmed_used);
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  vk_encoder.load(archive);
  text::QwenVisionTrace third_trace;
  vk_encoder.encode(image, &third_trace);
  CHECK(third_trace.block_residuals == cuda_trace.block_residuals);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  vk_encoder.unload();
  vk.collect();
  CHECK(vk.pooled_used_bytes() == warmed_used);
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  std::printf("qwen vision full27 S256 exact CUDA/Vulkan %.3f/%.3f s "
              "peak/scratch/activation/weight %.1f/%.1f/%.1f/%.1f MiB\n",
              cuda_seconds, vk_seconds, stats.allocator_peak_used_bytes / 1048576.0,
              stats.scratch_bytes / 1048576.0, stats.activation_bytes / 1048576.0,
              stats.max_streamed_weight_bytes / 1048576.0);
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_qwen_multimodal_full50_real, "integration") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const bool nv_requested = std::getenv("SLOPFAB_QWEN_MULTIMODAL_NV_REAL") != nullptr;
  if (!nv_requested && !std::getenv("SLOPFAB_QWEN_MULTIMODAL_REAL")) {
    SKIP_OPT_IN(
        "unavailable prerequisite: !nv_requested && !std::getenv(\"SLOPFAB_QWEN_MULTIMODAL_REAL\")");
    return;
  }
  const std::filesystem::path checkpoint_path =
      std::filesystem::path(SLOPFAB_TEST_SOURCE_DIR) /
      (nv_requested ? "weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
                    : "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors");
  int cuda_devices = 0;
  if (!std::filesystem::exists(checkpoint_path) ||
      cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(checkpoint_path) || cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
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
  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = true;
  device_options.enable_shader_float16 = true;
  device_options.enable_storage_buffer_16bit = true;
  device_options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(device_options);

  std::vector<uint8_t> rgb(size_t(256) * 256 * 3);
  for (size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = static_cast<uint8_t>((i * 37 + i / 17 + 23) & 255u);
  const text::QwenPixelValues image = text::qwen3vl_patchify_resized_rgb(rgb, 256, 256);
  std::vector<int32_t> token_ids = {7, 151652};
  token_ids.insert(token_ids.end(), 64, 151655);
  token_ids.push_back(151653);
  token_ids.push_back(8);
  CHECK(token_ids.size() == 68);
  SafeTensors archive;
  archive.open(checkpoint_path.string());
  const text::WeightFormat expected_format =
      nv_requested ? text::WeightFormat::kNVFP4Awq : text::WeightFormat::kI8ConvRot;
  CHECK(text::detect_weight_format(archive) == expected_format);
  if (nv_requested) {
    constexpr Sha256Digest nv_sha{0x33, 0xe6, 0x9e, 0x3e, 0xda, 0xb8, 0x46, 0xd5, 0x29, 0x49, 0xba,
                                  0xfd, 0xb0, 0x03, 0x78, 0xbd, 0x3f, 0x5a, 0x93, 0xf7, 0x81, 0x24,
                                  0xfc, 0x83, 0xd5, 0xef, 0x10, 0x9d, 0xc4, 0xa1, 0xfc, 0xbb};
    CHECK(sha256_file(checkpoint_path.string()) == nv_sha);
  }

  text::PromptEmbedding cuda_output;
  text::EncoderTrace cuda_trace;
  double cuda_seconds = 0.0;
  {
    text::Encoder encoder;
    text::EncoderConfig config;
    config.residency = text::Residency::kStreaming;
    config.arithmetic = text::EncoderArithmetic::kExact;
    encoder.load(archive, config);
    CHECK(encoder.format() == expected_format);
    const auto begin = std::chrono::steady_clock::now();
    cuda_output = encoder.encode(token_ids, {image}, &cuda_trace);
    cuda_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    encoder.unload();
  }

  TensorContextOptions options;
  options.max_batch_operators = 128;
  TensorContext vk(device, options);
  ExactQwenTextEncoder encoder = ExactQwenTextEncoder::create(vk);
  encoder.load(archive);
  CHECK(encoder.format() == expected_format);
  text::EncoderTrace vk_trace;
  const auto vk_begin = std::chrono::steady_clock::now();
  const text::PromptEmbedding vk_output = encoder.encode(token_ids, {image}, &vk_trace);
  const double vk_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - vk_begin).count();
  CHECK(vk_output.num_tokens == 68 && vk_output.hidden_size == 5120);
  CHECK(vk_output.modality_tags == cuda_output.modality_tags);
  CHECK(vk_output.data == cuda_output.data);
  CHECK(vk_trace.num_tokens == cuda_trace.num_tokens);
  CHECK(vk_trace.hidden_size == cuda_trace.hidden_size);
  CHECK(vk_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
  CHECK(vk_trace.layer_residual_bf16.size() == size_t(50) * 68 * 5120);
  CHECK(vk_output.modality_tags.front() == 1 && vk_output.modality_tags.back() == 1);
  for (size_t row = 1; row <= 66; ++row)
    CHECK(vk_output.modality_tags[row] == 0);
  auto fnv64 = [](const void* values, size_t bytes) {
    const auto* data = static_cast<const uint8_t*>(values);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
      hash ^= data[i];
      hash *= 1099511628211ull;
    }
    return hash;
  };
  std::array<uint64_t, 50> hashes{};
  const size_t layer_elements = size_t(68) * 5120;
  for (size_t layer = 0; layer < hashes.size(); ++layer)
    hashes[layer] = fnv64(vk_trace.layer_residual_bf16.data() + layer * layer_elements,
                          layer_elements * sizeof(uint16_t));
  const uint64_t final_hash = fnv64(vk_output.data.data(), vk_output.data.size() * sizeof(float));
  constexpr std::array<uint64_t, 50> expected_hashes{
      0x9bc6bf41395e5988ull, 0xc4d3d076719a2e86ull, 0xe4289d1b1a484315ull, 0x1de297a17774b2daull,
      0xaa8e33366da795cdull, 0xe617ee114012d5e7ull, 0x3b78f59733b170d7ull, 0x081ab2230db6d492ull,
      0xdde573a762aa566bull, 0x62e6cc60bb283460ull, 0xc294e7e6e0ad7c8dull, 0x89bb7aa5fbce690eull,
      0x7119d1c8da174f22ull, 0x63d787697ec85a40ull, 0x8399915f68486301ull, 0xfa1d8de1d422f176ull,
      0x58b8abc47facc8c8ull, 0x6144a5b72b70cf05ull, 0xdc368514a4f73d2bull, 0x1c0bddc1b4974754ull,
      0x6163c26eaf590ab0ull, 0xd5ffede93b76c96full, 0x5c6162ad0e7c2d26ull, 0xc115be698ebd82caull,
      0xd0ec1146b028af92ull, 0x36ddcdf97dcb35dbull, 0x906344b8641944b7ull, 0xccb1ef009b800e31ull,
      0xa5cae2f1510da93aull, 0xe4acfa2f20f0b6dcull, 0x0073b8158369f0a9ull, 0xae3c24faa60587feull,
      0xf96720c37d8c20a6ull, 0x00c978b6fd38ec76ull, 0x3284573098f8768bull, 0x91a355c911609dd1ull,
      0x13024b4727939ddbull, 0x61012e1835da4b67ull, 0x35571f785e52f5d7ull, 0x647e2a033f951255ull,
      0xb32089da85204802ull, 0x57a50f36bca3fcc1ull, 0xd1aaa9ff352d0ed8ull, 0x13b2b0a3ea604db5ull,
      0x319b4e0aef0e459eull, 0x548da104d1d5adbcull, 0x264608fbf678e286ull, 0x2101297dde74a635ull,
      0x1886e6cedc626cd1ull, 0x3021ad7836c55dcdull};
  constexpr std::array<uint64_t, 50> expected_nv_hashes{
      0x8545c17c86bce206ull, 0x246ae2760b600155ull, 0xe610e4e8b5d39c4aull, 0x9e9cca51ff9f1b31ull,
      0x3ed8d04b8628e290ull, 0x3a045bf327cb0f58ull, 0xcc7b4cf6684120cfull, 0x1c388d41e29d59e5ull,
      0xa3e936cc3beb6ba8ull, 0xf00c7d0aa0396719ull, 0x1fdded58a993f96eull, 0x394181b36114011dull,
      0xfc7f02830d74756bull, 0x774e1b387f7b4a88ull, 0x2a876f81a6dc117eull, 0xd463d9609b57c8adull,
      0x7a6dfc754330df17ull, 0xc5d5f683b706e28cull, 0xbd76353651031bc2ull, 0x86fbe18fd7a5eaa7ull,
      0xc819b641361167eaull, 0xa40257ea76268b3bull, 0x5901f48448c9767full, 0xe3dcb2a73978fdc8ull,
      0x8465406046da627aull, 0xbd47d4946aeb6204ull, 0x88ae879b09f86bf1ull, 0xe70b423dc0aab659ull,
      0x02d71bb1aa5f0ac8ull, 0x05b714cddfac51bdull, 0x32511519dd2b3cabull, 0x1c7e676f7fa8744dull,
      0x234672783a2eadbbull, 0x9bdf97e5124b239dull, 0x82844925d0bab31full, 0xc4e42865e8c2efafull,
      0x4933fa7eadd090cbull, 0xd0d58a632fc7b79dull, 0x00442d8822d97c6dull, 0x14918bf2aed37557ull,
      0x05f5a206b568e11cull, 0x4ccf85806ee0ab05ull, 0xfc02df0bb257c9a3ull, 0x2674158e28da5b05ull,
      0x92e1fcbe479e21c1ull, 0x051686ffbb7ca9f3ull, 0xc2a7ef431fc524fdull, 0x38957f3dd0122f65ull,
      0x1b949a92976536ebull, 0x1413a279c62a36c5ull};
  CHECK(hashes == (nv_requested ? expected_nv_hashes : expected_hashes));
  CHECK(final_hash == (nv_requested ? 0xebc9e36a30c843cdull : 0xa875c128aa7a0e9dull));
  const ExactQwenTextEncoderStats stats = encoder.stats();
  CHECK(stats.peak_device_bytes < 900ull * 1024 * 1024);
  CHECK(stats.max_layer_weight_bytes < 500ull * 1024 * 1024);
#ifdef _WIN32
  const text::QwenVisionCheckpoint vision = text::load_qwen3vl_vision_checkpoint(archive);
  const std::filesystem::path corrupt_visual = make_sparse_qwen_metadata_corruption(
      archive, expected_format, vision.prefix + "blocks.26.mlp.linear_fc2.weight", false, false,
      true);
  const uint64_t rollback_used = vk.pooled_used_bytes();
  const uint64_t rollback_reserved = vk.reserved_bytes();
  const uint64_t rollback_descriptors = vk.descriptor_set_allocations();
  bool visual_rejected = false;
  {
    SafeTensors corrupt;
    corrupt.open(corrupt_visual.string());
    try {
      encoder.load(corrupt);
    } catch (const std::runtime_error&) {
      visual_rejected = true;
    }
  }
  CHECK(visual_rejected && encoder.loaded());
  CHECK(vk.pooled_used_bytes() == rollback_used);
  CHECK(vk.reserved_bytes() == rollback_reserved);
  CHECK(vk.descriptor_set_allocations() == rollback_descriptors);
  const text::PromptEmbedding recovered = encoder.encode(token_ids, {image});
  CHECK(recovered.data == cuda_output.data);
  std::filesystem::remove(corrupt_visual);
#endif
  encoder.unload();
  vk.collect();
  const uint64_t warmed_used = vk.pooled_used_bytes();
  const uint64_t warmed_reserved = vk.reserved_bytes();
  const uint64_t warmed_descriptors = vk.descriptor_set_allocations();
  for (int cycle = 0; cycle < 2; ++cycle) {
    encoder.load(archive);
    const text::PromptEmbedding repeat = encoder.encode(token_ids, {image});
    CHECK(repeat.data == cuda_output.data);
    CHECK(vk.descriptor_set_allocations() == warmed_descriptors);
    encoder.unload();
    vk.collect();
    CHECK(vk.pooled_used_bytes() == warmed_used);
    CHECK(vk.reserved_bytes() == warmed_reserved);
    CHECK(vk.descriptor_set_allocations() == warmed_descriptors);
  }
  TensorContextOptions short_options;
  short_options.max_batch_operators = nv_requested ? 35 : 37;
  TensorContext short_vk(device, short_options);
  ExactQwenTextEncoder short_encoder = ExactQwenTextEncoder::create(short_vk);
  short_encoder.load(archive);
  const uint64_t short_used = short_vk.pooled_used_bytes();
  const uint64_t short_reserved = short_vk.reserved_bytes();
  const uint64_t short_descriptors = short_vk.descriptor_set_allocations();
  text::EncoderTrace rejected_trace;
  bool short_rejected = false;
  try {
    (void)short_encoder.encode(token_ids, {image}, &rejected_trace);
  } catch (const std::logic_error&) {
    short_rejected = true;
  }
  CHECK(short_rejected && rejected_trace.layer_residual_bf16.empty());
  CHECK(short_vk.pooled_used_bytes() == short_used);
  CHECK(short_vk.reserved_bytes() == short_reserved);
  CHECK(short_vk.descriptor_set_allocations() == short_descriptors);
  std::printf("qwen %s multimodal full50 L68 CUDA/Vulkan %.3f/%.3f s "
              "final %016llx peak %.1f MiB boundaries:",
              nv_requested ? "NVFP4" : "I8", cuda_seconds, vk_seconds,
              static_cast<unsigned long long>(final_hash), stats.peak_device_bytes / 1048576.0);
  for (uint64_t hash : hashes)
    std::printf(" %016llx", static_cast<unsigned long long>(hash));
  std::printf("\n");
}

SLOPFAB_TEST_CATEGORY(vulkan_qwen_multimodal_max_real, "integration") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const bool nv_requested = std::getenv("SLOPFAB_QWEN_MULTIMODAL_MAX_NV_REAL") != nullptr;
  const bool cuda_authority = std::getenv("SLOPFAB_QWEN_MULTIMODAL_MAX_CUDA_REAL") != nullptr;
  if (!nv_requested && !cuda_authority && !std::getenv("SLOPFAB_QWEN_MULTIMODAL_MAX_REAL")) {
    SKIP_OPT_IN(
        "unavailable prerequisite: !nv_requested && !cuda_authority && !std::getenv(\"SLOPFAB_QWEN_MULTIMODAL_MAX_REAL\")");
    return;
  }
  const std::filesystem::path checkpoint_path =
      std::filesystem::path(SLOPFAB_TEST_SOURCE_DIR) /
      (nv_requested ? "weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
                    : "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors");
  if (!std::filesystem::exists(checkpoint_path) || !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(checkpoint_path) || !Instance::available()");
    return;
  }
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
  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = true;
  device_options.enable_shader_float16 = true;
  device_options.enable_storage_buffer_16bit = true;
  device_options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(device_options);

  // A 2048x2048 RGB image is the production maximum visual grid: S16384
  // patches, 4096 merged decoder rows. This run intentionally has no trace
  // arena, so its high-water is representative of normal inference.
  std::vector<uint8_t> rgb(size_t(2048) * 2048 * 3);
  for (size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = static_cast<uint8_t>((i * 37 + i / 17 + 23) & 255u);
  const text::QwenPixelValues image = text::qwen3vl_patchify_resized_rgb(rgb, 2048, 2048);
  CHECK(image.grid.patch_count() == 16384);
  CHECK(image.grid.merged_token_count() == 4096);
  std::vector<int32_t> token_ids = {7, 151652};
  token_ids.insert(token_ids.end(), 4096, 151655);
  token_ids.push_back(151653);
  token_ids.push_back(8);
  CHECK(token_ids.size() == 4100);

  SafeTensors archive;
  archive.open(checkpoint_path.string());
  constexpr Sha256Digest i8_sha{0xbc, 0x2c, 0xed, 0x0f, 0xbe, 0xa6, 0x47, 0x57, 0xfa, 0x9a, 0xcd,
                                0xdc, 0xcf, 0xc0, 0xb3, 0xf4, 0x81, 0x9d, 0x1d, 0xcf, 0x1d, 0xa6,
                                0xc1, 0x24, 0xd6, 0x90, 0xd3, 0x68, 0xbe, 0x28, 0x39, 0x23};
  constexpr Sha256Digest nv_sha{0x33, 0xe6, 0x9e, 0x3e, 0xda, 0xb8, 0x46, 0xd5, 0x29, 0x49, 0xba,
                                0xfd, 0xb0, 0x03, 0x78, 0xbd, 0x3f, 0x5a, 0x93, 0xf7, 0x81, 0x24,
                                0xfc, 0x83, 0xd5, 0xef, 0x10, 0x9d, 0xc4, 0xa1, 0xfc, 0xbb};
  CHECK(sha256_file(checkpoint_path.string()) == (nv_requested ? nv_sha : i8_sha));
  if (nv_requested) {
    SafeTensors i8_archive;
    i8_archive.open((std::filesystem::path(SLOPFAB_TEST_SOURCE_DIR) /
                     "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors")
                        .string());
    size_t visual_tensor_count = 0;
    for (const auto& entry : archive.tensors()) {
      if (entry.first.rfind("visual.", 0) != 0)
        continue;
      const TensorView& nv_view = entry.second;
      const TensorView& i8_view = i8_archive.at(entry.first);
      CHECK(nv_view.dtype == i8_view.dtype);
      CHECK(nv_view.shape == i8_view.shape);
      CHECK(nv_view.nbytes == i8_view.nbytes);
      CHECK(std::memcmp(nv_view.data, i8_view.data, nv_view.nbytes) == 0);
      ++visual_tensor_count;
    }
    CHECK(visual_tensor_count == 351);
  }
  TensorContextOptions options;
  options.max_batch_operators = 128;
  TensorContext vk(device, options);
  auto fnv64 = [](const void* values, size_t bytes) {
    const auto* data = static_cast<const uint8_t*>(values);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
      hash ^= data[i];
      hash *= 1099511628211ull;
    }
    return hash;
  };

  text::PromptEmbedding cuda_max_output;
  double cuda_max_seconds = 0.0;
  if (cuda_authority) {
    text::Encoder cuda_encoder;
    text::EncoderConfig cuda_config;
    cuda_config.residency = text::Residency::kStreaming;
    cuda_config.arithmetic = text::EncoderArithmetic::kExact;
    cuda_encoder.load(archive, cuda_config);
    const auto cuda_begin = std::chrono::steady_clock::now();
    cuda_max_output = cuda_encoder.encode(token_ids, {image});
    cuda_max_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - cuda_begin).count();
    cuda_encoder.unload();
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Pin the four no-trace vision outputs at the same maximum grid. The visual
  // archive is shared by both shipped conditioner formats; these digests bind
  // the production patchifier, tower, and merger outputs to the file SHA.
  ExactQwenVisionEncoder vision = ExactQwenVisionEncoder::create(vk);
  vision.load(archive);
  vision.encode(image);
  std::array<uint64_t, 4> vision_hashes{};
  std::vector<uint16_t> vision_download(size_t(4096) * 5120);
  vk.download_bytes(vision.main_output(), vision_download.data(),
                    vision_download.size() * sizeof(uint16_t));
  vision_hashes[0] = fnv64(vision_download.data(), vision_download.size() * sizeof(uint16_t));
  for (uint32_t slot = 0; slot < 3; ++slot) {
    vk.download_bytes(vision.deepstack_output(slot), vision_download.data(),
                      vision_download.size() * sizeof(uint16_t));
    vision_hashes[slot + 1] =
        fnv64(vision_download.data(), vision_download.size() * sizeof(uint16_t));
  }
  constexpr std::array<uint64_t, 4> expected_vision_hashes{
      0x63a7dd4533d82d51ull, 0xda837b598ae29c53ull, 0x0edf2b1389f62ce2ull, 0xd8f970e5016f22b7ull};
  CHECK(vision_hashes == expected_vision_hashes);
  vision.unload();
  vk.collect();
  ExactQwenTextEncoder encoder = ExactQwenTextEncoder::create(vk);
  CHECK(vk.pooled_used_bytes() == 2 * vk.staging_capacity_bytes());
  encoder.load(archive);
  const auto begin = std::chrono::steady_clock::now();
  const text::PromptEmbedding first = encoder.encode(token_ids, {image});
  if (cuda_authority) {
    CHECK(first.modality_tags == cuda_max_output.modality_tags);
    CHECK(first.data == cuda_max_output.data);
  }
  const uint64_t final_hash = fnv64(first.data.data(), first.data.size() * sizeof(float));
  CHECK(final_hash == (nv_requested ? 0x4435938303e77287ull : 0xede491026c069661ull));
  const double first_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  const ExactQwenTextEncoderStats first_stats = encoder.stats();
  CHECK(first_stats.last_num_tokens == 4100);
  CHECK(first_stats.max_layer_weight_bytes < 500ull * 1024 * 1024);
  const uint64_t observed_peak =
      first_stats.allocator_peak_nonstaging_bytes >= first_stats.allocator_baseline_bytes
          ? first_stats.allocator_peak_nonstaging_bytes - first_stats.allocator_baseline_bytes
          : first_stats.allocator_peak_nonstaging_bytes;
  CHECK(first_stats.peak_device_bytes >= observed_peak);
  CHECK(first_stats.peak_device_bytes - observed_peak < 128ull * 1024 * 1024);
  CHECK(first_stats.peak_device_bytes < 4ull * 1024 * 1024 * 1024);
  const uint64_t warm_used = vk.pooled_used_bytes();
  const uint64_t warm_reserved = vk.reserved_bytes();
  const uint64_t warm_descriptors = vk.descriptor_set_allocations();

  const auto repeat_begin = std::chrono::steady_clock::now();
  const text::PromptEmbedding repeat = encoder.encode(token_ids, {image});
  const double repeat_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - repeat_begin).count();
  CHECK(repeat.data == first.data);
  CHECK(vk.pooled_used_bytes() == warm_used);
  CHECK(vk.reserved_bytes() == warm_reserved);
  CHECK(vk.descriptor_set_allocations() == warm_descriptors);
  const ExactQwenTextEncoderStats repeat_stats = encoder.stats();
  const uint64_t repeat_observed =
      repeat_stats.allocator_peak_nonstaging_bytes >= repeat_stats.allocator_baseline_bytes
          ? repeat_stats.allocator_peak_nonstaging_bytes - repeat_stats.allocator_baseline_bytes
          : repeat_stats.allocator_peak_nonstaging_bytes;
  CHECK(repeat_stats.peak_device_bytes >= repeat_observed);
  CHECK(repeat_stats.peak_device_bytes - repeat_observed < 128ull * 1024 * 1024);
  CHECK(repeat_stats.peak_device_bytes <= first_stats.peak_device_bytes + 128ull * 1024 * 1024);
  encoder.unload();
  vk.collect();
  CHECK(vk.pooled_used_bytes() == 2 * vk.staging_capacity_bytes());
  CHECK(vk.descriptor_set_allocations() == warm_descriptors);
  std::printf("qwen %s multimodal no-tap S16384/L4100 Vulkan %.3f/%.3f s "
              "logical/allocator %.1f/%.1f MiB weight %.1f MiB "
              "pool/reserved %.1f/%.1f MiB descriptors %llu final %016llx "
              "vision %016llx/%016llx/%016llx/%016llx\n",
              nv_requested ? "NVFP4" : "I8", first_seconds, repeat_seconds,
              first_stats.peak_device_bytes / 1048576.0, observed_peak / 1048576.0,
              first_stats.max_layer_weight_bytes / 1048576.0, warm_used / 1048576.0,
              warm_reserved / 1048576.0, static_cast<unsigned long long>(warm_descriptors),
              static_cast<unsigned long long>(final_hash),
              static_cast<unsigned long long>(vision_hashes[0]),
              static_cast<unsigned long long>(vision_hashes[1]),
              static_cast<unsigned long long>(vision_hashes[2]),
              static_cast<unsigned long long>(vision_hashes[3]));
  if (cuda_authority)
    std::printf("  qwen max direct CUDA/Vulkan %.3f/%.3f s exact\n", cuda_max_seconds,
                first_seconds);
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_qwen_full50_real_l132, "integration") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const std::filesystem::path source(SLOPFAB_TEST_SOURCE_DIR);
  const std::filesystem::path checkpoint_path =
      source / "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
  const std::filesystem::path capture_path = source / "tests/data/qwen_layer0_l132.vfqw";
  int cuda_devices = 0;
  if (!std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(capture_path) ||
      cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(capture_path) || cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }

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
  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = true;
  device_options.enable_shader_float16 = true;
  device_options.enable_storage_buffer_16bit = true;
  device_options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(device_options);

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  const text::QwenLayerCapture capture = text::read_qwen_layer_capture(capture_path.string());
  CHECK(capture.header.sequence == 132);
  CHECK(capture.token_ids.size() == 132);

  text::PromptEmbedding cuda_output;
  text::EncoderTrace cuda_trace;
  double cuda_seconds = 0.0;
  {
    text::Encoder encoder;
    text::EncoderConfig config;
    config.residency = text::Residency::kStreaming;
    config.arithmetic = text::EncoderArithmetic::kExact;
    encoder.load(checkpoint, config);
    const auto begin = std::chrono::steady_clock::now();
    cuda_output = encoder.encode(capture.token_ids, &cuda_trace);
    cuda_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    CHECK(encoder.format() == text::WeightFormat::kI8ConvRot);
    CHECK(encoder.residency() == text::Residency::kStreaming);
    encoder.unload();
  }

  TensorContextOptions context_options;
  // I8 L132 records 35 layer operators, one trace copy and the final widen.
  context_options.max_batch_operators = 37;
  TensorContext vk(device, context_options);
  const uint64_t unloaded_baseline = vk.pooled_used_bytes();
  ExactQwenTextEncoder encoder = ExactQwenTextEncoder::create(vk);
  encoder.load(checkpoint);
  text::EncoderTrace vk_trace;
  const auto vk_begin = std::chrono::steady_clock::now();
  const text::PromptEmbedding vk_output = encoder.encode(capture.token_ids, &vk_trace);
  const double vk_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - vk_begin).count();

  CHECK(cuda_output.num_tokens == vk_output.num_tokens);
  CHECK(cuda_output.hidden_size == vk_output.hidden_size);
  CHECK(cuda_output.modality_tags == vk_output.modality_tags);
  CHECK(cuda_output.data == vk_output.data);
  CHECK(cuda_trace.num_tokens == vk_trace.num_tokens);
  CHECK(cuda_trace.hidden_size == vk_trace.hidden_size);
  CHECK(cuda_trace.layer_residual_bf16 == vk_trace.layer_residual_bf16);
  CHECK(vk_trace.layer_residual_bf16.size() == size_t(50) * 132 * 5120);

  auto fnv64 = [](const void* values, size_t bytes) {
    const auto* data = static_cast<const uint8_t*>(values);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
      hash ^= data[i];
      hash *= 1099511628211ull;
    }
    return hash;
  };
  std::array<uint64_t, 50> hashes{};
  const size_t layer_elements = size_t(132) * 5120;
  for (size_t layer = 0; layer < hashes.size(); ++layer) {
    hashes[layer] = fnv64(vk_trace.layer_residual_bf16.data() + layer * layer_elements,
                          layer_elements * sizeof(uint16_t));
  }
  const uint64_t final_f32 = fnv64(vk_output.data.data(), vk_output.data.size() * sizeof(float));
  constexpr std::array<uint64_t, 50> expected_hashes{
      0xfb3966de636ac098ull, 0x16b0e53a0265959dull, 0x7e2377403c49f851ull, 0xdacca747d7db7ccaull,
      0xb3de1cdb9248573cull, 0xcb7ef7852ef4ce79ull, 0xfe09924dff391f53ull, 0x0ac372b850d058d8ull,
      0xe818f084c10d99d5ull, 0xeed04e74ca44abcfull, 0xe3e63048102b3e14ull, 0x6e5647916a5d3586ull,
      0x1b9a96cd7cb703c2ull, 0xe60df16cc8c0390dull, 0xcf624f216f397d84ull, 0x2e680a56b7030b0aull,
      0x8b8e97be1f24e88eull, 0x61e2e50339318a1eull, 0x410a6b1ab0a9c0e4ull, 0xaf70d3523ce3a525ull,
      0xfb6b26f66fc1eb31ull, 0x732c1c3ed4d7e6a6ull, 0x1b81b2dd36a55f5aull, 0x01f09dd599e38e09ull,
      0xc93c6edb4d56f691ull, 0xcb9bd628a2587064ull, 0x30c8bbdab894f8cfull, 0xd0c82e71b41c4e74ull,
      0xef88fb29b4c38602ull, 0x3b865ed94b23284eull, 0x160d4d0750485e84ull, 0xa3128750a0466a21ull,
      0xc1c6ba2884daa0e1ull, 0xbf2a42d54c6d336eull, 0x7ae8060855d02ab2ull, 0x2fe95298685c12c3ull,
      0x17610f06aabb0ce5ull, 0xd09a61dcf57388c9ull, 0xc363bc14f4fabe3bull, 0xc857cd797d07a823ull,
      0xb5ba41c7ec13df0bull, 0x153186eecfcb34e2ull, 0x89792cd842ae3215ull, 0x03cbb8ab112884f9ull,
      0x0d9c0669be4b1818ull, 0xc45536c76bfb5268ull, 0x465a47fdc0f38a3bull, 0xc6c70427de251f9dull,
      0x2082d9a03b0f2c88ull, 0x141e4954a3b02693ull};
  CHECK(hashes == expected_hashes);
  CHECK(final_f32 == 0x579170f52abfc8dbull);
  const ExactQwenTextEncoderStats stats = encoder.stats();
  CHECK(stats.last_num_tokens == 132);
  CHECK(stats.max_layer_weight_bytes < 500ull * 1024 * 1024);
  CHECK(stats.peak_device_bytes < 900ull * 1024 * 1024);
  CHECK(stats.allocator_peak_used_bytes >= stats.allocator_baseline_bytes);
  CHECK(stats.descriptor_set_allocations <= 40);
  const uint64_t stable_reserved = stats.allocator_reserved_bytes;
  const uint64_t stable_descriptors = stats.descriptor_set_allocations;
  encoder.unload();
  const uint64_t warm_unloaded_baseline = vk.pooled_used_bytes();
  CHECK(warm_unloaded_baseline >= unloaded_baseline);
  CHECK(warm_unloaded_baseline <= unloaded_baseline + 2 * vk.staging_capacity_bytes());
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);

  // A complete unload/reload preserves exact output and returns to the same
  // warmed context-only staging baseline. No descriptor or pool growth is
  // permitted on the second trajectory.
  encoder.load(checkpoint);
  text::EncoderTrace reload_trace;
  const text::PromptEmbedding reloaded = encoder.encode(capture.token_ids, &reload_trace);
  CHECK(reloaded.data == cuda_output.data);
  CHECK(reload_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
  const ExactQwenTextEncoderStats reload_stats = encoder.stats();
  CHECK(reload_stats.allocator_reserved_bytes >= stable_reserved);
  CHECK(reload_stats.allocator_reserved_bytes < 1536ull * 1024 * 1024);
  CHECK(reload_stats.peak_device_bytes < 900ull * 1024 * 1024);
  CHECK(reload_stats.descriptor_set_allocations == stable_descriptors);
  encoder.unload();
  CHECK(vk.pooled_used_bytes() == warm_unloaded_baseline);
  CHECK(vk.reserved_bytes() == reload_stats.allocator_reserved_bytes);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);

  // The first cold->warm transition may reserve an additional pool block due
  // to a different free-list order after complete shape destruction. A third
  // identical lifecycle must reuse that warmed high-water exactly.
  encoder.load(checkpoint);
  text::EncoderTrace third_trace;
  const text::PromptEmbedding third = encoder.encode(capture.token_ids, &third_trace);
  CHECK(third.data == cuda_output.data);
  CHECK(third_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
  CHECK(encoder.stats().allocator_reserved_bytes == reload_stats.allocator_reserved_bytes);
  CHECK(encoder.stats().descriptor_set_allocations == stable_descriptors);
#ifdef _WIN32
  // Aggregate load validation reaches the actual last layer before changing
  // the active archive or allocating. A corrupt canonical descriptor must
  // leave the loaded I8 model immediately usable and every allocator metric
  // unchanged.
  const std::filesystem::path corrupt_i8_path = make_sparse_qwen_metadata_corruption(
      checkpoint, text::WeightFormat::kI8ConvRot, "model.layers.49.mlp.down_proj.comfy_quant");
  const uint64_t rollback_used = vk.pooled_used_bytes();
  const uint64_t rollback_reserved = vk.reserved_bytes();
  const uint64_t rollback_descriptors = vk.descriptor_set_allocations();
  bool corrupt_i8_rejected = false;
  {
    SafeTensors corrupt_i8;
    corrupt_i8.open(corrupt_i8_path.string());
    try {
      encoder.load(corrupt_i8);
    } catch (const std::runtime_error&) {
      corrupt_i8_rejected = true;
    }
  }
  CHECK(corrupt_i8_rejected && encoder.loaded());
  CHECK(vk.pooled_used_bytes() == rollback_used);
  CHECK(vk.reserved_bytes() == rollback_reserved);
  CHECK(vk.descriptor_set_allocations() == rollback_descriptors);
  text::EncoderTrace rollback_trace;
  const text::PromptEmbedding rollback_output = encoder.encode(capture.token_ids, &rollback_trace);
  CHECK(rollback_output.data == cuda_output.data);
  CHECK(rollback_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
  std::filesystem::remove(corrupt_i8_path);
#endif
  encoder.unload();
  CHECK(vk.pooled_used_bytes() == warm_unloaded_baseline);
  CHECK(vk.reserved_bytes() == reload_stats.allocator_reserved_bytes);

  // Exact capacity is preflighted before activation allocation/upload. L132
  // I8 with a boundary trace requires 35+copy+final-widen = 37 operators.
  TensorContextOptions short_options;
  short_options.max_batch_operators = 36;
  TensorContext short_vk(device, short_options);
  const uint64_t short_used = short_vk.pooled_used_bytes();
  const uint64_t short_reserved = short_vk.reserved_bytes();
  ExactQwenTextEncoder short_encoder = ExactQwenTextEncoder::create(short_vk);
  short_encoder.load(checkpoint);
  text::EncoderTrace rejected_trace;
  bool short_rejected = false;
  try {
    (void)short_encoder.encode(capture.token_ids, &rejected_trace);
  } catch (const std::logic_error&) {
    short_rejected = true;
  }
  CHECK(short_rejected);
  CHECK(rejected_trace.layer_residual_bf16.empty());
  CHECK(short_vk.pooled_used_bytes() == short_used);
  CHECK(short_vk.reserved_bytes() == short_reserved);
  CHECK(short_vk.descriptor_set_allocations() == 0);

  std::printf(
      "  real exact Qwen full50 L132 CUDA/Vulkan %.2f/%.2f s final FNV64 %016llx, peak/used/reserved %.1f/%.1f/%.1f MiB descriptors %llu, boundaries:",
      cuda_seconds, vk_seconds, static_cast<unsigned long long>(final_f32),
      double(stats.peak_device_bytes) / 1048576.0, double(stats.allocator_used_bytes) / 1048576.0,
      double(stats.allocator_reserved_bytes) / 1048576.0,
      static_cast<unsigned long long>(stats.descriptor_set_allocations));
  for (uint64_t hash : hashes)
    std::printf(" %016llx", static_cast<unsigned long long>(hash));
  std::printf("\n");
}
