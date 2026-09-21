#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_qwen_full50_real_nvfp4_l132, "integration") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const std::filesystem::path source(SLOPFAB_TEST_SOURCE_DIR);
  const std::filesystem::path checkpoint_path =
      source / "weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
  const std::filesystem::path capture_path = source / "tests/data/qwen_layer0_l132.vfqw";
  int cuda_devices = 0;
  if (!std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(capture_path) ||
      cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(capture_path) || cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }

  constexpr Sha256Digest checkpoint_sha{0x33, 0xe6, 0x9e, 0x3e, 0xda, 0xb8, 0x46, 0xd5,
                                        0x29, 0x49, 0xba, 0xfd, 0xb0, 0x03, 0x78, 0xbd,
                                        0x3f, 0x5a, 0x93, 0xf7, 0x81, 0x24, 0xfc, 0x83,
                                        0xd5, 0xef, 0x10, 0x9d, 0xc4, 0xa1, 0xfc, 0xbb};
  CHECK(sha256_file(checkpoint_path.string()) == checkpoint_sha);
  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  CHECK(text::detect_weight_format(checkpoint) == text::WeightFormat::kNVFP4Awq);
  const text::QwenLayerCapture capture = text::read_qwen_layer_capture(capture_path.string());
  CHECK(capture.header.sequence == 132 && capture.token_ids.size() == 132);

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

  text::PromptEmbedding cuda_output;
  text::EncoderTrace cuda_trace;
  double cuda_seconds = 0.0;
  {
    text::Encoder cuda_encoder;
    text::EncoderConfig config;
    config.residency = text::Residency::kStreaming;
    config.arithmetic = text::EncoderArithmetic::kExact;
    cuda_encoder.load(checkpoint, config);
    const auto begin = std::chrono::steady_clock::now();
    cuda_output = cuda_encoder.encode(capture.token_ids, &cuda_trace);
    cuda_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    CHECK(cuda_encoder.format() == text::WeightFormat::kNVFP4Awq);
    cuda_encoder.unload();
  }

  // NV L132 needs 33 stage operators, one trace copy and the final widen.
  TensorContextOptions context_options;
  context_options.max_batch_operators = 35;
  TensorContext vk(device, context_options);
  const uint64_t cold_used = vk.pooled_used_bytes();
  ExactQwenTextEncoder encoder = ExactQwenTextEncoder::create(vk);
  encoder.load(checkpoint);
  text::EncoderTrace vk_trace;
  const auto vk_begin = std::chrono::steady_clock::now();
  const text::PromptEmbedding vk_output = encoder.encode(capture.token_ids, &vk_trace);
  const double vk_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - vk_begin).count();
  CHECK(vk_output.data == cuda_output.data);
  CHECK(vk_output.modality_tags == cuda_output.modality_tags);
  CHECK(vk_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
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
  const uint64_t final_hash = fnv64(vk_output.data.data(), vk_output.data.size() * sizeof(float));
  constexpr std::array<uint64_t, 50> expected_hashes{
      0x4f3d7ee0a06b9a5bull, 0x2ee44051eafbfdedull, 0x2136eeec9028bd54ull, 0x164957ffaa8dfc07ull,
      0x60e0255905821d34ull, 0xa3c509b5bb9d7733ull, 0xa6bf9965df0ceb31ull, 0x52114b3f7892944eull,
      0xb52a193f42f11ed7ull, 0xda67497cf617fb45ull, 0x6545b2d448a8e549ull, 0x73f278547b874f26ull,
      0x0c8f262349910885ull, 0x3e7996fc911ffac1ull, 0xeccd4e5fe5b9ac6eull, 0x2b7f444af528ec2dull,
      0x1fe608b480868fe3ull, 0xd8b7dbefaf13e2f5ull, 0x570cb9290ed47326ull, 0xc7741961b064baaaull,
      0x5df16682b0973a7aull, 0x2f0d7aaca3e8b7caull, 0xf70b206829d93c90ull, 0x54b02d3cd01f502dull,
      0x6bfe9e2b6fe1f02eull, 0x753267523cabf7b6ull, 0x22c7fea96d78d582ull, 0xbf17535d9b74b199ull,
      0x4765ec9c7e13c669ull, 0xd801f4ec20927ed8ull, 0x745e20ab4fe38a5eull, 0xa40ba39ab8bb10afull,
      0xc23561b38823b552ull, 0x5c572ce2f9ca1b89ull, 0xd08baefb73e43576ull, 0x2f420dbe4ae671b2ull,
      0xc503a9ae05447391ull, 0xd756d351b5d417e6ull, 0xb5c6a94bb94ba95full, 0x4183dffacb74d57dull,
      0xaf1051484be3604bull, 0x0d584ab455e75775ull, 0x758502e066122398ull, 0x17bdba736d540226ull,
      0xed068eb30c58681aull, 0x8c24098d6ccb1dd9ull, 0xb1fc7633590ec60aull, 0xdefffc251f98309full,
      0x95171c7e3779b64cull, 0xf3c5845b91289f24ull};
  CHECK(hashes == expected_hashes);
  CHECK(final_hash == 0xdd3fec8f152ad2d4ull);

  const ExactQwenTextEncoderStats stats = encoder.stats();
  CHECK(stats.last_num_tokens == 132);
  CHECK(stats.max_layer_weight_bytes < 300ull * 1024 * 1024);
  CHECK(stats.peak_device_bytes < 800ull * 1024 * 1024);
  CHECK(stats.descriptor_set_allocations <= 40);
  const uint64_t first_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();

#ifdef _WIN32
  auto rejected_reload = [&](const std::string& name, bool rank_one, bool zero_scalar) {
    const std::filesystem::path corrupt_path = make_sparse_qwen_metadata_corruption(
        checkpoint, text::WeightFormat::kNVFP4Awq, name, rank_one, zero_scalar);
    const uint64_t used = vk.pooled_used_bytes();
    const uint64_t reserved = vk.reserved_bytes();
    const uint64_t descriptors = vk.descriptor_set_allocations();
    bool rejected = false;
    {
      SafeTensors corrupt;
      corrupt.open(corrupt_path.string());
      try {
        encoder.load(corrupt);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
    }
    CHECK(rejected && encoder.loaded() && encoder.format() == text::WeightFormat::kNVFP4Awq);
    CHECK(vk.pooled_used_bytes() == used);
    CHECK(vk.reserved_bytes() == reserved);
    CHECK(vk.descriptor_set_allocations() == descriptors);
    std::filesystem::remove(corrupt_path);
  };
  const std::string global = "model.layers.49.mlp.down_proj.weight_scale_2";
  rejected_reload(global, true, false);
  rejected_reload(global, false, true);
  rejected_reload("model.layers.49.mlp.down_proj.comfy_quant", false, false);
#endif

  // Failed replacement leaves the active model immediately recordable.
  text::EncoderTrace recovery_trace;
  const text::PromptEmbedding recovery = encoder.encode(capture.token_ids, &recovery_trace);
  CHECK(recovery.data == cuda_output.data);
  CHECK(recovery_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
  const uint64_t recovery_reserved = vk.reserved_bytes();
  CHECK(recovery_reserved >= first_reserved);
  CHECK(recovery_reserved < 1024ull * 1024 * 1024);
  encoder.unload();
  const uint64_t warm_used = vk.pooled_used_bytes();
  CHECK(warm_used >= cold_used);
  CHECK(warm_used <= cold_used + 2 * vk.staging_capacity_bytes());
  CHECK(vk.reserved_bytes() == recovery_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);

  encoder.load(checkpoint);
  text::EncoderTrace reload_trace;
  const text::PromptEmbedding reload = encoder.encode(capture.token_ids, &reload_trace);
  CHECK(reload.data == cuda_output.data);
  CHECK(reload_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
  const uint64_t warmed_reserved = vk.reserved_bytes();
  CHECK(warmed_reserved >= recovery_reserved);
  CHECK(warmed_reserved < 1024ull * 1024 * 1024);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  encoder.unload();
  CHECK(vk.pooled_used_bytes() == warm_used);

  encoder.load(checkpoint);
  text::EncoderTrace third_trace;
  const text::PromptEmbedding third = encoder.encode(capture.token_ids, &third_trace);
  CHECK(third.data == cuda_output.data);
  CHECK(third_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
  CHECK(vk.reserved_bytes() == warmed_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  encoder.unload();
  CHECK(vk.pooled_used_bytes() == warm_used);

  TensorContextOptions short_options;
  short_options.max_batch_operators = 34;
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
  CHECK(short_rejected && rejected_trace.layer_residual_bf16.empty());
  CHECK(short_vk.pooled_used_bytes() == short_used);
  CHECK(short_vk.reserved_bytes() == short_reserved);
  CHECK(short_vk.descriptor_set_allocations() == 0);

  std::printf(
      "  real NVFP4 exact Qwen full50 L132 CUDA/Vulkan %.2f/%.2f s final FNV64 %016llx, peak/used/reserved %.1f/%.1f/%.1f MiB descriptors %llu, boundaries:",
      cuda_seconds, vk_seconds, static_cast<unsigned long long>(final_hash),
      double(stats.peak_device_bytes) / 1048576.0, double(stats.allocator_used_bytes) / 1048576.0,
      double(stats.allocator_reserved_bytes) / 1048576.0,
      static_cast<unsigned long long>(stats.descriptor_set_allocations));
  for (uint64_t hash : hashes)
    std::printf(" %016llx", static_cast<unsigned long long>(hash));
  std::printf("\n");
}
