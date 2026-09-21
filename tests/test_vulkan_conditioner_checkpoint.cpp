#include "detail/vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(vulkan_qwen_layer0_real_l132_capture_replay, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  constexpr std::array<uint64_t, 11> expected_hashes{
      0x6ca9b8c5e16917b5ull, 0xf9bf6e554a1e84e4ull, 0x9c2c264a60b4b8b1ull, 0xec21312eea810e06ull,
      0x73901fb1cb7f2cfbull, 0x5ab1cc9e26345fe2ull, 0x9e072ab6646a2a11ull, 0x74d47f5ed650356dull,
      0x8292d03af91a2025ull, 0x48f99e7549238eceull, 0xfb3966de636ac098ull};
  constexpr std::array<uint8_t, 32> checkpoint_sha{0xbc, 0x2c, 0xed, 0x0f, 0xbe, 0xa6, 0x47, 0x57,
                                                   0xfa, 0x9a, 0xcd, 0xdc, 0xcf, 0xc0, 0xb3, 0xf4,
                                                   0x81, 0x9d, 0x1d, 0xcf, 0x1d, 0xa6, 0xc1, 0x24,
                                                   0xd6, 0x90, 0xd3, 0x68, 0xbe, 0x28, 0x39, 0x23};
  constexpr std::array<uint8_t, 32> tokenizer_sha{0xa5, 0xd8, 0x5b, 0x6d, 0xcc, 0x53, 0x5e, 0x6b,
                                                  0x93, 0x11, 0x5a, 0x9e, 0xf2, 0x87, 0xe6, 0x13,
                                                  0x2f, 0xdb, 0xf3, 0x02, 0x70, 0xda, 0x62, 0x18,
                                                  0x19, 0x4b, 0xa7, 0x42, 0x26, 0x11, 0x73, 0xc7};
  const std::filesystem::path source(SLOPFAB_TEST_SOURCE_DIR);
  const std::filesystem::path checkpoint_path =
      source / "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
  const std::filesystem::path capture_path = source / "tests/data/qwen_layer0_l132.vfqw";
  const std::filesystem::path tokenizer_path = source / "ref/text_encoder/tokenizer.json";
  if (!std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(capture_path) ||
      !std::filesystem::exists(tokenizer_path) || !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(capture_path) || !std::filesystem::exists(tokenizer_path) || !Instance::available()");
    return;
  }

#if !defined(SLOPFAB_WITH_CUDA) || !SLOPFAB_WITH_CUDA
  constexpr Sha256Digest capture_sha{0xec, 0x13, 0xad, 0x62, 0xa7, 0xe2, 0x53, 0xd5,
                                     0x88, 0xbf, 0xac, 0x51, 0x85, 0x0b, 0x92, 0x48,
                                     0x7b, 0x7c, 0xb8, 0x8b, 0xa7, 0x3e, 0x78, 0x69,
                                     0xa2, 0xb0, 0x2c, 0xba, 0x79, 0x11, 0x04, 0xb3};
  const auto provenance_begin = std::chrono::steady_clock::now();
  CHECK(sha256_file(checkpoint_path.string()) == checkpoint_sha);
  CHECK(sha256_file(tokenizer_path.string()) == tokenizer_sha);
  CHECK(sha256_file(capture_path.string()) == capture_sha);
  const double provenance_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - provenance_begin)
          .count();
  std::printf("  portable Qwen checkpoint/tokenizer/capture SHA-256 %.1f ms\n", provenance_ms);
#endif

  const text::QwenLayerCapture capture = text::read_qwen_layer_capture(capture_path.string());
  CHECK(capture.header.sequence == 132);
  CHECK(capture.header.hidden == 5120);
  CHECK(capture.header.query_heads == 64);
  CHECK(capture.header.kv_heads == 8);
  CHECK(capture.header.head_dim == 128);
  CHECK(capture.header.intermediate == 25600);
  CHECK(capture.header.checkpoint_sha256 == checkpoint_sha);
  CHECK(capture.header.tokenizer_sha256 == tokenizer_sha);
  CHECK(capture.header.input_fnv64 == 0x617329501f3c87a1ull);
  CHECK(capture.header.rope_fnv64 == 0x693c23a9886dd147ull);
  CHECK(capture.header.boundary_fnv64 == expected_hashes);
  CHECK(fnv64_bytes(capture.input_bf16.data(), capture.input_bf16.size() * sizeof(uint16_t)) ==
        capture.header.input_fnv64);
  uint64_t rope_hash = fnv64_bytes(capture.cosine.data(), capture.cosine.size() * sizeof(float));
  const auto* sine_bytes = reinterpret_cast<const uint8_t*>(capture.sine.data());
  for (size_t i = 0; i < capture.sine.size() * sizeof(float); ++i) {
    rope_hash ^= sine_bytes[i];
    rope_hash *= 1099511628211ull;
  }
  CHECK(rope_hash == capture.header.rope_fnv64);

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty()");
    return;
  }
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !info.timeline_semaphore || !info.shader_int64");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = info.shader_float16;
  options.enable_storage_buffer_16bit = info.storage_buffer_16bit;
  options.enable_cooperative_matrix = info.cooperative_matrix_bf16_f32_16x16x16;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 46;
  TensorContext context(device, context_options);
  if (!context.exact_causal_gqa_attention() || !context.exact_fp32_vae_normalization() ||
      !context.exact_vae_pointwise()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !context.exact_causal_gqa_attention() || !context.exact_fp32_vae_normalization() || !context.exact_vae_pointwise()");
    return;
  }

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  QwenTextLayerConfig config;
  config.sequence = capture.header.sequence;
  ExactQwenTextLayerStage stage = ExactQwenTextLayerStage::create(context, config);
  stage.load(checkpoint, 0);
  ExactQwenTextLayerScratch scratch = ExactQwenTextLayerScratch::create(context, config);
  CHECK(stage.format() == text::WeightFormat::kI8ConvRot);
  CHECK(stage.persistent_bytes() < 500ull * 1024 * 1024);
  CHECK(scratch.dense_cache_bytes() == 250ull * 1024 * 1024);
  CHECK(stage.peak_device_bytes(scratch) == stage.persistent_bytes() + scratch.reserved_bytes());
  CHECK(stage.peak_device_bytes(scratch) < 800ull * 1024 * 1024);

  auto matrix = [](uint64_t rows, uint64_t cols) {
    const uint64_t shape[] = {rows, cols};
    return TensorLayout::contiguous(shape, 2);
  };
  auto heads = [](uint64_t rows, uint64_t count, uint64_t dim) {
    const uint64_t shape[] = {rows, count, dim};
    return TensorLayout::contiguous(shape, 3);
  };
  const uint32_t rows = capture.header.sequence;
  DeviceTensor tokens = context.allocate(matrix(rows, 5120), ScalarType::kBFloat16);
  DeviceTensor cosine = context.allocate(matrix(rows, 128));
  DeviceTensor sine = context.allocate(matrix(rows, 128));
  DeviceTensor norm = context.allocate(matrix(rows, 5120), ScalarType::kBFloat16);
  DeviceTensor query = context.allocate(heads(rows, 64, 128), ScalarType::kBFloat16);
  DeviceTensor key = context.allocate(heads(rows, 8, 128), ScalarType::kBFloat16);
  DeviceTensor value = context.allocate(heads(rows, 8, 128), ScalarType::kBFloat16);
  DeviceTensor attention = context.allocate(heads(rows, 64, 128), ScalarType::kBFloat16);
  DeviceTensor attention_residual = context.allocate(matrix(rows, 5120), ScalarType::kBFloat16);
  DeviceTensor post_norm = context.allocate(matrix(rows, 5120), ScalarType::kBFloat16);
  DeviceTensor gate = context.allocate(matrix(rows, 25600), ScalarType::kBFloat16);
  DeviceTensor up = context.allocate(matrix(rows, 25600), ScalarType::kBFloat16);
  DeviceTensor activation = context.allocate(matrix(rows, 25600), ScalarType::kBFloat16);
  DeviceTensor final = context.allocate(matrix(rows, 5120), ScalarType::kBFloat16);
  context.upload(cosine, capture.cosine.data(), capture.cosine.size());
  context.upload(sine, capture.sine.data(), capture.sine.size());
  QwenTextLayerTaps taps{&norm,      &query, &key, &value,      &attention, &attention_residual,
                         &post_norm, &gate,  &up,  &activation, &final};
  CHECK(stage.required_operators(&taps) == 46);

  std::array<DeviceTensor*, 11> outputs{
      &norm,      &query, &key, &value,      &attention, &attention_residual,
      &post_norm, &gate,  &up,  &activation, &final};
  const std::array<size_t, 11> counts{
      size_t(rows) * 5120,    size_t(rows) * 64 * 128, size_t(rows) * 8 * 128,
      size_t(rows) * 8 * 128, size_t(rows) * 64 * 128, size_t(rows) * 5120,
      size_t(rows) * 5120,    size_t(rows) * 25600,    size_t(rows) * 25600,
      size_t(rows) * 25600,   size_t(rows) * 5120};
  auto run = [&] {
    context.upload_bytes(tokens, capture.input_bf16.data(),
                         capture.input_bf16.size() * sizeof(uint16_t));
    TensorBatch batch = context.begin_batch();
    stage.record(batch, tokens, cosine, sine, scratch, &taps);
    CHECK(batch.remaining_operator_capacity() == 0);
    batch.submit().wait();
    std::array<uint64_t, 11> hashes{};
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::vector<uint16_t> values(counts[i]);
      context.download_bytes(*outputs[i], values.data(), values.size() * sizeof(uint16_t));
      hashes[i] = fnv64_bytes(values.data(), values.size() * sizeof(uint16_t));
    }
    return hashes;
  };
  const auto first = run();
  CHECK(first == expected_hashes);
  const uint64_t stable_descriptors = context.descriptor_set_allocations();
  const auto repeat = run();
  CHECK(repeat == first);
  CHECK(context.descriptor_set_allocations() == stable_descriptors);
}
