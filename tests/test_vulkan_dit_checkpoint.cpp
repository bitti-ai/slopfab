#include "detail/vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(viggle_vulkan_real_block_archive, "checkpoint") {
  const auto path = std::filesystem::path(SLOPFAB_TEST_SOURCE_DIR) /
      "weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot.safetensors";
  if (!std::filesystem::exists(path)) {
    SKIP_MISSING_FIXTURE("Viggle-Animate checkpoint absent");
    return;
  }
  slopfab::SafeTensors checkpoint;
  checkpoint.open(path.string());
  // Host-only archive validation runs even on machines with no Vulkan GPU.
  slopfab::vulkan::H3BlockConfig config;
  config.sequence = 1;
  for (uint32_t layer = 0; layer < 50; ++layer) {
    slopfab::vulkan::ExactH3BlockStage::validate_checkpoint(checkpoint, layer, config);
    CHECK(true);
  }
  for (uint32_t layer = 0; layer < 2; ++layer) {
    slopfab::vulkan::ExactH3BlockStage::validate_refiner_checkpoint(checkpoint, layer, config);
    CHECK(true);
  }
}

SLOPFAB_TEST_CATEGORY(vulkan_h3_quantized_weight_residency, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty()");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  for (const char* filename : {"fl2va_pruned_fp8_scaled.safetensors",
       "minimax_h3_fl2va_fasth3_dense_pruned_int8_convrot.safetensors"}) {
    const auto path = std::filesystem::path(SLOPFAB_TEST_SOURCE_DIR) /
        "weights/transformer" / filename;
    if (!std::filesystem::exists(path)) {
      SKIP_MISSING_FIXTURE("missing %s", filename);
      continue;
    }
    TensorContextOptions context_options;
    context_options.max_batch_operators = 2048;
    TensorContext context(device, context_options);
    if (!context.exact_h3_attention()) {
      SKIP_UNSUPPORTED_HARDWARE("exact H3 attention unavailable");
      return;
    }
    SafeTensors checkpoint;
    checkpoint.open(path.string());
    H3BlockConfig config;
    config.sequence = 65;  // Both the 64-row GEMM and its remainder use one expansion.
    auto stage = ExactH3BlockStage::create(context, config);
    stage.load(checkpoint, 0);
    auto scratch = ExactH3BlockScratch::create(context, config);
    stage.prepare(scratch);
    auto tensor = [&](std::initializer_list<uint64_t> shape,
                      ScalarType type = ScalarType::kFloat32) {
      return context.allocate(TensorLayout::contiguous(shape.begin(),
          static_cast<uint32_t>(shape.size())), type);
    };
    auto tokens = tensor({config.sequence, config.hidden}, ScalarType::kBFloat16);
    auto selectors = tensor({config.sequence}, ScalarType::kInt32);
    auto code = tensor({config.timesteps, config.adaln_rank});
    auto cosine = tensor({config.sequence, 96});
    auto sine = tensor({config.sequence, 96});
    std::vector<uint16_t> input(size_t(config.sequence) * config.hidden);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = f32_to_bf16(float(int(i % 61) - 30) / 64.0f);
    std::vector<int32_t> ids(config.sequence);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = int32_t(i % config.modalities);
    std::vector<float> codes(size_t(config.timesteps) * config.adaln_rank);
    for (size_t i = 0; i < codes.size(); ++i)
      codes[i] = float(int(i % 7) - 3) / 16.0f;
    std::vector<float> ones(size_t(config.sequence) * 96, 1.0f), zeros(ones.size());
    context.upload_bytes(selectors, ids.data(), ids.size() * 4);
    context.upload(code, codes.data(), codes.size());
    context.upload(cosine, ones.data(), ones.size());
    context.upload(sine, zeros.data(), zeros.size());
    auto replay = [&] {
      context.upload_bytes(tokens, input.data(), input.size() * 2);
      auto batch = context.begin_batch();
      stage.record(batch, tokens, selectors, code, cosine, sine, scratch);
      stage.record(batch, tokens, selectors, code, cosine, sine, scratch);
      batch.submit().wait();
      std::vector<uint16_t> result(input.size());
      context.download_bytes(tokens, result.data(), result.size() * 2);
      return result;
    };
    const auto output = replay();
    const uint64_t used = context.pooled_used_bytes();
    CHECK(output == replay());
    CHECK(context.pooled_used_bytes() == used);
    // Pinned on the former permanent-BF16 path. Covers six different shared
    // expansions, both GEMM row ranges, repeated blocks, and INT8 ConvRot.
    const uint64_t expected = std::strstr(filename, "int8")
        ? 0x730a94ce15dc8847ull : 0x3657399741e2efc5ull;
    CHECK(fnv64_bytes(output.data(), output.size() * 2) == expected);
    // One packed real block is below 400 MiB. A retained BF16 copy makes it
    // exceed 1 GiB and reproduces the 32-GB-card load failure at stack scale.
    CHECK(stage.persistent_bytes() < (400ull << 20));
    std::printf("  %s: block persistent %llu, scratch %llu, output %016llx\n",
        filename, static_cast<unsigned long long>(stage.persistent_bytes()),
        static_cast<unsigned long long>(scratch.reserved_bytes()),
        static_cast<unsigned long long>(fnv64_bytes(output.data(), output.size() * 2)));
  }
}
