#include "detail/vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(vulkan_h3_loaded_stage_cuda_off_contract, "synthetic") {
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
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 ||
      !info.storage_buffer_16bit ||
      !info.cooperative_matrix_bf16_f32_16x16x16) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 || !info.storage_buffer_16bit || !info.cooperative_matrix_bf16_f32_16x16x16");
    return;
  }
  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = true;
  device_options.enable_shader_float16 = true;
  device_options.enable_storage_buffer_16bit = true;
  device_options.enable_cooperative_matrix = true;
  device_options.enable_shader_int8 = info.shader_int8;
  Device device = physical.front().create_device(device_options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 64;
  TensorContext context(device, context_options);
  if (!context.exact_h3_attention() || !context.exact_vae_pointwise() ||
      !context.exact_fp32_vae_normalization()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !context.exact_h3_attention() || !context.exact_vae_pointwise() || !context.exact_fp32_vae_normalization()");
    return;
  }

  H3BlockConfig config;
  config.sequence = 65;
  config.hidden = 128;
  config.heads = 1;
  config.head_dim = 128;
  config.ffn = 128;
  config.timesteps = 1;
  config.modalities = 3;
  config.adaln_rank = 8;
  auto values = [](size_t count, uint32_t salt, float scale) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i)
      result[i] = float(int((i * 37 + salt) % 127) - 63) * scale;
    return result;
  };
  auto metadata = [](const std::string& text) {
    std::vector<float> result((text.size() + 3) / 4, 0.0f);
    std::memcpy(result.data(), text.data(), text.size());
    return result;
  };
  auto fixture = [&](uint32_t corruption) {
    const int64_t h = config.hidden, inner = config.heads * config.head_dim;
    const int64_t f = config.ffn;
    const int64_t adaln = int64_t(config.modalities) * 6 * h;
    std::vector<TensorWrite> tensors{
      {"blocks.0.norm1.weight", {h}, std::vector<float>(h, 1.0f)},
      {"blocks.0.norm2.weight", {h}, std::vector<float>(h, 1.0f)},
      {"blocks.0.attn.q_norm.weight", {config.head_dim},
       std::vector<float>(config.head_dim, 1.0f)},
      {"blocks.0.attn.k_norm.weight", {config.head_dim},
       std::vector<float>(config.head_dim, 1.0f)},
      {"blocks.0.attn.qkv_proj.weight", {3 * inner, h},
       values(size_t(3 * inner * h), 3, 1.0f / 4096.0f)},
      {"blocks.0.attn.out_proj.weight", {h, inner},
       values(size_t(h * inner), 5, 1.0f / 4096.0f)},
      // Exactly one AWQ activation pre-scale fixture.
      {"blocks.0.attn.out_proj.pre_quant_scale", {inner},
       values(size_t(inner), 71, 1.0f / 256.0f)},
      {"blocks.0.mlp.fc1.weight", {2 * f, h},
       values(size_t(2 * f * h), 7, 1.0f / 4096.0f)},
      {"blocks.0.mlp.fc2.weight", {h, f},
       values(size_t(h * f), 11, 1.0f / 4096.0f)},
      // Exactly one regular-H4 ConvRot metadata fixture.
      {"blocks.0.mlp.fc2.comfy_quant",
       {static_cast<int64_t>(metadata(
          corruption == 2 ? "{bad fc2 metadata" :
          "{\"convrot\":true,\"convrot_groupsize\":16}").size())},
       metadata(corruption == 2 ? "{bad fc2 metadata" :
                "{\"convrot\":true,\"convrot_groupsize\":16}")},
      {"blocks.0.adaln_proj.linear.weight", {adaln, config.adaln_rank},
       values(size_t(adaln * config.adaln_rank), 13, 1.0f / 8192.0f)},
      {"blocks.0.adaln_proj.linear.bias",
       {corruption == 1 ? adaln - 1 : adaln},
       values(size_t(corruption == 1 ? adaln - 1 : adaln), 17, 1.0f / 64.0f)}};
    return tensors;
  };
  const auto base = std::filesystem::temp_directory_path() /
      ("slopfab_h3_cuda_off_" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(base);
  const auto valid_path = base / "slopfab_h3_cuda_off_valid.safetensors";
  const auto corrupt_path = base / "slopfab_h3_cuda_off_corrupt.safetensors";
  const auto corrupt_fc2_path =
      base / "slopfab_h3_cuda_off_corrupt_fc2.safetensors";
  write_safetensors(valid_path.string(), fixture(0));
  write_safetensors(corrupt_path.string(), fixture(1));
  write_safetensors(corrupt_fc2_path.string(), fixture(2));
  SafeTensors valid, corrupt, corrupt_fc2;
  valid.open(valid_path.string());
  corrupt.open(corrupt_path.string());
  corrupt_fc2.open(corrupt_fc2_path.string());

  auto graph_fixture = [&](bool corrupt_last) {
    std::vector<TensorWrite> all;
    for (uint32_t layer = 0; layer < 3; ++layer) {
      std::vector<TensorWrite> one = fixture(corrupt_last && layer == 2 ? 2 : 0);
      for (TensorWrite& tensor : one) {
        tensor.name.replace(0, std::strlen("blocks.0"),
                            "blocks." + std::to_string(layer));
        all.push_back(std::move(tensor));
      }
    }
    return all;
  };
  const auto graph_path = base / "slopfab_h3_cuda_off_graph.safetensors";
  const auto corrupt_graph_path =
      base / "slopfab_h3_cuda_off_corrupt_graph.safetensors";
  write_safetensors(graph_path.string(), graph_fixture(false));
  write_safetensors(corrupt_graph_path.string(), graph_fixture(true));
  SafeTensors graph_checkpoint, corrupt_graph;
  graph_checkpoint.open(graph_path.string());
  corrupt_graph.open(corrupt_graph_path.string());

  auto transformer_fixture = [&](uint32_t corruption) {
    std::vector<TensorWrite> all = fixture(0);
    std::vector<TensorWrite> refiner = fixture(0);
    for (TensorWrite& tensor : refiner) {
      if (tensor.name.find(".adaln_proj.") != std::string::npos) continue;
      tensor.name.replace(0, std::strlen("blocks.0"),
                          "token_refiner.blocks.0");
      all.push_back(std::move(tensor));
    }
    const int64_t h = config.hidden;
    constexpr int64_t text_dim = 16, video_dim = 4, audio_dim = 2;
    const int64_t final_adaln = 2 * h;
    all.insert(all.end(), {
      {"adaln_t_table", {1025, 8}, values(1025 * 8, 97, 1.0f / 4096.0f)},
      {"condition_proj.weight", {h, text_dim},
       values(size_t(h * text_dim), 101, 1.0f / 1024.0f),
       corruption == 2 ? DType::kF32 : DType::kBF16},
      {"condition_proj.bias", {h}, values(h, 103, 1.0f / 128.0f),
       DType::kBF16},
      {"video_patch_proj.weight", {h, video_dim},
       values(size_t(h * video_dim), 107, 1.0f / 1024.0f)},
      {"video_patch_proj.bias", {h}, values(h, 109, 1.0f / 128.0f)},
      {"audio_patch_proj.weight", {h, audio_dim},
       values(size_t(h * audio_dim), 113, 1.0f / 1024.0f)},
      {"audio_patch_proj.bias", {h}, values(h, 127, 1.0f / 128.0f)},
      {"token_refiner.final_norm.weight", {h}, std::vector<float>(h, 1.0f),
       DType::kBF16},
      {"final_layer.norm.weight", {h}, std::vector<float>(h, 1.0f),
       DType::kBF16},
      {"final_layer.adaln_proj.linear.weight",
       {final_adaln, config.adaln_rank},
       values(size_t(final_adaln * config.adaln_rank), 131, 1.0f / 8192.0f),
       DType::kF16},
      {"final_layer.adaln_proj.linear.bias", {final_adaln},
       values(final_adaln, 137, 1.0f / 128.0f), DType::kF16},
      {"final_layer.video_out.weight", {video_dim, h},
       values(size_t(video_dim * h), 139, 1.0f / 1024.0f)},
      {"final_layer.video_out.bias", {video_dim},
       values(video_dim, 149, 1.0f / 128.0f)},
      {"final_layer.audio_out.weight", {audio_dim, h},
       values(size_t(audio_dim * h), 151, 1.0f / 1024.0f)},
      {"final_layer.audio_out.bias",
       {corruption == 1 ? audio_dim - 1 : audio_dim},
       values(size_t(corruption == 1 ? audio_dim - 1 : audio_dim),
              157, 1.0f / 128.0f)}
    });
    const char* metadata_suffix = nullptr;
    if (corruption == 3) metadata_suffix = ".pre_quant_scale";
    if (corruption == 4) metadata_suffix = ".weight_scale";
    if (corruption == 5) metadata_suffix = ".input_scale";
    if (corruption == 6) metadata_suffix = ".comfy_quant";
    if (metadata_suffix) {
      all.push_back({std::string("final_layer.adaln_proj.linear") +
                         metadata_suffix,
                     {1}, {1.0f}});
    }
    return all;
  };
  const auto transformer_path =
      base / "slopfab_h3_cuda_off_transformer.safetensors";
  const auto corrupt_transformer_path =
      base / "slopfab_h3_cuda_off_corrupt_transformer.safetensors";
  const auto corrupt_transformer_dtype_path =
      base / "slopfab_h3_cuda_off_corrupt_transformer_dtype.safetensors";
  const std::array<std::filesystem::path, 4> corrupt_transformer_metadata_paths{
      base / "slopfab_h3_cuda_off_corrupt_transformer_prequant.safetensors",
      base / "slopfab_h3_cuda_off_corrupt_transformer_weightscale.safetensors",
      base / "slopfab_h3_cuda_off_corrupt_transformer_inputscale.safetensors",
      base / "slopfab_h3_cuda_off_corrupt_transformer_comfy.safetensors"};
  write_safetensors(transformer_path.string(), transformer_fixture(0));
  write_safetensors(corrupt_transformer_path.string(),
                    transformer_fixture(1));
  write_safetensors(corrupt_transformer_dtype_path.string(),
                    transformer_fixture(2));
  for (uint32_t i = 0; i < corrupt_transformer_metadata_paths.size(); ++i)
    write_safetensors(corrupt_transformer_metadata_paths[i].string(),
                      transformer_fixture(3 + i));
  SafeTensors transformer_checkpoint, corrupt_transformer,
      corrupt_transformer_dtype;
  std::array<SafeTensors, 4> corrupt_transformer_metadata;
  transformer_checkpoint.open(transformer_path.string());
  corrupt_transformer.open(corrupt_transformer_path.string());
  corrupt_transformer_dtype.open(corrupt_transformer_dtype_path.string());
  for (uint32_t i = 0; i < corrupt_transformer_metadata.size(); ++i)
    corrupt_transformer_metadata[i].open(
        corrupt_transformer_metadata_paths[i].string());

  ExactH3BlockStage first = ExactH3BlockStage::create(context, config);
  ExactH3BlockStage second = ExactH3BlockStage::create(context, config);
  first.load(valid, 0);
  second.load(valid, 0);
  CHECK(first.required_operators() == 25u);
  CHECK(second.required_operators() == 25u);
  ExactH3BlockScratch scratch = ExactH3BlockScratch::create(context, config);
  first.prepare(scratch);
  second.prepare(scratch);
  CHECK(scratch.reserved_bytes() != 0u);
  const uint64_t token_shape[] = {config.sequence, config.hidden};
  const uint64_t selector_shape[] = {config.sequence};
  const uint64_t code_shape[] = {config.timesteps, config.adaln_rank};
  const uint64_t rope_shape[] = {config.sequence, 96};
  DeviceTensor tokens = context.allocate(
      TensorLayout::contiguous(token_shape, 2), ScalarType::kBFloat16);
  DeviceTensor selectors = context.allocate(
      TensorLayout::contiguous(selector_shape, 1), ScalarType::kInt32);
  DeviceTensor code = context.allocate(TensorLayout::contiguous(code_shape, 2));
  DeviceTensor cosine = context.allocate(TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor sine = context.allocate(TensorLayout::contiguous(rope_shape, 2));
  CHECK(static_cast<bool>(sine));
  std::vector<uint16_t> input(size_t(config.sequence) * config.hidden);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = f32_to_bf16(float(int(i % 47) - 23) / 32.0f);
  std::vector<int32_t> host_selectors(config.sequence);
  for (uint32_t i = 0; i < config.sequence; ++i)
    host_selectors[i] = static_cast<int32_t>(i % config.modalities);
  std::vector<float> host_code(config.adaln_rank);
  for (uint32_t i = 0; i < config.adaln_rank; ++i)
    host_code[i] = float(int(i) - 3) / 16.0f;
  std::vector<float> host_cos(size_t(config.sequence) * 96);
  std::vector<float> host_sin(host_cos.size());
  for (size_t i = 0; i < host_cos.size(); ++i) {
    const float angle = float((i * 13) % 101) / 997.0f;
    host_cos[i] = std::cos(angle);
    host_sin[i] = std::sin(angle);
  }
  context.upload_bytes(selectors, host_selectors.data(), host_selectors.size() * 4);
  context.upload(code, host_code.data(), host_code.size());
  context.upload(cosine, host_cos.data(), host_cos.size());
  context.upload(sine, host_sin.data(), host_sin.size());
  DeviceTensor dummy = context.allocate(
      TensorLayout::contiguous(token_shape, 2), ScalarType::kBFloat16);
  {
    TensorBatch exact_capacity = context.begin_batch();
    for (uint32_t i = first.required_operators(); i < 64; ++i)
      exact_capacity.copy(dummy, tokens);
    first.record(exact_capacity, tokens, selectors, code, cosine, sine, scratch);
    CHECK(exact_capacity.remaining_operator_capacity() == 0u);
    exact_capacity.submit().wait();
  }
  {
    TensorBatch short_capacity = context.begin_batch();
    for (uint32_t i = first.required_operators(); i <= 64; ++i)
      short_capacity.copy(dummy, tokens);
    bool rejected = false;
    try {
      first.record(short_capacity, tokens, selectors, code, cosine, sine, scratch);
    } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected && short_capacity.remaining_operator_capacity() == 24u);
  }
  TensorLayout bad_token_layout = TensorLayout::contiguous(token_shape, 2);
  bad_token_layout.stride[0] = config.hidden + 1;
  TensorLayout bad_selector_layout = TensorLayout::contiguous(selector_shape, 1);
  bad_selector_layout.stride[0] = 2;
  TensorLayout bad_code_layout = TensorLayout::contiguous(code_shape, 2);
  bad_code_layout.stride[0] = config.adaln_rank + 1;
  const uint64_t token_backing_shape[] = {config.sequence + 1, config.hidden};
  const uint64_t selector_backing_shape[] = {config.sequence * 2};
  const uint64_t code_backing_shape[] = {config.timesteps,
                                         config.adaln_rank + 1};
  DeviceTensor bad_tokens = context.allocate(
      TensorLayout::contiguous(token_backing_shape, 2), ScalarType::kBFloat16);
  DeviceTensor bad_selectors = context.allocate(
      TensorLayout::contiguous(selector_backing_shape, 1), ScalarType::kInt32);
  DeviceTensor bad_code = context.allocate(
      TensorLayout::contiguous(code_backing_shape, 2));
  // DeviceTensor intentionally exposes only validated contiguous allocation;
  // mutate test-only metadata on oversized backing allocations to inject the
  // otherwise-unconstructible adversarial views into stage preflight.
  const_cast<TensorLayout&>(bad_tokens.layout()) = bad_token_layout;
  const_cast<TensorLayout&>(bad_selectors.layout()) = bad_selector_layout;
  const_cast<TensorLayout&>(bad_code.layout()) = bad_code_layout;
  TensorContext foreign_context(device, context_options);
  const int32_t full_range[] = {0, 128, 0, 0};
  H3AttentionRanges foreign_ranges = H3AttentionRanges::create(
      foreign_context, config.sequence, full_range, 4);
  {
    TensorBatch invalid = context.begin_batch();
    auto rejected = [&](DeviceTensor& token_arg, DeviceTensor& selector_arg,
                        DeviceTensor& code_arg,
                        const H3AttentionRanges* ranges = nullptr) {
      bool threw = false;
      try {
        first.record(invalid, token_arg, selector_arg, code_arg, cosine, sine,
                     scratch, ranges);
      } catch (const std::invalid_argument&) { threw = true; }
      CHECK(threw && invalid.remaining_operator_capacity() == 64u);
    };
    rejected(bad_tokens, selectors, code);
    rejected(tokens, bad_selectors, code);
    rejected(tokens, selectors, bad_code);
    rejected(tokens, selectors, code, &foreign_ranges);
    // None of the rejected calls poisoned access tracking or consumed an op.
    first.record(invalid, tokens, selectors, code, cosine, sine, scratch);
    CHECK(invalid.remaining_operator_capacity() == 39u);
    invalid.submit().wait();
  }
  auto run_chain = [&] {
    context.upload_bytes(tokens, input.data(), input.size() * 2);
    TensorBatch batch = context.begin_batch();
    first.record(batch, tokens, selectors, code, cosine, sine, scratch);
    second.record(batch, tokens, selectors, code, cosine, sine, scratch);
    batch.submit().wait();
    std::vector<uint16_t> output(input.size());
    context.download_bytes(tokens, output.data(), output.size() * 2);
    return output;
  };
  const std::vector<uint16_t> output = run_chain();
  {
    // Two heads are essential: a one-head QKV permutation is the identity.
    config.heads = 2;
    auto contiguous = fixture(0);
    auto interleaved = contiguous;
    for (auto& tensor : interleaved) {
      if (tensor.name != "blocks.0.attn.qkv_proj.weight") continue;
      const auto original = tensor.data;
      size_t dst = 0;
      for (uint32_t head = 0; head < config.heads; ++head)
        for (uint32_t part = 0; part < 3; ++part)
          for (uint32_t channel = 0; channel < config.head_dim; ++channel)
            for (uint32_t col = 0; col < config.hidden; ++col)
              tensor.data[dst++] = original[size_t((part * config.heads + head) *
                  config.head_dim + channel) * config.hidden + col];
    }
    const auto pair_path = base / "slopfab_viggle_two_heads.safetensors";
    auto evaluate = [&](const std::vector<TensorWrite>& weights, const char* layout) {
      write_safetensors(pair_path.string(), weights, {{"qkv_layout", layout}});
      SafeTensors checkpoint;
      checkpoint.open(pair_path.string());
      auto stage = ExactH3BlockStage::create(context, config);
      stage.load(checkpoint, 0);
      auto pair_scratch = ExactH3BlockScratch::create(context, config);
      stage.prepare(pair_scratch);
      context.upload_bytes(tokens, input.data(), input.size() * 2);
      auto batch = context.begin_batch();
      stage.record(batch, tokens, selectors, code, cosine, sine, pair_scratch);
      batch.submit().wait();
      std::vector<uint16_t> result(input.size());
      context.download_bytes(tokens, result.data(), result.size() * 2);
      return result;
    };
    const auto expected = evaluate(contiguous, "contiguous");
    CHECK(expected == evaluate(interleaved, "interleaved"));
    std::filesystem::remove(pair_path);
    config.heads = 1;
  }
  uint64_t digest = 1469598103934665603ull;
  for (uint16_t bits : output) {
    digest ^= bits & 0xffu; digest *= 1099511628211ull;
    digest ^= bits >> 8; digest *= 1099511628211ull;
  }
  CHECK(digest == 0x70e1eaf01723020bull);
  if (context.h3_attention_supported(AttentionMode::kFlash2)) {
    // One video tile makes sparse attention dense. A zero compression gate
    // must then reproduce Flash attention; a learned gate must affect it.
    auto vsa_config = config;
    vsa_config.attention_mode = AttentionMode::kFlash2;
    dit::SequenceLayout geometry;
    geometry.num_text = 1;
    geometry.num_latent_frames = 4;
    geometry.latent_height = geometry.latent_width = 8;
    geometry.num_video_rows = 64;
    auto tile_map = std::make_shared<dit::VsaTiles>(dit::build_vsa_tiles(geometry));
    const auto vsa_path = base / "vsa.safetensors";
    auto evaluate = [&](bool sparse, float gate_scale) {
      auto tensors = fixture(0);
      if (sparse) {
        std::vector<float> gate(config.hidden * config.hidden, 0.0f);
        for (uint32_t i = 0; i < config.hidden; ++i) gate[i * config.hidden + i] = gate_scale;
        tensors.push_back({"blocks.0.attn.to_gate_compress.weight",
                           {config.hidden, config.hidden}, std::move(gate)});
      }
      write_safetensors(vsa_path.string(), tensors);
      SafeTensors checkpoint;
      checkpoint.open(vsa_path.string());
      vsa_config.vsa_tiles = sparse ? tile_map : nullptr;
      auto stage = ExactH3BlockStage::create(context, vsa_config);
      stage.load(checkpoint, 0);
      auto vsa_scratch = ExactH3BlockScratch::create(context, vsa_config);
      stage.prepare(vsa_scratch);
      context.upload_bytes(tokens, input.data(), input.size() * 2);
      auto batch = context.begin_batch();
      while (batch.remaining_operator_capacity() > stage.required_operators()) batch.copy(tokens, dummy);
      stage.record(batch, tokens, selectors, code, cosine, sine, vsa_scratch);
      CHECK(batch.remaining_operator_capacity() == 0);
      batch.submit().wait();
      std::vector<uint16_t> result(input.size());
      context.download_bytes(tokens, result.data(), result.size() * 2);
      return result;
    };
    const auto dense = evaluate(false, 0.0f);
    const auto zero_gate = evaluate(true, 0.0f);
    float error = 0;
    for (size_t i = 0; i < dense.size(); ++i)
      error = std::max(error, std::abs(bf16_to_f32(dense[i]) - bf16_to_f32(zero_gate[i])));
    CHECK_MSG(error < 0.016f, "VSA zero gate versus dense Flash error %g", error);
    CHECK(evaluate(true, 1.0f) != zero_gate);
    bool rejected = false;
    try { ExactH3BlockStage::validate_checkpoint(valid, 0, vsa_config); }
    catch (const std::runtime_error&) { rejected = true; }
    CHECK(rejected);
    std::filesystem::remove(vsa_path);
  }
  for (auto mode : {AttentionMode::kFlash2, AttentionMode::kSage2}) {
    if (!context.h3_attention_supported(mode)) continue;
    H3BlockConfig fast_config = config; fast_config.attention_mode = mode;
    auto fast_stage = ExactH3BlockStage::create(context, fast_config);
    fast_stage.load(valid, 0);
    auto fast_scratch = ExactH3BlockScratch::create(context, fast_config);
    fast_stage.prepare(fast_scratch);
    context.upload_bytes(tokens, input.data(), input.size()*2);
    auto batch = context.begin_batch();
    bool mismatch_rejected = false;
    try { fast_stage.record(batch, tokens, selectors, code, cosine, sine, scratch); }
    catch (const std::invalid_argument&) { mismatch_rejected = true; }
    CHECK(mismatch_rejected && batch.remaining_operator_capacity() == 64);
    fast_stage.record(batch, tokens, selectors, code, cosine, sine, fast_scratch);
    fast_stage.record(batch, tokens, selectors, code, cosine, sine, fast_scratch);
    CHECK(batch.remaining_operator_capacity() == 14);
    batch.submit().wait();
    std::vector<uint16_t> fast_output(input.size());
    context.download_bytes(tokens, fast_output.data(), fast_output.size()*2);
    double e2 = 0, r2 = 0;
    for (size_t i = 0; i < output.size(); ++i) {
      double r = bf16_to_f32(output[i]), e = bf16_to_f32(fast_output[i])-r;
      CHECK(std::isfinite(e)); e2 += e*e; r2 += r*r;
    }
    CHECK_MSG(std::sqrt(e2/std::max(r2, 1e-30)) < 0.01,
        "%s two-block residual differs from exact", attention_mode_name(mode));
  }
  CHECK(run_chain() == output);
  const uint64_t stable_used = context.pooled_used_bytes();
  const uint64_t stable_reserved = context.reserved_bytes();
  const uint64_t stable_descriptors = context.descriptor_set_allocations();
  CHECK(run_chain() == output);
  CHECK(context.pooled_used_bytes() == stable_used);
  CHECK(context.reserved_bytes() == stable_reserved);
  CHECK(context.descriptor_set_allocations() == stable_descriptors);

  // Corruption in the sixth and final logical projection is decoded before
  // the first Vulkan allocation. The active stage and allocator HWM remain
  // exactly unchanged.
  bool fc2_rejected = false;
  try { first.load(corrupt_fc2, 0); }
  catch (const std::exception&) { fc2_rejected = true; }
  CHECK(fc2_rejected && first.loaded());
  CHECK(context.pooled_used_bytes() == stable_used);
  CHECK(context.reserved_bytes() == stable_reserved);
  CHECK(context.descriptor_set_allocations() == stable_descriptors);
  CHECK(run_chain() == output);

  bool corrupt_rejected = false;
  try { first.load(corrupt, 0); }
  catch (const std::exception&) { corrupt_rejected = true; }
  CHECK(corrupt_rejected && first.loaded());
  CHECK(context.pooled_used_bytes() == stable_used);
  CHECK(context.reserved_bytes() == stable_reserved);
  CHECK(context.descriptor_set_allocations() == stable_descriptors);
  CHECK(run_chain() == output);
  const uint64_t persistent = first.persistent_bytes();
  first.unload();
  second.unload();

  // Multi-layer graph: one batch, one shared transformed-activation scratch,
  // exact capacity boundary, every device-only block boundary, transactional
  // corruption in the last projection of the last layer, and repeat reuse.
  TensorContextOptions graph_options;
  graph_options.max_batch_operators = 128;
  TensorContext graph_context(device, graph_options);
  H3MainGraphConfig graph_config;
  graph_config.block = config;
  graph_config.layers = 3;
  const uint64_t graph_create_used = graph_context.pooled_used_bytes();
  const uint64_t graph_create_reserved = graph_context.reserved_bytes();
  const uint64_t graph_create_descriptors =
      graph_context.descriptor_set_allocations();
  ExactH3MainGraph graph = ExactH3MainGraph::create(
      graph_context, graph_config);
  CHECK(!graph.loaded() && graph.layers() == 3u);
  CHECK(graph.persistent_bytes() == 0u && graph.scratch_bytes() == 0u &&
        graph.peak_device_bytes() == 0u);
  CHECK(graph_context.pooled_used_bytes() == graph_create_used);
  CHECK(graph_context.reserved_bytes() == graph_create_reserved);
  CHECK(graph_context.descriptor_set_allocations() == graph_create_descriptors);
  // Complete validation, including the last layer's fc2 metadata, is host-only
  // and precedes scratch or weight allocation on an initial load.
  bool initial_corrupt_rejected = false;
  try { graph.load(corrupt_graph); }
  catch (const std::exception&) { initial_corrupt_rejected = true; }
  CHECK(initial_corrupt_rejected && !graph.loaded());
  CHECK(graph.persistent_bytes() == 0u && graph.scratch_bytes() == 0u &&
        graph.peak_device_bytes() == 0u);
  CHECK(graph_context.pooled_used_bytes() == graph_create_used);
  CHECK(graph_context.reserved_bytes() == graph_create_reserved);
  CHECK(graph_context.descriptor_set_allocations() == graph_create_descriptors);
  const uint64_t graph_token_shape[] = {config.sequence, config.hidden};
  const uint64_t graph_selector_shape[] = {config.sequence};
  const uint64_t graph_code_shape[] = {config.timesteps, config.adaln_rank};
  const uint64_t graph_rope_shape[] = {config.sequence, 96};
  DeviceTensor graph_tokens = graph_context.allocate(
      TensorLayout::contiguous(graph_token_shape, 2), ScalarType::kBFloat16);
  DeviceTensor graph_selectors = graph_context.allocate(
      TensorLayout::contiguous(graph_selector_shape, 1), ScalarType::kInt32);
  DeviceTensor graph_code = graph_context.allocate(
      TensorLayout::contiguous(graph_code_shape, 2));
  DeviceTensor graph_cosine = graph_context.allocate(
      TensorLayout::contiguous(graph_rope_shape, 2));
  DeviceTensor graph_sine = graph_context.allocate(
      TensorLayout::contiguous(graph_rope_shape, 2));
  graph_context.upload_bytes(graph_selectors, host_selectors.data(),
                             host_selectors.size() * 4);
  graph_context.upload(graph_code, host_code.data(), host_code.size());
  graph_context.upload(graph_cosine, host_cos.data(), host_cos.size());
  graph_context.upload(graph_sine, host_sin.data(), host_sin.size());
  std::vector<DeviceTensor> boundaries;
  boundaries.reserve(graph_config.layers);
  for (uint32_t layer = 0; layer < graph_config.layers; ++layer)
    boundaries.push_back(graph_context.allocate(
        TensorLayout::contiguous(graph_token_shape, 2), ScalarType::kBFloat16));
  H3MainGraphReplayTaps graph_taps{boundaries.data(),
                                   static_cast<uint32_t>(boundaries.size())};
  DeviceTensor graph_dummy = graph_context.allocate(
      TensorLayout::contiguous(graph_token_shape, 2), ScalarType::kBFloat16);
  DeviceTensor graph_dummy_out = graph_context.allocate(
      TensorLayout::contiguous(graph_token_shape, 2), ScalarType::kBFloat16);
  const uint64_t graph_preload_used = graph_context.pooled_used_bytes();
  graph.load(graph_checkpoint);
  CHECK(graph.loaded() && graph.required_operators() == 75u);
  CHECK(graph.required_operators(&graph_taps) == 78u);
  {
    TensorBatch short_batch = graph_context.begin_batch();
    for (uint32_t i = graph.required_operators(&graph_taps); i <= 128; ++i)
      short_batch.copy(graph_dummy, graph_dummy_out);
    bool rejected = false;
    try {
      graph.record(short_batch, graph_tokens, graph_selectors, graph_code,
                   graph_cosine, graph_sine, nullptr, &graph_taps);
    } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected && short_batch.remaining_operator_capacity() == 77u);
  }
  auto run_graph = [&] {
    graph_context.upload_bytes(graph_tokens, input.data(), input.size() * 2);
    TensorBatch batch = graph_context.begin_batch();
    for (uint32_t i = graph.required_operators(&graph_taps); i < 128; ++i)
      batch.copy(graph_dummy, graph_dummy_out);
    graph.record(batch, graph_tokens, graph_selectors, graph_code,
                 graph_cosine, graph_sine, nullptr, &graph_taps);
    CHECK(batch.remaining_operator_capacity() == 0u);
    batch.submit().wait();
    std::vector<uint16_t> result(input.size());
    graph_context.download_bytes(graph_tokens, result.data(), result.size() * 2);
    return result;
  };
  const std::vector<uint16_t> graph_output = run_graph();
  auto digest_bf16 = [](const std::vector<uint16_t>& values) {
    uint64_t hash = 1469598103934665603ull;
    for (uint16_t bits : values) {
      hash ^= bits & 0xffu; hash *= 1099511628211ull;
      hash ^= bits >> 8; hash *= 1099511628211ull;
    }
    return hash;
  };
  std::vector<uint64_t> boundary_hashes;
  for (DeviceTensor& boundary : boundaries) {
    std::vector<uint16_t> boundary_values(input.size());
    graph_context.download_bytes(boundary, boundary_values.data(),
                                 boundary_values.size() * 2);
    boundary_hashes.push_back(digest_bf16(boundary_values));
  }
  CHECK(boundary_hashes == std::vector<uint64_t>({
      0xc40d66ec8f2b7f26ull, 0x70e1eaf01723020bull,
      0xe9b03bc5e1718291ull}));
  CHECK(digest_bf16(graph_output) == boundary_hashes.back());
  CHECK(graph.required_operators(0, 1, &graph_taps) == 26u);
  CHECK(graph.required_operators(1, 2, &graph_taps) == 52u);
  graph_context.upload_bytes(graph_tokens, input.data(), input.size() * 2);
  {
    TensorBatch split = graph_context.begin_batch();
    graph.record_layers(split, graph_tokens, graph_selectors, graph_code,
                        graph_cosine, graph_sine, 0, 1, nullptr, &graph_taps);
    graph.record_layers(split, graph_tokens, graph_selectors, graph_code,
                        graph_cosine, graph_sine, 1, 2, nullptr, &graph_taps);
    CHECK(split.remaining_operator_capacity() == 50u);
    split.submit().wait();
  }
  std::vector<uint16_t> split_output(input.size());
  graph_context.download_bytes(graph_tokens, split_output.data(),
                               split_output.size() * 2);
  CHECK(split_output == graph_output);
  {
    TensorBatch invalid_span = graph_context.begin_batch();
    bool rejected = false;
    try {
      graph.record_layers(invalid_span, graph_tokens, graph_selectors,
                          graph_code, graph_cosine, graph_sine, 2, 2);
    } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected && invalid_span.remaining_operator_capacity() == 128u);
  }
  const uint64_t graph_used = graph_context.pooled_used_bytes();
  const uint64_t graph_reserved = graph_context.reserved_bytes();
  const uint64_t graph_descriptors = graph_context.descriptor_set_allocations();
  CHECK(run_graph() == graph_output);
  CHECK(graph_context.pooled_used_bytes() == graph_used);
  CHECK(graph_context.reserved_bytes() == graph_reserved);
  CHECK(graph_context.descriptor_set_allocations() == graph_descriptors);
  const uint64_t graph_peak = graph.peak_device_bytes();
  bool active_reload_rejected = false;
  try { graph.load(graph_checkpoint); }
  catch (const std::logic_error&) { active_reload_rejected = true; }
  CHECK(active_reload_rejected && graph.loaded());
  CHECK(graph_context.pooled_used_bytes() == graph_used);
  CHECK(graph_context.reserved_bytes() == graph_reserved);
  CHECK(graph_context.descriptor_set_allocations() == graph_descriptors);
  CHECK(graph.peak_device_bytes() == graph_peak);
  CHECK(run_graph() == graph_output);
  const uint64_t graph_persistent = graph.persistent_bytes();
  const uint64_t graph_scratch = graph.scratch_bytes();
  graph.unload();
  CHECK(!graph.loaded() && graph.persistent_bytes() == 0u &&
        graph.scratch_bytes() == 0u && graph.peak_device_bytes() == 0u);
  CHECK_MSG(graph_context.pooled_used_bytes() == graph_preload_used,
            "H3 graph unload used %llu, baseline %llu",
            static_cast<unsigned long long>(graph_context.pooled_used_bytes()),
            static_cast<unsigned long long>(graph_preload_used));
  graph.load(graph_checkpoint);
  CHECK(graph.persistent_bytes() == graph_persistent &&
        graph.scratch_bytes() == graph_scratch &&
        graph.peak_device_bytes() == graph_peak);
  CHECK(run_graph() == graph_output);
  std::printf(
      "  CUDA-off H3 3-layer graph FNV64 %016llx, ops %u, persistent/scratch %.2f/%.2f MiB\n",
      static_cast<unsigned long long>(digest_bf16(graph_output)),
      graph.required_operators(), double(graph.persistent_bytes()) / 1048576.0,
      double(graph.scratch_bytes()) / 1048576.0);
  graph.unload();

  // Complete small/tail transformer evaluation: refiner cache, endpoint
  // projections/scatters, main graph and final modality heads.
  ExactH3TransformerConfig transformer_config;
  transformer_config.main.block = config;
  transformer_config.main.layers = 1;
  transformer_config.text_rows = 3;
  transformer_config.video_rows = 60;
  transformer_config.audio_rows = 2;
  transformer_config.text_dim = 16;
  transformer_config.video_dim = 4;
  transformer_config.audio_dim = 2;
  transformer_config.refiner_layers = 1;
  const uint64_t prompt_shape[] = {3, 16};
  const uint64_t video_input_shape[] = {60, 4};
  const uint64_t audio_input_shape[] = {2, 2};
  DeviceTensor prompt = graph_context.allocate(
      TensorLayout::contiguous(prompt_shape, 2));
  DeviceTensor transformer_video = graph_context.allocate(
      TensorLayout::contiguous(video_input_shape, 2));
  DeviceTensor transformer_audio = graph_context.allocate(
      TensorLayout::contiguous(audio_input_shape, 2));
  DeviceTensor transformer_video_out = graph_context.allocate(
      TensorLayout::contiguous(video_input_shape, 2));
  DeviceTensor transformer_audio_out = graph_context.allocate(
      TensorLayout::contiguous(audio_input_shape, 2));
  DeviceTensor transformer_text_idx = graph_context.allocate(
      TensorLayout::contiguous(prompt_shape, 1), ScalarType::kInt32);
  const uint64_t video_index_shape[] = {60};
  const uint64_t audio_index_shape[] = {2};
  DeviceTensor transformer_video_idx = graph_context.allocate(
      TensorLayout::contiguous(video_index_shape, 1), ScalarType::kInt32);
  DeviceTensor transformer_audio_idx = graph_context.allocate(
      TensorLayout::contiguous(audio_index_shape, 1), ScalarType::kInt32);
  DeviceTensor transformer_selectors = graph_context.allocate(
      TensorLayout::contiguous(selector_shape, 1), ScalarType::kInt32);
  DeviceTensor transformer_code = graph_context.allocate(
      TensorLayout::contiguous(code_shape, 2));
  DeviceTensor transformer_cosine = graph_context.allocate(
      TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor transformer_sine = graph_context.allocate(
      TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor transformer_video_ts = graph_context.allocate(
      TensorLayout::contiguous(video_index_shape, 1), ScalarType::kInt32);
  DeviceTensor transformer_audio_ts = graph_context.allocate(
      TensorLayout::contiguous(audio_index_shape, 1), ScalarType::kInt32);
  const std::vector<float> prompt_values = values(3 * 16, 163, 1.0f / 32.0f);
  const std::vector<float> video_values = values(60 * 4, 167, 1.0f / 32.0f);
  const std::vector<float> audio_values = values(2 * 2, 173, 1.0f / 32.0f);
  std::vector<int32_t> text_idx{0, 1, 2}, audio_idx{3, 4}, video_idx(60);
  for (int32_t i = 0; i < 60; ++i) video_idx[i] = i + 5;
  std::vector<int32_t> zero_video_ts(60, 0), zero_audio_ts(2, 0);
  graph_context.upload(prompt, prompt_values.data(), prompt_values.size());
  graph_context.upload(transformer_video, video_values.data(), video_values.size());
  graph_context.upload(transformer_audio, audio_values.data(), audio_values.size());
  graph_context.upload_bytes(transformer_text_idx, text_idx.data(), text_idx.size() * 4);
  graph_context.upload_bytes(transformer_video_idx, video_idx.data(), video_idx.size() * 4);
  graph_context.upload_bytes(transformer_audio_idx, audio_idx.data(), audio_idx.size() * 4);
  graph_context.upload_bytes(transformer_selectors, host_selectors.data(),
                             host_selectors.size() * 4);
  graph_context.upload(transformer_code, host_code.data(), host_code.size());
  graph_context.upload(transformer_cosine, host_cos.data(), host_cos.size());
  graph_context.upload(transformer_sine, host_sin.data(), host_sin.size());
  graph_context.upload_bytes(transformer_video_ts, zero_video_ts.data(),
                             zero_video_ts.size() * 4);
  graph_context.upload_bytes(transformer_audio_ts, zero_audio_ts.data(),
                             zero_audio_ts.size() * 4);
  DeviceTensor shared_forward_tap = graph_context.allocate(
      TensorLayout::contiguous(token_shape, 2), ScalarType::kBFloat16);
  const uint64_t transformer_baseline = graph_context.pooled_used_bytes();
  const uint64_t transformer_baseline_reserved = graph_context.reserved_bytes();
  const uint64_t transformer_baseline_descriptors =
      graph_context.descriptor_set_allocations();
  ExactH3Transformer transformer = ExactH3Transformer::create(
      graph_context, transformer_config);
  CHECK(!transformer.loaded() && transformer.persistent_bytes() == 0u &&
        transformer.scratch_bytes() == 0u);
  bool transformer_corrupt_rejected = false;
  try { transformer.load(corrupt_transformer); }
  catch (const std::exception&) { transformer_corrupt_rejected = true; }
  CHECK(transformer_corrupt_rejected && !transformer.loaded());
  CHECK(graph_context.pooled_used_bytes() == transformer_baseline);
  std::vector<SafeTensors*> invalid_endpoint_archives{
      &corrupt_transformer_dtype};
  for (SafeTensors& invalid : corrupt_transformer_metadata)
    invalid_endpoint_archives.push_back(&invalid);
  for (SafeTensors* invalid : invalid_endpoint_archives) {
    bool rejected = false;
    try { transformer.load(*invalid); }
    catch (const std::exception&) { rejected = true; }
    CHECK(rejected && !transformer.loaded());
    CHECK(graph_context.pooled_used_bytes() == transformer_baseline);
    CHECK(graph_context.reserved_bytes() == transformer_baseline_reserved);
    CHECK(graph_context.descriptor_set_allocations() ==
          transformer_baseline_descriptors);
  }
  transformer.load(transformer_checkpoint);
  CHECK(transformer.loaded() && !transformer.text_prepared());
  H3TransformerTextReplayTaps null_text_taps{nullptr, 6};
  H3TransformerTextReplayTaps short_text_taps{&prompt, 5};
  H3TransformerTextReplayTaps wrong_text_taps{&prompt, 6};
  for (const H3TransformerTextReplayTaps* invalid :
       {&null_text_taps, &short_text_taps, &wrong_text_taps}) {
    bool rejected = false;
    try { transformer.prepare_text(prompt, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected && !transformer.text_prepared());
  }
  transformer.prepare_text(prompt);
  CHECK(transformer.text_prepared());
  CHECK(transformer.required_forward_operators() == 42u);
  H3MainGraphReplayTaps null_main_taps{nullptr, 1};
  H3TransformerForwardReplayTaps null_forward_taps{
      nullptr, nullptr, &null_main_taps};
  H3MainGraphReplayTaps aliased_main_taps{&shared_forward_tap, 1};
  H3TransformerForwardReplayTaps aliased_forward_taps{
      &shared_forward_tap, nullptr, &aliased_main_taps};
  H3TransformerForwardReplayTaps valid_forward_taps{
      &shared_forward_tap, nullptr, nullptr};
  CHECK(transformer.required_forward_operators(nullptr) == 42u);
  CHECK(transformer.required_forward_operators(&valid_forward_taps) == 43u);
  {
    TensorBatch transactional = graph_context.begin_batch();
    auto reject_forward = [&](const H3AttentionRanges* test_ranges,
                              const H3TransformerForwardReplayTaps* test_taps) {
      bool rejected = false;
      try {
        transformer.record_forward(
            transactional, transformer_video, transformer_audio,
            transformer_selectors, transformer_code, transformer_cosine,
            transformer_sine, transformer_video_ts, transformer_audio_ts,
            transformer_video_out, transformer_audio_out, test_ranges,
            test_taps);
      } catch (const std::invalid_argument&) { rejected = true; }
      CHECK(rejected && transactional.remaining_operator_capacity() == 128u);
    };
    reject_forward(&foreign_ranges, nullptr);
    reject_forward(nullptr, &null_forward_taps);
    reject_forward(nullptr, &aliased_forward_taps);
    transformer.record_forward(
        transactional, transformer_video, transformer_audio,
        transformer_selectors, transformer_code, transformer_cosine,
        transformer_sine, transformer_video_ts, transformer_audio_ts,
        transformer_video_out, transformer_audio_out);
    CHECK(transactional.remaining_operator_capacity() == 86u);
    transactional.submit().wait();
  }
  {
    // A tapped forward that is exactly one operator short must not append any
    // work. The same batch remains usable by the untapped production path.
    TensorBatch one_short = graph_context.begin_batch();
    const uint32_t tapped_need =
        transformer.required_forward_operators(&valid_forward_taps);
    for (uint32_t i = tapped_need - 1; i < 128u; ++i)
      one_short.copy(transformer_video, transformer_video_out);
    const uint32_t remaining = one_short.remaining_operator_capacity();
    bool rejected = false;
    try {
      transformer.record_forward(
          one_short, transformer_video, transformer_audio,
          transformer_selectors, transformer_code, transformer_cosine,
          transformer_sine, transformer_video_ts, transformer_audio_ts,
          transformer_video_out, transformer_audio_out, nullptr,
          &valid_forward_taps);
    } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected && one_short.remaining_operator_capacity() == remaining);
    transformer.record_forward(
        one_short, transformer_video, transformer_audio,
        transformer_selectors, transformer_code, transformer_cosine,
        transformer_sine, transformer_video_ts, transformer_audio_ts,
        transformer_video_out, transformer_audio_out);
    CHECK(one_short.remaining_operator_capacity() == 0u);
    one_short.submit().wait();
  }
  auto run_transformer = [&] {
    TensorBatch batch = graph_context.begin_batch();
    transformer.record_forward(
        batch, transformer_video, transformer_audio, transformer_selectors,
        transformer_code, transformer_cosine, transformer_sine,
        transformer_video_ts, transformer_audio_ts, transformer_video_out,
        transformer_audio_out);
    CHECK(batch.remaining_operator_capacity() ==
          graph_options.max_batch_operators - transformer.required_forward_operators());
    batch.submit().wait();
    std::vector<float> video_result(60 * 4), audio_result(2 * 2);
    graph_context.download(transformer_video_out, video_result.data(),
                           video_result.size());
    graph_context.download(transformer_audio_out, audio_result.data(),
                           audio_result.size());
    video_result.insert(video_result.end(), audio_result.begin(), audio_result.end());
    return video_result;
  };
  const std::vector<float> transformer_output = run_transformer();
  const uint64_t transformer_digest = fnv64_floats(transformer_output);
  CHECK(transformer_digest == 0x30dec4598d352f88ull);
  std::printf("  CUDA-off H3 transformer S65 FNV64 %016llx, ops %u, persistent/scratch %.2f/%.2f MiB\n",
      static_cast<unsigned long long>(transformer_digest),
      transformer.required_forward_operators(),
      double(transformer.persistent_bytes()) / 1048576.0,
      double(transformer.scratch_bytes()) / 1048576.0);
  CHECK(run_transformer() == transformer_output);
  const uint64_t transformer_persistent = transformer.persistent_bytes();
  const uint64_t transformer_scratch = transformer.scratch_bytes();
  const uint64_t transformer_live_used = graph_context.pooled_used_bytes();
  const uint64_t transformer_reserved = graph_context.reserved_bytes();
  const uint64_t transformer_descriptors =
      graph_context.descriptor_set_allocations();
  bool transformer_reload_rejected = false;
  try { transformer.load(corrupt_transformer_metadata.back()); }
  catch (const std::logic_error&) { transformer_reload_rejected = true; }
  CHECK(transformer_reload_rejected && transformer.text_prepared());
  CHECK(graph_context.pooled_used_bytes() == transformer_live_used);
  CHECK(graph_context.reserved_bytes() == transformer_reserved);
  CHECK(graph_context.descriptor_set_allocations() == transformer_descriptors);
  CHECK(run_transformer() == transformer_output);
  transformer.unload();
  CHECK(!transformer.loaded() && transformer.persistent_bytes() == 0u &&
        transformer.scratch_bytes() == 0u &&
        transformer.peak_device_bytes() == 0u);
  CHECK(graph_context.pooled_used_bytes() == transformer_baseline);
  transformer.load(transformer_checkpoint);
  CHECK(transformer.persistent_bytes() == transformer_persistent &&
        transformer.scratch_bytes() == transformer_scratch);
  transformer.prepare_text(prompt);
  CHECK(run_transformer() == transformer_output);
  transformer.unload();

  // Singularity wraps every key and stores these endpoints as BF16. The
  // fixture values are exactly representable, so storage must not affect output.
  auto singularity_tensors = transformer_fixture(0);
  for (auto& tensor : singularity_tensors) {
    if (tensor.name == "video_patch_proj.weight" ||
        tensor.name == "audio_patch_proj.weight" ||
        tensor.name == "final_layer.adaln_proj.linear.weight" ||
        tensor.name == "final_layer.video_out.weight" ||
        tensor.name == "final_layer.audio_out.weight")
      tensor.dtype = DType::kBF16;
    tensor.name = "model.diffusion_model." + tensor.name;
  }
  const auto singularity_path = base / "Singularity_ref2va_Pruned.safetensors";
  write_safetensors(singularity_path.string(), singularity_tensors);
  SafeTensors singularity;
  singularity.open(singularity_path.string());
  transformer.load(singularity);
  transformer.prepare_text(prompt);
  CHECK(run_transformer() == transformer_output);
  transformer.unload();
  singularity.close();
  std::filesystem::remove(singularity_path);

  // Viggle keeps the AdaLN curve/projections in F32 and declares upstream
  // per-head QKV storage. The endpoint dtype must not change the result.
  auto viggle_tensors = transformer_fixture(0);
  for (auto& tensor : viggle_tensors)
    if (tensor.name.find("adaln_proj.linear") != std::string::npos)
      tensor.dtype = DType::kF32;
  const auto viggle_path = base / "Viggle-Animate-pruned_rank8_int8_convrot.safetensors";
  write_safetensors(viggle_path.string(), viggle_tensors,
      {{"source", "Viggle/Viggle-Animate"}, {"qkv_layout", "interleaved"}});
  SafeTensors viggle;
  viggle.open(viggle_path.string());
  transformer.load(viggle);
  transformer.prepare_text(prompt);
  CHECK(run_transformer() == transformer_output);
  transformer.unload();
  viggle.close();
  std::filesystem::remove(viggle_path);

  // Every Viggle target must affect the forward pass. In particular, accepting
  // proj_in/proj_out or split Q/K/V in the host loader must not drop the update.
  const auto viggle_adapter_path = base / "slopfab_viggle_vulkan_lora.safetensors";
  for (const auto& target : std::vector<std::pair<std::string, std::string>>{
      {"proj_in", "video_patch_proj"}, {"proj_out", "final_layer.video_out"},
      {"transformer_blocks.0.attn.to_q", "blocks.0.attn.qkv_proj"},
      {"transformer_blocks.0.attn.to_k", "blocks.0.attn.qkv_proj"},
      {"transformer_blocks.0.attn.to_v", "blocks.0.attn.qkv_proj"},
      {"transformer_blocks.0.attn.to_out.0", "blocks.0.attn.out_proj"},
      {"transformer_blocks.0.ff.net.0.proj", "blocks.0.mlp.fc1"},
      {"transformer_blocks.0.ff.net.2", "blocks.0.mlp.fc2"},
      {"blocks.0.adaln_proj.linear", "blocks.0.adaln_proj.linear"},
      {"final_layer.adaln_proj.linear", "final_layer.adaln_proj.linear"}}) {
    const auto& weight = transformer_checkpoint.at(target.second + ".weight");
    const int64_t in = weight.shape[1];
    const int64_t out = weight.shape[0] / (target.second == "blocks.0.attn.qkv_proj" ? 3 : 1);
    write_safetensors(viggle_adapter_path.string(), {
        {target.first + ".lora_A.weight", {16, in}, test::make_data(size_t(16 * in), 893, .25f)},
        {target.first + ".lora_B.weight", {out, 16}, test::make_data(size_t(out * 16), 981, .25f)}});
    LoraAdapters adapter;
    if (transformer_checkpoint.find(target.second + ".pre_quant_scale")) {
      bool rejected = false;
      try { adapter.load({{viggle_adapter_path.string(), 1}}, transformer_checkpoint); }
      catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("AWQ activation scaling") != std::string::npos;
      }
      CHECK(rejected);
      continue;
    }
    adapter.load({{viggle_adapter_path.string(), 1}}, transformer_checkpoint);
    auto adapter_config = transformer_config;
    adapter_config.main.block.loras = &adapter;
    transformer = ExactH3Transformer::create(graph_context, adapter_config);
    transformer.load(transformer_checkpoint);
    transformer.prepare_text(prompt);
    CHECK_MSG(run_transformer() != transformer_output, "%s update did not reach Vulkan forward", target.first.c_str());
    transformer.unload();
  }
  transformer = ExactH3Transformer::create(graph_context, transformer_config);
  std::filesystem::remove(viggle_adapter_path);

  // Complete CUDA-off T2VA denoise trajectory. Modality rows remain on the
  // device across every transformer evaluation and Euler update; only final
  // rows cross back to the host.
  TensorContextOptions denoise_options;
  denoise_options.max_batch_operators = 64;
  TensorContext denoise_context(device, denoise_options);
  ExactH3DenoiseConfig denoise_config;
  denoise_config.transformer = transformer_config;
  denoise_config.transformer.main.block.timesteps = 2;
  denoise_config.layout.num_text = 3;
  denoise_config.layout.num_audio_rows = 2;
  denoise_config.layout.num_video_rows = 60;
  denoise_config.layout.num_audio_latents = 1;
  denoise_config.layout.num_latent_frames = 15;
  denoise_config.layout.latent_height = 4;
  denoise_config.layout.latent_width = 4;
  denoise_config.indices = dit::build_indices(denoise_config.layout);
  denoise_config.position_ids = dit::build_position_ids(denoise_config.layout);
  denoise_config.attention_band = 1;

  // Ref2VA conditions are interleaved in packed order, while each modality
  // list is condition-prefix then generated suffix. Validate that schema
  // transactionally before any checkpoint or device allocation is touched.
  const std::vector<int32_t> ref_text_tags{
      dit::kTagText, dit::kTagVideo, dit::kTagText};
  const std::vector<dit::ReferenceGeometry> ref_geometries = {
      {dit::ReferenceKind::kImage, 1, 4, 4, 0},
      {dit::ReferenceKind::kAudio, 0, 0, 0, 1}};
  const dit::Ref2VAPackedSequence ref_packed =
      dit::build_ref2va_packed_sequence(
          ref_text_tags, ref_geometries, 1, 4, 4, 1);
  ExactH3DenoiseConfig ref_config = denoise_config;
  ref_config.layout = ref_packed.layout;
  ref_config.indices = ref_packed.indices;
  ref_config.position_ids = ref_packed.position_ids;
  ref_config.attention_band = 0;
  ref_config.transformer.text_rows =
      static_cast<uint32_t>(ref_config.layout.num_text);
  ref_config.transformer.video_rows =
      static_cast<uint32_t>(ref_config.indices.video.size());
  ref_config.transformer.audio_rows =
      static_cast<uint32_t>(ref_config.indices.audio.size());
  ref_config.transformer.video_output_rows =
      static_cast<uint32_t>(ref_config.layout.num_video_rows);
  ref_config.transformer.audio_output_rows =
      static_cast<uint32_t>(ref_config.layout.num_audio_rows);
  ref_config.transformer.video_output_start =
      static_cast<uint32_t>(ref_config.layout.video_start());
  ref_config.transformer.audio_output_start =
      static_cast<uint32_t>(ref_config.layout.audio_start());
  ref_config.transformer.main.block.sequence =
      static_cast<uint32_t>(ref_config.layout.total_rows());
  ref_config.transformer.main.block.timesteps = 4;
  const uint64_t ref_validate_used = denoise_context.pooled_used_bytes();
  const uint64_t ref_validate_reserved = denoise_context.reserved_bytes();
  const uint64_t ref_validate_descriptors =
      denoise_context.descriptor_set_allocations();
  {
    ExactH3Denoiser ref_denoiser = ExactH3Denoiser::create(
        denoise_context, ref_config);
    CHECK(!ref_denoiser.loaded());
  }
  auto reject_ref_config = [&](const ExactH3DenoiseConfig& invalid) {
    bool rejected = false;
    try {
      ExactH3Denoiser candidate = ExactH3Denoiser::create(
          denoise_context, invalid);
    } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    CHECK(denoise_context.pooled_used_bytes() == ref_validate_used);
    CHECK(denoise_context.reserved_bytes() == ref_validate_reserved);
    CHECK(denoise_context.descriptor_set_allocations() ==
          ref_validate_descriptors);
  };
  ExactH3DenoiseConfig invalid_ref = ref_config;
  invalid_ref.indices.audio[0] = invalid_ref.indices.video[0];
  reject_ref_config(invalid_ref);
  invalid_ref = ref_config;
  std::swap(invalid_ref.indices.video.front(),
            invalid_ref.indices.video.back());
  reject_ref_config(invalid_ref);
  invalid_ref = ref_config;
  ++invalid_ref.transformer.video_output_start;
  reject_ref_config(invalid_ref);
  invalid_ref = ref_config;
  invalid_ref.transformer.main.block.timesteps = 3;
  reject_ref_config(invalid_ref);
  invalid_ref = ref_config;
  invalid_ref.layout.num_condition_video = std::numeric_limits<int>::max();
  invalid_ref.layout.num_condition_audio = std::numeric_limits<int>::max();
  reject_ref_config(invalid_ref);

  const uint64_t denoise_baseline = denoise_context.pooled_used_bytes();
  const uint64_t denoise_baseline_reserved = denoise_context.reserved_bytes();
  const uint64_t denoise_baseline_descriptors =
      denoise_context.descriptor_set_allocations();
  ExactH3Denoiser denoiser = ExactH3Denoiser::create(
      denoise_context, denoise_config);
  bool denoise_corrupt_rejected = false;
  try { denoiser.load(corrupt_transformer_metadata.front()); }
  catch (const std::exception&) { denoise_corrupt_rejected = true; }
  CHECK(denoise_corrupt_rejected && !denoiser.loaded());
  CHECK_MSG(denoise_context.pooled_used_bytes() == denoise_baseline,
            "corrupt denoise load used %llu baseline %llu",
            static_cast<unsigned long long>(denoise_context.pooled_used_bytes()),
            static_cast<unsigned long long>(denoise_baseline));
  CHECK(denoise_context.reserved_bytes() == denoise_baseline_reserved);
  CHECK(denoise_context.descriptor_set_allocations() ==
        denoise_baseline_descriptors);
  denoiser.load(transformer_checkpoint);
  CHECK(denoiser.loaded() && !denoiser.prepared());
  CHECK(denoiser.required_step_operators() == 44u);
  const uint64_t capacity_shape[] = {1};
  DeviceTensor capacity_source = denoise_context.allocate(
      TensorLayout::contiguous(capacity_shape, 1));
  DeviceTensor capacity_destination = denoise_context.allocate(
      TensorLayout::contiguous(capacity_shape, 1));
  {
    // The complete transformer + two Euler updates fits exactly. Preflight is
    // observational: the same recording can still accept a smaller graph.
    TensorBatch exact_capacity = denoise_context.begin_batch();
    for (uint32_t i = denoiser.required_step_operators(); i < 64u; ++i)
      exact_capacity.copy(capacity_source, capacity_destination);
    exact_capacity.require_operator_capacity(denoiser.required_step_operators());
    CHECK(exact_capacity.remaining_operator_capacity() ==
          denoiser.required_step_operators());
    exact_capacity.copy(capacity_source, capacity_destination);
    exact_capacity.submit().wait();
  }
  {
    TensorBatch short_capacity = denoise_context.begin_batch();
    for (uint32_t i = denoiser.required_step_operators(); i <= 64u; ++i)
      short_capacity.copy(capacity_source, capacity_destination);
    bool rejected = false;
    try {
      short_capacity.require_operator_capacity(
          denoiser.required_step_operators());
    } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected && short_capacity.remaining_operator_capacity() + 1u ==
                          denoiser.required_step_operators());
    short_capacity.copy(capacity_source, capacity_destination);
    short_capacity.submit().wait();
  }
  sampler::FlowScheduler denoise_video(12.0f), denoise_audio(3.0f);
  denoise_video.set_timesteps(4);
  denoise_audio.set_timesteps(4);
  denoiser.prepare(prompt_values.data(), prompt_values.size(),
                   video_values.data(), video_values.size(),
                   audio_values.data(), audio_values.size());
  CHECK(denoiser.prepared());
  const ExactH3DenoiseResult denoise_output = denoiser.run(
      denoise_video, denoise_audio);
  CHECK(!denoise_output.cancelled && denoise_output.steps_completed == 3u);
  std::vector<float> denoise_joined = denoise_output.video_rows;
  denoise_joined.insert(denoise_joined.end(), denoise_output.audio_rows.begin(),
                        denoise_output.audio_rows.end());
  const uint64_t denoise_digest = fnv64_floats(denoise_joined);
  CHECK(denoise_digest == 0xad06feae77d1c494ull);
  const uint64_t denoise_used = denoise_context.pooled_used_bytes();
  const uint64_t denoise_reserved = denoise_context.reserved_bytes();
  const uint64_t denoise_descriptors =
      denoise_context.descriptor_set_allocations();
  denoiser.prepare(prompt_values.data(), prompt_values.size(),
                   video_values.data(), video_values.size(),
                   audio_values.data(), audio_values.size());
  ExactH3DenoiseResult denoise_repeat = denoiser.run(
      denoise_video, denoise_audio);
  CHECK(denoise_repeat.video_rows == denoise_output.video_rows &&
        denoise_repeat.audio_rows == denoise_output.audio_rows);
  CHECK(denoise_context.pooled_used_bytes() == denoise_used);
  CHECK(denoise_context.reserved_bytes() == denoise_reserved);
  CHECK(denoise_context.descriptor_set_allocations() == denoise_descriptors);
  denoiser.prepare(prompt_values.data(), prompt_values.size(),
                   video_values.data(), video_values.size(),
                   audio_values.data(), audio_values.size());
  ExactH3DenoiseResult cancelled = denoiser.run(
      denoise_video, denoise_audio,
      [](uint32_t step, uint32_t) { return step != 0; });
  CHECK(cancelled.cancelled && cancelled.steps_completed == 1u);
  sampler::FlowScheduler unsupported_audio = denoise_audio;
  unsupported_audio.set_sampler(sampler::SamplerKind::kAb2);
  bool ab2_rejected = false;
  try { (void)denoiser.run(denoise_video, unsupported_audio); }
  catch (const std::invalid_argument&) { ab2_rejected = true; }
  CHECK(ab2_rejected && denoiser.prepared());
  const uint64_t denoise_persistent = denoiser.persistent_bytes();
  const uint64_t denoise_scratch = denoiser.scratch_bytes();
  CHECK(denoise_persistent != 0 && denoise_scratch != 0 &&
        denoiser.peak_device_bytes() == denoise_persistent + denoise_scratch);
  denoiser.unload();
  CHECK(!denoiser.loaded() && !denoiser.prepared() &&
        denoiser.persistent_bytes() == 0u && denoiser.scratch_bytes() == 0u &&
        denoiser.peak_device_bytes() == 0u);
  // The context intentionally keeps its paired upload/readback buffers. Pool
  // usage includes driver memory-requirement rounding, so use the observed
  // unloaded watermark and prove it is bounded and stable across reload.
  const uint64_t denoise_staging_used = denoise_context.pooled_used_bytes();
  CHECK(denoise_context.staging_capacity_bytes() != 0 &&
        denoise_staging_used > denoise_baseline &&
        denoise_staging_used - denoise_baseline < denoise_persistent);
  denoiser.load(transformer_checkpoint);
  denoiser.prepare(prompt_values.data(), prompt_values.size(),
                   video_values.data(), video_values.size(),
                   audio_values.data(), audio_values.size());
  ExactH3DenoiseResult after_reload = denoiser.run(
      denoise_video, denoise_audio);
  CHECK(after_reload.video_rows == denoise_output.video_rows &&
        after_reload.audio_rows == denoise_output.audio_rows);
  CHECK(denoiser.persistent_bytes() == denoise_persistent &&
        denoiser.scratch_bytes() == denoise_scratch);
  std::printf(
      "  CUDA-off H3 denoise S65 x3 FNV64 %016llx, persistent/scratch %.2f/%.2f MiB\n",
      static_cast<unsigned long long>(denoise_digest),
      double(denoise_persistent) / 1048576.0,
      double(denoise_scratch) / 1048576.0);
  denoiser.unload();
  CHECK(denoise_context.pooled_used_bytes() == denoise_staging_used);

  // MotionCache leaves the disabled trajectory bit-identical, and its split
  // forward/update submissions must also agree when warmup prevents reuse.
  {
    auto mc_config = denoise_config;
    sampler::FlowScheduler mv(12), ma(3);
    mv.set_timesteps(12); ma.set_timesteps(12);
    auto run_motion = [&](const ExactH3DenoiseConfig& config, bool cancel) {
      auto model = ExactH3Denoiser::create(denoise_context, config);
      model.load(transformer_checkpoint);
      model.prepare(prompt_values.data(), prompt_values.size(),
                    video_values.data(), video_values.size(),
                    audio_values.data(), audio_values.size());
      const auto output = model.run(mv, ma, [&](uint32_t step, uint32_t) {
        return !cancel || step < 3;
      });
      CHECK(output.steps_completed == output.steps_computed + output.steps_skipped);
      if (!cancel) {
        model.prepare(prompt_values.data(), prompt_values.size(),
                      video_values.data(), video_values.size(),
                      audio_values.data(), audio_values.size());
        const auto repeat = model.run(mv, ma);
        CHECK(output.video_rows == repeat.video_rows);
        CHECK(output.audio_rows == repeat.audio_rows);
        CHECK(output.steps_skipped == repeat.steps_skipped);
      }
      return output;
    };
    const auto baseline = run_motion(mc_config, false);
    CHECK(baseline.steps_skipped == 0);
    mc_config.motion_cache.enabled = true;
    mc_config.motion_cache.warmup_steps = 20;
    const auto warmup = run_motion(mc_config, false);
    CHECK(warmup.steps_skipped == 0);
    CHECK(warmup.video_rows == baseline.video_rows);
    CHECK(warmup.audio_rows == baseline.audio_rows);
    mc_config.motion_cache.warmup_steps = 2;
    mc_config.motion_cache.reuse_threshold = 1;
    mc_config.motion_cache.start_percent = 0;
    mc_config.motion_cache.end_percent = 1;
    const auto reused = run_motion(mc_config, false);
    CHECK(reused.steps_skipped > 0 && reused.steps_computed >= 3);
    const auto motion_cancelled = run_motion(mc_config, true);
    CHECK(motion_cancelled.cancelled && motion_cancelled.steps_completed == 4);
  }

  // Dedicated still trajectory: one video latent frame and no audio modality.
  // Empty audio host spans stay empty all the way through the transformer,
  // scheduler and result; no one-row device placeholder is introduced.
  ExactH3DenoiseConfig still_config = denoise_config;
  still_config.layout.num_audio_rows = 0;
  still_config.layout.num_audio_latents = 0;
  still_config.layout.num_latent_frames = 1;
  still_config.layout.num_video_rows = still_config.layout.rows_per_frame();
  still_config.indices = dit::build_indices(still_config.layout);
  still_config.position_ids = dit::build_position_ids(still_config.layout);
  still_config.attention_band = 0;
  still_config.transformer.video_rows =
      static_cast<uint32_t>(still_config.layout.num_video_rows);
  still_config.transformer.audio_rows = 0;
  still_config.transformer.main.block.sequence =
      static_cast<uint32_t>(still_config.layout.total_rows());
  ExactH3Denoiser still_denoiser = ExactH3Denoiser::create(
      denoise_context, still_config);
  still_denoiser.load(transformer_checkpoint);
  CHECK_MSG(still_denoiser.required_step_operators() == 30u,
            "video-only Vulkan denoiser recorded %u operators, expected 30",
            still_denoiser.required_step_operators());
  const std::vector<float> still_video(
      video_values.begin(),
      video_values.begin() +
          static_cast<ptrdiff_t>(still_config.layout.num_video_rows * 4));
  still_denoiser.prepare(prompt_values.data(), prompt_values.size(),
                         still_video.data(), still_video.size(), nullptr, 0);
  const ExactH3DenoiseResult still_output = still_denoiser.run(
      denoise_video, denoise_audio);
  CHECK(!still_output.cancelled && still_output.steps_completed == 3u);
  CHECK(still_output.video_rows.size() == still_video.size());
  CHECK(still_output.audio_rows.empty());
  still_denoiser.prepare(prompt_values.data(), prompt_values.size(),
                         still_video.data(), still_video.size(), nullptr, 0);
  const ExactH3DenoiseResult still_repeat = still_denoiser.run(
      denoise_video, denoise_audio);
  CHECK(still_repeat.video_rows == still_output.video_rows);
  CHECK(still_repeat.audio_rows.empty());
  still_denoiser.unload();
  CHECK(denoise_context.pooled_used_bytes() == denoise_staging_used);

  // The same zero-audio contract must survive Ref2VA's indexed packing, where
  // fixed image rows precede the generated still rows in the video modality.
  const dit::Ref2VAPackedSequence still_ref_packed =
      dit::build_ref2va_packed_sequence(
          ref_text_tags,
          {{dit::ReferenceKind::kImage, 1, 4, 4, 0}},
          1, 4, 4, 0);
  ExactH3DenoiseConfig still_ref_config = still_config;
  still_ref_config.layout = still_ref_packed.layout;
  still_ref_config.indices = still_ref_packed.indices;
  still_ref_config.position_ids = still_ref_packed.position_ids;
  still_ref_config.transformer.video_rows =
      static_cast<uint32_t>(still_ref_config.indices.video.size());
  still_ref_config.transformer.video_output_rows =
      static_cast<uint32_t>(still_ref_config.layout.num_video_rows);
  still_ref_config.transformer.video_output_start =
      static_cast<uint32_t>(still_ref_config.layout.video_start());
  still_ref_config.transformer.audio_output_rows = 0;
  still_ref_config.transformer.audio_output_start =
      static_cast<uint32_t>(still_ref_config.layout.audio_start());
  still_ref_config.transformer.main.block.sequence =
      static_cast<uint32_t>(still_ref_config.layout.total_rows());
  still_ref_config.transformer.main.block.timesteps = 4;
  ExactH3Denoiser still_ref_denoiser = ExactH3Denoiser::create(
      denoise_context, still_ref_config);
  still_ref_denoiser.load(transformer_checkpoint);
  const std::vector<float> still_ref_video(
      video_values.begin(),
      video_values.begin() + static_cast<ptrdiff_t>(
          still_ref_config.transformer.video_rows * 4));
  still_ref_denoiser.prepare(
      prompt_values.data(), prompt_values.size(), still_ref_video.data(),
      still_ref_video.size(), nullptr, 0);
  const ExactH3DenoiseResult still_ref_output = still_ref_denoiser.run(
      denoise_video, denoise_audio);
  CHECK(still_ref_output.video_rows.size() ==
        static_cast<size_t>(still_ref_config.layout.num_video_rows) * 4);
  CHECK(still_ref_output.audio_rows.empty());
  still_ref_denoiser.unload();
  CHECK(denoise_context.pooled_used_bytes() == denoise_staging_used);

  CHECK(first.persistent_bytes() == 0u && second.persistent_bytes() == 0u);
  CHECK(context.pooled_used_bytes() + 2 * persistent <= stable_used);
  first.load(valid, 0);
  second.load(valid, 0);
  CHECK(run_chain() == output);
  CHECK(first.persistent_bytes() == persistent);
  std::printf(
      "  CUDA-off H3 S65 two-stage FNV64 %016llx, persistent/scratch %.2f/%.2f MiB\n",
      static_cast<unsigned long long>(digest),
      double(first.persistent_bytes() + second.persistent_bytes()) / 1048576.0,
      double(scratch.reserved_bytes()) / 1048576.0);
  first.unload();
  second.unload();

  // This executable links no CUDA code.  When the shipped checkpoint is
  // present, additionally load its real NVFP4 block 0 and pin the canonical
  // S65 result used by the CUDA-enabled replay suite.
  const std::filesystem::path real_path = std::filesystem::path(
      SLOPFAB_TEST_SOURCE_DIR) /
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  if (std::filesystem::exists(real_path)) {
    H3BlockConfig real_config;
    real_config.sequence = 65;
    SafeTensors real_checkpoint;
    real_checkpoint.open(real_path.string());
    ExactH3BlockStage real_stage = ExactH3BlockStage::create(context, real_config);
    real_stage.load(real_checkpoint, 0);
    ExactH3BlockScratch real_scratch =
        ExactH3BlockScratch::create(context, real_config);
    real_stage.prepare(real_scratch);
    const uint64_t real_token_shape[] = {real_config.sequence, real_config.hidden};
    const uint64_t real_selector_shape[] = {real_config.sequence};
    const uint64_t real_code_shape[] = {real_config.timesteps,
                                        real_config.adaln_rank};
    const uint64_t real_rope_shape[] = {real_config.sequence, 96};
    DeviceTensor real_tokens = context.allocate(
        TensorLayout::contiguous(real_token_shape, 2), ScalarType::kBFloat16);
    DeviceTensor real_selectors = context.allocate(
        TensorLayout::contiguous(real_selector_shape, 1), ScalarType::kInt32);
    DeviceTensor real_code = context.allocate(
        TensorLayout::contiguous(real_code_shape, 2));
    DeviceTensor real_cosine = context.allocate(
        TensorLayout::contiguous(real_rope_shape, 2));
    DeviceTensor real_sine = context.allocate(
        TensorLayout::contiguous(real_rope_shape, 2));
    std::vector<uint16_t> real_input(
        size_t(real_config.sequence) * real_config.hidden);
    for (size_t i = 0; i < real_input.size(); ++i)
      real_input[i] = f32_to_bf16(float(int(i % 61) - 30) / 64.0f);
    std::vector<int32_t> real_selector_values(real_config.sequence);
    for (uint32_t i = 0; i < real_config.sequence; ++i)
      real_selector_values[i] = static_cast<int32_t>(i % real_config.modalities);
    std::vector<float> real_code_values(
        size_t(real_config.timesteps) * real_config.adaln_rank);
    for (size_t i = 0; i < real_code_values.size(); ++i)
      real_code_values[i] = float(int(i % 7) - 3) / 16.0f;
    std::vector<float> real_cosine_values(size_t(real_config.sequence) * 96, 1.0f);
    std::vector<float> real_sine_values(real_cosine_values.size(), 0.0f);
    context.upload_bytes(real_selectors, real_selector_values.data(),
                         real_selector_values.size() * 4);
    context.upload(real_code, real_code_values.data(), real_code_values.size());
    context.upload(real_cosine, real_cosine_values.data(), real_cosine_values.size());
    context.upload(real_sine, real_sine_values.data(), real_sine_values.size());
    auto run_real = [&] {
      context.upload_bytes(real_tokens, real_input.data(), real_input.size() * 2);
      TensorBatch batch = context.begin_batch();
      real_stage.record(batch, real_tokens, real_selectors, real_code,
                        real_cosine, real_sine, real_scratch);
      batch.submit().wait();
      std::vector<uint16_t> result(real_input.size());
      context.download_bytes(real_tokens, result.data(), result.size() * 2);
      return result;
    };
    const std::vector<uint16_t> real_output = run_real();
    uint64_t real_digest = 1469598103934665603ull;
    for (uint16_t bits : real_output) {
      real_digest ^= bits & 0xffu; real_digest *= 1099511628211ull;
      real_digest ^= bits >> 8; real_digest *= 1099511628211ull;
    }
    CHECK(real_digest == 0x191929c14480e873ull);
    const uint64_t real_reserved = context.reserved_bytes();
    const uint64_t real_descriptors = context.descriptor_set_allocations();
    CHECK(run_real() == real_output);
    CHECK(context.reserved_bytes() == real_reserved);
    CHECK(context.descriptor_set_allocations() == real_descriptors);
    const uint64_t real_used = context.pooled_used_bytes();
    real_stage.unload();
    CHECK(context.pooled_used_bytes() < real_used);
    std::printf("  CUDA-off real H3 block0 S65 FNV64 %016llx\n",
                static_cast<unsigned long long>(real_digest));

    H3MainGraphConfig real_graph_config;
    real_graph_config.block = real_config;
    real_graph_config.layers = 2;
    const uint64_t real_graph_preload = context.pooled_used_bytes();
    ExactH3MainGraph real_graph = ExactH3MainGraph::create(
        context, real_graph_config);
    CHECK(context.pooled_used_bytes() == real_graph_preload);
    real_graph.load(real_checkpoint);
    CHECK(real_graph.required_operators() == 58u);
    auto run_real_graph = [&] {
      context.upload_bytes(real_tokens, real_input.data(), real_input.size() * 2);
      TensorBatch batch = context.begin_batch();
      real_graph.record(batch, real_tokens, real_selectors, real_code,
                        real_cosine, real_sine);
      batch.submit().wait();
      std::vector<uint16_t> result(real_input.size());
      context.download_bytes(real_tokens, result.data(), result.size() * 2);
      return result;
    };
    const std::vector<uint16_t> real_graph_output = run_real_graph();
    uint64_t real_graph_digest = 1469598103934665603ull;
    for (uint16_t bits : real_graph_output) {
      real_graph_digest ^= bits & 0xffu; real_graph_digest *= 1099511628211ull;
      real_graph_digest ^= bits >> 8; real_graph_digest *= 1099511628211ull;
    }
    CHECK(real_graph_digest == 0x7f940c81104e7471ull);
    const uint64_t real_graph_reserved = context.reserved_bytes();
    const uint64_t real_graph_descriptors = context.descriptor_set_allocations();
    CHECK(run_real_graph() == real_graph_output);
    CHECK(context.reserved_bytes() == real_graph_reserved);
    CHECK(context.descriptor_set_allocations() == real_graph_descriptors);
    const uint64_t real_graph_used = context.pooled_used_bytes();
    real_graph.unload();
    CHECK(context.pooled_used_bytes() == real_graph_preload);
    CHECK(real_graph_used > real_graph_preload);
    CHECK(real_graph.persistent_bytes() == 0u &&
          real_graph.scratch_bytes() == 0u &&
          real_graph.peak_device_bytes() == 0u);
    std::printf("  CUDA-off real H3 main2 S65 FNV64 %016llx\n",
                static_cast<unsigned long long>(real_graph_digest));

    {
    TensorContextOptions real_transformer_options;
    real_transformer_options.max_batch_operators = 2048;
    TensorContext context(device, real_transformer_options);
    ExactH3TransformerConfig real_transformer_config;
    real_transformer_config.main.block = real_config;
    real_transformer_config.main.layers = 50;
    real_transformer_config.text_rows = 3;
    real_transformer_config.video_rows = 60;
    real_transformer_config.audio_rows = 2;
    real_transformer_config.refiner_layers = 2;
    const uint64_t real_prompt_shape[] = {3, 5120};
    const uint64_t real_video_shape[] = {60, 96};
    const uint64_t real_audio_shape[] = {2, 32};
    const uint64_t real_video_index_shape[] = {60};
    const uint64_t real_audio_index_shape[] = {2};
    DeviceTensor real_prompt = context.allocate(
        TensorLayout::contiguous(real_prompt_shape, 2));
    DeviceTensor real_video = context.allocate(
        TensorLayout::contiguous(real_video_shape, 2));
    DeviceTensor real_audio = context.allocate(
        TensorLayout::contiguous(real_audio_shape, 2));
    DeviceTensor real_video_out = context.allocate(
        TensorLayout::contiguous(real_video_shape, 2));
    DeviceTensor real_audio_out = context.allocate(
        TensorLayout::contiguous(real_audio_shape, 2));
    DeviceTensor real_transformer_selectors = context.allocate(
        TensorLayout::contiguous(real_selector_shape, 1), ScalarType::kInt32);
    DeviceTensor real_transformer_code = context.allocate(
        TensorLayout::contiguous(real_code_shape, 2));
    DeviceTensor real_transformer_cosine = context.allocate(
        TensorLayout::contiguous(real_rope_shape, 2));
    DeviceTensor real_transformer_sine = context.allocate(
        TensorLayout::contiguous(real_rope_shape, 2));
    DeviceTensor real_video_ts = context.allocate(
        TensorLayout::contiguous(real_video_index_shape, 1), ScalarType::kInt32);
    DeviceTensor real_audio_ts = context.allocate(
        TensorLayout::contiguous(real_audio_index_shape, 1), ScalarType::kInt32);
    std::vector<float> real_prompt_values(3 * 5120);
    std::vector<float> real_video_values(60 * 96);
    std::vector<float> real_audio_values(2 * 32);
    for (size_t i = 0; i < real_prompt_values.size(); ++i)
      real_prompt_values[i] = float(int(i % 251) - 125) / 128.0f;
    for (size_t i = 0; i < real_video_values.size(); ++i)
      real_video_values[i] = float(int(i % 127) - 63) / 64.0f;
    for (size_t i = 0; i < real_audio_values.size(); ++i)
      real_audio_values[i] = float(int(i % 61) - 30) / 32.0f;
    std::vector<int32_t> real_video_ts_values(60, 0),
        real_audio_ts_values(2, 0);
    context.upload(real_prompt, real_prompt_values.data(), real_prompt_values.size());
    context.upload(real_video, real_video_values.data(), real_video_values.size());
    context.upload(real_audio, real_audio_values.data(), real_audio_values.size());
    context.upload_bytes(real_transformer_selectors, real_selector_values.data(),
                         real_selector_values.size() * 4);
    context.upload(real_transformer_code, real_code_values.data(),
                   real_code_values.size());
    context.upload(real_transformer_cosine, real_cosine_values.data(),
                   real_cosine_values.size());
    context.upload(real_transformer_sine, real_sine_values.data(),
                   real_sine_values.size());
    context.upload_bytes(real_video_ts, real_video_ts_values.data(),
                         real_video_ts_values.size() * 4);
    context.upload_bytes(real_audio_ts, real_audio_ts_values.data(),
                         real_audio_ts_values.size() * 4);
    const uint64_t real_transformer_baseline = context.pooled_used_bytes();
    ExactH3Transformer real_transformer = ExactH3Transformer::create(
        context, real_transformer_config);
    real_transformer.load(real_checkpoint);
    real_transformer.prepare_text(real_prompt);
    auto run_real_transformer = [&] {
      TensorBatch batch = context.begin_batch();
      real_transformer.record_forward(
          batch, real_video, real_audio, real_transformer_selectors,
          real_transformer_code, real_transformer_cosine,
          real_transformer_sine, real_video_ts, real_audio_ts,
          real_video_out, real_audio_out);
      batch.submit().wait();
      std::vector<float> result(real_video_values.size() + real_audio_values.size());
      context.download(real_video_out, result.data(), real_video_values.size());
      context.download(real_audio_out, result.data() + real_video_values.size(),
                       real_audio_values.size());
      return result;
    };
    const std::vector<float> real_transformer_output = run_real_transformer();
    const uint64_t real_transformer_digest =
        fnv64_floats(real_transformer_output);
    CHECK(real_transformer_digest == 0x42764ebbb3850be4ull);
    const uint64_t real_transformer_reserved = context.reserved_bytes();
    const uint64_t real_transformer_descriptors =
        context.descriptor_set_allocations();
    CHECK(run_real_transformer() == real_transformer_output);
    CHECK(context.reserved_bytes() == real_transformer_reserved);
    CHECK(context.descriptor_set_allocations() == real_transformer_descriptors);
    real_transformer.unload();
    CHECK(context.pooled_used_bytes() == real_transformer_baseline);
    CHECK(real_transformer.persistent_bytes() == 0u &&
          real_transformer.scratch_bytes() == 0u);
    std::printf("  CUDA-off real H3 transformer S65 FNV64 %016llx\n",
                static_cast<unsigned long long>(real_transformer_digest));
    }
  }
  std::error_code ignored;
  std::filesystem::remove(valid_path, ignored);
  std::filesystem::remove(corrupt_path, ignored);
  std::filesystem::remove(corrupt_fc2_path, ignored);
  std::filesystem::remove(graph_path, ignored);
  std::filesystem::remove(corrupt_graph_path, ignored);
  std::filesystem::remove(transformer_path, ignored);
  std::filesystem::remove(corrupt_transformer_path, ignored);
  std::filesystem::remove(corrupt_transformer_dtype_path, ignored);
  for (const auto& path : corrupt_transformer_metadata_paths)
    std::filesystem::remove(path, ignored);
  std::filesystem::remove(base, ignored);
}
