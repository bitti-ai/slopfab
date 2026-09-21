#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_dit_real_block0_replay, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const std::filesystem::path path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  if (!std::filesystem::exists(path) || !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(path) || !Instance::available()");
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
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 64;
  TensorContext vk(device, context_options);
  if (!vk.exact_h3_attention() || !vk.exact_vae_pointwise() || !vk.exact_fp32_vae_normalization()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !vk.exact_h3_attention() || !vk.exact_vae_pointwise() || !vk.exact_fp32_vae_normalization()");
    return;
  }

  uint32_t sequence = 65;
  if (const char* requested = std::getenv("SLOPFAB_DIT_BLOCK_SEQUENCE")) {
    const unsigned long parsed = std::strtoul(requested, nullptr, 10);
    if (parsed == 0 || parsed > UINT32_MAX)
      throw std::invalid_argument("SLOPFAB_DIT_BLOCK_SEQUENCE is invalid");
    sequence = static_cast<uint32_t>(parsed);
  }
  H3BlockConfig config;
  config.sequence = sequence;
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  const auto load_begin = std::chrono::steady_clock::now();
  ExactH3BlockStage stage = ExactH3BlockStage::create(vk, config);
  stage.load(checkpoint, 0);
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_begin)
          .count();
  ExactH3BlockScratch scratch = ExactH3BlockScratch::create(vk, config);

  std::vector<uint16_t> token_bits(size_t(sequence) * config.hidden);
  for (size_t i = 0; i < token_bits.size(); ++i)
    token_bits[i] = f32_to_bf16(float(int(i % 61) - 30) / 64.0f);
  std::vector<int32_t> selectors(sequence);
  for (uint32_t i = 0; i < sequence; ++i)
    selectors[i] = static_cast<int32_t>(i % config.modalities);
  std::vector<float> code(size_t(config.timesteps) * config.adaln_rank);
  for (size_t i = 0; i < code.size(); ++i)
    code[i] = float(int(i % 7) - 3) / 16.0f;
  std::vector<float> cosine(size_t(sequence) * 96, 1.0f);
  std::vector<float> sine(size_t(sequence) * 96, 0.0f);
  const uint64_t token_shape[] = {sequence, config.hidden};
  const uint64_t selector_shape[] = {sequence};
  const uint64_t code_shape[] = {config.timesteps, config.adaln_rank};
  const uint64_t rope_shape[] = {sequence, 96};
  DeviceTensor tokens =
      vk.allocate(TensorLayout::contiguous(token_shape, 2), ScalarType::kBFloat16);
  DeviceTensor selector_tensor =
      vk.allocate(TensorLayout::contiguous(selector_shape, 1), ScalarType::kInt32);
  DeviceTensor code_tensor = vk.allocate(TensorLayout::contiguous(code_shape, 2));
  DeviceTensor cosine_tensor = vk.allocate(TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor sine_tensor = vk.allocate(TensorLayout::contiguous(rope_shape, 2));
  vk.upload_bytes(tokens, token_bits.data(), token_bits.size() * 2);
  vk.upload_bytes(selector_tensor, selectors.data(), selectors.size() * 4);
  vk.upload(code_tensor, code.data(), code.size());
  vk.upload(cosine_tensor, cosine.data(), cosine.size());
  vk.upload(sine_tensor, sine.data(), sine.size());

  auto run = [&] {
    const auto begin = std::chrono::steady_clock::now();
    TensorBatch batch = vk.begin_batch();
    stage.record(batch, tokens, selector_tensor, code_tensor, cosine_tensor, sine_tensor, scratch);
    batch.submit().wait();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
        .count();
  };
  const double first_ms = run();
  std::vector<uint16_t> first(token_bits.size()), repeated(token_bits.size());
  vk.download_bytes(tokens, first.data(), first.size() * 2);
  vk.upload_bytes(tokens, token_bits.data(), token_bits.size() * 2);
  const uint64_t descriptors = vk.descriptor_set_allocations();
  const uint64_t reserved = vk.reserved_bytes();
  const double repeat_ms = run();
  vk.download_bytes(tokens, repeated.data(), repeated.size() * 2);
  CHECK(first == repeated);
  CHECK(vk.descriptor_set_allocations() == descriptors);
  CHECK(vk.reserved_bytes() == reserved);

  if (!cuda::deterministic_h3_attention_available()) {
    SKIP_UNSUPPORTED_HARDWARE("exact CUDA H3 attention is unavailable");
    return;
  }
  const uint32_t hidden = config.hidden;
  const uint32_t inner = config.heads * config.head_dim;
  const uint32_t ffn = config.ffn;
  const uint32_t modulation_rows = config.timesteps * config.modalities;
  auto bf16_host = [&](const std::string& name, uint32_t count) {
    std::vector<float> wide = to_f32(checkpoint.at(name));
    CHECK(wide.size() == count);
    std::vector<uint16_t> bits(count);
    for (uint32_t i = 0; i < count; ++i)
      bits[i] = f32_to_bf16(wide[i]);
    return bits;
  };
  const std::vector<uint16_t> norm1 = bf16_host("blocks.0.norm1.weight", hidden);
  const std::vector<uint16_t> norm2 = bf16_host("blocks.0.norm2.weight", hidden);
  const std::vector<uint16_t> q_norm = bf16_host("blocks.0.attn.q_norm.weight", config.head_dim);
  const std::vector<uint16_t> k_norm = bf16_host("blocks.0.attn.k_norm.weight", config.head_dim);
  std::vector<float> adaln_w = to_f32(checkpoint.at("blocks.0.adaln_proj.linear.weight"));
  std::vector<float> adaln_b = to_f32(checkpoint.at("blocks.0.adaln_proj.linear.bias"));
  const size_t largest_weight = std::max({size_t(inner) * hidden, size_t(hidden) * inner,
                                          size_t(2) * ffn * hidden, size_t(hidden) * ffn});
  const uint32_t largest_input = std::max({hidden, inner, ffn});
  cuda::DeviceBuffer<uint8_t> cuda_stored(largest_weight / 2), cuda_scale(largest_weight / 16);
  cuda::DeviceBuffer<uint16_t> cuda_dense(largest_weight), cuda_pre(largest_input),
      cuda_transform(size_t(sequence) * largest_input);
  cuda::DeviceBuffer<uint16_t> cuda_tokens(token_bits.size()),
      cuda_normed(size_t(sequence) * hidden), cuda_q(size_t(sequence) * inner),
      cuda_k(size_t(sequence) * inner), cuda_v(size_t(sequence) * inner),
      cuda_attention(size_t(sequence) * inner), cuda_branch(size_t(sequence) * hidden),
      cuda_fused(size_t(sequence) * 2 * ffn), cuda_activation(size_t(sequence) * ffn),
      cuda_norm1(hidden), cuda_norm2(hidden), cuda_q_norm(config.head_dim),
      cuda_k_norm(config.head_dim);
  cuda::DeviceBuffer<float> cuda_adaln_w(adaln_w.size()), cuda_adaln_b(adaln_b.size()),
      cuda_code(code.size()), cuda_modulation(size_t(6) * modulation_rows * hidden),
      cuda_cosine(cosine.size()), cuda_sine(sine.size());
  cuda::DeviceBuffer<int32_t> cuda_selectors(selectors.size());
  cuda_tokens.copy_from_host(token_bits.data(), token_bits.size());
  cuda_norm1.copy_from_host(norm1.data(), norm1.size());
  cuda_norm2.copy_from_host(norm2.data(), norm2.size());
  cuda_q_norm.copy_from_host(q_norm.data(), q_norm.size());
  cuda_k_norm.copy_from_host(k_norm.data(), k_norm.size());
  cuda_adaln_w.copy_from_host(adaln_w.data(), adaln_w.size());
  cuda_adaln_b.copy_from_host(adaln_b.data(), adaln_b.size());
  cuda_code.copy_from_host(code.data(), code.size());
  cuda_cosine.copy_from_host(cosine.data(), cosine.size());
  cuda_sine.copy_from_host(sine.data(), sine.size());
  cuda_selectors.copy_from_host(selectors.data(), selectors.size());

  auto cuda_projection = [&](const std::string& name, uint32_t out, uint32_t in,
                             uint32_t source_out, uint32_t row_offset, const uint16_t* input,
                             uint16_t* output) {
    const TensorView& weight = checkpoint.at(name + ".weight");
    const TensorView& scale = checkpoint.at(name + ".weight_scale");
    CHECK(weight.dtype == DType::kU8 && scale.dtype == DType::kF8E4M3);
    CHECK(weight.shape == std::vector<int64_t>({source_out, in / 2}));
    CHECK(scale.shape == std::vector<int64_t>({source_out, in / 16}));
    const size_t elements = size_t(out) * in;
    const size_t element_offset = size_t(row_offset) * in;
    cuda_stored.copy_from_host(static_cast<const uint8_t*>(weight.data) + element_offset / 2,
                               elements / 2);
    cuda_scale.copy_from_host(static_cast<const uint8_t*>(scale.data) + element_offset / 16,
                              elements / 16);
    const std::vector<float> global_values = to_f32(checkpoint.at(name + ".weight_scale_2"));
    CHECK(global_values.size() == 1);
    const uint16_t* source = input;
    if (const TensorView* pre = checkpoint.find(name + ".pre_quant_scale")) {
      std::vector<float> wide = to_f32(*pre);
      CHECK(wide.size() == in);
      std::vector<uint16_t> bits(in);
      for (uint32_t i = 0; i < in; ++i)
        bits[i] = f32_to_bf16(wide[i]);
      cuda_pre.copy_from_host(bits.data(), bits.size());
      cuda::launch_pre_quant_scale(reinterpret_cast<const __nv_bfloat16*>(input),
                                   reinterpret_cast<const __nv_bfloat16*>(cuda_pre.get()),
                                   reinterpret_cast<__nv_bfloat16*>(cuda_transform.get()), sequence,
                                   in, nullptr);
      source = cuda_transform.get();
    }
    cuda::launch_dequant_nvfp4(cuda_stored.get(), cuda_scale.get(), global_values.front(),
                               reinterpret_cast<__nv_bfloat16*>(cuda_dense.get()), out, in,
                               nullptr);
    const uint32_t tiled_rows = sequence / 64 * 64;
    if (tiled_rows != 0) {
      cuda::launch_deterministic_bf16_gemm_nt(
          reinterpret_cast<const __nv_bfloat16*>(source),
          reinterpret_cast<const __nv_bfloat16*>(cuda_dense.get()), nullptr,
          reinterpret_cast<__nv_bfloat16*>(output), tiled_rows, out, in, DenseGemmBias::kNone);
    }
    if (tiled_rows != sequence) {
      cuda::launch_deterministic_scalar_gemm_nt(
          source, cuda_dense.get(), nullptr, output, sequence - tiled_rows, out, in,
          DenseGemmMode::kBFloat16, DenseGemmBias::kNone, tiled_rows, tiled_rows);
    }
  };
  const auto cuda_begin = std::chrono::steady_clock::now();
  cuda::launch_adaln_expand(cuda_adaln_w.get(), cuda_adaln_b.get(), cuda_code.get(),
                            cuda_modulation.get(), config.timesteps, config.modalities, 6, hidden,
                            config.adaln_rank, nullptr);
  const size_t table = size_t(modulation_rows) * hidden;
  cuda::launch_rmsnorm_modulate(reinterpret_cast<const __nv_bfloat16*>(cuda_tokens.get()),
                                reinterpret_cast<const __nv_bfloat16*>(cuda_norm1.get()),
                                cuda_modulation.get() + table, cuda_modulation.get(),
                                cuda_selectors.get(),
                                reinterpret_cast<__nv_bfloat16*>(cuda_normed.get()), sequence,
                                hidden, config.epsilon, nullptr);
  const std::string qkv = "blocks.0.attn.qkv_proj";
  cuda_projection(qkv, inner, hidden, 3 * inner, 0, cuda_normed.get(), cuda_q.get());
  cuda_projection(qkv, inner, hidden, 3 * inner, inner, cuda_normed.get(), cuda_k.get());
  cuda_projection(qkv, inner, hidden, 3 * inner, 2 * inner, cuda_normed.get(), cuda_v.get());
  cuda::launch_head_rmsnorm(reinterpret_cast<__nv_bfloat16*>(cuda_q.get()),
                            reinterpret_cast<const __nv_bfloat16*>(cuda_q_norm.get()), sequence,
                            config.heads, config.head_dim, config.epsilon, nullptr);
  cuda::launch_head_rmsnorm(reinterpret_cast<__nv_bfloat16*>(cuda_k.get()),
                            reinterpret_cast<const __nv_bfloat16*>(cuda_k_norm.get()), sequence,
                            config.heads, config.head_dim, config.epsilon, nullptr);
  cuda::launch_rope_h3(reinterpret_cast<__nv_bfloat16*>(cuda_q.get()), cuda_cosine.get(),
                       cuda_sine.get(), sequence, config.heads, config.head_dim, nullptr);
  cuda::launch_rope_h3(reinterpret_cast<__nv_bfloat16*>(cuda_k.get()), cuda_cosine.get(),
                       cuda_sine.get(), sequence, config.heads, config.head_dim, nullptr);
  cuda::launch_deterministic_h3_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cuda_q.get()),
      reinterpret_cast<const __nv_bfloat16*>(cuda_k.get()),
      reinterpret_cast<const __nv_bfloat16*>(cuda_v.get()),
      reinterpret_cast<__nv_bfloat16*>(cuda_attention.get()), nullptr, sequence, config.heads,
      config.head_dim, exact_attention_scale(config.head_dim));
  cuda_projection("blocks.0.attn.out_proj", hidden, inner, hidden, 0, cuda_attention.get(),
                  cuda_branch.get());
  cuda::launch_add_gated(reinterpret_cast<__nv_bfloat16*>(cuda_tokens.get()),
                         reinterpret_cast<const __nv_bfloat16*>(cuda_branch.get()),
                         cuda_modulation.get() + 2 * table, cuda_selectors.get(), sequence, hidden,
                         nullptr);
  cuda::launch_rmsnorm_modulate(reinterpret_cast<const __nv_bfloat16*>(cuda_tokens.get()),
                                reinterpret_cast<const __nv_bfloat16*>(cuda_norm2.get()),
                                cuda_modulation.get() + 4 * table,
                                cuda_modulation.get() + 3 * table, cuda_selectors.get(),
                                reinterpret_cast<__nv_bfloat16*>(cuda_normed.get()), sequence,
                                hidden, config.epsilon, nullptr);
  cuda_projection("blocks.0.mlp.fc1", 2 * ffn, hidden, 2 * ffn, 0, cuda_normed.get(),
                  cuda_fused.get());
  cuda::launch_swiglu_exact(reinterpret_cast<const __nv_bfloat16*>(cuda_fused.get()),
                            reinterpret_cast<__nv_bfloat16*>(cuda_activation.get()), sequence, ffn,
                            nullptr);
  cuda_projection("blocks.0.mlp.fc2", hidden, ffn, hidden, 0, cuda_activation.get(),
                  cuda_branch.get());
  cuda::launch_add_gated(reinterpret_cast<__nv_bfloat16*>(cuda_tokens.get()),
                         reinterpret_cast<const __nv_bfloat16*>(cuda_branch.get()),
                         cuda_modulation.get() + 5 * table, cuda_selectors.get(), sequence, hidden,
                         nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_begin)
          .count();
  std::vector<uint16_t> cuda_output(token_bits.size());
  cuda_tokens.copy_to_host(cuda_output.data(), cuda_output.size());
  size_t mismatch = first.size();
  for (size_t i = 0; i < first.size(); ++i) {
    if (first[i] != cuda_output[i]) {
      mismatch = i;
      break;
    }
  }
  CHECK_MSG(mismatch == first.size(), "real H3 block0 CUDA/Vulkan mismatch at %zu: %04x != %04x",
            mismatch, mismatch == first.size() ? 0 : cuda_output[mismatch],
            mismatch == first.size() ? 0 : first[mismatch]);
  uint64_t digest = 1469598103934665603ull;
  for (uint16_t bits : first) {
    digest ^= bits & 0xffu;
    digest *= 1099511628211ull;
    digest ^= bits >> 8;
    digest *= 1099511628211ull;
  }
  if (sequence == 65)
    CHECK(digest == 0x191929c14480e873ull);
  const uint64_t persistent = stage.persistent_bytes();
  bool failed_reload = false;
  try {
    stage.load(checkpoint, 50);
  } catch (const std::exception&) {
    failed_reload = true;
  }
  CHECK(failed_reload && stage.loaded());
  CHECK(stage.persistent_bytes() == persistent);
  vk.upload_bytes(tokens, token_bits.data(), token_bits.size() * 2);
  (void)run();
  std::vector<uint16_t> after_failure(first.size());
  vk.download_bytes(tokens, after_failure.data(), after_failure.size() * 2);
  CHECK(after_failure == first);
  stage.unload();
  CHECK(!stage.loaded() && stage.persistent_bytes() == 0);
  stage.load(checkpoint, 0);
  CHECK(stage.loaded() && stage.persistent_bytes() == persistent);
  vk.upload_bytes(tokens, token_bits.data(), token_bits.size() * 2);
  (void)run();
  std::vector<uint16_t> after_reload(first.size());
  vk.download_bytes(tokens, after_reload.data(), after_reload.size() * 2);
  CHECK(after_reload == first);
  std::printf(
      "  real H3 block0 S%u: load %.3f ms, CUDA %.3f ms, Vulkan first/repeat %.3f/%.3f ms, FNV64 %016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, pool used/reserved %.2f/%.2f MiB, descriptors %llu\n",
      sequence, load_ms, cuda_ms, first_ms, repeat_ms, static_cast<unsigned long long>(digest),
      double(stage.persistent_bytes()) / 1048576.0, double(scratch.reserved_bytes()) / 1048576.0,
      double(stage.peak_device_bytes(scratch)) / 1048576.0,
      double(vk.pooled_used_bytes()) / 1048576.0, double(vk.reserved_bytes()) / 1048576.0,
      static_cast<unsigned long long>(vk.descriptor_set_allocations()));

  if (const char* production = std::getenv("SLOPFAB_DIT_GRAPH_PRODUCTION");
      production && production[0] == '1') {
    TensorContextOptions prod_context_options;
    prod_context_options.max_batch_operators = 2048;
    TensorContext prod_vk(device, prod_context_options);
    H3MainGraphConfig prod_config;
    prod_config.layers = 50;
    prod_config.block.sequence = 9864;
    ExactH3MainGraph prod = ExactH3MainGraph::create(prod_vk, prod_config);
    const auto prod_load_begin = std::chrono::steady_clock::now();
    prod.load(checkpoint);
    const double prod_load_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - prod_load_begin)
                                    .count();
    const uint64_t prod_token_shape[] = {prod_config.block.sequence, prod_config.block.hidden};
    const uint64_t prod_selector_shape[] = {prod_config.block.sequence};
    const uint64_t prod_code_shape[] = {prod_config.block.timesteps, prod_config.block.adaln_rank};
    const uint64_t prod_rope_shape[] = {prod_config.block.sequence, 96};
    DeviceTensor prod_tokens =
        prod_vk.allocate(TensorLayout::contiguous(prod_token_shape, 2), ScalarType::kBFloat16);
    DeviceTensor prod_selectors =
        prod_vk.allocate(TensorLayout::contiguous(prod_selector_shape, 1), ScalarType::kInt32);
    DeviceTensor prod_code = prod_vk.allocate(TensorLayout::contiguous(prod_code_shape, 2));
    DeviceTensor prod_cosine = prod_vk.allocate(TensorLayout::contiguous(prod_rope_shape, 2));
    DeviceTensor prod_sine = prod_vk.allocate(TensorLayout::contiguous(prod_rope_shape, 2));
    std::vector<uint16_t> prod_input(size_t(prod_config.block.sequence) * prod_config.block.hidden);
    for (size_t i = 0; i < prod_input.size(); ++i)
      prod_input[i] = f32_to_bf16(float(int(i % 61) - 30) / 64.0f);
    std::vector<int32_t> prod_selector_values(prod_config.block.sequence);
    for (uint32_t i = 0; i < prod_config.block.sequence; ++i)
      prod_selector_values[i] = static_cast<int32_t>(i % prod_config.block.modalities);
    std::vector<float> prod_code_values(size_t(prod_config.block.timesteps) *
                                        prod_config.block.adaln_rank);
    for (size_t i = 0; i < prod_code_values.size(); ++i)
      prod_code_values[i] = float(int(i % 7) - 3) / 16.0f;
    std::vector<float> prod_cosine_values(size_t(prod_config.block.sequence) * 96, 1.0f);
    std::vector<float> prod_sine_values(prod_cosine_values.size(), 0.0f);
    prod_vk.upload_bytes(prod_selectors, prod_selector_values.data(),
                         prod_selector_values.size() * 4);
    prod_vk.upload(prod_code, prod_code_values.data(), prod_code_values.size());
    prod_vk.upload(prod_cosine, prod_cosine_values.data(), prod_cosine_values.size());
    prod_vk.upload(prod_sine, prod_sine_values.data(), prod_sine_values.size());
    auto run_prod = [&] {
      prod_vk.upload_bytes(prod_tokens, prod_input.data(), prod_input.size() * 2);
      const auto begin = std::chrono::steady_clock::now();
      TensorBatch prod_batch = prod_vk.begin_batch();
      prod.record(prod_batch, prod_tokens, prod_selectors, prod_code, prod_cosine, prod_sine);
      CHECK(prod_batch.remaining_operator_capacity() == 598u);
      prod_batch.submit().wait();
      const double elapsed =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
              .count();
      std::vector<uint16_t> output(prod_input.size());
      prod_vk.download_bytes(prod_tokens, output.data(), output.size() * 2);
      return std::pair<std::vector<uint16_t>, double>(std::move(output), elapsed);
    };
    auto prod_first = run_prod();
    uint64_t prod_digest = 1469598103934665603ull;
    for (uint16_t bits : prod_first.first) {
      prod_digest ^= bits & 0xffu;
      prod_digest *= 1099511628211ull;
      prod_digest ^= bits >> 8;
      prod_digest *= 1099511628211ull;
    }
    // This durable pin follows the separately asserted CUDA/Vulkan block-0
    // S9864 boundary `51e414a3b2556e88`; the full CUDA 50-block capture at
    // S526 remains the cross-backend authority for every graph boundary.
    CHECK(prod_digest == 0x3d59d01afa11ba77ull);
    const uint64_t prod_used = prod_vk.pooled_used_bytes();
    const uint64_t prod_reserved = prod_vk.reserved_bytes();
    const uint64_t prod_descriptors = prod_vk.descriptor_set_allocations();
    auto prod_repeat = run_prod();
    CHECK(prod_repeat.first == prod_first.first);
    CHECK(prod_vk.pooled_used_bytes() == prod_used);
    CHECK(prod_vk.reserved_bytes() == prod_reserved);
    CHECK(prod_vk.descriptor_set_allocations() == prod_descriptors);
    std::printf(
        "  production H3 main50 S9864: load %.3f ms, first/repeat %.3f/%.3f ms, final %016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, pool %.2f/%.2f MiB, descriptors %llu\n",
        prod_load_ms, prod_first.second, prod_repeat.second,
        static_cast<unsigned long long>(prod_digest), double(prod.persistent_bytes()) / 1048576.0,
        double(prod.scratch_bytes()) / 1048576.0, double(prod.peak_device_bytes()) / 1048576.0,
        double(prod_vk.pooled_used_bytes()) / 1048576.0,
        double(prod_vk.reserved_bytes()) / 1048576.0,
        static_cast<unsigned long long>(prod_vk.descriptor_set_allocations()));
  }
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_dit_real_capture_replay, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::dit;
  using namespace slopfab::vulkan;
  const std::filesystem::path checkpoint_path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  const std::filesystem::path capture_path = "tests/data/h3_block0_step0_seed424242_256.vfh3";
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
  const std::array<uint8_t, 32> capture_sha = sha256_mapping(capture.data(), capture.size());
  const std::array<uint8_t, 32> expected_capture_sha{
      0xe0, 0x9e, 0x29, 0x7b, 0x8d, 0xc7, 0x30, 0x06, 0xf6, 0x75, 0x7b,
      0x0a, 0x1a, 0x5b, 0xf0, 0xa6, 0x05, 0x53, 0xf0, 0x05, 0x92, 0xea,
      0xf2, 0xa2, 0xf0, 0x48, 0x4d, 0x2e, 0x11, 0x03, 0x82, 0x75};
  CHECK(capture_sha == expected_capture_sha);
#endif
  if (capture.size() < sizeof(H3BlockCaptureHeader))
    throw std::runtime_error("truncated H3 block capture");
  H3BlockCaptureHeader header{};
  std::memcpy(&header, capture.data(), sizeof(header));
  CHECK(std::memcmp(header.magic, "VFH3BLK\0", 8) == 0);
  CHECK(header.version == 1 && header.header_bytes == sizeof(header));
  CHECK(header.sequence == 538 && header.hidden == 5376 && header.heads == 56 &&
        header.head_dim == 128 && header.ffn == 14336);
  CHECK(header.timesteps == 1 && header.modalities == 3 && header.adaln_rank == 8 &&
        header.denoise_step == 0 && header.layer == 0 && header.range_values == 20);
  CHECK(header.input_fnv64 == 0x7c9f5a55cc5266ebull);
  CHECK(header.qkv_fnv64 == 0x0ce5a1f4d191bdd1ull);
  CHECK(header.attention_fnv64 == 0x550f1253844cd657ull);
  CHECK(header.attention_residual_fnv64 == 0xe8a9ee4dfec51636ull);
  CHECK(header.final_fnv64 == 0x2fd91fe15c281f00ull);
  size_t cursor = sizeof(header);
  auto take = [&](auto& values, size_t count) {
    using Value = typename std::decay_t<decltype(values)>::value_type;
    if (count > (capture.size() - cursor) / sizeof(Value))
      throw std::runtime_error("truncated H3 block capture payload");
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
  CHECK(std::any_of(cosine.begin(), cosine.end(), [](float x) {
    return x != 1.0f;
  }));
  CHECK(std::any_of(sine.begin(), sine.end(), [](float x) {
    return x != 0.0f;
  }));
  CHECK(std::all_of(selectors.begin(), selectors.end(), [](int32_t x) {
    return x >= 0 && x < 3;
  }));

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
#ifdef _WIN32
  const std::array<uint8_t, 32> checkpoint_sha =
      sha256_mapping(checkpoint.mapping_base(), checkpoint.file_size());
  const std::array<uint8_t, 32> expected_checkpoint_sha{
      0x6a, 0xb7, 0xf0, 0xc4, 0x81, 0x41, 0xe7, 0x91, 0x9b, 0x32, 0xf9,
      0x25, 0xca, 0x3d, 0xef, 0x22, 0xe0, 0x6a, 0x6a, 0xeb, 0xeb, 0x9e,
      0x0b, 0x6f, 0x5a, 0x0b, 0xe0, 0xfe, 0x84, 0x09, 0x97, 0x6f};
  CHECK(checkpoint_sha == expected_checkpoint_sha);
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
  context_options.max_batch_operators = 64;
  TensorContext vk(device, context_options);
  if (!vk.exact_h3_attention()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_h3_attention()");
    return;
  }
  H3BlockConfig config;
  config.sequence = header.sequence;
  config.hidden = header.hidden;
  config.heads = header.heads;
  config.head_dim = header.head_dim;
  config.ffn = header.ffn;
  config.timesteps = header.timesteps;
  config.modalities = header.modalities;
  config.adaln_rank = header.adaln_rank;
  ExactH3BlockStage stage = ExactH3BlockStage::create(vk, config);
  stage.load(checkpoint, header.layer);
  ExactH3BlockScratch scratch = ExactH3BlockScratch::create(vk, config);
  const uint64_t residual_shape[] = {header.sequence, header.hidden};
  const uint64_t selector_shape[] = {header.sequence};
  const uint64_t code_shape[] = {header.timesteps, header.adaln_rank};
  const uint64_t rope_shape[] = {header.sequence, 96};
  const uint64_t qkv_shape[] = {header.sequence, header.heads, header.head_dim};
  const uint64_t attention_shape[] = {header.sequence, header.heads * header.head_dim};
  auto bf = [&](const uint64_t* shape, uint32_t rank) {
    return vk.allocate(TensorLayout::contiguous(shape, rank), ScalarType::kBFloat16);
  };
  DeviceTensor tokens = bf(residual_shape, 2);
  DeviceTensor selector_tensor =
      vk.allocate(TensorLayout::contiguous(selector_shape, 1), ScalarType::kInt32);
  DeviceTensor code_tensor = vk.allocate(TensorLayout::contiguous(code_shape, 2));
  DeviceTensor cosine_tensor = vk.allocate(TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor sine_tensor = vk.allocate(TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor q_tap = bf(qkv_shape, 3), k_tap = bf(qkv_shape, 3), v_tap = bf(qkv_shape, 3),
               attention_tap = bf(attention_shape, 2),
               attention_residual_tap = bf(residual_shape, 2), final_tap = bf(residual_shape, 2);
  vk.upload_bytes(tokens, input.data(), input.size() * 2);
  vk.upload_bytes(selector_tensor, selectors.data(), selectors.size() * 4);
  vk.upload(code_tensor, code.data(), code.size());
  vk.upload(cosine_tensor, cosine.data(), cosine.size());
  vk.upload(sine_tensor, sine.data(), sine.size());
  H3AttentionRanges ranges =
      H3AttentionRanges::create(vk, header.sequence, range_values.data(), header.range_values);
  H3BlockReplayTaps taps{&q_tap,    &k_tap, &v_tap, &attention_tap, &attention_residual_tap,
                         &final_tap};
  CHECK(stage.required_operators(&taps) == 35);
  // Fill exactly the spare capacity. A one-operation larger prefix below is
  // rejected before the stage changes tokens or the batch's access state.
  DeviceTensor dummy_a = bf(residual_shape, 2), dummy_b = bf(residual_shape, 2);
  TensorBatch exact = vk.begin_batch();
  for (uint32_t i = stage.required_operators(&taps); i < 64; ++i)
    exact.copy(dummy_a, dummy_b);
  stage.record(exact, tokens, selector_tensor, code_tensor, cosine_tensor, sine_tensor, scratch,
               &ranges, &taps);
  CHECK(exact.remaining_operator_capacity() == 0);
  exact.submit().wait();
  auto fnv = [](const std::vector<uint16_t>& values, uint64_t hash = 1469598103934665603ull) {
    for (uint16_t bits : values) {
      hash ^= bits & 0xffu;
      hash *= 1099511628211ull;
      hash ^= bits >> 8;
      hash *= 1099511628211ull;
    }
    return hash;
  };
  std::vector<uint16_t> q(header.qkv_elements), k(header.qkv_elements), v(header.qkv_elements),
      attention(header.qkv_elements), attention_residual(header.residual_elements),
      final(header.residual_elements), tokens_final(header.residual_elements);
  vk.download_bytes(q_tap, q.data(), q.size() * 2);
  vk.download_bytes(k_tap, k.data(), k.size() * 2);
  vk.download_bytes(v_tap, v.data(), v.size() * 2);
  vk.download_bytes(attention_tap, attention.data(), attention.size() * 2);
  vk.download_bytes(attention_residual_tap, attention_residual.data(),
                    attention_residual.size() * 2);
  vk.download_bytes(final_tap, final.data(), final.size() * 2);
  vk.download_bytes(tokens, tokens_final.data(), tokens_final.size() * 2);
  uint64_t qkv_hash = fnv(q);
  qkv_hash = fnv(k, qkv_hash);
  qkv_hash = fnv(v, qkv_hash);
  CHECK(qkv_hash == header.qkv_fnv64);
  CHECK(fnv(attention) == header.attention_fnv64);
  CHECK(fnv(attention_residual) == header.attention_residual_fnv64);
  CHECK(fnv(final) == header.final_fnv64);
  CHECK(final == expected_final && tokens_final == expected_final);

  // Late-shape and range faults are transactional and consume no operator.
  const uint64_t bad_rope_shape[] = {header.sequence, 95};
  DeviceTensor bad_sine = vk.allocate(TensorLayout::contiguous(bad_rope_shape, 2));
  std::vector<int32_t> wrong_values(4 * ((header.sequence + 127) / 128), 0);
  for (size_t i = 0; i < wrong_values.size(); i += 4)
    wrong_values[i + 1] = 576;
  H3AttentionRanges wrong_ranges = H3AttentionRanges::create(
      vk, header.sequence + 1, wrong_values.data(), static_cast<uint32_t>(wrong_values.size()));
  {
    TensorBatch invalid = vk.begin_batch();
    bool bad_rope_threw = false;
    try {
      stage.record(invalid, tokens, selector_tensor, code_tensor, cosine_tensor, bad_sine, scratch,
                   &ranges);
    } catch (const std::invalid_argument&) {
      bad_rope_threw = true;
    }
    CHECK(bad_rope_threw && invalid.remaining_operator_capacity() == 64);
    bool bad_ranges_threw = false;
    try {
      stage.record(invalid, tokens, selector_tensor, code_tensor, cosine_tensor, sine_tensor,
                   scratch, &wrong_ranges);
    } catch (const std::invalid_argument&) {
      bad_ranges_threw = true;
    }
    CHECK(bad_ranges_threw && invalid.remaining_operator_capacity() == 64);
  }
  {
    TensorBatch short_capacity = vk.begin_batch();
    for (uint32_t i = stage.required_operators(); i <= 64; ++i)
      short_capacity.copy(dummy_a, dummy_b);
    bool capacity_threw = false;
    try {
      stage.record(short_capacity, tokens, selector_tensor, code_tensor, cosine_tensor, sine_tensor,
                   scratch, &ranges);
    } catch (const std::logic_error&) {
      capacity_threw = true;
    }
    CHECK(capacity_threw &&
          short_capacity.remaining_operator_capacity() == stage.required_operators() - 1);
  }

  bool overflow_threw = false;
  try {
    H3BlockConfig overflow = config;
    overflow.heads = UINT32_MAX;
    (void)ExactH3BlockStage::create(vk, overflow);
  } catch (const std::invalid_argument&) {
    overflow_threw = true;
  }
  CHECK(overflow_threw);
}

SLOPFAB_TEST_CATEGORY(cuda_ref2va_text_l4100_authority, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::dit;
  const char* enabled = std::getenv("SLOPFAB_REF2VA_TEXT_CUDA_REAL");
  if (!enabled || enabled[0] != '1') {
    SKIP_OPT_IN("unavailable prerequisite: !enabled || enabled[0] != '1'");
    return;
  }
  const std::filesystem::path checkpoint_path =
      "weights/transformer/minimax_h3_ref2va_pruned_nvfp4.safetensors";
  const std::filesystem::path source_capture =
      "tests/data/h3_transformer_step0_seed424242_256.vfh3f";
  if (!std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(source_capture)) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(source_capture)");
    return;
  }
  std::ifstream input(source_capture, std::ios::binary);
  H3TransformerCaptureHeader source{};
  CHECK(static_cast<bool>(input.read(reinterpret_cast<char*>(&source), sizeof(source))));
  CHECK(std::memcmp(source.magic, "VFH3FWD\0", 8) == 0 && source.text_rows == 4 &&
        source.text_dim == 5120 &&
        source.prompt_elements == uint64_t(source.text_rows) * source.text_dim);
  std::vector<float> base_prompt(source.prompt_elements);
  CHECK(static_cast<bool>(
      input.read(reinterpret_cast<char*>(base_prompt.data()),
                 static_cast<std::streamsize>(base_prompt.size() * sizeof(float)))));
  constexpr uint32_t text_rows = 4100;
  std::vector<float> prompt(uint64_t(text_rows) * source.text_dim);
  for (size_t i = 0; i < prompt.size(); ++i)
    prompt[i] = base_prompt[i % base_prompt.size()];
  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  Transformer transformer;
  transformer.load(checkpoint);
  transformer.set_attention_mode(AttentionMode::kExact);
  const auto begin = std::chrono::steady_clock::now();
  const std::vector<Transformer::DebugStage> stages =
      transformer.debug_text_stages(prompt.data(), text_rows);
  const double milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  CHECK(stages.size() == 6u);
  std::array<uint64_t, 6> hashes{};
  for (size_t stage = 0; stage < stages.size(); ++stage) {
    CHECK(stages[stage].rows == static_cast<int>(text_rows) && stages[stage].dim == 5376 &&
          stages[stage].data.size() == uint64_t(text_rows) * 5376);
    uint64_t hash = 1469598103934665603ull;
    for (float value : stages[stage].data) {
      const uint16_t bits = f32_to_bf16(value);
      hash ^= bits & 0xffu;
      hash *= 1099511628211ull;
      hash ^= bits >> 8;
      hash *= 1099511628211ull;
    }
    hashes[stage] = hash;
  }
  std::printf("  Ref2VA CUDA text L4100 %.3f ms stage FNV64:", milliseconds);
  for (uint64_t hash : hashes)
    std::printf(" %016llx", static_cast<unsigned long long>(hash));
  std::printf("\n");
  transformer.unload();
}
