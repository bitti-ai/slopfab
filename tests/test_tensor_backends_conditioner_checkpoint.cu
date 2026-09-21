#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_qwen_vision_real_block0, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
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

  SafeTensors archive;
  archive.open(checkpoint_path.string());
  const text::QwenVisionCheckpoint vision = text::load_qwen3vl_vision_checkpoint(archive);
  constexpr uint32_t rows = 64, hidden = 1152, head_dim = 72;
  constexpr uint32_t intermediate = 4304;
  const std::string p = vision.prefix + "blocks.0.";
  auto upload_cuda = [&](const std::string& suffix) {
    const TensorView& view = archive.at(p + suffix);
    cuda::DeviceBuffer<uint16_t> result(view.nbytes / 2);
    result.copy_from_host(static_cast<const uint16_t*>(view.data), view.nbytes / 2);
    return result;
  };
  cuda::DeviceBuffer<uint16_t> n1w = upload_cuda("norm1.weight"), n1b = upload_cuda("norm1.bias"),
                               n2w = upload_cuda("norm2.weight"), n2b = upload_cuda("norm2.bias"),
                               qw = upload_cuda("attn.qkv.weight"),
                               qb = upload_cuda("attn.qkv.bias"),
                               ow = upload_cuda("attn.proj.weight"),
                               ob = upload_cuda("attn.proj.bias"),
                               f1w = upload_cuda("mlp.linear_fc1.weight"),
                               f1b = upload_cuda("mlp.linear_fc1.bias"),
                               f2w = upload_cuda("mlp.linear_fc2.weight"),
                               f2b = upload_cuda("mlp.linear_fc2.bias");
  auto dense = [](cuda::DeviceBuffer<uint16_t>& weight, cuda::DeviceBuffer<uint16_t>& bias, int out,
                  int in) {
    cuda::QuantWeight result;
    result.format = cuda::QuantFormat::kBF16;
    result.data = weight.get();
    result.bias = bias.get();
    result.bias_format = cuda::QuantFormat::kBF16;
    result.out_features = out;
    result.in_features = in;
    return result;
  };
  cuda::QwenVisionBlockWeights cuda_weights;
  cuda_weights.norm1_weight = reinterpret_cast<__nv_bfloat16*>(n1w.get());
  cuda_weights.norm1_bias = reinterpret_cast<__nv_bfloat16*>(n1b.get());
  cuda_weights.norm2_weight = reinterpret_cast<__nv_bfloat16*>(n2w.get());
  cuda_weights.norm2_bias = reinterpret_cast<__nv_bfloat16*>(n2b.get());
  cuda_weights.qkv = dense(qw, qb, 3 * hidden, hidden);
  cuda_weights.attention_out = dense(ow, ob, hidden, hidden);
  cuda_weights.mlp_fc1 = dense(f1w, f1b, intermediate, hidden);
  cuda_weights.mlp_fc2 = dense(f2w, f2b, hidden, intermediate);

  std::vector<uint16_t> input(size_t(rows) * hidden);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = f32_to_bf16(float(int(i * 29 % 257) - 128) / 128.0f);
  const text::QwenVisionPositions positions = text::qwen3vl_vision_positions({1, 8, 8});
  std::vector<float> cosine, sine;
  text::qwen3vl_vision_rope_tables(positions, cosine, sine);
  cuda::DeviceBuffer<uint16_t> cx(input.size()), cn(size_t(rows) * hidden),
      cqkv(size_t(rows) * 3 * hidden), cq(size_t(rows) * hidden), ck(size_t(rows) * hidden),
      cv(size_t(rows) * hidden), cbranch(size_t(rows) * hidden), cmlp(size_t(rows) * intermediate);
  cuda::DeviceBuffer<float> cc(cosine.size()), cs(sine.size());
  cx.copy_from_host(input.data(), input.size());
  cc.copy_from_host(cosine.data(), cosine.size());
  cs.copy_from_host(sine.data(), sine.size());
  cuda::QwenVisionBlockScratch cuda_scratch{
      reinterpret_cast<__nv_bfloat16*>(cn.get()),  reinterpret_cast<__nv_bfloat16*>(cqkv.get()),
      reinterpret_cast<__nv_bfloat16*>(cq.get()),  reinterpret_cast<__nv_bfloat16*>(ck.get()),
      reinterpret_cast<__nv_bfloat16*>(cv.get()),  reinterpret_cast<__nv_bfloat16*>(cbranch.get()),
      reinterpret_cast<__nv_bfloat16*>(cmlp.get())};
  const auto cuda_begin = std::chrono::steady_clock::now();
  cuda::qwen_vision_block_forward_exact(nullptr, cuda_weights, cc.get(), cs.get(),
                                        reinterpret_cast<__nv_bfloat16*>(cx.get()), rows,
                                        cuda_scratch);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_begin)
          .count();
  std::vector<uint16_t> expected(input.size());
  cx.copy_to_host(expected.data(), expected.size());

  TensorContextOptions context_options;
  context_options.max_batch_operators = 20;
  TensorContext vk(device, context_options);
  QwenVisionStageConfig config;
  config.sequence = rows;
  ExactQwenVisionScratch scratch = ExactQwenVisionScratch::create(vk, config);
  ExactQwenVisionBlockStage stage = ExactQwenVisionBlockStage::create(vk, config);
  auto mat = [](uint64_t a, uint64_t b) {
    const uint64_t shape[] = {a, b};
    return TensorLayout::contiguous(shape, 2);
  };
  DeviceTensor residual = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16);
  DeviceTensor vc = vk.allocate(mat(rows, head_dim), ScalarType::kFloat32);
  DeviceTensor vs = vk.allocate(mat(rows, head_dim), ScalarType::kFloat32);
  vk.upload_bytes(residual, input.data(), input.size() * 2);
  vk.upload(vc, cosine.data(), cosine.size());
  vk.upload(vs, sine.data(), sine.size());
  const uint64_t unloaded_used = vk.pooled_used_bytes();
  stage.load(vision, 0);
  CHECK(stage.loaded() && stage.block() == 0);
  CHECK(stage.required_operators() == 16);
  CHECK(stage.persistent_bytes() < 32ull * 1024 * 1024);

  // One-short preflight leaves the same batch usable and does not touch the
  // residual. Five harmless preceding copies reduce cap20 to cap15.
  const uint64_t scalar_shape[] = {1};
  DeviceTensor dummy_a =
      vk.allocate(TensorLayout::contiguous(scalar_shape, 1), ScalarType::kBFloat16);
  DeviceTensor dummy_b =
      vk.allocate(TensorLayout::contiguous(scalar_shape, 1), ScalarType::kBFloat16);
  vk.upload_bytes(residual, input.data(), input.size() * 2);
  TensorBatch short_batch = vk.begin_batch();
  short_batch.copy(dummy_a, dummy_b);
  short_batch.copy(dummy_b, dummy_a);
  short_batch.copy(dummy_a, dummy_b);
  short_batch.copy(dummy_b, dummy_a);
  short_batch.copy(dummy_a, dummy_b);
  CHECK(short_batch.remaining_operator_capacity() == 15);
  bool short_rejected = false;
  try {
    stage.record(short_batch, residual, vc, vs, scratch);
  } catch (const std::logic_error&) {
    short_rejected = true;
  }
  CHECK(short_rejected && short_batch.remaining_operator_capacity() == 15);
  short_batch.submit().wait();
  std::vector<uint16_t> unchanged(input.size());
  vk.download_bytes(residual, unchanged.data(), unchanged.size() * 2);
  CHECK(unchanged == input);

  const auto vk_begin = std::chrono::steady_clock::now();
  TensorBatch batch = vk.begin_batch();
  {
    test::HostAllocationGuard no_host_allocations;
    stage.record(batch, residual, vc, vs, scratch);
  }
  CHECK(batch.remaining_operator_capacity() == 4);
  batch.submit().wait();
  const double vk_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vk_begin)
          .count();
  std::vector<uint16_t> actual(input.size());
  vk.download_bytes(residual, actual.data(), actual.size() * 2);
  CHECK(std::memcmp(expected.data(), actual.data(), actual.size() * 2) == 0);
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  const uint64_t stable_used = vk.pooled_used_bytes();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_persistent = stage.persistent_bytes();
  text::QwenVisionCheckpoint invalid = vision;
  invalid.prefix = "missing.visual.";
  bool reload_rejected = false;
  try {
    stage.load(invalid, 26);
  } catch (const std::exception&) {
    reload_rejected = true;
  }
  CHECK(reload_rejected);
  CHECK(stage.loaded() && stage.block() == 0 && stage.persistent_bytes() == stable_persistent);
  CHECK(vk.pooled_used_bytes() == stable_used && vk.reserved_bytes() == stable_reserved &&
        vk.descriptor_set_allocations() == stable_descriptors);
  vk.upload_bytes(residual, input.data(), input.size() * 2);
  TensorBatch repeat = vk.begin_batch();
  stage.record(repeat, residual, vc, vs, scratch);
  repeat.submit().wait();
  vk.download_bytes(residual, actual.data(), actual.size() * 2);
  CHECK(actual == expected);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  dummy_a = DeviceTensor();
  dummy_b = DeviceTensor();
  vk.collect();
  stage.unload();
  vk.collect();
  CHECK(!stage.loaded() && stage.persistent_bytes() == 0);
  CHECK(vk.pooled_used_bytes() == unloaded_used);

  // Patch embedding and both merger norm orders use the same strict archive
  // and shared arena. They are checked against explicit exact CUDA, not the
  // shipped cuBLAS/native-tanh tower.
  std::vector<uint16_t> pixels(size_t(rows) * 1536);
  for (size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = f32_to_bf16(float(int(i * 17 % 193) - 96) / 96.0f);
  const TensorView& patch_w_view = archive.at(vision.prefix + "patch_embed.proj.weight");
  const TensorView& patch_b_view = archive.at(vision.prefix + "patch_embed.proj.bias");
  const TensorView& position_view = archive.at(vision.prefix + "pos_embed.weight");
  cuda::DeviceBuffer<uint16_t> cpatch_w(patch_w_view.nbytes / 2), cpatch_b(patch_b_view.nbytes / 2),
      cposition(position_view.nbytes / 2), cpixels(pixels.size()),
      cpatch_out(size_t(rows) * hidden);
  cuda::DeviceBuffer<int32_t> cposition_index(positions.learned.size());
  cpatch_w.copy_from_host(static_cast<const uint16_t*>(patch_w_view.data), patch_w_view.nbytes / 2);
  cpatch_b.copy_from_host(static_cast<const uint16_t*>(patch_b_view.data), patch_b_view.nbytes / 2);
  cposition.copy_from_host(static_cast<const uint16_t*>(position_view.data),
                           position_view.nbytes / 2);
  cpixels.copy_from_host(pixels.data(), pixels.size());
  cposition_index.copy_from_host(positions.learned.data(), positions.learned.size());
  cuda::QuantWeight patch_weight = dense(cpatch_w, cpatch_b, hidden, 1536);
  cuda::qwen_vision_patch_embed_exact(
      nullptr, patch_weight, reinterpret_cast<const __nv_bfloat16*>(cpixels.get()),
      reinterpret_cast<const __nv_bfloat16*>(cposition.get()), cposition_index.get(),
      reinterpret_cast<__nv_bfloat16*>(cpatch_out.get()), rows);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> expected_patch(size_t(rows) * hidden);
  cpatch_out.copy_to_host(expected_patch.data(), expected_patch.size());

  DeviceTensor vpixels = vk.allocate(mat(rows, 1536), ScalarType::kBFloat16);
  const uint64_t index_shape[] = {rows};
  DeviceTensor vposition_index =
      vk.allocate(TensorLayout::contiguous(index_shape, 1), ScalarType::kInt32);
  DeviceTensor vpatch_out = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16);
  vk.upload_bytes(vpixels, pixels.data(), pixels.size() * 2);
  vk.upload_bytes(vposition_index, positions.learned.data(),
                  positions.learned.size() * sizeof(int32_t));
  ExactQwenVisionPatchStage patch_stage = ExactQwenVisionPatchStage::create(vk, config);
  patch_stage.load(vision);
  CHECK(patch_stage.required_operators() == 2);
  TensorBatch patch_batch = vk.begin_batch();
  {
    test::HostAllocationGuard no_alloc;
    patch_stage.record(patch_batch, vpixels, vposition_index, vpatch_out, scratch);
  }
  patch_batch.submit().wait();
  std::vector<uint16_t> actual_patch(expected_patch.size());
  vk.download_bytes(vpatch_out, actual_patch.data(), actual_patch.size() * 2);
  CHECK(actual_patch == expected_patch);

  ExactQwenVisionMergerStage merger_stage = ExactQwenVisionMergerStage::create(vk, config);
  DeviceTensor vmerged_out = vk.allocate(mat(rows / 4, 5120), ScalarType::kBFloat16);
  auto exact_cuda_merger = [&](int slot_index) {
    const std::string base = slot_index < 0 ? vision.prefix + "merger."
                                            : vision.prefix + "deepstack_merger_list." +
                                                  std::to_string(slot_index) + ".";
    auto component = [&](const char* suffix) {
      const TensorView& view = archive.at(base + suffix);
      cuda::DeviceBuffer<uint16_t> result(view.nbytes / 2);
      result.copy_from_host(static_cast<const uint16_t*>(view.data), view.nbytes / 2);
      return result;
    };
    cuda::DeviceBuffer<uint16_t> nw = component("norm.weight"), nb = component("norm.bias"),
                                 mw1 = component("linear_fc1.weight"),
                                 mb1 = component("linear_fc1.bias"),
                                 mw2 = component("linear_fc2.weight"),
                                 mb2 = component("linear_fc2.bias");
    cuda::DeviceBuffer<uint16_t> mn(size_t(rows) * hidden), mm(size_t(rows / 4) * 4608),
        mh(size_t(rows / 4) * 4608), mo(size_t(rows / 4) * 5120);
    cuda::QwenVisionMergerWeights weights;
    weights.norm_weight = reinterpret_cast<__nv_bfloat16*>(nw.get());
    weights.norm_bias = reinterpret_cast<__nv_bfloat16*>(nb.get());
    weights.fc1 = dense(mw1, mb1, 4608, 4608);
    weights.fc2 = dense(mw2, mb2, 5120, 4608);
    weights.norm_before_merge = slot_index < 0;
    cuda::qwen_vision_merger_forward_exact(
        nullptr, weights, reinterpret_cast<const __nv_bfloat16*>(cpatch_out.get()),
        reinterpret_cast<__nv_bfloat16*>(mn.get()), reinterpret_cast<__nv_bfloat16*>(mm.get()),
        reinterpret_cast<__nv_bfloat16*>(mh.get()), reinterpret_cast<__nv_bfloat16*>(mo.get()),
        rows);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> result(size_t(rows / 4) * 5120);
    mo.copy_to_host(result.data(), result.size());
    return result;
  };
  for (const int merger_slot : {-1, 0}) {
    const std::vector<uint16_t> expected_merger = exact_cuda_merger(merger_slot);
    merger_stage.load(vision, merger_slot);
    CHECK(merger_stage.slot() == merger_slot);
    CHECK(merger_stage.required_operators() == 5);
    CHECK(merger_stage.persistent_bytes() < 96ull * 1024 * 1024);
    TensorBatch merger_batch = vk.begin_batch();
    {
      test::HostAllocationGuard no_alloc;
      merger_stage.record(merger_batch, vpatch_out, vmerged_out, scratch);
    }
    merger_batch.submit().wait();
    std::vector<uint16_t> actual_merger(expected_merger.size());
    vk.download_bytes(vmerged_out, actual_merger.data(), actual_merger.size() * 2);
    CHECK(actual_merger == expected_merger);
  }
  patch_stage.unload();
  merger_stage.unload();
  std::printf("qwen vision real block0 S64 CUDA %.3f ms Vulkan %.3f ms "
              "scratch %llu bytes\n",
              cuda_ms, vk_ms, static_cast<unsigned long long>(scratch.reserved_bytes()));
}

SLOPFAB_TEST_CATEGORY(vulkan_qwen_real_layer0_synthetic_activation, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const std::filesystem::path checkpoint_path =
      "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
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
  // S3 I8: q/k/v and gate/up share their identical ConvRot activation, saving
  // three operators, plus eleven requested diagnostic copies.
  context_options.max_batch_operators = 39;
  TensorContext vk(device, context_options);
  if (!vk.exact_causal_gqa_attention() || !vk.exact_fp32_vae_normalization() ||
      !vk.exact_vae_pointwise()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !vk.exact_causal_gqa_attention() || !vk.exact_fp32_vae_normalization() || !vk.exact_vae_pointwise()");
    return;
  }

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  QwenTextLayerConfig config;
  config.sequence = 3;
  const auto load_begin = std::chrono::steady_clock::now();
  ExactQwenTextLayerStage stage = ExactQwenTextLayerStage::create(vk, config);
  stage.load(checkpoint, 0);
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_begin)
          .count();
  ExactQwenTextLayerScratch scratch = ExactQwenTextLayerScratch::create(vk, config);
  CHECK(stage.loaded());
  CHECK(stage.format() == text::WeightFormat::kI8ConvRot);
  CHECK(scratch.dense_cache_bytes() == 250ull * 1024 * 1024);

  constexpr uint32_t rows = 3, hidden = 5120, q_heads = 64, kv_heads = 8, head_dim = 128,
                     ffn = 25600;
  std::vector<uint16_t> input(size_t(rows) * hidden);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = f32_to_bf16(float(int((i * 19) % 101) - 50) / 64.0f);
  std::vector<float> cosine, sine;
  text::build_rope_tables(rows, text::rope_inv_freq(head_dim, 5.0e6f), cosine, sine);
  const uint64_t token_shape[] = {rows, hidden};
  const uint64_t rope_shape[] = {rows, head_dim};
  DeviceTensor tokens =
      vk.allocate(TensorLayout::contiguous(token_shape, 2), ScalarType::kBFloat16);
  DeviceTensor cos_tensor =
      vk.allocate(TensorLayout::contiguous(rope_shape, 2), ScalarType::kFloat32);
  DeviceTensor sin_tensor =
      vk.allocate(TensorLayout::contiguous(rope_shape, 2), ScalarType::kFloat32);
  vk.upload(cos_tensor, cosine.data(), cosine.size());
  vk.upload(sin_tensor, sine.data(), sine.size());

  auto mat = [&](uint64_t a, uint64_t b) {
    const uint64_t shape[] = {a, b};
    return TensorLayout::contiguous(shape, 2);
  };
  auto three = [&](uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t shape[] = {a, b, c};
    return TensorLayout::contiguous(shape, 3);
  };
  DeviceTensor tap_norm = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16);
  DeviceTensor tap_q = vk.allocate(three(rows, q_heads, head_dim), ScalarType::kBFloat16);
  DeviceTensor tap_k = vk.allocate(three(rows, kv_heads, head_dim), ScalarType::kBFloat16);
  DeviceTensor tap_v = vk.allocate(three(rows, kv_heads, head_dim), ScalarType::kBFloat16);
  DeviceTensor tap_attention = vk.allocate(three(rows, q_heads, head_dim), ScalarType::kBFloat16);
  DeviceTensor tap_attention_residual = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16);
  DeviceTensor tap_post_norm = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16);
  DeviceTensor tap_gate = vk.allocate(mat(rows, ffn), ScalarType::kBFloat16);
  DeviceTensor tap_up = vk.allocate(mat(rows, ffn), ScalarType::kBFloat16);
  DeviceTensor tap_activation = vk.allocate(mat(rows, ffn), ScalarType::kBFloat16);
  DeviceTensor tap_final = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16);
  QwenTextLayerTaps taps{
      &tap_norm,      &tap_q,    &tap_k,  &tap_v,          &tap_attention, &tap_attention_residual,
      &tap_post_norm, &tap_gate, &tap_up, &tap_activation, &tap_final};
  CHECK(stage.required_operators(&taps) == 39);

  auto run = [&] {
    vk.upload_bytes(tokens, input.data(), input.size() * 2);
    const auto begin = std::chrono::steady_clock::now();
    TensorBatch batch = vk.begin_batch();
    {
      test::HostAllocationGuard allocation_guard;
      stage.record(batch, tokens, cos_tensor, sin_tensor, scratch, &taps);
    }
    CHECK(batch.remaining_operator_capacity() ==
          context_options.max_batch_operators - stage.required_operators(&taps));
    batch.submit().wait();
    const double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    std::vector<uint16_t> result(input.size());
    vk.download_bytes(tokens, result.data(), result.size() * 2);
    std::vector<uint16_t> tapped(result.size());
    vk.download_bytes(tap_final, tapped.data(), tapped.size() * 2);
    CHECK(result == tapped);
    return std::pair<std::vector<uint16_t>, double>(std::move(result), elapsed);
  };

  // One-short capacity is rejected before the first stage operator. The
  // caller can still submit its preceding copy and the tokens remain intact.
  const uint64_t one_shape[] = {1};
  DeviceTensor dummy_a = vk.allocate(TensorLayout::contiguous(one_shape, 1), ScalarType::kBFloat16);
  DeviceTensor dummy_b = vk.allocate(TensorLayout::contiguous(one_shape, 1), ScalarType::kBFloat16);
  const uint16_t dummy = f32_to_bf16(1.0f);
  vk.upload_bytes(dummy_a, &dummy, sizeof(dummy));
  vk.upload_bytes(tokens, input.data(), input.size() * 2);
  TensorBatch short_batch = vk.begin_batch();
  short_batch.copy(dummy_a, dummy_b);
  bool short_rejected = false;
  try {
    stage.record(short_batch, tokens, cos_tensor, sin_tensor, scratch, &taps);
  } catch (const std::logic_error&) {
    short_rejected = true;
  }
  CHECK(short_rejected);
  CHECK(short_batch.remaining_operator_capacity() == 38);
  short_batch.submit().wait();
  std::vector<uint16_t> unchanged(input.size());
  vk.download_bytes(tokens, unchanged.data(), unchanged.size() * 2);
  CHECK(unchanged == input);

  // Establish the exact non-weight pool baseline with caller scratch,
  // activations and taps retained. Every unload below must return here.
  stage.unload();
  const uint64_t unloaded_pool_baseline = vk.pooled_used_bytes();
  const uint64_t unloaded_descriptor_baseline = vk.descriptor_set_allocations();
  stage.load(checkpoint, 0);

  const auto first = run();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  const auto repeat = run();
  CHECK(first.first == repeat.first);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);

  uint64_t digest = 1469598103934665603ull;
  for (uint16_t value : first.first) {
    digest ^= value & 0xffu;
    digest *= 1099511628211ull;
    digest ^= value >> 8u;
    digest *= 1099511628211ull;
  }
  CHECK(digest == 0x42a875f728d707f3ull);
  const uint64_t persistent = stage.persistent_bytes();
  const uint64_t scratch_bytes = scratch.reserved_bytes();
  CHECK(stage.peak_device_bytes(scratch) == persistent + scratch_bytes);
  CHECK(persistent < 500ull * 1024 * 1024);
  CHECK(scratch_bytes < 270ull * 1024 * 1024);

  // A malformed replacement fails during host validation, preserving active
  // weights, pool high-water and the executable output.
  const std::filesystem::path bad_path =
      std::filesystem::temp_directory_path() /
      ("slopfab_qwen_layer_bad_reload_" +
       std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
       ".safetensors");
  write_safetensors(bad_path.string(), {{"unrelated", {1}, {0.0f}, DType::kF32}});
  const uint64_t before_bad_persistent = stage.persistent_bytes();
  const uint64_t before_bad_reserved = vk.reserved_bytes();
  bool bad_rejected = false;
  {
    SafeTensors bad;
    bad.open(bad_path.string());
    try {
      stage.load(bad, 0);
    } catch (const std::exception&) {
      bad_rejected = true;
    }
  }
  CHECK(bad_rejected);
  CHECK(stage.persistent_bytes() == before_bad_persistent);
  CHECK(vk.reserved_bytes() == before_bad_reserved);
  const auto after_bad = run();
  CHECK(after_bad.first == first.first);
  std::filesystem::remove(bad_path);

  // A valid full manifest whose final projection descriptor is corrupted is
  // rejected just as transactionally. The sparse fixture retains the real
  // archive header/offset contract without duplicating a 27 GiB checkpoint.
  const std::filesystem::path late_i8_path = make_sparse_qwen_metadata_corruption(
      checkpoint, text::WeightFormat::kI8ConvRot, "model.layers.0.mlp.down_proj.comfy_quant");
  const uint64_t before_late_i8_used = vk.pooled_used_bytes();
  const uint64_t before_late_i8_reserved = vk.reserved_bytes();
  const uint64_t before_late_i8_descriptors = vk.descriptor_set_allocations();
  bool late_i8_rejected = false;
  {
    SafeTensors bad;
    bad.open(late_i8_path.string());
    try {
      stage.load(bad, 0);
    } catch (const std::exception&) {
      late_i8_rejected = true;
    }
  }
  CHECK(late_i8_rejected);
  CHECK(vk.pooled_used_bytes() == before_late_i8_used);
  CHECK(vk.reserved_bytes() == before_late_i8_reserved);
  CHECK(vk.descriptor_set_allocations() == before_late_i8_descriptors);
  CHECK(run().first == first.first);
  std::filesystem::remove(late_i8_path);

  const uint64_t i8_high_water = vk.reserved_bytes();
  stage.unload();
  CHECK(!stage.loaded());
  CHECK(stage.persistent_bytes() == 0);
  CHECK(vk.pooled_used_bytes() == unloaded_pool_baseline);
  CHECK(vk.descriptor_set_allocations() >= unloaded_descriptor_baseline);
  stage.load(checkpoint, 0);
  const auto reloaded = run();
  CHECK(reloaded.first == first.first);
  CHECK(vk.reserved_bytes() == i8_high_water);

  // The same stage/scratch executes the shipped NVFP4+AWQ contract. Only the
  // two declared AWQ transforms record; all seven matrices still use the one
  // shared dense slot.
  const std::filesystem::path nvfp4_path =
      "weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
  SafeTensors nvfp4;
  nvfp4.open(nvfp4_path.string());
  stage.unload();
  CHECK(vk.pooled_used_bytes() == unloaded_pool_baseline);
  CHECK(vk.reserved_bytes() == i8_high_water);
  const auto nv_load_begin = std::chrono::steady_clock::now();
  stage.load(nvfp4, 0);
  const double nv_load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - nv_load_begin)
          .count();
  CHECK(stage.format() == text::WeightFormat::kNVFP4Awq);
  CHECK(stage.required_operators(&taps) == 37);
  const auto nv_first = run();
  const uint64_t nv_stable_descriptors = vk.descriptor_set_allocations();
  const auto nv_repeat = run();
  CHECK(nv_first.first == nv_repeat.first);
  CHECK(vk.descriptor_set_allocations() == nv_stable_descriptors);
  const std::filesystem::path late_nv_path = make_sparse_qwen_metadata_corruption(
      nvfp4, text::WeightFormat::kNVFP4Awq, "model.layers.0.mlp.down_proj.comfy_quant");
  const uint64_t before_late_nv_used = vk.pooled_used_bytes();
  const uint64_t before_late_nv_reserved = vk.reserved_bytes();
  const uint64_t before_late_nv_descriptors = vk.descriptor_set_allocations();
  bool late_nv_rejected = false;
  {
    SafeTensors bad;
    bad.open(late_nv_path.string());
    try {
      stage.load(bad, 0);
    } catch (const std::exception&) {
      late_nv_rejected = true;
    }
  }
  CHECK(late_nv_rejected);
  CHECK(vk.pooled_used_bytes() == before_late_nv_used);
  CHECK(vk.reserved_bytes() == before_late_nv_reserved);
  CHECK(vk.descriptor_set_allocations() == before_late_nv_descriptors);
  CHECK(run().first == nv_first.first);
  std::filesystem::remove(late_nv_path);
  uint64_t nv_digest = 1469598103934665603ull;
  for (uint16_t value : nv_first.first) {
    nv_digest ^= value & 0xffu;
    nv_digest *= 1099511628211ull;
    nv_digest ^= value >> 8u;
    nv_digest *= 1099511628211ull;
  }
  CHECK(nv_digest == 0x935104aeb9432d6aull);
  const uint64_t nv_persistent = stage.persistent_bytes();
  CHECK(nv_persistent < 270ull * 1024 * 1024);
  const uint64_t nv_high_water = vk.reserved_bytes();
  stage.unload();
  CHECK(vk.pooled_used_bytes() == unloaded_pool_baseline);
  CHECK(vk.reserved_bytes() == nv_high_water);
  stage.load(nvfp4, 0);
  CHECK(run().first == nv_first.first);
  CHECK(vk.reserved_bytes() == nv_high_water);
  CHECK(vk.descriptor_set_allocations() == nv_stable_descriptors);
  stage.unload();
  CHECK(stage.persistent_bytes() == 0);
  CHECK(vk.pooled_used_bytes() == unloaded_pool_baseline);
  CHECK(vk.reserved_bytes() == nv_high_water);
  std::printf(
      "  exact Vulkan Qwen layer0 S3: I8 load/first/repeat %.1f/%.1f/%.1f ms FNV64 %016llx persistent %.1f MiB; NVFP4 load/first/repeat %.1f/%.1f/%.1f ms FNV64 %016llx persistent %.1f MiB; shared scratch %.1f MiB, unloaded pool baseline/reserved HWM %.1f/%.1f MiB, descriptors %llu\n",
      load_ms, first.second, repeat.second, static_cast<unsigned long long>(digest),
      double(persistent) / 1048576.0, nv_load_ms, nv_first.second, nv_repeat.second,
      static_cast<unsigned long long>(nv_digest), double(nv_persistent) / 1048576.0,
      double(scratch_bytes) / 1048576.0, double(unloaded_pool_baseline) / 1048576.0,
      double(vk.reserved_bytes()) / 1048576.0, static_cast<unsigned long long>(stable_descriptors));
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_qwen_layer0_real_l132, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  constexpr uint32_t rows = 132, hidden = 5120, q_heads = 64, kv_heads = 8, head_dim = 128,
                     ffn = 25600;
  const std::filesystem::path source(SLOPFAB_TEST_SOURCE_DIR);
  const std::filesystem::path checkpoint_path =
      source / "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
  const std::filesystem::path tokenizer_path = source / "ref/text_encoder/tokenizer.json";
  int cuda_devices = 0;
  if (!std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(tokenizer_path) ||
      cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(checkpoint_path)|| !std::filesystem::exists(tokenizer_path)|| cudaGetDeviceCount(&cuda_devices)!=cudaSuccess||cuda_devices==0|| !Instance::available()");
    return;
  }
  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  text::EncoderConfig encoder_config;
  encoder_config.format = text::detect_weight_format(checkpoint);
  CHECK(encoder_config.format == text::WeightFormat::kI8ConvRot);
  text::Tokenizer tokenizer;
  tokenizer.load(tokenizer_path.string());
  std::string prompt;
  const std::string sentence =
      "A cinematic tracking shot follows a copper airship over snowy forests, "
      "with warm sunrise reflections and natural motion. ";
  while (tokenizer.encode(prompt).size() < rows)
    prompt += sentence;
  std::vector<int32_t> token_ids = tokenizer.encode(prompt);
  token_ids.resize(rows);
  std::vector<uint16_t> input;
  text::gather_embedding_rows(checkpoint.at("model.embed_tokens.weight"), nullptr, token_ids,
                              input);
  CHECK(input.size() == size_t(rows) * hidden);
  std::vector<float> cosine, sine;
  text::build_rope_tables(rows, text::rope_inv_freq(head_dim, 5.0e6f), cosine, sine);

  // Upload the checkpoint-native packed layer once and invoke the reusable
  // canonical exact CUDA path. This is the authority, not shipped cuBLAS.
  const text::LayerLayout layer_layout = text::make_layer_layout(encoder_config);
  cuda::DeviceBuffer<uint8_t> cuda_layer(layer_layout.total_bytes);
  text::upload_layer_direct(checkpoint, encoder_config, 0, layer_layout, cuda_layer.get(), nullptr);
  const text::LayerGlobalScales globals = text::read_global_scales(checkpoint, encoder_config, 0);
  const text::LayerWeights cuda_weights =
      text::layer_weights_from_blob(cuda_layer.get(), layer_layout, encoder_config, globals);
  text::LayerDims dims;
  dims.format = encoder_config.format;
  dims.num_tokens = rows;
  dims.hidden = hidden;
  dims.num_heads = q_heads;
  dims.num_kv_heads = kv_heads;
  dims.head_dim = head_dim;
  dims.intermediate = ffn;
  dims.rms_norm_eps = 1.0e-6f;
  cuda::Workspace cuda_workspace;
  cuda_workspace.reserve(text::exact_layer_workspace_bytes(cuda_weights, dims));
  cuda::DeviceBuffer<uint16_t> cuda_tokens(input.size());
  cuda::DeviceBuffer<float> cuda_cosine(cosine.size()), cuda_sine(sine.size());
  cuda_tokens.copy_from_host(input.data(), input.size());
  cuda_cosine.copy_from_host(cosine.data(), cosine.size());
  cuda_sine.copy_from_host(sine.data(), sine.size());
  cuda::DeviceBuffer<uint16_t> c_norm(size_t(rows) * hidden),
      c_q(size_t(rows) * q_heads * head_dim), c_k(size_t(rows) * kv_heads * head_dim),
      c_v(size_t(rows) * kv_heads * head_dim), c_attention(size_t(rows) * q_heads * head_dim),
      c_attention_residual(size_t(rows) * hidden), c_post_norm(size_t(rows) * hidden),
      c_gate(size_t(rows) * ffn), c_up(size_t(rows) * ffn), c_activation(size_t(rows) * ffn),
      c_final(size_t(rows) * hidden);
  text::ExactLayerTaps cuda_taps{reinterpret_cast<__nv_bfloat16*>(c_norm.get()),
                                 reinterpret_cast<__nv_bfloat16*>(c_q.get()),
                                 reinterpret_cast<__nv_bfloat16*>(c_k.get()),
                                 reinterpret_cast<__nv_bfloat16*>(c_v.get()),
                                 reinterpret_cast<__nv_bfloat16*>(c_attention.get()),
                                 reinterpret_cast<__nv_bfloat16*>(c_attention_residual.get()),
                                 reinterpret_cast<__nv_bfloat16*>(c_post_norm.get()),
                                 reinterpret_cast<__nv_bfloat16*>(c_gate.get()),
                                 reinterpret_cast<__nv_bfloat16*>(c_up.get()),
                                 reinterpret_cast<__nv_bfloat16*>(c_activation.get()),
                                 reinterpret_cast<__nv_bfloat16*>(c_final.get())};
  const auto cuda_begin = std::chrono::steady_clock::now();
  text::encoder_layer_forward_exact(nullptr, cuda_weights, dims, cuda_cosine.get(), cuda_sine.get(),
                                    reinterpret_cast<__nv_bfloat16*>(cuda_tokens.get()),
                                    cuda_workspace, &cuda_taps);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_begin)
          .count();

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty()");
    return;
  }
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !info.timeline_semaphore||!info.shader_int64");
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
  TensorContext vk(device, context_options);
  if (!vk.exact_causal_gqa_attention() || !vk.exact_fp32_vae_normalization() ||
      !vk.exact_vae_pointwise()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !vk.exact_causal_gqa_attention()||!vk.exact_fp32_vae_normalization()|| !vk.exact_vae_pointwise()");
    return;
  }
  QwenTextLayerConfig config;
  config.sequence = rows;
  ExactQwenTextLayerStage stage = ExactQwenTextLayerStage::create(vk, config);
  const auto load_begin = std::chrono::steady_clock::now();
  stage.load(checkpoint, 0);
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_begin)
          .count();
  ExactQwenTextLayerScratch scratch = ExactQwenTextLayerScratch::create(vk, config);
  auto mat = [&](uint64_t a, uint64_t b) {
    const uint64_t s[] = {a, b};
    return TensorLayout::contiguous(s, 2);
  };
  auto three = [&](uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t s[] = {a, b, c};
    return TensorLayout::contiguous(s, 3);
  };
  DeviceTensor v_tokens = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16),
               v_cos = vk.allocate(mat(rows, head_dim)), v_sin = vk.allocate(mat(rows, head_dim)),
               v_norm = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16),
               v_q = vk.allocate(three(rows, q_heads, head_dim), ScalarType::kBFloat16),
               v_k = vk.allocate(three(rows, kv_heads, head_dim), ScalarType::kBFloat16),
               v_v = vk.allocate(three(rows, kv_heads, head_dim), ScalarType::kBFloat16),
               v_attention = vk.allocate(three(rows, q_heads, head_dim), ScalarType::kBFloat16),
               v_attention_residual = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16),
               v_post_norm = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16),
               v_gate = vk.allocate(mat(rows, ffn), ScalarType::kBFloat16),
               v_up = vk.allocate(mat(rows, ffn), ScalarType::kBFloat16),
               v_activation = vk.allocate(mat(rows, ffn), ScalarType::kBFloat16),
               v_final = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16);
  vk.upload_bytes(v_tokens, input.data(), input.size() * 2);
  vk.upload(v_cos, cosine.data(), cosine.size());
  vk.upload(v_sin, sine.data(), sine.size());
  QwenTextLayerTaps vk_taps{
      &v_norm,      &v_q,    &v_k,  &v_v,          &v_attention, &v_attention_residual,
      &v_post_norm, &v_gate, &v_up, &v_activation, &v_final};
  CHECK(stage.required_operators(&vk_taps) == 46);
  const auto vk_begin = std::chrono::steady_clock::now();
  TensorBatch batch = vk.begin_batch();
  stage.record(batch, v_tokens, v_cos, v_sin, scratch, &vk_taps);
  CHECK(batch.remaining_operator_capacity() == 0);
  batch.submit().wait();
  const double vk_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vk_begin)
          .count();

  auto compare = [&](const char* name, cuda::DeviceBuffer<uint16_t>& expected_buffer,
                     DeviceTensor& actual_tensor, size_t count) {
    std::vector<uint16_t> expected(count), actual(count);
    expected_buffer.copy_to_host(expected.data(), count);
    vk.download_bytes(actual_tensor, actual.data(), count * 2);
    size_t mismatch = count;
    for (size_t i = 0; i < count; ++i)
      if (expected[i] != actual[i]) {
        mismatch = i;
        break;
      }
    CHECK_MSG(mismatch == count, "Qwen L132 %s mismatch at %zu: CUDA %04x Vulkan %04x", name,
              mismatch, mismatch == count ? 0u : expected[mismatch],
              mismatch == count ? 0u : actual[mismatch]);
    uint64_t hash = 1469598103934665603ull;
    for (uint16_t value : actual) {
      hash ^= value & 0xffu;
      hash *= 1099511628211ull;
      hash ^= value >> 8u;
      hash *= 1099511628211ull;
    }
    return hash;
  };
  std::array<uint64_t, 11> hashes{
      compare("input norm", c_norm, v_norm, size_t(rows) * hidden),
      compare("query", c_q, v_q, size_t(rows) * q_heads * head_dim),
      compare("key", c_k, v_k, size_t(rows) * kv_heads * head_dim),
      compare("value", c_v, v_v, size_t(rows) * kv_heads * head_dim),
      compare("attention", c_attention, v_attention, size_t(rows) * q_heads * head_dim),
      compare("attention residual", c_attention_residual, v_attention_residual,
              size_t(rows) * hidden),
      compare("post norm", c_post_norm, v_post_norm, size_t(rows) * hidden),
      compare("gate", c_gate, v_gate, size_t(rows) * ffn),
      compare("up", c_up, v_up, size_t(rows) * ffn),
      compare("activation", c_activation, v_activation, size_t(rows) * ffn),
      compare("final", c_final, v_final, size_t(rows) * hidden)};
  constexpr std::array<uint64_t, 11> expected_hashes{
      0x6ca9b8c5e16917b5ull, 0xf9bf6e554a1e84e4ull, 0x9c2c264a60b4b8b1ull, 0xec21312eea810e06ull,
      0x73901fb1cb7f2cfbull, 0x5ab1cc9e26345fe2ull, 0x9e072ab6646a2a11ull, 0x74d47f5ed650356dull,
      0x8292d03af91a2025ull, 0x48f99e7549238eceull, 0xfb3966de636ac098ull};
  CHECK(hashes == expected_hashes);

  auto fnv_bytes = [](const void* data, size_t bytes, uint64_t hash = 1469598103934665603ull) {
    const auto* cursor = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) {
      hash ^= cursor[i];
      hash *= 1099511628211ull;
    }
    return hash;
  };
  const uint64_t input_hash = fnv_bytes(input.data(), input.size() * sizeof(uint16_t));
  uint64_t rope_hash = fnv_bytes(cosine.data(), cosine.size() * sizeof(float));
  rope_hash = fnv_bytes(sine.data(), sine.size() * sizeof(float), rope_hash);
  CHECK(input_hash == 0x617329501f3c87a1ull);
  CHECK(rope_hash == 0x693c23a9886dd147ull);
  constexpr std::array<uint8_t, 32> checkpoint_sha{0xbc, 0x2c, 0xed, 0x0f, 0xbe, 0xa6, 0x47, 0x57,
                                                   0xfa, 0x9a, 0xcd, 0xdc, 0xcf, 0xc0, 0xb3, 0xf4,
                                                   0x81, 0x9d, 0x1d, 0xcf, 0x1d, 0xa6, 0xc1, 0x24,
                                                   0xd6, 0x90, 0xd3, 0x68, 0xbe, 0x28, 0x39, 0x23};
  constexpr std::array<uint8_t, 32> tokenizer_sha{0xa5, 0xd8, 0x5b, 0x6d, 0xcc, 0x53, 0x5e, 0x6b,
                                                  0x93, 0x11, 0x5a, 0x9e, 0xf2, 0x87, 0xe6, 0x13,
                                                  0x2f, 0xdb, 0xf3, 0x02, 0x70, 0xda, 0x62, 0x18,
                                                  0x19, 0x4b, 0xa7, 0x42, 0x26, 0x11, 0x73, 0xc7};
  text::QwenLayerCapture generated;
  generated.header.sequence = rows;
  generated.header.hidden = hidden;
  generated.header.query_heads = q_heads;
  generated.header.kv_heads = kv_heads;
  generated.header.head_dim = head_dim;
  generated.header.intermediate = ffn;
  generated.header.checkpoint_sha256 = checkpoint_sha;
  generated.header.tokenizer_sha256 = tokenizer_sha;
  generated.header.input_fnv64 = input_hash;
  generated.header.rope_fnv64 = rope_hash;
  generated.header.boundary_fnv64 = hashes;
  generated.token_ids = token_ids;
  generated.input_bf16 = input;
  generated.cosine = cosine;
  generated.sine = sine;
  const std::filesystem::path capture_path =
      std::filesystem::path(SLOPFAB_TEST_SOURCE_DIR) / "tests/data/qwen_layer0_l132.vfqw";
  if (const char* write = std::getenv("SLOPFAB_WRITE_QWEN_LAYER_CAPTURE");
      write && std::strcmp(write, "1") == 0) {
    text::write_qwen_layer_capture(capture_path.string(), generated);
  }
  CHECK(std::filesystem::exists(capture_path));
#ifdef _WIN32
  std::ifstream capture_stream(capture_path, std::ios::binary);
  const std::vector<uint8_t> capture_bytes{std::istreambuf_iterator<char>(capture_stream),
                                           std::istreambuf_iterator<char>()};
  constexpr std::array<uint8_t, 32> capture_sha{0xec, 0x13, 0xad, 0x62, 0xa7, 0xe2, 0x53, 0xd5,
                                                0x88, 0xbf, 0xac, 0x51, 0x85, 0x0b, 0x92, 0x48,
                                                0x7b, 0x7c, 0xb8, 0x8b, 0xa7, 0x3e, 0x78, 0x69,
                                                0xa2, 0xb0, 0x2c, 0xba, 0x79, 0x11, 0x04, 0xb3};
  CHECK(sha256_mapping(capture_bytes.data(), capture_bytes.size()) == capture_sha);
#endif
  const text::QwenLayerCapture captured = text::read_qwen_layer_capture(capture_path.string());
  CHECK(captured.header.sequence == rows);
  CHECK(captured.header.hidden == hidden);
  CHECK(captured.header.query_heads == q_heads);
  CHECK(captured.header.kv_heads == kv_heads);
  CHECK(captured.header.head_dim == head_dim);
  CHECK(captured.header.intermediate == ffn);
  CHECK(captured.header.checkpoint_sha256 == checkpoint_sha);
  CHECK(captured.header.tokenizer_sha256 == tokenizer_sha);
  CHECK(captured.header.input_fnv64 == input_hash);
  CHECK(captured.header.rope_fnv64 == rope_hash);
  CHECK(captured.header.boundary_fnv64 == expected_hashes);
  CHECK(captured.token_ids == token_ids);
  CHECK(captured.input_bf16 == input);
  CHECK(captured.cosine == cosine);
  CHECK(captured.sine == sine);

  // Run the identical captured activation through the shipped NVFP4+AWQ
  // layer. L132 exercises the 128-row cooperative tile plus a four-row GEMM
  // tail, while o/down execute their checkpoint-declared AWQ pre-scales.
  const std::filesystem::path nv_path =
      source / "weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
  const uint64_t i8_persistent_bytes = stage.persistent_bytes();
  CHECK(std::filesystem::exists(nv_path));
  SafeTensors nv_checkpoint;
  nv_checkpoint.open(nv_path.string());
  text::EncoderConfig nv_config;
  nv_config.format = text::detect_weight_format(nv_checkpoint);
  CHECK(nv_config.format == text::WeightFormat::kNVFP4Awq);
  const text::LayerLayout nv_layout = text::make_layer_layout(nv_config);
  cuda::DeviceBuffer<uint8_t> nv_cuda_layer(nv_layout.total_bytes);
  text::upload_layer_direct(nv_checkpoint, nv_config, 0, nv_layout, nv_cuda_layer.get(), nullptr);
  const text::LayerGlobalScales nv_globals = text::read_global_scales(nv_checkpoint, nv_config, 0);
  const text::LayerWeights nv_cuda_weights =
      text::layer_weights_from_blob(nv_cuda_layer.get(), nv_layout, nv_config, nv_globals);
  dims.format = nv_config.format;
  cuda_tokens.copy_from_host(input.data(), input.size());
  const auto nv_cuda_begin = std::chrono::steady_clock::now();
  text::encoder_layer_forward_exact(
      nullptr, nv_cuda_weights, dims, cuda_cosine.get(), cuda_sine.get(),
      reinterpret_cast<__nv_bfloat16*>(cuda_tokens.get()), cuda_workspace, &cuda_taps);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double nv_cuda_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - nv_cuda_begin)
          .count();

  stage.load(nv_checkpoint, 0);
  CHECK(stage.format() == text::WeightFormat::kNVFP4Awq);
  CHECK(stage.required_operators(&vk_taps) == 44);
  vk.upload_bytes(v_tokens, input.data(), input.size() * sizeof(uint16_t));
  const auto nv_vk_begin = std::chrono::steady_clock::now();
  TensorBatch nv_batch = vk.begin_batch();
  {
    test::HostAllocationGuard allocation_guard;
    stage.record(nv_batch, v_tokens, v_cos, v_sin, scratch, &vk_taps);
  }
  CHECK(nv_batch.remaining_operator_capacity() == 2);
  nv_batch.submit().wait();
  const double nv_vk_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - nv_vk_begin)
          .count();
  std::array<uint64_t, 11> nv_hashes{
      compare("NV input norm", c_norm, v_norm, size_t(rows) * hidden),
      compare("NV query", c_q, v_q, size_t(rows) * q_heads * head_dim),
      compare("NV key", c_k, v_k, size_t(rows) * kv_heads * head_dim),
      compare("NV value", c_v, v_v, size_t(rows) * kv_heads * head_dim),
      compare("NV attention", c_attention, v_attention, size_t(rows) * q_heads * head_dim),
      compare("NV attention residual", c_attention_residual, v_attention_residual,
              size_t(rows) * hidden),
      compare("NV post norm", c_post_norm, v_post_norm, size_t(rows) * hidden),
      compare("NV gate", c_gate, v_gate, size_t(rows) * ffn),
      compare("NV up", c_up, v_up, size_t(rows) * ffn),
      compare("NV activation", c_activation, v_activation, size_t(rows) * ffn),
      compare("NV final", c_final, v_final, size_t(rows) * hidden)};
  constexpr std::array<uint64_t, 11> expected_nv_hashes{
      0x395ca0928bc2fe96ull, 0xbe792d7f50dffcd8ull, 0xabf4d02a691c93feull, 0xa7c505fccf740093ull,
      0x3c771911cbcbba19ull, 0xafcdcae26c9edd26ull, 0x503531dfaee1b283ull, 0x241abe4e317283c4ull,
      0xb3f789dffb6d190cull, 0xdd56acc48480841eull, 0xa389ed1f9e8067e8ull};
  CHECK(nv_hashes == expected_nv_hashes);
  std::printf(
      "  real Qwen layer0 L132 exact CUDA %.1f ms Vulkan %.1f ms load %.1f ms, persistent/scratch %.1f/%.1f MiB, boundary FNV64:",
      cuda_ms, vk_ms, load_ms, double(i8_persistent_bytes) / 1048576.0,
      double(scratch.reserved_bytes()) / 1048576.0);
  for (uint64_t hash : hashes)
    std::printf(" %016llx", static_cast<unsigned long long>(hash));
  std::printf("; input/rope %016llx/%016llx\n", static_cast<unsigned long long>(input_hash),
              static_cast<unsigned long long>(rope_hash));
  std::printf(
      "  real NVFP4+AWQ Qwen layer0 L132 exact CUDA %.1f ms Vulkan %.1f ms, boundary FNV64:",
      nv_cuda_ms, nv_vk_ms);
  for (uint64_t hash : nv_hashes)
    std::printf(" %016llx", static_cast<unsigned long long>(hash));
  std::printf("\n");
}
