#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_exact_vae_vit_block_stage, "synthetic") {
  using namespace slopfab;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !vulkan::Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !vulkan::Instance::available()");
    return;
  }
  vulkan::Instance instance = vulkan::Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  vulkan::DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = physical.front().info().shader_int64;
  vulkan::Device device = physical.front().create_device(device_options);
  vulkan::TensorContextOptions context_options;
  context_options.max_batch_operators = 64;
  vulkan::TensorContext context(device, context_options);
  if (!context.exact_fp32_vae_normalization() || !context.exact_vae_pointwise() ||
      !context.exact_blocked_attention()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !context.exact_fp32_vae_normalization() || !context.exact_vae_pointwise() || !context.exact_blocked_attention()");
    return;
  }

  vae::ViTBlockConfig config;
  config.sequence = 69;
  config.num_patches = 64;
  config.dim = 64;
  config.heads = 1;
  config.head_dim = 64;
  config.ffn_inner = 128;
  config.rope_dim = 48;
  vae::ViTBlockWeights weights;
  weights.norm1.resize(config.dim);
  weights.norm2.resize(config.dim);
  weights.scale1.resize(config.dim);
  weights.scale2.resize(config.dim);
  weights.qkv_weight.resize(size_t(3) * config.dim * config.dim);
  weights.qkv_bias.resize(3 * config.dim);
  weights.out_weight.resize(size_t(config.dim) * config.dim);
  weights.out_bias.resize(config.dim);
  weights.w1_weight.resize(size_t(2) * config.ffn_inner * config.dim);
  weights.w1_bias.resize(2 * config.ffn_inner);
  weights.w2_weight.resize(size_t(config.dim) * config.ffn_inner);
  weights.w2_bias.resize(config.dim);
  for (uint32_t i = 0; i < config.dim; ++i) {
    weights.norm1[i] = 0.75f + float(i % 11) / 32.0f;
    weights.norm2[i] = 0.875f + float(i % 7) / 32.0f;
    weights.scale1[i] = 0.01f + float(i % 5) / 1024.0f;
    weights.scale2[i] = 0.0125f + float(i % 3) / 1024.0f;
    weights.out_bias[i] = float(int(i % 13) - 6) / 512.0f;
    weights.w2_bias[i] = float(int(i % 17) - 8) / 512.0f;
  }
  auto fill_half = [](std::vector<uint16_t>& values, uint32_t multiplier) {
    for (size_t i = 0; i < values.size(); ++i) {
      const float value = float(int((i * multiplier) % 31) - 15) / 512.0f;
      values[i] = f32_to_f16(value);
    }
  };
  fill_half(weights.qkv_weight, 7);
  fill_half(weights.out_weight, 11);
  fill_half(weights.w1_weight, 13);
  fill_half(weights.w2_weight, 17);
  for (size_t i = 0; i < weights.qkv_bias.size(); ++i)
    weights.qkv_bias[i] = float(int(i % 19) - 9) / 512.0f;
  for (size_t i = 0; i < weights.w1_bias.size(); ++i)
    weights.w1_bias[i] = float(int(i % 23) - 11) / 512.0f;

  const size_t token_count = size_t(config.sequence) * config.dim;
  const size_t rope_count = size_t(config.sequence) * config.rope_dim;
  std::vector<float> input(token_count), cosine(rope_count, 1.0f), sine(rope_count, 0.0f),
      cuda_once(token_count), vulkan_once(token_count), cuda_twice(token_count),
      vulkan_twice(token_count);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = float(int((i * 29) % 251) - 125) / 128.0f;

  auto cuda_stage = cuda::create_exact_vae_vit_block_stage(config);
  cuda_stage->load(weights.view());
  vulkan::ExactViTBlockStage vk_stage = vulkan::ExactViTBlockStage::create(context, config);
  vk_stage.load(weights.view());
  CHECK(cuda_stage->backend() == DeviceBackend::kCuda);
  CHECK(vk_stage.backend() == DeviceBackend::kVulkan);
  CHECK(cuda_stage->persistent_bytes() == weights.bytes());
  CHECK(vk_stage.persistent_bytes() == weights.bytes());
  {
    vae::ViTBlockWeights noncanonical = weights;
    noncanonical.qkv_weight[0] = 0x0001u;
    bool cuda_rejected = false, vulkan_rejected = false;
    try {
      auto bad = cuda::create_exact_vae_vit_block_stage(config);
      bad->load(noncanonical.view());
    } catch (const std::invalid_argument&) {
      cuda_rejected = true;
    }
    try {
      vulkan::ExactViTBlockStage bad = vulkan::ExactViTBlockStage::create(context, config);
      bad.load(noncanonical.view());
    } catch (const std::invalid_argument&) {
      vulkan_rejected = true;
    }
    CHECK(cuda_rejected && vulkan_rejected);
  }
  cuda_stage->forward(input.data(), cosine.data(), sine.data(), cuda_once.data());
  vk_stage.forward(input.data(), cosine.data(), sine.data(), vulkan_once.data());
  CHECK(std::memcmp(cuda_once.data(), vulkan_once.data(), token_count * 4) == 0);

  // The production seam chains two block records through one device tensor,
  // one command buffer and one shared scratch arena. No host boundary occurs.
  const uint64_t token_shape[] = {config.sequence, config.dim};
  const uint64_t rope_shape[] = {config.sequence, config.rope_dim};
  vulkan::DeviceTensor tokens = context.allocate(TensorLayout::contiguous(token_shape, 2));
  vulkan::DeviceTensor vk_cosine = context.allocate(TensorLayout::contiguous(rope_shape, 2));
  vulkan::DeviceTensor vk_sine = context.allocate(TensorLayout::contiguous(rope_shape, 2));
  context.upload(tokens, input.data(), input.size());
  context.upload(vk_cosine, cosine.data(), cosine.size());
  context.upload(vk_sine, sine.data(), sine.size());
  vulkan::ExactViTBlockScratch scratch = vulkan::ExactViTBlockScratch::create(context, config);
  const uint64_t wrong_rope_shape[] = {config.sequence, config.rope_dim - 1};
  vulkan::DeviceTensor wrong_cosine =
      context.allocate(TensorLayout::contiguous(wrong_rope_shape, 2));
  vulkan::TensorBatch chained = context.begin_batch();
  const uint32_t capacity_before_rejection = chained.remaining_operator_capacity();
  bool shape_rejected = false;
  try {
    vk_stage.record(chained, tokens, wrong_cosine, vk_sine, scratch);
  } catch (const std::invalid_argument&) {
    shape_rejected = true;
  }
  CHECK(shape_rejected);
  CHECK(chained.remaining_operator_capacity() == capacity_before_rejection);
  vk_stage.record(chained, tokens, vk_cosine, vk_sine, scratch);
  vk_stage.record(chained, tokens, vk_cosine, vk_sine, scratch);
  chained.submit().wait();
  context.download(tokens, vulkan_twice.data(), vulkan_twice.size());
  cuda_stage->forward(cuda_once.data(), cosine.data(), sine.data(), cuda_twice.data());
  CHECK(std::memcmp(cuda_twice.data(), vulkan_twice.data(), token_count * 4) == 0);

  const uint64_t stable_reserved = context.reserved_bytes();
  const uint64_t stable_descriptors = context.descriptor_set_allocations();
  vk_stage.forward(input.data(), cosine.data(), sine.data(), vulkan_twice.data());
  CHECK(std::memcmp(cuda_once.data(), vulkan_twice.data(), token_count * 4) == 0);
  CHECK(context.reserved_bytes() == stable_reserved);
  CHECK(context.descriptor_set_allocations() == stable_descriptors);

  // Three real graph nodes share one scratch arena. Vulkan records all sixty
  // operators into a single submission and exactly matches the CUDA graph.
  cuda::ExactViTBlockGraph cuda_graph = cuda::ExactViTBlockGraph::create(config, 3);
  vulkan::ExactViTBlockGraph vk_graph = vulkan::ExactViTBlockGraph::create(context, config, 3);
  for (uint32_t layer = 0; layer < 3; ++layer) {
    cuda_graph.load_layer(layer, weights.view());
    vk_graph.load_layer(layer, weights.view());
  }
  std::vector<float> cuda_graph_output(token_count), vk_graph_output(token_count);
  cuda_graph.forward(input.data(), cosine.data(), sine.data(), cuda_graph_output.data());
  vk_graph.forward(input.data(), cosine.data(), sine.data(), vk_graph_output.data());
  CHECK(std::memcmp(cuda_graph_output.data(), vk_graph_output.data(),
                    token_count * sizeof(float)) == 0);
  CHECK(cuda_graph.layers() == 3 && vk_graph.layers() == 3);
  CHECK(cuda_graph.persistent_bytes() == 3 * weights.bytes());
  CHECK(vk_graph.persistent_bytes() == 3 * weights.bytes());
  CHECK(cuda_graph.peak_device_bytes() < 3 * cuda_stage->peak_device_bytes());
  CHECK(vk_graph.peak_device_bytes() < 3 * vk_stage.peak_device_bytes());
  const uint64_t graph_reserved = context.reserved_bytes();
  const uint64_t graph_descriptors = context.descriptor_set_allocations();
  vk_graph.forward(input.data(), cosine.data(), sine.data(), vk_graph_output.data());
  CHECK(context.reserved_bytes() == graph_reserved);
  CHECK(context.descriptor_set_allocations() == graph_descriptors);

  // A full graph is transactional with respect to load state: neither backend
  // may execute its loaded prefix when a later layer is missing.
  cuda::ExactViTBlockGraph partial_cuda = cuda::ExactViTBlockGraph::create(config, 3);
  vulkan::ExactViTBlockGraph partial_vk = vulkan::ExactViTBlockGraph::create(context, config, 3);
  partial_cuda.load_layer(0, weights.view());
  partial_vk.load_layer(0, weights.view());
  cuda::DeviceBuffer<float> partial_tokens(token_count), partial_cosine(rope_count),
      partial_sine(rope_count);
  cuda::Stream partial_stream;
  partial_tokens.copy_from_host(input.data(), input.size(), partial_stream.get());
  partial_cosine.copy_from_host(cosine.data(), cosine.size(), partial_stream.get());
  partial_sine.copy_from_host(sine.data(), sine.size(), partial_stream.get());
  partial_stream.synchronize();
  bool partial_cuda_rejected = false;
  try {
    partial_cuda.forward_device(partial_tokens.get(), partial_cosine.get(), partial_sine.get(),
                                partial_stream.get());
  } catch (const std::logic_error&) {
    partial_cuda_rejected = true;
  }
  CHECK(partial_cuda_rejected);
  CHECK(cudaStreamQuery(partial_stream.get()) == cudaSuccess);
  std::vector<float> partial_after(token_count);
  partial_tokens.copy_to_host(partial_after.data(), partial_after.size(), partial_stream.get());
  partial_stream.synchronize();
  CHECK(std::memcmp(input.data(), partial_after.data(), token_count * 4) == 0);
  {
    vulkan::TensorBatch partial_batch = context.begin_batch();
    const uint32_t partial_capacity = partial_batch.remaining_operator_capacity();
    bool partial_vk_rejected = false;
    try {
      partial_vk.record(partial_batch, tokens, vk_cosine, vk_sine);
    } catch (const std::logic_error&) {
      partial_vk_rejected = true;
    }
    CHECK(partial_vk_rejected);
    CHECK(partial_batch.remaining_operator_capacity() == partial_capacity);
  }

  // The CUDA ragged-shape cache is hard-bounded to two arenas. The largest
  // shape below supplies a monotonic upper bound for every transition.
  cuda::ExactViTBlockGraph shape_graph = cuda::ExactViTBlockGraph::create(config, 3);
  for (uint32_t layer = 0; layer < 3; ++layer)
    shape_graph.load_layer(layer, weights.view());
  const uint64_t shape_persistent = shape_graph.persistent_bytes();
  const uint64_t base_peak = shape_graph.peak_device_bytes();
  const uint64_t base_scratch = base_peak - shape_persistent;
  CHECK(shape_graph.cached_scratch_shapes() == 1);
  shape_graph.prepare_shape(101, 96);
  const uint64_t full_peak = shape_graph.peak_device_bytes();
  const uint64_t full_scratch = full_peak - shape_persistent - base_scratch;
  CHECK(shape_graph.cached_scratch_shapes() == 2);
  const uint64_t two_full_bound = shape_persistent + 2 * full_scratch;
  CHECK(full_peak <= two_full_bound);
  shape_graph.prepare_shape(85, 80);
  CHECK(shape_graph.cached_scratch_shapes() == 2);
  CHECK(shape_graph.peak_device_bytes() <= two_full_bound);
  shape_graph.prepare_shape(93, 88);
  CHECK(shape_graph.cached_scratch_shapes() == 2);
  CHECK(shape_graph.peak_device_bytes() <= two_full_bound);
  vulkan::ExactViTBlockGraph vk_shape_graph =
      vulkan::ExactViTBlockGraph::create(context, config, 3);
  for (uint32_t layer = 0; layer < 3; ++layer)
    vk_shape_graph.load_layer(layer, weights.view());
  const uint64_t vk_shape_persistent = vk_shape_graph.persistent_bytes();
  const uint64_t vk_base_scratch = vk_shape_graph.peak_device_bytes() - vk_shape_persistent;
  CHECK(vk_shape_graph.cached_scratch_shapes() == 1);
  vk_shape_graph.prepare_shape(101, 96);
  const uint64_t vk_full_scratch =
      vk_shape_graph.peak_device_bytes() - vk_shape_persistent - vk_base_scratch;
  const uint64_t vk_two_full_bound = vk_shape_persistent + 2 * vk_full_scratch;
  CHECK(vk_shape_graph.cached_scratch_shapes() == 2);
  CHECK(vk_shape_graph.peak_device_bytes() <= vk_two_full_bound);
  vk_shape_graph.prepare_shape(85, 80);
  CHECK(vk_shape_graph.cached_scratch_shapes() == 2);
  CHECK(vk_shape_graph.peak_device_bytes() <= vk_two_full_bound);
  vk_shape_graph.prepare_shape(93, 88);
  CHECK(vk_shape_graph.cached_scratch_shapes() == 2);
  CHECK(vk_shape_graph.peak_device_bytes() <= vk_two_full_bound);

  // Queue work against the current arena, switch twice so that arena is the
  // eviction victim, and then queue another shape on a second stream. Slot
  // completion events make both reuse and eviction safe without a device-wide
  // synchronization in the execution path.
  auto make_cuda_activation = [&](uint32_t sequence) {
    return cuda::DeviceBuffer<float>(size_t(sequence) * config.dim);
  };
  auto make_cuda_rope = [&](uint32_t sequence) {
    return cuda::DeviceBuffer<float>(size_t(sequence) * config.rope_dim);
  };
  cuda::DeviceBuffer<float> queued_tokens = make_cuda_activation(93);
  cuda::DeviceBuffer<float> queued_cosine = make_cuda_rope(93);
  cuda::DeviceBuffer<float> queued_sine = make_cuda_rope(93);
  cuda::DeviceBuffer<float> base_tokens = make_cuda_activation(69);
  cuda::DeviceBuffer<float> base_cosine = make_cuda_rope(69);
  cuda::DeviceBuffer<float> base_sine = make_cuda_rope(69);
  cuda::Stream queued_stream, switched_stream;
  SLOPFAB_CUDA_CHECK(
      cudaMemsetAsync(queued_tokens.get(), 0, queued_tokens.nbytes(), queued_stream.get()));
  SLOPFAB_CUDA_CHECK(
      cudaMemsetAsync(queued_cosine.get(), 0, queued_cosine.nbytes(), queued_stream.get()));
  SLOPFAB_CUDA_CHECK(
      cudaMemsetAsync(queued_sine.get(), 0, queued_sine.nbytes(), queued_stream.get()));
  shape_graph.forward_device(queued_tokens.get(), queued_cosine.get(), queued_sine.get(),
                             queued_stream.get());
  shape_graph.prepare_shape(69, 64);
  SLOPFAB_CUDA_CHECK(
      cudaMemsetAsync(base_tokens.get(), 0, base_tokens.nbytes(), switched_stream.get()));
  SLOPFAB_CUDA_CHECK(
      cudaMemsetAsync(base_cosine.get(), 0, base_cosine.nbytes(), switched_stream.get()));
  SLOPFAB_CUDA_CHECK(
      cudaMemsetAsync(base_sine.get(), 0, base_sine.nbytes(), switched_stream.get()));
  shape_graph.forward_device(base_tokens.get(), base_cosine.get(), base_sine.get(),
                             switched_stream.get());
  shape_graph.prepare_shape(101, 96); // evicts and fences queued R93
  queued_stream.synchronize();
  switched_stream.synchronize();
  CHECK(shape_graph.cached_scratch_shapes() == 2);
  CHECK(shape_graph.peak_device_bytes() <= two_full_bound);

  {
    vulkan::ExactViTBlockGraph oversized_graph =
        vulkan::ExactViTBlockGraph::create(context, config, 4);
    vulkan::TensorBatch insufficient = context.begin_batch();
    const uint32_t insufficient_capacity = insufficient.remaining_operator_capacity();
    bool capacity_rejected = false;
    try {
      oversized_graph.record(insufficient, tokens, vk_cosine, vk_sine);
    } catch (const std::logic_error&) {
      capacity_rejected = true;
    }
    CHECK(capacity_rejected);
    CHECK(insufficient.remaining_operator_capacity() == insufficient_capacity);
  }

  if (!std::getenv("SLOPFAB_VAE_VIT_BLOCK_REAL")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_VAE_VIT_BLOCK_REAL\")");
    return;
  }
  const std::filesystem::path checkpoint_path = "weights/vae/minimax_h3_video_vae_fp16.safetensors";
  if (!std::filesystem::exists(checkpoint_path)) {
    SKIP_MISSING_FIXTURE("unavailable prerequisite: !std::filesystem::exists(checkpoint_path)");
    return;
  }
  vae::ViTBlockConfig real_config;
  real_config.sequence = 1797;
  real_config.num_patches = 1792;
  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_checkpoint_sha{
      0x7c, 0x1f, 0x13, 0x14, 0x92, 0xe7, 0xed, 0xda, 0xca, 0xac, 0x90,
      0x69, 0xa6, 0x1b, 0x81, 0xbd, 0xd3, 0x9d, 0xe5, 0xcc, 0x96, 0x56,
      0x1e, 0x67, 0x7c, 0x5e, 0xab, 0x1c, 0xdc, 0xe5, 0xe5, 0x22};
  CHECK(sha256_mapping(checkpoint.mapping_base(), checkpoint.file_size()) ==
        expected_checkpoint_sha);
#endif
  const std::array<const char*, 4> real_matrix_names{
      "decoder.transformer_blocks.0.attn.to_qkv.weight",
      "decoder.transformer_blocks.0.attn.to_out.weight",
      "decoder.transformer_blocks.0.ff.w1.weight", "decoder.transformer_blocks.0.ff.w2.weight"};
  size_t raw_fp16_subnormals = 0;
  for (const char* name : real_matrix_names) {
    const TensorView& tensor = checkpoint.at(name);
    CHECK(tensor.dtype == DType::kF16);
    const auto* words = static_cast<const uint16_t*>(tensor.data);
    for (size_t i = 0; i < tensor.nbytes / sizeof(uint16_t); ++i)
      raw_fp16_subnormals += (words[i] & 0x7c00u) == 0 && (words[i] & 0x03ffu) != 0;
  }
  CHECK(raw_fp16_subnormals == 330659u);
  vae::ViTBlockWeights real_weights = vae::load_vit_block_weights(checkpoint, 0, real_config);
  CHECK(real_weights.bytes() == 134356992ull);
  size_t fp32_subnormals = 0, fp16_subnormals = 0;
  auto scan_float = [&](const std::vector<float>& values) {
    for (float value : values) {
      uint32_t bits = 0;
      std::memcpy(&bits, &value, 4);
      fp32_subnormals += (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0;
    }
  };
  auto scan_half = [&](const std::vector<uint16_t>& values) {
    for (uint16_t bits : values)
      fp16_subnormals += (bits & 0x7c00u) == 0 && (bits & 0x03ffu) != 0;
  };
  scan_float(real_weights.norm1);
  scan_float(real_weights.norm2);
  scan_float(real_weights.scale1);
  scan_float(real_weights.scale2);
  scan_float(real_weights.qkv_bias);
  scan_float(real_weights.out_bias);
  scan_float(real_weights.w1_bias);
  scan_float(real_weights.w2_bias);
  scan_half(real_weights.qkv_weight);
  scan_half(real_weights.out_weight);
  scan_half(real_weights.w1_weight);
  scan_half(real_weights.w2_weight);
  CHECK(fp32_subnormals == 0);
  CHECK(fp16_subnormals == 0);
  std::printf("  real block weight subnormals fp32=%zu fp16=%zu\n", fp32_subnormals,
              fp16_subnormals);
  std::vector<float> real_input(size_t(real_config.sequence) * real_config.dim),
      real_cosine(size_t(real_config.sequence) * real_config.rope_dim, 1.0f),
      real_sine(real_cosine.size(), 0.0f), real_cuda(real_input.size()),
      real_vulkan(real_input.size());
  for (size_t i = 0; i < real_input.size(); ++i)
    real_input[i] = float(int((i * 29) % 509) - 254) / 512.0f;
  auto real_cuda_stage = cuda::create_exact_vae_vit_block_stage(real_config);
  real_cuda_stage->load(real_weights.view());
  vulkan::ExactViTBlockStage real_vk_stage =
      vulkan::ExactViTBlockStage::create(context, real_config);
  real_vk_stage.load(real_weights.view());
  const auto cuda_begin = std::chrono::steady_clock::now();
  real_cuda_stage->forward(real_input.data(), real_cosine.data(), real_sine.data(),
                           real_cuda.data());
  const double cuda_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_begin)
          .count();
  const uint64_t real_token_shape[] = {real_config.sequence, real_config.dim};
  const uint64_t real_rope_shape[] = {real_config.sequence, real_config.rope_dim};
  vulkan::DeviceTensor real_tokens =
      context.allocate(TensorLayout::contiguous(real_token_shape, 2));
  vulkan::DeviceTensor real_vk_cosine =
      context.allocate(TensorLayout::contiguous(real_rope_shape, 2));
  vulkan::DeviceTensor real_vk_sine =
      context.allocate(TensorLayout::contiguous(real_rope_shape, 2));
  vulkan::ExactViTBlockScratch real_scratch =
      vulkan::ExactViTBlockScratch::create(context, real_config);
  context.upload(real_tokens, real_input.data(), real_input.size());
  context.upload(real_vk_cosine, real_cosine.data(), real_cosine.size());
  context.upload(real_vk_sine, real_sine.data(), real_sine.size());
  const auto vk_begin = std::chrono::steady_clock::now();
  vulkan::TensorBatch real_batch = context.begin_batch();
  real_vk_stage.record(real_batch, real_tokens, real_vk_cosine, real_vk_sine, real_scratch);
  real_batch.submit().wait();
  const double vk_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vk_begin)
          .count();
  context.download(real_tokens, real_vulkan.data(), real_vulkan.size());
  size_t mismatch = real_input.size();
  for (size_t i = 0; i < real_input.size(); ++i) {
    if (std::memcmp(&real_cuda[i], &real_vulkan[i], 4) != 0) {
      mismatch = i;
      break;
    }
  }
  if (mismatch != real_input.size()) {
    uint32_t cb = 0, vb = 0;
    std::memcpy(&cb, &real_cuda[mismatch], 4);
    std::memcpy(&vb, &real_vulkan[mismatch], 4);
    std::printf("  first real block mismatch %zu: CUDA %08x Vulkan %08x\n", mismatch, cb, vb);
  }
  CHECK_MSG(mismatch == real_input.size(), "real VAE ViT block mismatch at %zu/%zu", mismatch,
            real_input.size());
  uint64_t fnv = 1469598103934665603ull;
  for (float value : real_vulkan) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    for (int byte = 0; byte < 4; ++byte) {
      fnv ^= (bits >> (byte * 8)) & 0xffu;
      fnv *= 1099511628211ull;
    }
  }
  CHECK(fnv == 0xc8a7ac3241effbb7ull);
  std::printf(
      "  real VAE ViT block0 R1797/D2048/I8192: CUDA %.3f ms, Vulkan %.3f ms, exact %zu words, FNV64 %016llx, weights %.1f MiB, Vulkan peak %.1f MiB\n",
      cuda_ms, vk_ms, real_vulkan.size(), static_cast<unsigned long long>(fnv),
      double(real_weights.bytes()) / 1048576.0,
      double(real_vk_stage.persistent_bytes() + real_scratch.reserved_bytes() +
             real_input.size() * sizeof(float) + 2 * real_cosine.size() * sizeof(float)) /
          1048576.0);

  if (!std::getenv("SLOPFAB_VAE_VIT_GRAPH_REAL")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_VAE_VIT_GRAPH_REAL\")");
    return;
  }
  vulkan::TensorContextOptions graph_options;
  graph_options.max_batch_operators = 1024;
  vulkan::TensorContext graph_context(device, graph_options);
  if (!graph_context.exact_fp32_vae_normalization() || !graph_context.exact_vae_pointwise() ||
      !graph_context.exact_blocked_attention()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !graph_context.exact_fp32_vae_normalization() || !graph_context.exact_vae_pointwise() || !graph_context.exact_blocked_attention()");
    return;
  }
  constexpr uint32_t kGraphLayers = 36;
  cuda::ExactViTBlockGraph real_cuda_graph =
      cuda::ExactViTBlockGraph::create(real_config, kGraphLayers);
  vulkan::ExactViTBlockGraph real_vk_graph =
      vulkan::ExactViTBlockGraph::create(graph_context, real_config, kGraphLayers);
  size_t graph_raw_subnormals = 0, graph_loaded_subnormals = 0, graph_loaded_fp32_subnormals = 0;
  const auto graph_load_begin = std::chrono::steady_clock::now();
  for (uint32_t layer = 0; layer < kGraphLayers; ++layer) {
    const std::array<std::string, 4> names{
        "decoder.transformer_blocks." + std::to_string(layer) + ".attn.to_qkv.weight",
        "decoder.transformer_blocks." + std::to_string(layer) + ".attn.to_out.weight",
        "decoder.transformer_blocks." + std::to_string(layer) + ".ff.w1.weight",
        "decoder.transformer_blocks." + std::to_string(layer) + ".ff.w2.weight"};
    for (const std::string& name : names) {
      const TensorView& tensor = checkpoint.at(name);
      CHECK(tensor.dtype == DType::kF16);
      const auto* words = static_cast<const uint16_t*>(tensor.data);
      for (size_t i = 0; i < tensor.nbytes / sizeof(uint16_t); ++i)
        graph_raw_subnormals += (words[i] & 0x7c00u) == 0 && (words[i] & 0x03ffu) != 0;
    }
    vae::ViTBlockWeights layer_weights =
        vae::load_vit_block_weights(checkpoint, layer, real_config);
    auto count_loaded = [&](const std::vector<uint16_t>& values) {
      for (uint16_t word : values)
        graph_loaded_subnormals += (word & 0x7c00u) == 0 && (word & 0x03ffu) != 0;
    };
    count_loaded(layer_weights.qkv_weight);
    count_loaded(layer_weights.out_weight);
    count_loaded(layer_weights.w1_weight);
    count_loaded(layer_weights.w2_weight);
    auto count_loaded_float = [&](const std::vector<float>& values) {
      for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        graph_loaded_fp32_subnormals += (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0;
      }
    };
    count_loaded_float(layer_weights.norm1);
    count_loaded_float(layer_weights.norm2);
    count_loaded_float(layer_weights.scale1);
    count_loaded_float(layer_weights.scale2);
    count_loaded_float(layer_weights.qkv_bias);
    count_loaded_float(layer_weights.out_bias);
    count_loaded_float(layer_weights.w1_bias);
    count_loaded_float(layer_weights.w2_bias);
    real_cuda_graph.load_layer(layer, layer_weights.view());
    real_vk_graph.load_layer(layer, layer_weights.view());
  }
  const double graph_load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - graph_load_begin)
          .count();
  CHECK(graph_raw_subnormals == 8495330u);
  CHECK(graph_loaded_subnormals == 0);
  CHECK(graph_loaded_fp32_subnormals == 0);
  CHECK(real_cuda_graph.persistent_bytes() == uint64_t(kGraphLayers) * real_weights.bytes());
  CHECK(real_vk_graph.persistent_bytes() == uint64_t(kGraphLayers) * real_weights.bytes());

  cuda::Stream graph_cuda_stream;
  cuda::DeviceBuffer<float> graph_cuda_tokens(real_input.size()),
      graph_cuda_cosine(real_cosine.size()), graph_cuda_sine(real_sine.size());
  graph_cuda_tokens.copy_from_host(real_input.data(), real_input.size(), graph_cuda_stream.get());
  graph_cuda_cosine.copy_from_host(real_cosine.data(), real_cosine.size(), graph_cuda_stream.get());
  graph_cuda_sine.copy_from_host(real_sine.data(), real_sine.size(), graph_cuda_stream.get());
  const uint64_t graph_token_shape[] = {real_config.sequence, real_config.dim};
  const uint64_t graph_rope_shape[] = {real_config.sequence, real_config.rope_dim};
  vulkan::DeviceTensor graph_vk_tokens =
      graph_context.allocate(TensorLayout::contiguous(graph_token_shape, 2));
  vulkan::DeviceTensor graph_vk_cosine =
      graph_context.allocate(TensorLayout::contiguous(graph_rope_shape, 2));
  vulkan::DeviceTensor graph_vk_sine =
      graph_context.allocate(TensorLayout::contiguous(graph_rope_shape, 2));
  graph_context.upload(graph_vk_tokens, real_input.data(), real_input.size());
  graph_context.upload(graph_vk_cosine, real_cosine.data(), real_cosine.size());
  graph_context.upload(graph_vk_sine, real_sine.data(), real_sine.size());
  std::vector<float> graph_cuda_boundary(real_input.size()), graph_vk_boundary(real_input.size());
  std::array<uint64_t, kGraphLayers> boundary_fnv{};
  auto fnv_words = [](const std::vector<float>& values) {
    uint64_t digest = 1469598103934665603ull;
    for (float value : values) {
      uint32_t bits = 0;
      std::memcpy(&bits, &value, 4);
      for (int byte = 0; byte < 4; ++byte) {
        digest ^= (bits >> (byte * 8)) & 0xffu;
        digest *= 1099511628211ull;
      }
    }
    return digest;
  };
  const auto boundary_begin = std::chrono::steady_clock::now();
  for (uint32_t layer = 0; layer < kGraphLayers; ++layer) {
    real_cuda_graph.forward_layer_device(layer, graph_cuda_tokens.get(), graph_cuda_cosine.get(),
                                         graph_cuda_sine.get(), graph_cuda_stream.get());
    graph_cuda_tokens.copy_to_host(graph_cuda_boundary.data(), graph_cuda_boundary.size(),
                                   graph_cuda_stream.get());
    graph_cuda_stream.synchronize();
    vulkan::TensorBatch boundary_batch = graph_context.begin_batch();
    real_vk_graph.record_layer(layer, boundary_batch, graph_vk_tokens, graph_vk_cosine,
                               graph_vk_sine);
    boundary_batch.submit().wait();
    graph_context.download(graph_vk_tokens, graph_vk_boundary.data(), graph_vk_boundary.size());
    CHECK_MSG(std::memcmp(graph_cuda_boundary.data(), graph_vk_boundary.data(),
                          graph_vk_boundary.size() * sizeof(float)) == 0,
              "real VAE ViT graph boundary %u mismatch", layer);
    boundary_fnv[layer] = fnv_words(graph_vk_boundary);
  }
  constexpr std::array<uint64_t, kGraphLayers> kExpectedBoundaryFnv{
      0xc8a7ac3241effbb7ull, 0xe6bdd67a8d48bff8ull, 0xfe02227922d69136ull, 0x4a4e37dbdc31d38eull,
      0x904f376dd2c88994ull, 0x38f3a254dcaa4b58ull, 0x1adf972aa1b5ad86ull, 0xcf07b32f966af880ull,
      0x41216b0be919cf32ull, 0xde0123fcd7230526ull, 0x768a973a476b9f3dull, 0x39f8eec6518d09ecull,
      0xd96137acfd15c70bull, 0x258870835106c183ull, 0x2b9269a616840b60ull, 0x2f80754cca866626ull,
      0xfa59d4e2bb5827fcull, 0x9fce240021df0759ull, 0x569c14bf07d8f02full, 0x72148c477fbd0555ull,
      0xc2e6327af72e9180ull, 0x866d0f23de08c438ull, 0xa4af356bb581aa36ull, 0x9115b305a680b5a3ull,
      0x5d2f5f5b25608da1ull, 0x7dba3acd266c90d6ull, 0x1bf5ac0ceb451f2aull, 0xf3f0972ba2242d72ull,
      0xa00df63f64401882ull, 0x44c2c7fbe814ef3full, 0xf408320a5e9c4dc4ull, 0xbb6a8a106ede9fdbull,
      0x2b0694c284a43fdfull, 0x5ece5abf0e1457b4ull, 0x5cfe899e8a5da408ull, 0x50d92f167ac90922ull};
  CHECK(boundary_fnv == kExpectedBoundaryFnv);
  const double boundary_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - boundary_begin)
          .count();

  graph_cuda_tokens.copy_from_host(real_input.data(), real_input.size(), graph_cuda_stream.get());
  graph_context.upload(graph_vk_tokens, real_input.data(), real_input.size());
  const auto cuda_graph_begin = std::chrono::steady_clock::now();
  real_cuda_graph.forward_device(graph_cuda_tokens.get(), graph_cuda_cosine.get(),
                                 graph_cuda_sine.get(), graph_cuda_stream.get());
  graph_cuda_tokens.copy_to_host(graph_cuda_boundary.data(), graph_cuda_boundary.size(),
                                 graph_cuda_stream.get());
  graph_cuda_stream.synchronize();
  const double cuda_graph_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_graph_begin)
          .count();
  const auto vk_graph_begin = std::chrono::steady_clock::now();
  vulkan::TensorBatch graph_batch = graph_context.begin_batch();
  real_vk_graph.record(graph_batch, graph_vk_tokens, graph_vk_cosine, graph_vk_sine);
  graph_batch.submit().wait();
  const double vk_graph_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vk_graph_begin)
          .count();
  graph_context.download(graph_vk_tokens, graph_vk_boundary.data(), graph_vk_boundary.size());
  CHECK(std::memcmp(graph_cuda_boundary.data(), graph_vk_boundary.data(),
                    graph_vk_boundary.size() * sizeof(float)) == 0);
  CHECK(fnv_words(graph_vk_boundary) == boundary_fnv.back());

  const uint64_t full_graph_reserved = graph_context.reserved_bytes();
  const uint64_t full_graph_used = graph_context.pooled_used_bytes();
  const uint64_t full_graph_descriptors = graph_context.descriptor_set_allocations();
  graph_context.upload(graph_vk_tokens, real_input.data(), real_input.size());
  vulkan::TensorBatch repeat_batch = graph_context.begin_batch();
  real_vk_graph.record(repeat_batch, graph_vk_tokens, graph_vk_cosine, graph_vk_sine);
  repeat_batch.submit().wait();
  CHECK(graph_context.reserved_bytes() == full_graph_reserved);
  CHECK(graph_context.pooled_used_bytes() == full_graph_used);
  CHECK(graph_context.descriptor_set_allocations() == full_graph_descriptors);
  std::printf(
      "  real 36-block graph: load %.1f ms, boundary replay %.1f ms, CUDA %.1f ms, Vulkan %.1f ms, raw subnormals %zu, final FNV64 %016llx, persistent %.1f MiB, Vulkan peak %.1f MiB, pool used/reserved %.1f/%.1f MiB, descriptors %llu\n",
      graph_load_ms, boundary_ms, cuda_graph_ms, vk_graph_ms, graph_raw_subnormals,
      static_cast<unsigned long long>(boundary_fnv.back()),
      double(real_vk_graph.persistent_bytes()) / 1048576.0,
      double(real_vk_graph.peak_device_bytes()) / 1048576.0, double(full_graph_used) / 1048576.0,
      double(full_graph_reserved) / 1048576.0,
      static_cast<unsigned long long>(full_graph_descriptors));
  std::printf("  graph boundary FNV64:");
  for (uint64_t digest : boundary_fnv)
    std::printf(" %016llx", static_cast<unsigned long long>(digest));
  std::printf("\n");
}
