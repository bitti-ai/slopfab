#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_linear_weight_real_nvfp4_slab, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const std::filesystem::path path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  int cuda_devices = 0;
  if (!std::filesystem::exists(path) || cudaGetDeviceCount(&cuda_devices) != cudaSuccess ||
      cuda_devices == 0 || !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(path) || cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  const std::string prefix = "blocks.0.attn.qkv_proj";
  const TensorView& stored = checkpoint.at(prefix + ".weight");
  const TensorView& scale = checkpoint.at(prefix + ".weight_scale");
  const TensorView& global_view = checkpoint.at(prefix + ".weight_scale_2");
  CHECK(stored.shape.size() == 2);
  constexpr uint32_t slab_out = 384;
  const uint32_t full_out = static_cast<uint32_t>(stored.shape[0]);
  const uint32_t in = static_cast<uint32_t>(stored.shape[1] * 2);
  CHECK(full_out >= slab_out && full_out % 128 == 0 && in % 64 == 0);
  const size_t elements = static_cast<size_t>(slab_out) * in;
  const size_t stored_bytes = elements / 2;
  const size_t scale_bytes = elements / 16;
  CHECK(stored.nbytes >= stored_bytes && scale.nbytes >= scale_bytes);
  float global = 0.0f;
  std::memcpy(&global, global_view.data, sizeof(global));

  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kNVFloat4;
  upload.out_features = slab_out;
  upload.in_features = in;
  upload.data = stored.data;
  upload.data_bytes = stored_bytes;
  upload.block_scale = static_cast<const uint8_t*>(scale.data);
  upload.block_scale_count = scale_bytes;
  upload.global_scale = global;
  const auto upload_begin = std::chrono::steady_clock::now();
  LinearWeight weight = LinearWeight::upload(vk, upload);
  const double upload_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - upload_begin)
          .count();
  CHECK(weight.stored_bytes() == stored_bytes);
  CHECK(weight.resident_bytes() == stored_bytes + scale_bytes);
  const uint64_t shape[] = {slab_out, in};
  DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);

  cuda::DeviceBuffer<uint8_t> cuda_stored(stored_bytes), cuda_scale(scale_bytes);
  cuda::DeviceBuffer<uint16_t> cuda_dense(elements);
  cuda_stored.copy_from_host(static_cast<const uint8_t*>(stored.data), stored_bytes);
  cuda_scale.copy_from_host(static_cast<const uint8_t*>(scale.data), scale_bytes);
  cuda::launch_dequant_nvfp4(cuda_stored.get(), cuda_scale.get(), global,
                             reinterpret_cast<__nv_bfloat16*>(cuda_dense.get()), slab_out, in,
                             nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  {
    TensorBatch batch = vk.begin_batch();
    weight.materialize_bf16(batch, dense);
    batch.submit().wait();
  }
  std::vector<uint16_t> cuda_bits(elements), vulkan_bits(elements);
  cuda_dense.copy_to_host(cuda_bits.data(), elements);
  vk.download_bytes(dense, vulkan_bits.data(), elements * sizeof(uint16_t));
  CHECK(std::memcmp(cuda_bits.data(), vulkan_bits.data(), elements * sizeof(uint16_t)) == 0);

  constexpr int iterations = 20;
  auto cuda_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    cuda::launch_dequant_nvfp4(cuda_stored.get(), cuda_scale.get(), global,
                               reinterpret_cast<__nv_bfloat16*>(cuda_dense.get()), slab_out, in,
                               nullptr);
  }
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_begin)
          .count() /
      iterations;
  auto vulkan_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    TensorBatch batch = vk.begin_batch();
    weight.materialize_bf16(batch, dense);
    batch.submit().wait();
  }
  const double vulkan_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vulkan_begin)
          .count() /
      iterations;
  std::printf("  real NVFP4 slab %ux%u: upload %.3f ms, CUDA %.3f ms, "
              "Vulkan %.3f ms, persistent %.2f MiB, dense %.2f MiB\n",
              slab_out, in, upload_ms, cuda_ms, vulkan_ms,
              static_cast<double>(weight.resident_bytes()) / 1048576.0,
              static_cast<double>(elements * 2) / 1048576.0);
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_streamed_nvfp4_gemm_real_slab, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const std::filesystem::path path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  int cuda_devices = 0;
  if (!std::filesystem::exists(path) || cudaGetDeviceCount(&cuda_devices) != cudaSuccess ||
      cuda_devices == 0 || !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(path) || cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().cooperative_matrix_bf16_f32_16x16x16) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore || !physical.front().info().cooperative_matrix_bf16_f32_16x16x16");
    return;
  }
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  const std::string prefix = "blocks.0.attn.qkv_proj";
  const TensorView& stored = checkpoint.at(prefix + ".weight");
  const TensorView& scale = checkpoint.at(prefix + ".weight_scale");
  const TensorView& global_view = checkpoint.at(prefix + ".weight_scale_2");
  constexpr uint32_t n = 384, k = 5376, rows = 66, output_rows = 68;
  const size_t weight_elements = size_t(n) * k;
  const size_t stored_bytes = weight_elements / 2;
  const size_t scale_bytes = weight_elements / 16;
  CHECK(stored.nbytes >= stored_bytes && scale.nbytes >= scale_bytes);
  float global = 0.0f;
  std::memcpy(&global, global_view.data, sizeof(global));

  std::vector<uint16_t> input(size_t(rows) * k), bias(n);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = f32_to_bf16(float(int(i % 29) - 14) / 32.0f);
  for (uint32_t i = 0; i < n; ++i)
    bias[i] = f32_to_bf16(float(int(i % 17) - 8) / 64.0f);
  constexpr uint16_t sentinel = 0x7fc1;
  std::vector<uint16_t> initial(size_t(output_rows) * n, sentinel);

  cuda::DeviceBuffer<uint8_t> cw(stored_bytes), cs(scale_bytes);
  cuda::DeviceBuffer<uint16_t> cdense(weight_elements), ci(input.size()), cbias(bias.size()),
      co(initial.size());
  cw.copy_from_host(static_cast<const uint8_t*>(stored.data), stored_bytes);
  cs.copy_from_host(static_cast<const uint8_t*>(scale.data), scale_bytes);
  ci.copy_from_host(input.data(), input.size());
  cbias.copy_from_host(bias.data(), bias.size());
  co.copy_from_host(initial.data(), initial.size());
  auto cuda_run = [&] {
    cuda::launch_dequant_nvfp4(cw.get(), cs.get(), global,
                               reinterpret_cast<__nv_bfloat16*>(cdense.get()), n, k, nullptr);
    cuda::launch_deterministic_bf16_gemm_nt(reinterpret_cast<const __nv_bfloat16*>(ci.get()),
                                            reinterpret_cast<const __nv_bfloat16*>(cdense.get()),
                                            cbias.get(), reinterpret_cast<__nv_bfloat16*>(co.get()),
                                            64, n, k, DenseGemmBias::kBFloat16);
    cuda::launch_deterministic_scalar_gemm_nt(ci.get(), cdense.get(), cbias.get(), co.get(), 2, n,
                                              k, DenseGemmMode::kBFloat16, DenseGemmBias::kBFloat16,
                                              64, 64);
  };
  cuda_run();
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> cuda_output(initial.size());
  co.copy_to_host(cuda_output.data(), cuda_output.size());

  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = true;
  options.enable_storage_buffer_16bit = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kNVFloat4;
  upload.out_features = n;
  upload.in_features = k;
  upload.data = stored.data;
  upload.data_bytes = stored_bytes;
  upload.block_scale = static_cast<const uint8_t*>(scale.data);
  upload.block_scale_count = scale_bytes;
  upload.global_scale = global;
  LinearWeight weight = LinearWeight::upload(context, upload);
  const uint64_t input_shape[] = {rows, k}, output_shape[] = {output_rows, n};
  const uint64_t bias_shape[] = {n};
  DeviceTensor vi =
      context.allocate(TensorLayout::contiguous(input_shape, 2), ScalarType::kBFloat16);
  DeviceTensor vo =
      context.allocate(TensorLayout::contiguous(output_shape, 2), ScalarType::kBFloat16);
  DeviceTensor vb =
      context.allocate(TensorLayout::contiguous(bias_shape, 1), ScalarType::kBFloat16);
  context.upload_bytes(vi, input.data(), input.size() * 2);
  context.upload_bytes(vo, initial.data(), initial.size() * 2);
  context.upload_bytes(vb, bias.data(), bias.size() * 2);
  DenseGemmPlan plan = DenseGemmPlan::create(
      context, {64, n, k, DenseGemmMode::kBFloat16, DenseGemmBias::kBFloat16});
  StreamedNVFP4WeightCache cache = StreamedNVFP4WeightCache::create(context, weight_elements);
  auto vulkan_run = [&] {
    TensorBatch batch = context.begin_batch();
    PreparedNVFP4WeightView prepared = cache.prepare(batch, weight, plan);
    plan.record(batch, vi, prepared, vo, 64, 0, 0, &vb);
    plan.record(batch, vi, prepared, vo, 2, 64, 64, &vb);
    return batch.submit();
  };
  vulkan_run().wait();
  std::vector<uint16_t> vulkan_output(initial.size());
  context.download_bytes(vo, vulkan_output.data(), vulkan_output.size() * 2);
  size_t mismatch = vulkan_output.size();
  for (size_t i = 0; i < vulkan_output.size(); ++i) {
    if (cuda_output[i] != vulkan_output[i]) {
      mismatch = i;
      break;
    }
  }
  CHECK_MSG(mismatch == vulkan_output.size(),
            "streamed real NVFP4 GEMM mismatch at %zu: %04x != %04x", mismatch,
            mismatch == vulkan_output.size() ? 0u : cuda_output[mismatch],
            mismatch == vulkan_output.size() ? 0u : vulkan_output[mismatch]);

  constexpr int iterations = 10;
  cudaEvent_t begin = nullptr, end = nullptr;
  SLOPFAB_CUDA_CHECK(cudaEventCreate(&begin));
  SLOPFAB_CUDA_CHECK(cudaEventCreate(&end));
  SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i)
    cuda_run();
  SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
  SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
  float cuda_elapsed = 0.0f;
  SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&cuda_elapsed, begin, end));
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  const auto vk_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    vulkan_run().wait();
  const double vulkan_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vk_begin)
          .count() /
      iterations;
  std::printf("  streamed real NVFP4 dequant+GEMM %ux%ux%u: CUDA %.3f ms, "
              "Vulkan %.3f ms, compressed %.2f MiB, one dense slot %.2f MiB\n",
              rows, n, k, cuda_elapsed / iterations, vulkan_ms,
              double(weight.resident_bytes()) / 1048576.0, double(cache.dense_bytes()) / 1048576.0);
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_linear_weight_real_nf4_conv, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const std::filesystem::path path = "weights/vae/video_vae_nf4.safetensors";
  int cuda_devices = 0;
  if (!std::filesystem::exists(path) || cudaGetDeviceCount(&cuda_devices) != cudaSuccess ||
      cuda_devices == 0 || !Instance::available()) {
    SKIP_MISSING_FIXTURE(
        "unavailable prerequisite: !std::filesystem::exists(path) || cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  constexpr std::string_view state_suffix = ".quant_state.bitsandbytes__nf4";
  std::string name;
  uint64_t largest_elements = 0;
  for (const auto& entry : checkpoint.tensors()) {
    const std::string& candidate = entry.first;
    if (candidate.size() <= state_suffix.size() ||
        candidate.compare(candidate.size() - state_suffix.size(), state_suffix.size(),
                          state_suffix) != 0)
      continue;
    const std::string weight_name = candidate.substr(0, candidate.size() - state_suffix.size());
    const NF4State candidate_state = read_nf4_state(checkpoint, weight_name, "Vulkan weight test");
    uint64_t candidate_elements = 1;
    for (int64_t extent : candidate_state.shape)
      candidate_elements *= static_cast<uint64_t>(extent);
    if (candidate_elements > largest_elements) {
      largest_elements = candidate_elements;
      name = weight_name;
    }
  }
  CHECK(!name.empty());
  if (name.empty()) {
    SKIP_MISSING_FIXTURE("checkpoint has no compatible NF4 convolution");
    return;
  }
  const TensorView& stored = checkpoint.at(name);
  const TensorView& absmax = checkpoint.at(name + ".absmax");
  const TensorView& map = checkpoint.at(name + ".quant_map");
  const TensorView& nested_map = checkpoint.at(name + ".nested_quant_map");
  const TensorView& nested_absmax = checkpoint.at(name + ".nested_absmax");
  const NF4State state = read_nf4_state(checkpoint, name, "Vulkan weight test");
  CHECK(state.shape.size() >= 2 && state.shape[0] > 0);
  uint64_t elements = 1;
  for (int64_t extent : state.shape)
    elements *= static_cast<uint64_t>(extent);
  const uint32_t out = static_cast<uint32_t>(state.shape[0]);
  const uint32_t in = static_cast<uint32_t>(elements / out);
  CHECK(elements == static_cast<uint64_t>(out) * in && stored.nbytes * 2 == elements);

  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kNF4;
  upload.out_features = out;
  upload.in_features = in;
  upload.data = stored.data;
  upload.data_bytes = stored.nbytes;
  upload.nf4_absmax = static_cast<const uint8_t*>(absmax.data);
  upload.nf4_absmax_count = absmax.nbytes;
  upload.nf4_quant_map = static_cast<const float*>(map.data);
  upload.nf4_quant_map_count = static_cast<uint64_t>(map.numel());
  upload.nf4_nested_quant_map = static_cast<const float*>(nested_map.data);
  upload.nf4_nested_quant_map_count = static_cast<uint64_t>(nested_map.numel());
  upload.nf4_nested_absmax = static_cast<const float*>(nested_absmax.data);
  upload.nf4_nested_absmax_count = static_cast<uint64_t>(nested_absmax.numel());
  upload.nf4_block_size = state.block_size;
  upload.nf4_nested_block_size = state.nested_block_size;
  upload.nf4_nested_offset = state.nested_offset;
  const auto upload_begin = std::chrono::steady_clock::now();
  LinearWeight weight = LinearWeight::upload(vk, upload);
  const double upload_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - upload_begin)
          .count();
  const uint64_t shape[] = {out, in};
  DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kFloat16);

  cuda::DeviceBuffer<uint8_t> d_stored(stored.nbytes), d_absmax(absmax.nbytes);
  cuda::DeviceBuffer<float> d_map(map.numel()), d_nested_map(nested_map.numel()),
      d_nested_absmax(nested_absmax.numel());
  cuda::DeviceBuffer<uint16_t> d_dense(elements);
  d_stored.copy_from_host(static_cast<const uint8_t*>(stored.data), stored.nbytes);
  d_absmax.copy_from_host(static_cast<const uint8_t*>(absmax.data), absmax.nbytes);
  d_map.copy_from_host(static_cast<const float*>(map.data), map.numel());
  d_nested_map.copy_from_host(static_cast<const float*>(nested_map.data), nested_map.numel());
  d_nested_absmax.copy_from_host(static_cast<const float*>(nested_absmax.data),
                                 nested_absmax.numel());
  cuda::launch_dequant_nf4_f16(d_stored.get(), d_absmax.get(), d_map.get(), d_nested_map.get(),
                               d_nested_absmax.get(), state.block_size, state.nested_block_size,
                               state.nested_offset, reinterpret_cast<__half*>(d_dense.get()),
                               elements, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  {
    TensorBatch batch = vk.begin_batch();
    weight.materialize_f16(batch, dense);
    batch.submit().wait();
  }
  std::vector<uint16_t> cuda_bits(elements), vulkan_bits(elements);
  d_dense.copy_to_host(cuda_bits.data(), elements);
  vk.download_bytes(dense, vulkan_bits.data(), elements * sizeof(uint16_t));
  CHECK(std::memcmp(cuda_bits.data(), vulkan_bits.data(), elements * 2) == 0);

  constexpr int iterations = 10;
  auto cuda_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    cuda::launch_dequant_nf4_f16(d_stored.get(), d_absmax.get(), d_map.get(), d_nested_map.get(),
                                 d_nested_absmax.get(), state.block_size, state.nested_block_size,
                                 state.nested_offset, reinterpret_cast<__half*>(d_dense.get()),
                                 elements, nullptr);
  }
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_begin)
          .count() /
      iterations;
  auto vulkan_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    TensorBatch batch = vk.begin_batch();
    weight.materialize_f16(batch, dense);
    batch.submit().wait();
  }
  const double vulkan_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vulkan_begin)
          .count() /
      iterations;
  std::printf("  real NF4 conv %ux%u: upload %.3f ms, CUDA %.3f ms, "
              "Vulkan %.3f ms, persistent %.2f MiB, dense %.2f MiB\n",
              out, in, upload_ms, cuda_ms, vulkan_ms,
              static_cast<double>(weight.resident_bytes()) / 1048576.0,
              static_cast<double>(elements * 2) / 1048576.0);
}
