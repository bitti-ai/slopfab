#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_h3_capture_replay, "checkpoint") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const char* path = std::getenv("SLOPFAB_H3_ATTENTION_CAPTURE");
  if (!path || !*path) {
    SKIP_OPT_IN("unavailable prerequisite: !path || !*path");
    return;
  }
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }

  SolCaptureHeader header{};
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
      std::memcmp(header.magic, "VFSOLQKV", 8) != 0 || header.version != 1 ||
      header.header_bytes != sizeof(header) || header.head_dim != 128) {
    throw std::runtime_error("invalid H3 capture replay header");
  }
  const uint64_t elements64 = uint64_t(header.seq_len) * header.num_heads * header.head_dim;
  if (header.tensor_elements != elements64 || elements64 > SIZE_MAX / 6)
    throw std::runtime_error("invalid H3 capture replay shape");
  const size_t count = static_cast<size_t>(elements64);
  std::vector<uint16_t> captured(count * 3);
  if (!input.read(reinterpret_cast<char*>(captured.data()),
                  static_cast<std::streamsize>(captured.size() * 2)) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("truncated H3 capture replay");
  }
  uint64_t input_hash = 1469598103934665603ull;
  uint64_t subnormal[3]{}, nonfinite[3]{};
  float maximum[3]{};
  for (size_t i = 0; i < captured.size(); ++i) {
    const uint16_t bits = captured[i];
    input_hash ^= bits & 0xffu;
    input_hash *= 1099511628211ull;
    input_hash ^= bits >> 8;
    input_hash *= 1099511628211ull;
    const size_t tensor = i / count;
    const uint16_t exponent = bits & 0x7f80u;
    subnormal[tensor] += exponent == 0 && (bits & 0x007fu) != 0;
    nonfinite[tensor] += exponent == 0x7f80u;
    if (exponent != 0x7f80u)
      maximum[tensor] = std::max(maximum[tensor], std::abs(bf16_to_f32(bits)));
  }

  cuda::DeviceBuffer<uint16_t> cq(count), ck(count), cv(count), exact_output(count),
      shipped_output(count);
  cq.copy_from_host(captured.data(), count);
  ck.copy_from_host(captured.data() + count, count);
  cv.copy_from_host(captured.data() + count * 2, count);
  auto cuda_time = [&](auto&& launch) {
    cudaEvent_t begin{}, end{};
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&begin));
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&end));
    SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
    launch();
    SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
    SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0;
    SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds;
  };
  const float scale = exact_attention_scale(header.head_dim);
  const float exact_cuda_ms = cuda_time([&] {
    cuda::launch_deterministic_h3_attention(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
        reinterpret_cast<const __nv_bfloat16*>(ck.get()),
        reinterpret_cast<const __nv_bfloat16*>(cv.get()),
        reinterpret_cast<__nv_bfloat16*>(exact_output.get()), nullptr, header.seq_len,
        header.num_heads, header.head_dim, scale);
  });
  cublasHandle_t blas{};
  SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_create(&blas));
  cuda::Workspace workspace;
  cuda::AttentionConfig config;
  config.seq_len = header.seq_len;
  config.num_heads = header.num_heads;
  config.head_dim = header.head_dim;
  config.scale = scale;
  const float shipped_ms = cuda_time([&] {
    cuda::attention_forward(blas, nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
                            reinterpret_cast<const __nv_bfloat16*>(ck.get()),
                            reinterpret_cast<const __nv_bfloat16*>(cv.get()),
                            reinterpret_cast<__nv_bfloat16*>(shipped_output.get()), config,
                            cuda::AttentionBackend::kFused, workspace);
  });
  slopfab::cuda::cublas_destroy(blas);
  std::vector<uint16_t> expected(count), shipped(count);
  exact_output.copy_to_host(expected.data(), count);
  shipped_output.copy_to_host(shipped.data(), count);
  size_t differences = 0;
  double error2 = 0, reference2 = 0;
  float max_abs = 0;
  uint64_t output_hash = 1469598103934665603ull;
  for (size_t i = 0; i < count; ++i) {
    output_hash ^= expected[i] & 0xffu;
    output_hash *= 1099511628211ull;
    output_hash ^= expected[i] >> 8;
    output_hash *= 1099511628211ull;
    differences += expected[i] != shipped[i];
    const double exact = bf16_to_f32(expected[i]);
    const double approximate = bf16_to_f32(shipped[i]);
    const double error = exact - approximate;
    error2 += error * error;
    reference2 += exact * exact;
    max_abs = std::max(max_abs, static_cast<float>(std::abs(error)));
  }

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty()");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  CHECK(vk.exact_h3_attention());
  const uint64_t shape[] = {header.seq_len, header.num_heads, header.head_dim};
  const TensorLayout layout = TensorLayout::contiguous(shape, 3);
  DeviceTensor q = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor k = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor v = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor output = vk.allocate(layout, ScalarType::kBFloat16);
  vk.upload_bytes(q, captured.data(), count * 2);
  vk.upload_bytes(k, captured.data() + count, count * 2);
  vk.upload_bytes(v, captured.data() + count * 2, count * 2);
  H3AttentionPlan plan =
      H3AttentionPlan::create(vk, {header.seq_len, header.num_heads, header.head_dim, scale});
  const auto begin = std::chrono::steady_clock::now();
  TensorBatch batch = vk.begin_batch();
  plan.record(batch, q, k, v, output);
  batch.submit().wait();
  const double vulkan_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  std::vector<uint16_t> got(count);
  vk.download_bytes(output, got.data(), count * 2);
  CHECK(got == expected);
  std::printf(
      "  H3 capture S%u H%u D%u step%d layer%d: input FNV64 %016llx, exact output %016llx; max Q/K/V %.7g/%.7g/%.7g, subnormal %llu/%llu/%llu, nonfinite %llu/%llu/%llu; exact CUDA %.3f ms Vulkan %.3f ms shipped %.3f ms, shipped drift %zu/%zu relL2 %.7g maxabs %.7g\n",
      header.seq_len, header.num_heads, header.head_dim, header.denoise_step, header.layer,
      static_cast<unsigned long long>(input_hash), static_cast<unsigned long long>(output_hash),
      maximum[0], maximum[1], maximum[2], static_cast<unsigned long long>(subnormal[0]),
      static_cast<unsigned long long>(subnormal[1]), static_cast<unsigned long long>(subnormal[2]),
      static_cast<unsigned long long>(nonfinite[0]), static_cast<unsigned long long>(nonfinite[1]),
      static_cast<unsigned long long>(nonfinite[2]), exact_cuda_ms, vulkan_ms, shipped_ms,
      differences, count, reference2 == 0 ? 0 : std::sqrt(error2 / reference2), max_abs);
}
