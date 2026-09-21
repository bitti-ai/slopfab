#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_bf16_gemm_5376_baseline, "benchmark") {
  constexpr int m = 64, n = 5376, k = 5376;
  slopfab::cuda::DeviceBuffer<__nv_bfloat16> a(size_t(m) * k), w(size_t(n) * k), c(size_t(m) * n);
  SLOPFAB_CUDA_CHECK(cudaMemset(a.get(), 0, size_t(m) * k * sizeof(__nv_bfloat16)));
  SLOPFAB_CUDA_CHECK(cudaMemset(w.get(), 0, size_t(n) * k * sizeof(__nv_bfloat16)));
  cublasHandle_t handle = nullptr;
  SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_create(&handle));
  const float alpha = 1.0f, beta = 0.0f;
  auto launch = [&] {
    SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_gemm_ex(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w.get(), CUDA_R_16BF, k, a.get(),
        CUDA_R_16BF, k, &beta, c.get(), CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  launch();
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin = nullptr, end = nullptr;
  SLOPFAB_CUDA_CHECK(cudaEventCreate(&begin));
  SLOPFAB_CUDA_CHECK(cudaEventCreate(&end));
  SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < 20; ++i)
    launch();
  SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
  SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed = 0.0f;
  SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
  std::printf("  cuBLAS BF16 GEMM 64x5376x5376: %.3f ms\n", elapsed / 20.0f);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  slopfab::cuda::cublas_destroy(handle);
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_dense_gemm_production_timing, "benchmark") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_shader_float16 = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);

  cublasHandle_t handle = nullptr;
  SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_create(&handle));
  cudaEvent_t begin = nullptr, end = nullptr;
  SLOPFAB_CUDA_CHECK(cudaEventCreate(&begin));
  SLOPFAB_CUDA_CHECK(cudaEventCreate(&end));
  const float alpha = 1.0f, beta = 0.0f;

  {
    constexpr uint32_t m = 64, n = 6144, k = 2048;
    cuda::DeviceBuffer<float> caf(size_t(m) * k);
    cuda::DeviceBuffer<__half> ca(size_t(m) * k), cw(size_t(n) * k);
    cuda::DeviceBuffer<float> co(size_t(m) * n);
    SLOPFAB_CUDA_CHECK(cudaMemset(caf.get(), 0, size_t(m) * k * 4));
    SLOPFAB_CUDA_CHECK(cudaMemset(ca.get(), 0, size_t(m) * k * 2));
    SLOPFAB_CUDA_CHECK(cudaMemset(cw.get(), 0, size_t(n) * k * 2));
    auto cuda_launch = [&] {
      SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_gemm_ex(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, cw.get(), CUDA_R_16F, k, ca.get(),
          CUDA_R_16F, k, &beta, co.get(), CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    cuda_launch();
    SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
    for (int i = 0; i < 20; ++i)
      cuda_launch();
    SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
    SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
    float cuda_ms = 0;
    SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, begin, end));
    cuda_ms /= 20.0f;
    auto cuda_total_launch = [&] {
      cuda::launch_narrow_f16(caf.get(), ca.get(), size_t(m) * k, nullptr);
      cuda_launch();
    };
    cuda_total_launch();
    SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
    for (int i = 0; i < 20; ++i)
      cuda_total_launch();
    SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
    SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
    float cuda_total_ms = 0;
    SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&cuda_total_ms, begin, end));
    cuda_total_ms /= 20.0f;

    const uint64_t as[] = {m, k}, ws[] = {n, k}, os[] = {m, n};
    DeviceTensor a = context.allocate(TensorLayout::contiguous(as, 2), ScalarType::kFloat32);
    DeviceTensor w = context.allocate(TensorLayout::contiguous(ws, 2), ScalarType::kFloat16);
    DeviceTensor o = context.allocate(TensorLayout::contiguous(os, 2), ScalarType::kFloat32);
    std::vector<float> ah(size_t(m) * k, 0.0f);
    std::vector<uint16_t> wh(size_t(n) * k, 0);
    context.upload(a, ah.data(), ah.size());
    context.upload_bytes(w, wh.data(), wh.size() * 2);
    PreparedF16Activation slot = PreparedF16Activation::create(context, m, k);
    DenseGemmPlan plan =
        DenseGemmPlan::create(context, {m, n, k, DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone});
    double prepare_ms = 0.0;
    for (int i = -2; i < 10; ++i) {
      const auto start = std::chrono::steady_clock::now();
      TensorBatch batch = context.begin_batch();
      (void)slot.prepare(batch, a, m);
      batch.submit().wait();
      if (i >= 0)
        prepare_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
    }
    prepare_ms /= 10.0;
    double vulkan_total_ms = 0.0;
    for (int i = -2; i < 10; ++i) {
      const auto total_start = std::chrono::steady_clock::now();
      TensorBatch total_batch = context.begin_batch();
      PreparedF16ActivationView total_prepared = slot.prepare(total_batch, a, m);
      plan.record(total_batch, total_prepared, w, o);
      total_batch.submit().wait();
      if (i >= 0)
        vulkan_total_ms += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - total_start)
                               .count();
    }
    vulkan_total_ms /= 10.0;
    const auto start = std::chrono::steady_clock::now();
    TensorBatch batch = context.begin_batch();
    PreparedF16ActivationView prepared = slot.prepare(batch, a, m);
    for (int i = 0; i < 16; ++i)
      plan.record(batch, prepared, w, o);
    batch.submit().wait();
    const double vulkan_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count() /
        16.0;
    std::printf("  F16 VAE M64 N6144 K2048: GEMM CUDA %.3f ms/Vulkan %.3f ms, "
                "narrow+GEMM CUDA %.3f ms/Vulkan %.3f ms, prepare %.3f ms\n",
                cuda_ms, vulkan_ms, cuda_total_ms, vulkan_total_ms, prepare_ms);
  }

  {
    constexpr uint32_t m = 64, n = 2048, k = 2048;
    cuda::DeviceBuffer<float> ca(size_t(m) * k), cw(size_t(n) * k), co(size_t(m) * n);
    SLOPFAB_CUDA_CHECK(cudaMemset(ca.get(), 0, size_t(m) * k * 4));
    SLOPFAB_CUDA_CHECK(cudaMemset(cw.get(), 0, size_t(n) * k * 4));
    auto cuda_launch = [&] {
      SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_sgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                                       &alpha, cw.get(), k, ca.get(), k, &beta,
                                                       co.get(), n));
    };
    cuda_launch();
    SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
    for (int i = 0; i < 20; ++i)
      cuda_launch();
    SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
    SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
    float cuda_ms = 0;
    SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, begin, end));
    cuda_ms /= 20.0f;

    const uint64_t as[] = {m, k}, ws[] = {n, k}, os[] = {m, n};
    DeviceTensor a = context.allocate(TensorLayout::contiguous(as, 2));
    DeviceTensor w = context.allocate(TensorLayout::contiguous(ws, 2));
    DeviceTensor o = context.allocate(TensorLayout::contiguous(os, 2));
    std::vector<float> ah(size_t(m) * k, 0.0f), wh(size_t(n) * k, 0.0f);
    context.upload(a, ah.data(), ah.size());
    context.upload(w, wh.data(), wh.size());
    DenseGemmPlan plan =
        DenseGemmPlan::create(context, {m, n, k, DenseGemmMode::kFloat32, DenseGemmBias::kNone});
    const auto start = std::chrono::steady_clock::now();
    TensorBatch batch = context.begin_batch();
    for (int i = 0; i < 8; ++i)
      plan.record(batch, a, w, o, m);
    batch.submit().wait();
    const double vulkan_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count() /
        8.0;
    std::printf("  F32 SGEMM M64 N2048 K2048: CUDA %.3f ms, Vulkan %.3f ms\n", cuda_ms, vulkan_ms);
  }
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  slopfab::cuda::cublas_destroy(handle);
}
