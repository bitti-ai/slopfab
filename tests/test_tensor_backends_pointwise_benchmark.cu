#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_vae_pointwise_real_timing, "benchmark") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!std::getenv("SLOPFAB_VAE_POINTWISE_BENCH")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_VAE_POINTWISE_BENCH\")");
    return;
  }
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }

  constexpr int rows = 1797;
  constexpr int columns = 2048;
  constexpr int inner = 8192;
  constexpr int channels = 24;
  constexpr int voxels = 1792;
  constexpr int repeats = 8;
  const size_t matrix_count = size_t(rows) * columns;
  const size_t swiglu_input_count = size_t(rows) * 2 * inner;
  const size_t swiglu_output_count = size_t(rows) * inner;
  const size_t latent_count = size_t(channels) * voxels;
  std::vector<float> x(matrix_count), y(matrix_count), bias(columns),
      scale(columns), swiglu_input(swiglu_input_count),
      swiglu_bias(2 * inner), latent(latent_count), mean(channels),
      std_dev(channels);
  for (size_t index = 0; index < matrix_count; ++index) {
    x[index] = float(int(index % 127) - 63) / 128.0f;
    y[index] = float(int(index % 109) - 54) / 128.0f;
  }
  for (int index = 0; index < columns; ++index) {
    bias[index] = float((index % 31) - 15) / 256.0f;
    scale[index] = 0.01f + float(index % 13) / 1024.0f;
  }
  for (size_t index = 0; index < swiglu_input_count; ++index)
    swiglu_input[index] = float(int(index % 251) - 125) / 64.0f;
  for (size_t index = 0; index < swiglu_bias.size(); ++index)
    swiglu_bias[index] = float(int(index % 37) - 18) / 256.0f;
  for (size_t index = 0; index < latent_count; ++index)
    latent[index] = float(int(index % 97) - 48) / 32.0f;
  for (int index = 0; index < channels; ++index) {
    mean[index] = float(index - 12) / 64.0f;
    std_dev[index] = 0.5f + float(index % 7) / 16.0f;
  }

  cuda::DeviceBuffer<float> cx(matrix_count), cy(matrix_count), cbias(columns),
      cscale(columns), cinput(swiglu_input_count), csbias(2 * inner),
      clegacy(swiglu_output_count), cexact(swiglu_output_count),
      clatent(latent_count), cmean(channels), cstd(channels),
      cout(latent_count);
  cx.copy_from_host(x.data(), x.size());
  cy.copy_from_host(y.data(), y.size());
  cbias.copy_from_host(bias.data(), bias.size());
  cscale.copy_from_host(scale.data(), scale.size());
  cinput.copy_from_host(swiglu_input.data(), swiglu_input.size());
  csbias.copy_from_host(swiglu_bias.data(), swiglu_bias.size());
  clatent.copy_from_host(latent.data(), latent.size());
  cmean.copy_from_host(mean.data(), mean.size());
  cstd.copy_from_host(std_dev.data(), std_dev.size());

  auto time_cuda = [&](auto&& launch) {
    launch();
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t begin{}, end{};
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&begin));
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&end));
    SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
    for (int repeat = 0; repeat < repeats; ++repeat) launch();
    SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
    SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds / repeats;
  };
  const dim3 swiglu_grid((inner + 255) / 256, rows);
  const float legacy_swiglu_ms = time_cuda([&] {
    legacy_vae_swiglu_probe<<<swiglu_grid, 256>>>(
        cinput.get(), csbias.get(), clegacy.get(), inner);
    SLOPFAB_CUDA_CHECK(cudaGetLastError());
  });
  const float exact_swiglu_ms = time_cuda([&] {
    cuda::launch_swiglu(cinput.get(), csbias.get(), cexact.get(), rows, inner,
                        nullptr);
  });
  cx.copy_from_host(x.data(), x.size());
  const float residual_ms = time_cuda([&] {
    cuda::launch_layerscale_residual(cx.get(), cy.get(), cbias.get(),
                                     cscale.get(), rows, columns, nullptr);
  });
  const float denorm_ms = time_cuda([&] {
    cuda::launch_latent_denorm(clatent.get(), cmean.get(), cstd.get(),
                               cout.get(), channels, voxels, nullptr);
  });

  std::vector<float> legacy(swiglu_output_count), exact(swiglu_output_count);
  clegacy.copy_to_host(legacy.data(), legacy.size());
  cexact.copy_to_host(exact.data(), exact.size());
  size_t legacy_differences = 0;
  float legacy_max_absolute = 0.0f;
  for (size_t index = 0; index < exact.size(); ++index) {
    if (std::memcmp(&legacy[index], &exact[index], sizeof(float)) != 0)
      ++legacy_differences;
    legacy_max_absolute = std::max(
        legacy_max_absolute, std::abs(legacy[index] - exact[index]));
  }

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  CHECK(vk.exact_vae_pointwise());
  const uint64_t matrix_shape[] = {rows, columns};
  const uint64_t swiglu_input_shape[] = {rows, 2 * inner};
  const uint64_t swiglu_output_shape[] = {rows, inner};
  const uint64_t latent_shape[] = {channels, voxels};
  const uint64_t column_shape = columns, swiglu_bias_shape = 2 * inner,
                 channel_shape = channels;
  DeviceTensor vx = vk.allocate(TensorLayout::contiguous(matrix_shape, 2));
  DeviceTensor vy = vk.allocate(TensorLayout::contiguous(matrix_shape, 2));
  DeviceTensor vb = vk.allocate(TensorLayout::contiguous(&column_shape, 1));
  DeviceTensor vs = vk.allocate(TensorLayout::contiguous(&column_shape, 1));
  DeviceTensor vi = vk.allocate(TensorLayout::contiguous(swiglu_input_shape, 2));
  DeviceTensor vsb = vk.allocate(TensorLayout::contiguous(&swiglu_bias_shape, 1));
  DeviceTensor vo = vk.allocate(TensorLayout::contiguous(swiglu_output_shape, 2));
  DeviceTensor vl = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
  DeviceTensor vm = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
  DeviceTensor vsd = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
  DeviceTensor vlo = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
  vk.upload(vx, x.data(), x.size());
  vk.upload(vy, y.data(), y.size());
  vk.upload(vb, bias.data(), bias.size());
  vk.upload(vs, scale.data(), scale.size());
  vk.upload(vi, swiglu_input.data(), swiglu_input.size());
  vk.upload(vsb, swiglu_bias.data(), swiglu_bias.size());
  vk.upload(vl, latent.data(), latent.size());
  vk.upload(vm, mean.data(), mean.size());
  vk.upload(vsd, std_dev.data(), std_dev.size());
  auto time_vulkan = [&](auto&& record) {
    { TensorBatch warm = vk.begin_batch(); record(warm); warm.submit().wait(); }
    const auto begin = std::chrono::steady_clock::now();
    TensorBatch batch = vk.begin_batch();
    for (int repeat = 0; repeat < repeats; ++repeat) record(batch);
    batch.submit().wait();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - begin).count() /
           repeats;
  };
  const double vk_swiglu_ms = time_vulkan(
      [&](TensorBatch& batch) { batch.swiglu_bias_f32(vi, vsb, vo); });
  const double vk_residual_ms = time_vulkan([&](TensorBatch& batch) {
    batch.layer_scale_residual_f32(vx, vy, vb, vs);
  });
  const double vk_denorm_ms = time_vulkan([&](TensorBatch& batch) {
    batch.latent_denorm_f32(vl, vm, vsd, vlo);
  });
  std::vector<float> vulkan_swiglu(swiglu_output_count);
  vk.download(vo, vulkan_swiglu.data(), vulkan_swiglu.size());
  CHECK(std::memcmp(vulkan_swiglu.data(), exact.data(),
                    exact.size() * sizeof(float)) == 0);
  const double direct_mib =
      double((matrix_count * 2 + columns * 2 + swiglu_input_count +
              swiglu_output_count + swiglu_bias.size() + latent_count * 2 +
              channels * 2) *
             sizeof(float)) /
      1048576.0;
  std::printf(
      "  VAE pointwise R%d C%d I%d: SwiGLU old CUDA %.3f ms, exact CUDA %.3f ms, Vulkan %.3f ms (x36 %.1f/%.1f/%.1f ms); residual CUDA %.3f/Vulkan %.3f ms; denorm CUDA %.3f/Vulkan %.3f ms; old/exact drift %zu/%zu maxabs %.7g; direct %.1f MiB, scratch 0, reserved %.1f MiB\n",
      rows, columns, inner, legacy_swiglu_ms, exact_swiglu_ms, vk_swiglu_ms,
      legacy_swiglu_ms * 36.0f, exact_swiglu_ms * 36.0f,
      vk_swiglu_ms * 36.0, residual_ms, vk_residual_ms, denorm_ms,
      vk_denorm_ms, legacy_differences, exact.size(), legacy_max_absolute,
      direct_mib, double(vk.reserved_bytes()) / 1048576.0);
}
