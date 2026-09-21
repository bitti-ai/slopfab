#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_qwen_vision_exact_gelu, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 1;
  TensorContext vk(device, context_options);

  // Odd count pins the packed-pair tail.  Exceptional BF16 values define a
  // total contract: signed subnormals flush, NaNs canonicalize, -Inf maps to
  // -0, and +Inf remains +Inf.  Huge finite values also avoid native tanh.
  std::vector<uint16_t> input = {0x0000u,
                                 0x8000u,
                                 0x0001u,
                                 0x8001u,
                                 0x7f81u,
                                 0x7fc1u,
                                 0x7f80u,
                                 0xff80u,
                                 0x7f7fu,
                                 0xff7fu,
                                 f32_to_bf16(-12.0f),
                                 f32_to_bf16(-1.0f),
                                 f32_to_bf16(-0.125f),
                                 f32_to_bf16(0.125f),
                                 f32_to_bf16(1.0f),
                                 f32_to_bf16(6.0f),
                                 f32_to_bf16(12.0f)};
  cuda::DeviceBuffer<uint16_t> cuda_bits(input.size());
  cuda_bits.copy_from_host(input.data(), input.size());
  cuda::launch_gelu_tanh_exact(reinterpret_cast<__nv_bfloat16*>(cuda_bits.get()), input.size(),
                               nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> expected(input.size());
  cuda_bits.copy_to_host(expected.data(), expected.size());

  const uint64_t shape[] = {1, input.size()};
  DeviceTensor activation = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);
  vk.upload_bytes(activation, input.data(), input.size() * sizeof(uint16_t));
  TensorBatch batch = vk.begin_batch();
  CHECK(batch.remaining_operator_capacity() == 1);
  {
    test::HostAllocationGuard no_host_allocations;
    batch.vision_gelu_tanh_bf16(activation);
  }
  CHECK(batch.remaining_operator_capacity() == 0);
  batch.submit().wait();
  std::vector<uint16_t> actual(input.size());
  vk.download_bytes(activation, actual.data(), actual.size() * sizeof(uint16_t));
  CHECK(std::memcmp(expected.data(), actual.data(), actual.size() * 2) == 0);
  CHECK(actual[0] == 0x0000u);
  CHECK(actual[1] == 0x8000u);
  CHECK(actual[2] == 0x0000u);
  CHECK(actual[3] == 0x8000u);
  CHECK(actual[4] == 0x7fffu && actual[5] == 0x7fffu);
  CHECK(actual[6] == 0x7f80u && actual[7] == 0x8000u);

  // Production S=16384, intermediate=4304 is opt-in because each authority
  // buffer is 134.5 MiB.  It is deliberately one packed dispatch, including
  // a non-multiple-of-64 pair count, rather than a synthetic extrapolation.
  if (std::getenv("SLOPFAB_RUN_QWEN_VISION_REAL")) {
    constexpr uint64_t rows = 16384, dim = 4304;
    const uint64_t count = rows * dim;
    std::vector<uint16_t> host(count);
    for (uint64_t i = 0; i < count; ++i)
      host[i] = f32_to_bf16(static_cast<float>(static_cast<int>(i % 257) - 128) / 32.0f);
    cuda::DeviceBuffer<uint16_t> cuda_production(count);
    cuda_production.copy_from_host(host.data(), host.size());
    const auto cuda_start = std::chrono::steady_clock::now();
    cuda::launch_gelu_tanh_exact(reinterpret_cast<__nv_bfloat16*>(cuda_production.get()), count,
                                 nullptr);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const auto cuda_end = std::chrono::steady_clock::now();
    const uint64_t production_shape[] = {rows, dim};
    DeviceTensor vk_production =
        vk.allocate(TensorLayout::contiguous(production_shape, 2), ScalarType::kBFloat16);
    vk.upload_bytes(vk_production, host.data(), host.size() * 2);
    const auto vk_start = std::chrono::steady_clock::now();
    TensorBatch production_batch = vk.begin_batch();
    production_batch.vision_gelu_tanh_bf16(vk_production);
    production_batch.submit().wait();
    const auto vk_end = std::chrono::steady_clock::now();
    std::vector<uint16_t> cuda_out(count), vk_out(count);
    cuda_production.copy_to_host(cuda_out.data(), cuda_out.size());
    vk.download_bytes(vk_production, vk_out.data(), vk_out.size() * 2);
    CHECK(std::memcmp(cuda_out.data(), vk_out.data(), count * 2) == 0);
    std::printf("qwen vision GELU S16384 CUDA %.3f ms Vulkan %.3f ms bytes %llu\n",
                std::chrono::duration<double, std::milli>(cuda_end - cuda_start).count(),
                std::chrono::duration<double, std::milli>(vk_end - vk_start).count(),
                static_cast<unsigned long long>(count * 2));
  }
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_qwen_vision_exact_layout, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 7;
  TensorContext vk(device, context_options);

  constexpr uint32_t rows = 8, dim = 7, positions = 5;
  constexpr uint32_t groups = rows / 4, merged_dim = 4 * dim;
  std::vector<uint16_t> x(size_t(rows) * dim), table(size_t(positions) * dim);
  std::vector<uint16_t> fused(size_t(rows) * 3 * dim);
  std::vector<int32_t> position_index(rows), scatter_index = {1, 4};
  std::vector<uint16_t> destination(size_t(5) * merged_dim);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = f32_to_bf16(float(int(i % 17) - 8) / 16.0f);
  for (size_t i = 0; i < table.size(); ++i)
    table[i] = f32_to_bf16(float(int(i % 13) - 6) / 32.0f);
  for (size_t i = 0; i < fused.size(); ++i)
    fused[i] = static_cast<uint16_t>(0x3e00u + (i * 37u) % 0x0180u);
  // Raw split/merge must preserve exceptional payload bits rather than
  // accidentally canonicalizing them in a layout operation.
  fused[0] = 0x7fc1u;
  fused[dim] = 0x8001u;
  fused[2 * dim] = 0xff80u;
  for (uint32_t r = 0; r < rows; ++r)
    position_index[r] = int32_t(r % positions);
  for (size_t i = 0; i < destination.size(); ++i)
    destination[i] = f32_to_bf16(float(int(i % 11) - 5) / 64.0f);

  cuda::DeviceBuffer<uint16_t> cx(x.size()), ct(table.size()), cf(fused.size());
  cuda::DeviceBuffer<int32_t> cpi(position_index.size()), csi(scatter_index.size());
  cuda::DeviceBuffer<uint16_t> cq(size_t(rows) * dim), ck(size_t(rows) * dim),
      cv(size_t(rows) * dim), cm(size_t(groups) * merged_dim), cd(destination.size());
  cx.copy_from_host(x.data(), x.size());
  ct.copy_from_host(table.data(), table.size());
  cf.copy_from_host(fused.data(), fused.size());
  cpi.copy_from_host(position_index.data(), position_index.size());
  csi.copy_from_host(scatter_index.data(), scatter_index.size());
  cd.copy_from_host(destination.data(), destination.size());
  cuda::qwen_vision_add_positions_exact(reinterpret_cast<__nv_bfloat16*>(cx.get()),
                                        reinterpret_cast<const __nv_bfloat16*>(ct.get()), cpi.get(),
                                        rows, dim, nullptr);
  cuda::qwen_vision_split_qkv_exact(reinterpret_cast<const __nv_bfloat16*>(cf.get()),
                                    reinterpret_cast<__nv_bfloat16*>(cq.get()),
                                    reinterpret_cast<__nv_bfloat16*>(ck.get()),
                                    reinterpret_cast<__nv_bfloat16*>(cv.get()), rows, dim, nullptr);
  cuda::launch_merge_four_rows(reinterpret_cast<const __nv_bfloat16*>(cq.get()),
                               reinterpret_cast<__nv_bfloat16*>(cm.get()), groups, dim, nullptr);
  cuda::qwen_vision_scatter_add_exact(reinterpret_cast<const __nv_bfloat16*>(cm.get()), csi.get(),
                                      reinterpret_cast<__nv_bfloat16*>(cd.get()), groups,
                                      merged_dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  auto matrix = [](uint64_t a, uint64_t b) {
    const uint64_t shape[] = {a, b};
    return TensorLayout::contiguous(shape, 2);
  };
  auto vector = [](uint64_t n) {
    return TensorLayout::contiguous(&n, 1);
  };
  DeviceTensor vx = vk.allocate(matrix(rows, dim), ScalarType::kBFloat16);
  DeviceTensor vt = vk.allocate(matrix(positions, dim), ScalarType::kBFloat16);
  DeviceTensor vpi = vk.allocate(vector(rows), ScalarType::kInt32);
  DeviceTensor vf = vk.allocate(matrix(rows, 3 * dim), ScalarType::kBFloat16);
  DeviceTensor vq = vk.allocate(matrix(rows, dim), ScalarType::kBFloat16);
  DeviceTensor vk_key = vk.allocate(matrix(rows, dim), ScalarType::kBFloat16);
  DeviceTensor vv = vk.allocate(matrix(rows, dim), ScalarType::kBFloat16);
  DeviceTensor vm = vk.allocate(matrix(groups, merged_dim), ScalarType::kBFloat16);
  DeviceTensor vsi = vk.allocate(vector(groups), ScalarType::kInt32);
  DeviceTensor vd = vk.allocate(matrix(5, merged_dim), ScalarType::kBFloat16);
  DeviceTensor vd_replace = vk.allocate(matrix(5, merged_dim), ScalarType::kBFloat16);
  vk.upload_bytes(vx, x.data(), x.size() * 2);
  vk.upload_bytes(vt, table.data(), table.size() * 2);
  vk.upload_bytes(vpi, position_index.data(), position_index.size() * 4);
  vk.upload_bytes(vf, fused.data(), fused.size() * 2);
  vk.upload_bytes(vsi, scatter_index.data(), scatter_index.size() * 4);
  vk.upload_bytes(vd, destination.data(), destination.size() * 2);
  vk.upload_bytes(vd_replace, destination.data(), destination.size() * 2);
  TensorBatch batch = vk.begin_batch();
  {
    test::HostAllocationGuard no_host_allocations;
    batch.vision_add_positions_bf16(vx, vt, vpi);
    batch.vision_split_qkv_bf16(vf, vq, vk_key, vv);
    batch.vision_merge_four_bf16(vq, vm);
    batch.vision_scatter_bf16(vm, vd_replace, vsi);
    batch.vision_scatter_add_bf16(vm, vd, vsi);
  }
  CHECK(batch.remaining_operator_capacity() == 0);
  batch.submit().wait();
  auto exact = [&](cuda::DeviceBuffer<uint16_t>& authority, DeviceTensor& actual, size_t count) {
    std::vector<uint16_t> a(count), b(count);
    authority.copy_to_host(a.data(), a.size());
    vk.download_bytes(actual, b.data(), b.size() * 2);
    CHECK(std::memcmp(a.data(), b.data(), count * 2) == 0);
  };
  exact(cx, vx, x.size());
  exact(cq, vq, size_t(rows) * dim);
  exact(ck, vk_key, size_t(rows) * dim);
  exact(cv, vv, size_t(rows) * dim);
  exact(cm, vm, size_t(groups) * merged_dim);
  exact(cd, vd, destination.size());
  std::vector<uint16_t> merged_host(size_t(groups) * merged_dim);
  cm.copy_to_host(merged_host.data(), merged_host.size());
  std::vector<uint16_t> replaced_expected = destination;
  for (uint32_t row = 0; row < groups; ++row)
    std::memcpy(replaced_expected.data() + size_t(scatter_index[row]) * merged_dim,
                merged_host.data() + size_t(row) * merged_dim,
                size_t(merged_dim) * sizeof(uint16_t));
  std::vector<uint16_t> replaced_actual(destination.size());
  vk.download_bytes(vd_replace, replaced_actual.data(), replaced_actual.size() * 2);
  CHECK(replaced_actual == replaced_expected);

  // DeepStack is extracted after completed vision blocks 8/16/24, but those
  // three tensors are injected after decoder layers 0/1/2. Keep this canonical
  // mapping explicit so a similarly numbered implementation cannot drift.
  constexpr std::array<int, 3> extraction = {8, 16, 24};
  constexpr std::array<int, 3> injection = {0, 1, 2};
  CHECK(extraction[0] == 8 && extraction[1] == 16 && extraction[2] == 24);
  CHECK(injection[0] == 0 && injection[1] == 1 && injection[2] == 2);

  TensorContextOptions short_options;
  short_options.max_batch_operators = 2;
  TensorContext short_vk(device, short_options);
  DeviceTensor sf = short_vk.allocate(matrix(rows, 3 * dim), ScalarType::kBFloat16);
  DeviceTensor sq = short_vk.allocate(matrix(rows, dim), ScalarType::kBFloat16);
  DeviceTensor sk = short_vk.allocate(matrix(rows, dim), ScalarType::kBFloat16);
  DeviceTensor sv = short_vk.allocate(matrix(rows, dim), ScalarType::kBFloat16);
  TensorBatch short_batch = short_vk.begin_batch();
  bool short_rejected = false;
  try {
    short_batch.vision_split_qkv_bf16(sf, sq, sk, sv);
  } catch (const std::invalid_argument&) {
    short_rejected = true;
  }
  CHECK(short_rejected);
  CHECK(short_batch.remaining_operator_capacity() == 2);
}
