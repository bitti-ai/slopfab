#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_tensor_exact_vae_norms, "synthetic") {
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
  TensorContext vk(device);
  if (!vk.exact_normalization()) {
    bool rejected = false;
    try {
      vk.require_exact_normalization();
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    CHECK(rejected);
    return;
  }
  vk.require_exact_normalization();

  auto run_case = [&](int rows, int dim, float epsilon, int pattern) {
    const size_t count = static_cast<size_t>(rows) * dim;
    std::vector<float> input(count), weight(dim), bias(dim);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<float>(static_cast<int>((i * 17) % 251) - 125) / 32.0f;
    for (int i = 0; i < dim; ++i) {
      weight[i] = 0.5f + static_cast<float>(i % 29) / 32.0f;
      bias[i] = static_cast<float>((i % 17) - 8) / 64.0f;
    }
    if (pattern == 1) {
      // Constant/zero-variance row and signed-zero values exercise degenerate
      // LayerNorm and sign preservation without leaving the advertised domain.
      for (int col = 0; col < dim; ++col)
        input[col] = 2.0f;
      for (int col = 0; col < dim; ++col)
        input[dim + col] = (col & 1) ? -0.0f : 0.0f;
    } else if (pattern == 2) {
      // Large values whose squares and reduction totals remain finite.
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = (i & 1) ? -1.0e16f : 1.0e16f;
    } else if (pattern == 3) {
      std::fill(input.begin(), input.end(), 0.0f);
    } else if (pattern == 4) {
      for (size_t i = 0; i < input.size(); ++i) {
        const uint32_t exponent = 97u + static_cast<uint32_t>(i % 61u);
        const uint32_t mantissa = static_cast<uint32_t>(i * 2654435761u) & 0x007fffffu;
        const uint32_t value_bits = exponent << 23u | mantissa;
        std::memcpy(&input[i], &value_bits, sizeof(value_bits));
      }
    }

    cuda::DeviceBuffer<float> c_input(count), c_weight(dim), c_bias(dim), c_rms(count),
        c_layer(count), c_rms_repeat(count), c_layer_repeat(count);
    c_input.copy_from_host(input.data(), input.size());
    c_weight.copy_from_host(weight.data(), weight.size());
    c_bias.copy_from_host(bias.data(), bias.size());
    cuda::launch_rmsnorm(c_input.get(), c_weight.get(), c_rms.get(), rows, dim, epsilon, nullptr);
    cuda::launch_layernorm(c_input.get(), c_weight.get(), c_bias.get(), c_layer.get(), rows, dim,
                           epsilon, nullptr);
    cuda::launch_rmsnorm(c_input.get(), c_weight.get(), c_rms_repeat.get(), rows, dim, epsilon,
                         nullptr);
    cuda::launch_layernorm(c_input.get(), c_weight.get(), c_bias.get(), c_layer_repeat.get(), rows,
                           dim, epsilon, nullptr);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> rms_once(count), rms_twice(count), layer_once(count), layer_twice(count);
    c_rms.copy_to_host(rms_once.data(), count);
    c_rms_repeat.copy_to_host(rms_twice.data(), count);
    c_layer.copy_to_host(layer_once.data(), count);
    c_layer_repeat.copy_to_host(layer_twice.data(), count);
    CHECK(std::memcmp(rms_once.data(), rms_twice.data(), count * sizeof(float)) == 0);
    CHECK(std::memcmp(layer_once.data(), layer_twice.data(), count * sizeof(float)) == 0);

    const uint64_t shape_extents[] = {static_cast<uint64_t>(rows), static_cast<uint64_t>(dim)};
    const uint64_t feature_extent = static_cast<uint64_t>(dim);
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(shape_extents, 2));
    DeviceTensor v_weight = vk.allocate(TensorLayout::contiguous(&feature_extent, 1));
    DeviceTensor v_bias = vk.allocate(TensorLayout::contiguous(&feature_extent, 1));
    DeviceTensor v_rms = vk.allocate(TensorLayout::contiguous(shape_extents, 2));
    DeviceTensor v_layer = vk.allocate(TensorLayout::contiguous(shape_extents, 2));
    vk.upload(v_input, input.data(), input.size());
    vk.upload(v_weight, weight.data(), weight.size());
    vk.upload(v_bias, bias.data(), bias.size());
    TensorBatch batch = vk.begin_batch();
    batch.rms_norm(v_input, v_weight, v_rms, epsilon);
    batch.layer_norm(v_input, v_weight, v_bias, v_layer, epsilon);
    batch.submit().wait();

    auto compare = [&](auto& cuda_buffer, DeviceTensor& vulkan_tensor, const char* label) {
      std::vector<float> cuda_host(count), vulkan_host(count);
      cuda_buffer.copy_to_host(cuda_host.data(), cuda_host.size());
      vk.download(vulkan_tensor, vulkan_host.data(), vulkan_host.size());
      size_t mismatch = count;
      for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(&cuda_host[i], &vulkan_host[i], sizeof(float)) != 0) {
          mismatch = i;
          break;
        }
      }
      uint32_t cuda_bits = 0, vulkan_bits = 0;
      if (mismatch != count) {
        std::memcpy(&cuda_bits, &cuda_host[mismatch], sizeof(cuda_bits));
        std::memcpy(&vulkan_bits, &vulkan_host[mismatch], sizeof(vulkan_bits));
      }
      CHECK_MSG(mismatch == count, "CUDA/Vulkan %s %dx%d mismatch at %zu: %08x != %08x", label,
                rows, dim, mismatch, cuda_bits, vulkan_bits);
    };
    compare(c_rms, v_rms, "VAE RMSNorm");
    compare(c_layer, v_layer, "VAE LayerNorm");
  };

  run_case(3, 513, 1.0e-6f, 0);  // odd scalar and workgroup tail
  run_case(2, 128, 1.0e-5f, 1);  // test-model width, signed zero/variance zero
  run_case(2, 2048, 1.0e-6f, 2); // shipped video-VAE width
  run_case(2, 7, 1.0e-6f, 1);    // dimension below a warp
  run_case(1, 9, std::numeric_limits<float>::min(), 3);
  run_case(4099, 1, 1.0e-6f, 4); // dense exponent/mantissa and dispatch tail

  // Host-known values outside the exact domain fail before recording and do
  // not poison an otherwise valid batch.
  const uint64_t extents[] = {2, 7};
  const uint64_t feature = 7;
  DeviceTensor input = vk.allocate(TensorLayout::contiguous(extents, 2));
  DeviceTensor weight = vk.allocate(TensorLayout::contiguous(&feature, 1));
  DeviceTensor bias = vk.allocate(TensorLayout::contiguous(&feature, 1));
  DeviceTensor rms_a = vk.allocate(TensorLayout::contiguous(extents, 2));
  DeviceTensor rms_b = vk.allocate(TensorLayout::contiguous(extents, 2));
  DeviceTensor layer_a = vk.allocate(TensorLayout::contiguous(extents, 2));
  DeviceTensor layer_b = vk.allocate(TensorLayout::contiguous(extents, 2));
  std::vector<float> values(14, 1.0f), affine(7, 1.0f), offsets(7, 0.25f);
  vk.upload(input, values.data(), values.size());
  vk.upload(weight, affine.data(), affine.size());
  vk.upload(bias, offsets.data(), offsets.size());
  TensorBatch recoverable = vk.begin_batch();
  bool subnormal_epsilon_rejected = false;
  try {
    recoverable.rms_norm(input, weight, rms_a, std::numeric_limits<float>::denorm_min());
  } catch (const std::invalid_argument&) {
    subnormal_epsilon_rejected = true;
  }
  CHECK(subnormal_epsilon_rejected);
  recoverable.rms_norm(input, weight, rms_a, 1.0e-6f);
  recoverable.submit().wait();

  // Warm both flight slots with the same two-pipeline sequence. Two jobs are
  // submitted before either token is waited, then a third begin exercises
  // bounded oldest-slot backpressure without queue/device idle.
  auto submit_pair = [&](DeviceTensor& rms, DeviceTensor& layer) {
    TensorBatch batch = vk.begin_batch();
    batch.rms_norm(input, weight, rms, 1.0e-6f);
    batch.layer_norm(input, weight, bias, layer, 1.0e-6f);
    return batch.submit();
  };
  Submission first = submit_pair(rms_a, layer_a);
  Submission second = submit_pair(rms_b, layer_b);
  CHECK(first.value() != 0 && second.value() > first.value());
  Submission third = submit_pair(rms_a, layer_a);
  third.wait();
  second.wait();
  const uint64_t warm_reserved = vk.reserved_bytes();
  const uint64_t warm_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 8; ++repeat) {
    Submission a = submit_pair(rms_a, layer_a);
    Submission b = submit_pair(rms_b, layer_b);
    a.wait();
    b.wait();
    CHECK(vk.reserved_bytes() == warm_reserved);
    CHECK(vk.descriptor_set_allocations() == warm_descriptors);
  }
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_tensor_exact_shared_norms, "synthetic") {
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
  TensorContext vk(device);
  if (!vk.exact_normalization()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_normalization()");
    return;
  }
  auto exact_bf16 = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
  };

  auto compare_bf16 = [&](const std::vector<uint16_t>& expected,
                          const std::vector<uint16_t>& actual, const char* label, int rows,
                          int dim) {
    size_t mismatch = expected.size();
    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != actual[i]) {
        mismatch = i;
        break;
      }
    }
    CHECK_MSG(mismatch == expected.size(), "CUDA/Vulkan %s %dx%d mismatch at %zu: %04x != %04x",
              label, rows, dim, mismatch, mismatch == expected.size() ? 0 : expected[mismatch],
              mismatch == expected.size() ? 0 : actual[mismatch]);
  };

  auto make_bf16_data = [&](size_t count, int pattern) {
    std::vector<uint16_t> values(count);
    for (size_t i = 0; i < count; ++i) {
      float value = static_cast<float>(static_cast<int>((i * 17) % 61) - 30) / 16.0f;
      if (pattern == 1 && i < 16)
        value = (i & 1) ? -0.0f : 0.0f;
      if (pattern == 2)
        value = (i & 1) ? -1.0e16f : 1.0e16f;
      values[i] = exact_bf16(value);
    }
    return values;
  };

  auto run_rms = [&](int rows, int dim, int pattern, bool in_place) {
    const size_t count = static_cast<size_t>(rows) * dim;
    auto input = make_bf16_data(count, pattern);
    std::vector<uint16_t> weight(dim);
    for (int i = 0; i < dim; ++i)
      weight[i] = exact_bf16(0.5f + static_cast<float>(i % 8) / 8.0f);
    cuda::DeviceBuffer<uint16_t> c_input(count), c_weight(dim), c_output(count), c_repeat(count);
    c_input.copy_from_host(input.data(), count);
    c_weight.copy_from_host(weight.data(), dim);
    cuda::launch_rmsnorm(reinterpret_cast<const __nv_bfloat16*>(c_input.get()),
                         reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
                         reinterpret_cast<__nv_bfloat16*>(c_output.get()), rows, dim, 1.0e-5f,
                         nullptr);
    cuda::launch_rmsnorm(reinterpret_cast<const __nv_bfloat16*>(c_input.get()),
                         reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
                         reinterpret_cast<__nv_bfloat16*>(c_repeat.get()), rows, dim, 1.0e-5f,
                         nullptr);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> expected(count), actual(count);
    c_output.copy_to_host(expected.data(), count);
    std::vector<uint16_t> repeated(count);
    c_repeat.copy_to_host(repeated.data(), count);
    CHECK(repeated == expected);
    const uint64_t extents[] = {static_cast<uint64_t>(rows), static_cast<uint64_t>(dim)};
    const uint64_t feature = dim;
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(extents, 2), ScalarType::kBFloat16);
    DeviceTensor v_weight =
        vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kBFloat16);
    DeviceTensor v_output =
        in_place ? DeviceTensor()
                 : vk.allocate(TensorLayout::contiguous(extents, 2), ScalarType::kBFloat16);
    vk.upload_bytes(v_input, input.data(), input.size() * sizeof(uint16_t));
    vk.upload_bytes(v_weight, weight.data(), weight.size() * sizeof(uint16_t));
    TensorBatch batch = vk.begin_batch();
    batch.rms_norm_bf16(v_input, v_weight, in_place ? v_input : v_output, 1.0e-5f);
    batch.submit().wait();
    DeviceTensor& result = in_place ? v_input : v_output;
    vk.download_bytes(result, actual.data(), actual.size() * sizeof(uint16_t));
    compare_bf16(expected, actual, "BF16 RMSNorm", rows, dim);
  };

  run_rms(9, 128, 1, true);   // head path, rows%8 and in-place
  run_rms(2, 5120, 0, false); // Qwen text width
  run_rms(3, 5376, 0, false); // DiT width and pack/workgroup tail
  run_rms(3, 31, 1, false);   // scalar narrow and odd packed rows
  run_rms(2, 513, 0, false);  // scalar block and odd packed rows

  auto run_layer = [&](int rows, int dim, bool constant_row, bool in_place = false) {
    const size_t count = static_cast<size_t>(rows) * dim;
    auto input = make_bf16_data(count, 0);
    if (constant_row)
      std::fill(input.begin(), input.begin() + dim, exact_bf16(2.0f));
    std::vector<uint16_t> weight(dim), bias(dim);
    for (int i = 0; i < dim; ++i) {
      weight[i] = exact_bf16(0.5f + static_cast<float>(i % 4) / 4.0f);
      bias[i] = exact_bf16(static_cast<float>((i % 9) - 4) / 16.0f);
    }
    cuda::DeviceBuffer<uint16_t> c_input(count), c_weight(dim), c_bias(dim), c_output(count),
        c_repeat(count);
    c_input.copy_from_host(input.data(), count);
    c_weight.copy_from_host(weight.data(), dim);
    c_bias.copy_from_host(bias.data(), dim);
    cuda::launch_layernorm_affine(reinterpret_cast<const __nv_bfloat16*>(c_input.get()),
                                  reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
                                  reinterpret_cast<const __nv_bfloat16*>(c_bias.get()),
                                  reinterpret_cast<__nv_bfloat16*>(c_output.get()), rows, dim,
                                  1.0e-6f, nullptr);
    cuda::launch_layernorm_affine(reinterpret_cast<const __nv_bfloat16*>(c_input.get()),
                                  reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
                                  reinterpret_cast<const __nv_bfloat16*>(c_bias.get()),
                                  reinterpret_cast<__nv_bfloat16*>(c_repeat.get()), rows, dim,
                                  1.0e-6f, nullptr);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> expected(count), actual(count);
    c_output.copy_to_host(expected.data(), count);
    std::vector<uint16_t> repeated(count);
    c_repeat.copy_to_host(repeated.data(), count);
    CHECK(repeated == expected);
    const uint64_t extents[] = {static_cast<uint64_t>(rows), static_cast<uint64_t>(dim)};
    const uint64_t feature = dim;
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(extents, 2), ScalarType::kBFloat16);
    DeviceTensor v_weight =
        vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kBFloat16);
    DeviceTensor v_bias = vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kBFloat16);
    DeviceTensor v_output =
        in_place ? DeviceTensor()
                 : vk.allocate(TensorLayout::contiguous(extents, 2), ScalarType::kBFloat16);
    vk.upload_bytes(v_input, input.data(), input.size() * sizeof(uint16_t));
    vk.upload_bytes(v_weight, weight.data(), weight.size() * sizeof(uint16_t));
    vk.upload_bytes(v_bias, bias.data(), bias.size() * sizeof(uint16_t));
    TensorBatch batch = vk.begin_batch();
    batch.layer_norm_bf16(v_input, v_weight, v_bias, in_place ? v_input : v_output, 1.0e-6f);
    batch.submit().wait();
    DeviceTensor& result = in_place ? v_input : v_output;
    vk.download_bytes(result, actual.data(), actual.size() * sizeof(uint16_t));
    compare_bf16(expected, actual, "BF16 LayerNorm", rows, dim);
  };
  run_layer(2, 1152, true);
  run_layer(1, 4608, false);
  run_layer(3, 129, true, true);

  auto run_mod = [&](bool fp32, bool invalid_selectors = false, bool contraction_fixture = false,
                     int rows = 3, int dim = 5376, bool in_place = false) {
    constexpr int mod_rows = 4;
    const size_t count = static_cast<size_t>(rows) * dim;
    auto bf_input = make_bf16_data(count, 0);
    std::vector<float> f_input(count);
    for (size_t i = 0; i < count; ++i) {
      uint32_t bits = static_cast<uint32_t>(bf_input[i]) << 16u;
      std::memcpy(&f_input[i], &bits, sizeof(bits));
    }
    std::vector<uint16_t> weight(dim);
    std::vector<float> scale(static_cast<size_t>(mod_rows) * dim);
    std::vector<float> shift(scale.size());
    std::vector<int32_t> valid_selectors(rows), invalid_selector_values(rows);
    for (int row = 0; row < rows; ++row) {
      valid_selectors[row] = (row * 3 + 1) % mod_rows;
      invalid_selector_values[row] = (row & 1) ? mod_rows + row : -1 - row;
    }
    const int32_t* selectors =
        invalid_selectors ? invalid_selector_values.data() : valid_selectors.data();
    for (int i = 0; i < dim; ++i)
      weight[i] = exact_bf16(0.5f + static_cast<float>(i % 8) / 8.0f);
    for (size_t i = 0; i < scale.size(); ++i) {
      scale[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32.0f;
      shift[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 64.0f;
    }
    if (contraction_fixture) {
      std::fill(f_input.begin(), f_input.begin() + dim, 0.5f);
      weight[0] = exact_bf16(1.5f);
      const uint32_t scale_bits = 0xbebf60eeu;
      const uint32_t shift_bits = 0x3714c343u;
      std::memcpy(&scale[static_cast<size_t>(selectors[0]) * dim], &scale_bits, 4);
      std::memcpy(&shift[static_cast<size_t>(selectors[0]) * dim], &shift_bits, 4);
    }
    const float epsilon = contraction_fixture ? 0.75f : 1.0e-5f;
    cuda::DeviceBuffer<uint16_t> c_bf_input(count), c_weight(dim), c_bf_output(count),
        c_bf_repeat(count);
    cuda::DeviceBuffer<float> c_f_input(count), c_scale(scale.size()), c_shift(shift.size()),
        c_f_output(count), c_f_repeat(count);
    cuda::DeviceBuffer<int32_t> c_selectors(rows);
    c_bf_input.copy_from_host(bf_input.data(), count);
    c_f_input.copy_from_host(f_input.data(), count);
    c_weight.copy_from_host(weight.data(), dim);
    c_scale.copy_from_host(scale.data(), scale.size());
    c_shift.copy_from_host(shift.data(), shift.size());
    c_selectors.copy_from_host(selectors, rows);
    if (!invalid_selectors && fp32) {
      cuda::launch_rmsnorm_modulate_f32(
          c_f_input.get(), reinterpret_cast<const __nv_bfloat16*>(c_weight.get()), c_scale.get(),
          c_shift.get(), c_selectors.get(), c_f_output.get(), rows, dim, epsilon, nullptr);
      cuda::launch_rmsnorm_modulate_f32(
          c_f_input.get(), reinterpret_cast<const __nv_bfloat16*>(c_weight.get()), c_scale.get(),
          c_shift.get(), c_selectors.get(), c_f_repeat.get(), rows, dim, epsilon, nullptr);
    } else if (!invalid_selectors) {
      cuda::launch_rmsnorm_modulate(reinterpret_cast<const __nv_bfloat16*>(c_bf_input.get()),
                                    reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
                                    c_scale.get(), c_shift.get(), c_selectors.get(),
                                    reinterpret_cast<__nv_bfloat16*>(c_bf_output.get()), rows, dim,
                                    1.0e-5f, nullptr);
      cuda::launch_rmsnorm_modulate(reinterpret_cast<const __nv_bfloat16*>(c_bf_input.get()),
                                    reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
                                    c_scale.get(), c_shift.get(), c_selectors.get(),
                                    reinterpret_cast<__nv_bfloat16*>(c_bf_repeat.get()), rows, dim,
                                    1.0e-5f, nullptr);
    }
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    if (!invalid_selectors) {
      if (fp32) {
        std::vector<float> once(count), twice(count);
        c_f_output.copy_to_host(once.data(), count);
        c_f_repeat.copy_to_host(twice.data(), count);
        CHECK(std::memcmp(once.data(), twice.data(), count * sizeof(float)) == 0);
      } else {
        std::vector<uint16_t> once(count), twice(count);
        c_bf_output.copy_to_host(once.data(), count);
        c_bf_repeat.copy_to_host(twice.data(), count);
        CHECK(once == twice);
      }
    }
    const uint64_t extents[] = {static_cast<uint64_t>(rows), static_cast<uint64_t>(dim)};
    const uint64_t feature = dim;
    const uint64_t mod_extents[] = {mod_rows, static_cast<uint64_t>(dim)};
    const uint64_t selector_extent = rows;
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(extents, 2),
                                       fp32 ? ScalarType::kFloat32 : ScalarType::kBFloat16);
    DeviceTensor v_weight =
        vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kBFloat16);
    DeviceTensor v_scale = vk.allocate(TensorLayout::contiguous(mod_extents, 2));
    DeviceTensor v_shift = vk.allocate(TensorLayout::contiguous(mod_extents, 2));
    DeviceTensor v_selectors =
        vk.allocate(TensorLayout::contiguous(&selector_extent, 1), ScalarType::kInt32);
    DeviceTensor v_output = in_place
                                ? DeviceTensor()
                                : vk.allocate(TensorLayout::contiguous(extents, 2),
                                              fp32 ? ScalarType::kFloat32 : ScalarType::kBFloat16);
    if (fp32)
      vk.upload(v_input, f_input.data(), f_input.size());
    else
      vk.upload_bytes(v_input, bf_input.data(), bf_input.size() * sizeof(uint16_t));
    vk.upload_bytes(v_weight, weight.data(), weight.size() * sizeof(uint16_t));
    vk.upload(v_scale, scale.data(), scale.size());
    vk.upload(v_shift, shift.data(), shift.size());
    vk.upload_bytes(v_selectors, selectors, rows * sizeof(int32_t));
    TensorBatch batch = vk.begin_batch();
    if (fp32)
      batch.rms_norm_modulate_f32(v_input, v_weight, v_scale, v_shift, v_selectors,
                                  in_place ? v_input : v_output, epsilon);
    else
      batch.rms_norm_modulate_bf16(v_input, v_weight, v_scale, v_shift, v_selectors,
                                   in_place ? v_input : v_output, epsilon);
    batch.submit().wait();
    DeviceTensor& result = in_place ? v_input : v_output;
    if (fp32) {
      std::vector<float> expected(count), actual(count);
      if (!invalid_selectors)
        c_f_output.copy_to_host(expected.data(), count);
      vk.download(result, actual.data(), count);
      if (contraction_fixture) {
        uint32_t fused_bits = 0;
        std::memcpy(&fused_bits, &expected[0], 4);
        CHECK_MSG(fused_bits == 0x3ef07877u, "CUDA AdaLN must select fused result, got %08x",
                  fused_bits);
      }
      size_t mismatch = count;
      for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0) {
          mismatch = i;
          break;
        }
      }
      uint32_t expected_bits = 0, actual_bits = 0;
      if (mismatch != count) {
        std::memcpy(&expected_bits, &expected[mismatch], sizeof(expected_bits));
        std::memcpy(&actual_bits, &actual[mismatch], sizeof(actual_bits));
      }
      CHECK_MSG(mismatch == count, "CUDA/Vulkan fp32 AdaLN mismatch at %zu: %08x != %08x", mismatch,
                expected_bits, actual_bits);
    } else {
      std::vector<uint16_t> expected(count), actual(count);
      if (!invalid_selectors)
        c_bf_output.copy_to_host(expected.data(), count);
      vk.download_bytes(result, actual.data(), actual.size() * sizeof(uint16_t));
      compare_bf16(expected, actual, "BF16 AdaLN", rows, dim);
    }
  };
  run_mod(false);
  run_mod(true);
  run_mod(false, true);
  run_mod(true, true);
  run_mod(true, false, true);
  run_mod(false, false, false, 5, 31);
  run_mod(false, false, false, 3, 513, true);
  run_mod(true, false, false, 5, 31, true);
  run_mod(true, false, false, 3, 513);

  const uint64_t used_before_mixed = vk.pooled_used_bytes();
  {
    // All five shared-normalization pipelines share the same bounded two-slot
    // submission context. Keep two mixed jobs outstanding, force oldest-slot
    // reuse with a third, and prove descriptor/pool high-water stays stable.
    const uint64_t narrow_shape[] = {9, 128};
    const uint64_t wide_shape[] = {2, 513};
    const uint64_t layer_shape[] = {2, 129};
    const uint64_t mod_shape[] = {3, 31};
    const uint64_t narrow_feature = 128, wide_feature = 513, layer_feature = 129;
    const uint64_t mod_feature = 31, mod_rows = 4, selector_rows = 3;
    const uint64_t mod_parameter_shape[] = {mod_rows, mod_feature};
    DeviceTensor narrow_input =
        vk.allocate(TensorLayout::contiguous(narrow_shape, 2), ScalarType::kBFloat16);
    DeviceTensor narrow_weight =
        vk.allocate(TensorLayout::contiguous(&narrow_feature, 1), ScalarType::kBFloat16);
    DeviceTensor wide_input =
        vk.allocate(TensorLayout::contiguous(wide_shape, 2), ScalarType::kBFloat16);
    DeviceTensor wide_weight =
        vk.allocate(TensorLayout::contiguous(&wide_feature, 1), ScalarType::kBFloat16);
    DeviceTensor layer_input =
        vk.allocate(TensorLayout::contiguous(layer_shape, 2), ScalarType::kBFloat16);
    DeviceTensor layer_weight =
        vk.allocate(TensorLayout::contiguous(&layer_feature, 1), ScalarType::kBFloat16);
    DeviceTensor layer_bias =
        vk.allocate(TensorLayout::contiguous(&layer_feature, 1), ScalarType::kBFloat16);
    DeviceTensor mod_bf_input =
        vk.allocate(TensorLayout::contiguous(mod_shape, 2), ScalarType::kBFloat16);
    DeviceTensor mod_f_input = vk.allocate(TensorLayout::contiguous(mod_shape, 2));
    DeviceTensor mod_weight =
        vk.allocate(TensorLayout::contiguous(&mod_feature, 1), ScalarType::kBFloat16);
    DeviceTensor mod_scale = vk.allocate(TensorLayout::contiguous(mod_parameter_shape, 2));
    DeviceTensor mod_shift = vk.allocate(TensorLayout::contiguous(mod_parameter_shape, 2));
    DeviceTensor mod_selectors =
        vk.allocate(TensorLayout::contiguous(&selector_rows, 1), ScalarType::kInt32);
    std::array<std::array<DeviceTensor, 5>, 2> mixed_outputs;
    for (auto& outputs : mixed_outputs) {
      outputs[0] = vk.allocate(TensorLayout::contiguous(narrow_shape, 2), ScalarType::kBFloat16);
      outputs[1] = vk.allocate(TensorLayout::contiguous(wide_shape, 2), ScalarType::kBFloat16);
      outputs[2] = vk.allocate(TensorLayout::contiguous(layer_shape, 2), ScalarType::kBFloat16);
      outputs[3] = vk.allocate(TensorLayout::contiguous(mod_shape, 2), ScalarType::kBFloat16);
      outputs[4] = vk.allocate(TensorLayout::contiguous(mod_shape, 2));
    }
    std::vector<uint16_t> narrow_zero(9 * 128), narrow_one(128, exact_bf16(1.0f));
    std::vector<uint16_t> wide_zero(2 * 513), wide_one(513, exact_bf16(1.0f));
    std::vector<uint16_t> layer_zero(2 * 129), layer_one(129, exact_bf16(1.0f));
    std::vector<uint16_t> layer_zero_bias(129), mod_bf_zero(3 * 31), mod_one(31, exact_bf16(1.0f));
    std::vector<float> mod_f_zero(3 * 31), mod_parameter_zero(4 * 31);
    const int32_t mixed_selectors[] = {3, 1, 0};
    vk.upload_bytes(narrow_input, narrow_zero.data(), narrow_zero.size() * 2);
    vk.upload_bytes(narrow_weight, narrow_one.data(), narrow_one.size() * 2);
    vk.upload_bytes(wide_input, wide_zero.data(), wide_zero.size() * 2);
    vk.upload_bytes(wide_weight, wide_one.data(), wide_one.size() * 2);
    vk.upload_bytes(layer_input, layer_zero.data(), layer_zero.size() * 2);
    vk.upload_bytes(layer_weight, layer_one.data(), layer_one.size() * 2);
    vk.upload_bytes(layer_bias, layer_zero_bias.data(), layer_zero_bias.size() * 2);
    vk.upload_bytes(mod_bf_input, mod_bf_zero.data(), mod_bf_zero.size() * 2);
    vk.upload(mod_f_input, mod_f_zero.data(), mod_f_zero.size());
    vk.upload_bytes(mod_weight, mod_one.data(), mod_one.size() * 2);
    vk.upload(mod_scale, mod_parameter_zero.data(), mod_parameter_zero.size());
    vk.upload(mod_shift, mod_parameter_zero.data(), mod_parameter_zero.size());
    vk.upload_bytes(mod_selectors, mixed_selectors, sizeof(mixed_selectors));
    auto submit_mixed = [&](std::array<DeviceTensor, 5>& outputs) {
      TensorBatch batch = vk.begin_batch();
      batch.rms_norm_bf16(narrow_input, narrow_weight, outputs[0], 1.0e-5f);
      batch.rms_norm_bf16(wide_input, wide_weight, outputs[1], 1.0e-5f);
      batch.layer_norm_bf16(layer_input, layer_weight, layer_bias, outputs[2], 1.0e-6f);
      batch.rms_norm_modulate_bf16(mod_bf_input, mod_weight, mod_scale, mod_shift, mod_selectors,
                                   outputs[3], 1.0e-5f);
      batch.rms_norm_modulate_f32(mod_f_input, mod_weight, mod_scale, mod_shift, mod_selectors,
                                  outputs[4], 1.0e-5f);
      return batch.submit();
    };
    {
      Submission first = submit_mixed(mixed_outputs[0]);
      Submission second = submit_mixed(mixed_outputs[1]);
      CHECK(first.value() != 0 && second.value() > first.value());
      first.wait();
      second.wait();
    }
    const uint64_t mixed_reserved = vk.reserved_bytes();
    const uint64_t mixed_descriptors = vk.descriptor_set_allocations();
    for (int iteration = 0; iteration < 8; ++iteration) {
      Submission first = submit_mixed(mixed_outputs[0]);
      Submission second = submit_mixed(mixed_outputs[1]);
      Submission third = submit_mixed(mixed_outputs[0]);
      CHECK(second.value() > first.value() && third.value() > second.value());
      first.wait();
      second.wait();
      third.wait();
      CHECK(vk.reserved_bytes() == mixed_reserved);
      CHECK(vk.descriptor_set_allocations() == mixed_descriptors);
    }

    // Every new API rejects invalid metadata before recording. The same batch
    // remains usable after each rejection, proving no partial descriptor/barrier
    // mutation was committed.
    DeviceTensor wrong_type = vk.allocate(TensorLayout::contiguous(&narrow_feature, 1));
    const uint64_t short_shape[] = {9, 127};
    DeviceTensor wrong_shape =
        vk.allocate(TensorLayout::contiguous(short_shape, 2), ScalarType::kBFloat16);
    auto valid_after_rejection = [&](auto&& invalid) {
      TensorBatch batch = vk.begin_batch();
      bool rejected = false;
      try {
        invalid(batch);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      CHECK(rejected);
      batch.rms_norm_bf16(narrow_input, narrow_weight, mixed_outputs[0][0], 1.0e-5f);
      batch.submit().wait();
    };
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rms_norm_bf16(narrow_input, wrong_type, mixed_outputs[0][0], 1.0e-5f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rms_norm_bf16(narrow_input, narrow_weight, wrong_shape, 1.0e-5f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.layer_norm_bf16(layer_input, layer_weight, layer_input, mixed_outputs[0][2], 1.0e-6f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rms_norm_bf16(narrow_input, narrow_weight, mixed_outputs[0][0], 0.0f);
    });
  }
  // Submission records retain every referenced tensor until its exact token
  // completes. Once jobs and caller wrappers are gone, all pooled spans are
  // returned while the context itself remains alive.
  {
    TensorBatch collect_completed_slots = vk.begin_batch();
  }
  CHECK_MSG(vk.pooled_used_bytes() == used_before_mixed,
            "mixed pooled bytes not released: %llu != %llu",
            static_cast<unsigned long long>(vk.pooled_used_bytes()),
            static_cast<unsigned long long>(used_before_mixed));
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_tensor_exact_group_norm_silu, "synthetic") {
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
  TensorContext vk(device);
  if (!vk.exact_normalization()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_normalization()");
    return;
  }

  auto run_case = [&](int channels, int height, int width, int groups, int pattern, bool in_place) {
    const size_t count = static_cast<size_t>(channels) * height * width;
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i)
      input[i] = static_cast<float>(static_cast<int>((i * 29) % 103) - 51) / 16.0f;
    if (pattern == 1)
      std::fill(input.begin(), input.end(), 2.0f);
    if (pattern == 2) {
      for (size_t i = 0; i < count; ++i)
        input[i] = 4096.0f + static_cast<float>(static_cast<int>(i % 7) - 3) / 8.0f;
    }
    if (pattern == 3) {
      for (size_t i = 0; i < count; ++i)
        input[i] = (i & 1) ? -0.0f : 0.0f;
    }
    std::vector<__half> weight(channels), bias(channels);
    std::vector<uint16_t> weight_bits(channels), bias_bits(channels);
    for (int c = 0; c < channels; ++c) {
      weight[c] = __float2half(0.5f + static_cast<float>(c % 11) / 16.0f);
      bias[c] = __float2half(static_cast<float>((c % 13) - 6) / 32.0f);
      if (pattern == 3 && c < 4) {
        const float edge_weight[] = {-0.0f, 0.0f, 65504.0f, -65504.0f};
        const float edge_bias[] = {-0.0f, 0.0f, 1.0f, -1.0f};
        weight[c] = __float2half(edge_weight[c]);
        bias[c] = __float2half(edge_bias[c]);
      }
      std::memcpy(&weight_bits[c], &weight[c], 2);
      std::memcpy(&bias_bits[c], &bias[c], 2);
    }
    cuda::DeviceBuffer<float> c_input(count), c_output(count);
    cuda::DeviceBuffer<__half> c_weight(channels), c_bias(channels);
    c_input.copy_from_host(input.data(), count);
    c_weight.copy_from_host(weight.data(), channels);
    c_bias.copy_from_host(bias.data(), channels);
    cuda::launch_keyframe_groupnorm_silu(c_input.get(), c_weight.get(), c_bias.get(),
                                         c_output.get(), channels, height, width, groups, 1.0e-6f,
                                         nullptr);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> expected(count), actual(count);
    c_output.copy_to_host(expected.data(), count);

    const uint64_t shape[] = {static_cast<uint64_t>(channels), static_cast<uint64_t>(height),
                              static_cast<uint64_t>(width)};
    const uint64_t feature = channels;
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(shape, 3));
    DeviceTensor v_weight =
        vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kFloat16);
    DeviceTensor v_bias = vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kFloat16);
    DeviceTensor v_output =
        in_place ? DeviceTensor() : vk.allocate(TensorLayout::contiguous(shape, 3));
    vk.upload(v_input, input.data(), count);
    vk.upload_bytes(v_weight, weight_bits.data(), weight_bits.size() * 2);
    vk.upload_bytes(v_bias, bias_bits.data(), bias_bits.size() * 2);
    TensorBatch batch = vk.begin_batch();
    batch.group_norm_silu_f16_affine(v_input, v_weight, v_bias, in_place ? v_input : v_output,
                                     static_cast<uint32_t>(groups), 1.0e-6f);
    batch.submit().wait();
    DeviceTensor& result = in_place ? v_input : v_output;
    vk.download(result, actual.data(), count);
    size_t mismatch = count;
    for (size_t i = 0; i < count; ++i) {
      if (std::memcmp(&expected[i], &actual[i], 4) != 0) {
        mismatch = i;
        break;
      }
    }
    uint32_t expected_bits = 0, actual_bits = 0;
    if (mismatch != count) {
      std::memcpy(&expected_bits, &expected[mismatch], 4);
      std::memcpy(&actual_bits, &actual[mismatch], 4);
    }
    CHECK_MSG(mismatch == count,
              "CUDA/Vulkan GroupNorm+SiLU %dx%dx%d g%d mismatch at %zu: %08x != %08x", channels,
              height, width, groups, mismatch, expected_bits, actual_bits);
  };
  run_case(128, 17, 19, 32, 0, false);
  run_case(96, 5, 7, 32, 1, true);
  run_case(128, 3, 11, 16, 2, false);
  run_case(128, 4, 13, 32, 3, true);

  bool cuda_limit_rejected = false;
  try {
    cuda::launch_keyframe_groupnorm_silu(
        reinterpret_cast<const float*>(uintptr_t{1}), reinterpret_cast<const __half*>(uintptr_t{1}),
        reinterpret_cast<const __half*>(uintptr_t{1}), reinterpret_cast<float*>(uintptr_t{1}), 128,
        32768, 32768, 32, 1.0e-6f, nullptr);
  } catch (const std::runtime_error&) {
    cuda_limit_rejected = true;
  }
  CHECK(cuda_limit_rejected);

  const uint64_t used_before_reuse = vk.pooled_used_bytes();
  {
    const uint64_t shape[] = {128, 3, 5}, feature = 128;
    DeviceTensor input = vk.allocate(TensorLayout::contiguous(shape, 3));
    DeviceTensor weight = vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kFloat16);
    DeviceTensor bias = vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kFloat16);
    std::array<DeviceTensor, 2> outputs{vk.allocate(TensorLayout::contiguous(shape, 3)),
                                        vk.allocate(TensorLayout::contiguous(shape, 3))};
    std::vector<float> zeros(128 * 3 * 5);
    std::vector<uint16_t> affine(128, 0x3c00u), offsets(128);
    vk.upload(input, zeros.data(), zeros.size());
    vk.upload_bytes(weight, affine.data(), affine.size() * 2);
    vk.upload_bytes(bias, offsets.data(), offsets.size() * 2);
    auto submit = [&](DeviceTensor& output) {
      TensorBatch batch = vk.begin_batch();
      batch.group_norm_silu_f16_affine(input, weight, bias, output, 32, 1.0e-6f);
      return batch.submit();
    };
    Submission warm_a = submit(outputs[0]), warm_b = submit(outputs[1]);
    warm_a.wait();
    warm_b.wait();
    const uint64_t stable_reserved = vk.reserved_bytes();
    const uint64_t stable_descriptors = vk.descriptor_set_allocations();
    for (int repeat = 0; repeat < 8; ++repeat) {
      Submission first = submit(outputs[0]);
      Submission second = submit(outputs[1]);
      Submission third = submit(outputs[0]);
      CHECK(second.value() > first.value() && third.value() > second.value());
      first.wait();
      second.wait();
      third.wait();
      CHECK(vk.reserved_bytes() == stable_reserved);
      CHECK_MSG(vk.descriptor_set_allocations() == stable_descriptors,
                "GroupNorm descriptors grew: %llu != %llu",
                static_cast<unsigned long long>(vk.descriptor_set_allocations()),
                static_cast<unsigned long long>(stable_descriptors));
    }
    DeviceTensor wrong_type = vk.allocate(TensorLayout::contiguous(&feature, 1));
    const uint64_t short_shape[] = {128, 3, 4};
    DeviceTensor wrong_shape = vk.allocate(TensorLayout::contiguous(short_shape, 3));
    auto valid_after_rejection = [&](auto&& invalid) {
      TensorBatch batch = vk.begin_batch();
      bool rejected = false;
      try {
        invalid(batch);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      CHECK(rejected);
      batch.group_norm_silu_f16_affine(input, weight, bias, outputs[0], 32, 1.0e-6f);
      batch.submit().wait();
    };
    valid_after_rejection([&](TensorBatch& batch) {
      batch.group_norm_silu_f16_affine(input, wrong_type, bias, outputs[0], 32, 1.0e-6f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.group_norm_silu_f16_affine(input, weight, bias, wrong_shape, 32, 1.0e-6f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.group_norm_silu_f16_affine(input, weight, bias, outputs[0], 31, 1.0e-6f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.group_norm_silu_f16_affine(input, weight, weight, outputs[0], 32, 1.0e-6f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.group_norm_silu_f16_affine(input, weight, bias, outputs[0], 32, 0.0f);
    });
  }
  {
    TensorBatch collect_completed_slots = vk.begin_batch();
  }
  CHECK(vk.pooled_used_bytes() == used_before_reuse);
}
