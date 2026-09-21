#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_tensor_exact_copy_and_add, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("CUDA and Vulkan devices are required");
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
  bool full_exact_rejected = false;
  try {
    vk.require_full_fp32_arithmetic_exactness();
  } catch (const std::runtime_error&) {
    full_exact_rejected = true;
  }
  CHECK(full_exact_rejected == !physical.front().info().fp32_denorm_preserve);
  CHECK(vk.full_fp32_arithmetic_exactness() == physical.front().info().fp32_denorm_preserve);
  CHECK(vk.full_fp32_add_exactness() == vk.full_fp32_arithmetic_exactness());

  constexpr uint64_t count = 259;
  const TensorLayout layout = TensorLayout::contiguous(&count, 1);
  std::vector<float> a(count), b(count);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 41) - 20) / 32.0f;
    b[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 64.0f;
  }
  // Copy must preserve every bit pattern. Arithmetic includes signed zero,
  // subnormal and infinity inputs whose selected sums have stable exact bits
  // on both APIs. NaN payload preservation is covered by copy: GLSL fp32 add
  // is permitted to canonicalise a NaN and is therefore not advertised as an
  // exact-payload arithmetic operation.
  const uint32_t copy_special[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x00800000u, 0x3f800000u,
                                   0x3f800000u, 0x7f800000u, 0xff800000u, 0x7fc12345u, 0x7fa54321u};
  const uint32_t add_rhs[] = {0x80000000u, 0x80000000u, 0x00000001u, 0x807fffffu, 0x33800000u,
                              0x33800001u, 0x3f800000u, 0xbf800000u, 0x00000000u, 0x00000000u};
  std::memcpy(a.data(), copy_special, sizeof(copy_special));
  std::memcpy(b.data(), add_rhs, sizeof(add_rhs));

  cuda::DeviceBuffer<float> cuda_a(count), cuda_b(count), cuda_copy(count), cuda_sum(count);
  cuda_a.copy_from_host(a.data(), count);
  cuda_b.copy_from_host(b.data(), count);
  SLOPFAB_CUDA_CHECK(
      cudaMemcpy(cuda_copy.get(), cuda_a.get(), count * sizeof(float), cudaMemcpyDeviceToDevice));
  cuda::launch_add(cuda_a.get(), cuda_b.get(), cuda_sum.get(), count, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> cuda_copy_host(count), cuda_sum_host(count);
  cuda_copy.copy_to_host(cuda_copy_host.data(), count);
  cuda_sum.copy_to_host(cuda_sum_host.data(), count);

  DeviceTensor vk_a = vk.allocate(layout);
  DeviceTensor vk_b = vk.allocate(layout);
  DeviceTensor vk_copy = vk.allocate(layout);
  DeviceTensor vk_sum = vk.allocate(layout);
  vk.upload(vk_a, a.data(), count);
  vk.upload(vk_b, b.data(), count);
  TensorBatch batch = vk.begin_batch();
  batch.copy(vk_a, vk_copy);
  batch.add(vk_a, vk_b, vk_sum);
  batch.submit().wait();
  std::vector<float> vk_copy_host(count), vk_sum_host(count);
  vk.download(vk_copy, vk_copy_host.data(), count);
  vk.download(vk_sum, vk_sum_host.data(), count);

  CHECK(std::memcmp(cuda_copy_host.data(), vk_copy_host.data(), count * sizeof(float)) == 0);
  // NaN results are excluded from the arithmetic exactness claim; every
  // finite/infinite result, including signed-zero/subnormal inputs, is exact.
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits = 0;
    std::memcpy(&bits, &a[i], sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0)
      continue;
    // Indices 2/3 require denormal-preserving arithmetic. On devices without
    // that Vulkan mode, the explicit capability gate above fails rather than
    // claiming those results are CUDA-exact.
    if (!vk.full_fp32_arithmetic_exactness() && (i == 2 || i == 3))
      continue;
    uint32_t cuda_bits = 0, vk_bits = 0;
    std::memcpy(&cuda_bits, &cuda_sum_host[i], sizeof(cuda_bits));
    std::memcpy(&vk_bits, &vk_sum_host[i], sizeof(vk_bits));
    CHECK_MSG(cuda_bits == vk_bits, "CUDA/Vulkan fp32 add mismatch at %zu: %08x != %08x", i,
              cuda_bits, vk_bits);
  }
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_tensor_exact_conversion_and_layout_ops, "synthetic") {
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
  Device device = physical.front().create_device(options);
  TensorContext vk(device);

  // 335 values exercise a non-workgroup tail; the prefix pins the CUDA NaN,
  // Inf, signed-zero, subnormal and half-way conversion policy exactly.
  constexpr int rows = 5;
  constexpr int cols = 67;
  constexpr size_t count = static_cast<size_t>(rows) * cols;
  std::vector<float> input(count), bias(cols);
  for (size_t i = 0; i < count; ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 101) - 50) / 64.0f;
  }
  const uint32_t special[] = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x007fffffu, 0x00800000u, 0x33800000u, 0x33800001u,
      0x387fffffu, 0x38800000u, 0x3f800000u, 0x477fe000u, 0x7f800000u, 0xff800000u, 0x7fc12345u,
      0x7fa54321u, 0xffc12345u, 0x33000000u, 0x33000001u, 0x3f808000u, 0x3f808001u, 0x00800000u};
  std::memcpy(input.data(), special, sizeof(special));
  for (int i = 0; i < cols; ++i)
    bias[i] = static_cast<float>((i % 13) - 6) / 32.0f;
  for (int i = 0; i < 4; ++i)
    bias[i] = 0.0f;
  // Two normal operands whose exact sum is the smallest subnormal.
  const uint32_t normal_above_min = 0x00800001u;
  const uint32_t negative_min_normal = 0x80800000u;
  std::memcpy(&input[4], &normal_above_min, sizeof(float));
  std::memcpy(&bias[4], &negative_min_normal, sizeof(float));

  cuda::DeviceBuffer<float> c_input(count), c_bf16_back(count), c_f16_back(count);
  cuda::DeviceBuffer<uint16_t> c_bf16(count), c_f16(count);
  c_input.copy_from_host(input.data(), count);
  cuda::launch_narrow_to_bf16(c_input.get(), reinterpret_cast<__nv_bfloat16*>(c_bf16.get()), count,
                              nullptr);
  cuda::launch_widen_bf16(reinterpret_cast<const __nv_bfloat16*>(c_bf16.get()), c_bf16_back.get(),
                          count, nullptr);
  cuda::launch_narrow_f16(c_input.get(), c_f16.get(), count, nullptr);
  cuda::launch_widen_f16(c_f16.get(), c_f16_back.get(), count, nullptr);

  cuda::DeviceBuffer<float> c_transpose(count), c_biased(count);
  cuda::DeviceBuffer<float> c_bias(cols);
  c_bias.copy_from_host(bias.data(), bias.size());
  SLOPFAB_CUDA_CHECK(
      cudaMemcpy(c_biased.get(), c_input.get(), count * sizeof(float), cudaMemcpyDeviceToDevice));
  cuda::launch_transpose_cn_to_nc(c_input.get(), c_transpose.get(), rows, cols, nullptr);
  cuda::launch_add_bias(c_biased.get(), c_bias.get(), rows, cols, nullptr);

  constexpr int matrix_rows = 7;
  constexpr int selected_rows = 5;
  constexpr int matrix_cols = 67;
  constexpr size_t matrix_count = static_cast<size_t>(matrix_rows) * matrix_cols;
  constexpr size_t selected_count = static_cast<size_t>(selected_rows) * matrix_cols;
  std::vector<float> matrix(matrix_count), scatter_initial(matrix_count, -99.0f);
  for (size_t i = 0; i < matrix.size(); ++i)
    matrix[i] = static_cast<float>(static_cast<int>(i) - 170) / 16.0f;
  const int32_t indices_host[selected_rows] = {6, 0, 4, 1, 3};
  cuda::DeviceBuffer<float> c_matrix(matrix_count), c_gathered(selected_count),
      c_scattered(matrix_count);
  cuda::DeviceBuffer<int32_t> c_indices(selected_rows);
  c_matrix.copy_from_host(matrix.data(), matrix.size());
  c_indices.copy_from_host(indices_host, selected_rows);
  c_scattered.copy_from_host(scatter_initial.data(), scatter_initial.size());
  cuda::launch_gather_rows_f32(c_matrix.get(), c_indices.get(), c_gathered.get(), selected_rows,
                               matrix_cols, nullptr);
  cuda::launch_scatter_rows_f32(c_gathered.get(), c_indices.get(), c_scattered.get(), selected_rows,
                                matrix_cols, nullptr);

  constexpr int heads = 3, sequence = 5, head_dim = 7;
  constexpr size_t heads_count = static_cast<size_t>(heads) * sequence * head_dim;
  std::vector<float> heads_input(heads_count);
  for (size_t i = 0; i < heads_count; ++i)
    heads_input[i] = input[i];
  cuda::DeviceBuffer<float> c_heads(heads_count);
  cuda::DeviceBuffer<uint16_t> c_tokens(heads_count);
  c_heads.copy_from_host(heads_input.data(), heads_input.size());
  cuda::launch_heads_to_tokens_bf16(c_heads.get(), reinterpret_cast<__nv_bfloat16*>(c_tokens.get()),
                                    sequence, heads, head_dim, nullptr);

  // Every axis is nontrivial, including temporal patching. 288 output values
  // leave a 32-invocation workgroup tail and catch an output-plane expression
  // that accidentally omits patch_t.
  constexpr int depth_t = 2, depth_h = 2, depth_w = 3, depth_channels = 3;
  constexpr int patch_t = 2, patch = 2;
  constexpr size_t depth_count =
      static_cast<size_t>(depth_t) * depth_h * depth_w * depth_channels * patch_t * patch * patch;
  std::vector<float> depth_input(depth_count);
  for (size_t i = 0; i < depth_input.size(); ++i)
    depth_input[i] = static_cast<float>(i) + 0.25f;
  cuda::DeviceBuffer<float> c_depth_input(depth_count), c_depth_output(depth_count);
  c_depth_input.copy_from_host(depth_input.data(), depth_input.size());
  cuda::launch_depth_to_space(c_depth_input.get(), c_depth_output.get(), depth_t, depth_h, depth_w,
                              depth_channels, patch_t, patch, nullptr);

  // Widening also has an explicit CUDA NaN policy for arbitrary checkpoint
  // half bits, independently of values produced by the narrowing kernel.
  constexpr size_t raw_half_count = 1u << 16;
  std::vector<uint16_t> raw_half_host(raw_half_count);
  for (size_t i = 0; i < raw_half_count; ++i)
    raw_half_host[i] = static_cast<uint16_t>(i);
  cuda::DeviceBuffer<uint16_t> c_raw_half(raw_half_count);
  cuda::DeviceBuffer<float> c_raw_half_wide(raw_half_count);
  c_raw_half.copy_from_host(raw_half_host.data(), raw_half_count);
  cuda::launch_widen_f16(c_raw_half.get(), c_raw_half_wide.get(), raw_half_count, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  const uint64_t shape_extents[] = {rows, cols};
  const uint64_t transpose_extents[] = {cols, rows};
  const TensorLayout shape = TensorLayout::contiguous(shape_extents, 2);
  DeviceTensor v_input = vk.allocate(shape);
  DeviceTensor v_bf16 = vk.allocate(shape, ScalarType::kBFloat16);
  DeviceTensor v_f16 = vk.allocate(shape, ScalarType::kFloat16);
  DeviceTensor v_bf16_back = vk.allocate(shape);
  DeviceTensor v_f16_back = vk.allocate(shape);
  DeviceTensor v_transpose = vk.allocate(TensorLayout::contiguous(transpose_extents, 2));
  DeviceTensor v_bias = vk.allocate(TensorLayout::contiguous(&transpose_extents[0], 1));
  DeviceTensor v_biased = vk.allocate(shape);
  vk.upload(v_input, input.data(), count);
  vk.upload(v_bias, bias.data(), bias.size());

  const uint64_t matrix_extents[] = {matrix_rows, matrix_cols};
  const uint64_t selected_extents[] = {selected_rows, matrix_cols};
  const uint64_t index_extent = selected_rows;
  DeviceTensor v_matrix = vk.allocate(TensorLayout::contiguous(matrix_extents, 2));
  DeviceTensor v_indices =
      vk.allocate(TensorLayout::contiguous(&index_extent, 1), ScalarType::kInt32);
  DeviceTensor v_gathered = vk.allocate(TensorLayout::contiguous(selected_extents, 2));
  DeviceTensor v_scattered = vk.allocate(TensorLayout::contiguous(matrix_extents, 2));
  vk.upload(v_matrix, matrix.data(), matrix.size());
  vk.upload_bytes(v_indices, indices_host, sizeof(indices_host));
  vk.upload(v_scattered, scatter_initial.data(), scatter_initial.size());

  const uint64_t heads_extent = heads_count;
  DeviceTensor v_heads = vk.allocate(TensorLayout::contiguous(&heads_extent, 1));
  DeviceTensor v_tokens =
      vk.allocate(TensorLayout::contiguous(&heads_extent, 1), ScalarType::kBFloat16);
  vk.upload(v_heads, heads_input.data(), heads_input.size());
  const uint64_t depth_extent = depth_count;
  DeviceTensor v_depth_input = vk.allocate(TensorLayout::contiguous(&depth_extent, 1));
  DeviceTensor v_depth_output = vk.allocate(TensorLayout::contiguous(&depth_extent, 1));
  vk.upload(v_depth_input, depth_input.data(), depth_input.size());
  const uint64_t raw_half_extent = raw_half_count;
  DeviceTensor v_raw_half =
      vk.allocate(TensorLayout::contiguous(&raw_half_extent, 1), ScalarType::kFloat16);
  DeviceTensor v_raw_half_wide = vk.allocate(TensorLayout::contiguous(&raw_half_extent, 1));
  vk.upload_bytes(v_raw_half, raw_half_host.data(), raw_half_host.size() * sizeof(uint16_t));

  // All operators are one device-only Vulkan batch: there is no host boundary
  // between conversion, indexed movement, elementwise, and layout work.
  TensorBatch batch = vk.begin_batch();
  batch.convert(v_input, v_bf16);
  batch.convert(v_bf16, v_bf16_back);
  batch.convert(v_input, v_f16);
  batch.convert(v_f16, v_f16_back);
  batch.transpose_2d(v_input, v_transpose);
  batch.add_bias(v_input, v_bias, v_biased);
  batch.gather_rows(v_matrix, v_indices, v_gathered);
  batch.scatter_rows(v_gathered, v_indices, v_scattered);
  batch.heads_to_tokens_bf16(v_heads, v_tokens, heads, sequence, head_dim);
  batch.depth_to_space(v_depth_input, v_depth_output, depth_t, depth_h, depth_w, depth_channels,
                       patch_t, patch);
  batch.convert(v_raw_half, v_raw_half_wide);
  batch.submit().wait();

  auto compare_bytes = [&](auto& cuda_buffer, DeviceTensor& vulkan_tensor, size_t elements,
                           const char* label) {
    using Value = std::remove_pointer_t<decltype(cuda_buffer.get())>;
    std::vector<Value> cuda_host(elements), vulkan_host(elements);
    cuda_buffer.copy_to_host(cuda_host.data(), elements);
    vk.download_bytes(vulkan_tensor, vulkan_host.data(), elements * sizeof(Value));
    size_t mismatch = elements;
    for (size_t i = 0; i < elements; ++i) {
      if (std::memcmp(&cuda_host[i], &vulkan_host[i], sizeof(Value)) != 0) {
        mismatch = i;
        break;
      }
    }
    uint64_t cuda_bits = 0, vulkan_bits = 0;
    if (mismatch != elements) {
      std::memcpy(&cuda_bits, &cuda_host[mismatch], sizeof(Value));
      std::memcpy(&vulkan_bits, &vulkan_host[mismatch], sizeof(Value));
    }
    CHECK_MSG(mismatch == elements, "CUDA/Vulkan %s mismatch at %zu: %llx != %llx", label, mismatch,
              static_cast<unsigned long long>(cuda_bits),
              static_cast<unsigned long long>(vulkan_bits));
  };
  compare_bytes(c_bf16, v_bf16, count, "fp32-to-bf16");
  compare_bytes(c_bf16_back, v_bf16_back, count, "bf16-to-fp32");
  compare_bytes(c_f16, v_f16, count, "fp32-to-fp16");
  compare_bytes(c_f16_back, v_f16_back, count, "fp16-to-fp32");
  compare_bytes(c_transpose, v_transpose, count, "transpose");
  compare_bytes(c_gathered, v_gathered, selected_count, "gather");
  compare_bytes(c_scattered, v_scattered, matrix_count, "scatter");
  compare_bytes(c_tokens, v_tokens, heads_count, "heads-to-tokens");
  compare_bytes(c_depth_output, v_depth_output, depth_count, "depth-to-space");
  compare_bytes(c_raw_half_wide, v_raw_half_wide, raw_half_count, "arbitrary-fp16-widen");

  bool full_arithmetic_gate_rejected = false;
  try {
    vk.require_full_fp32_arithmetic_exactness();
  } catch (const std::runtime_error&) {
    full_arithmetic_gate_rejected = true;
  }
  CHECK(full_arithmetic_gate_rejected == !vk.full_fp32_arithmetic_exactness());
  std::vector<float> biased_cuda(count), biased_vulkan(count);
  c_biased.copy_to_host(biased_cuda.data(), biased_cuda.size());
  vk.download(v_biased, biased_vulkan.data(), biased_vulkan.size());
  size_t subnormal_cases = 0;
  auto bits_of = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  auto is_nan = [](uint32_t bits) {
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
  };
  auto is_subnormal = [](uint32_t bits) {
    return (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0;
  };
  for (size_t i = 0; i < count; ++i) {
    const uint32_t input_bits = bits_of(input[i]);
    const uint32_t bias_bits = bits_of(bias[i % cols]);
    const uint32_t cuda_bits = bits_of(biased_cuda[i]);
    if (is_nan(input_bits) || is_nan(bias_bits) || is_nan(cuda_bits))
      continue;
    const bool uses_subnormal =
        is_subnormal(input_bits) || is_subnormal(bias_bits) || is_subnormal(cuda_bits);
    if (uses_subnormal)
      ++subnormal_cases;
    if (uses_subnormal && !vk.full_fp32_arithmetic_exactness())
      continue;
    CHECK_MSG(cuda_bits == bits_of(biased_vulkan[i]),
              "CUDA/Vulkan add-bias mismatch at %zu: %08x != %08x", i, cuda_bits,
              bits_of(biased_vulkan[i]));
  }
  CHECK(subnormal_cases >= 3);

  std::vector<float> depth_vulkan(depth_count);
  vk.download(v_depth_output, depth_vulkan.data(), depth_vulkan.size());
  const int out_t = depth_t * patch_t;
  const int out_h = depth_h * patch;
  const int out_w = depth_w * patch;
  const int patch_volume = patch_t * patch * patch;
  for (int channel = 0; channel < depth_channels; ++channel) {
    for (int ot = 0; ot < out_t; ++ot) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          const size_t output_index =
              ((static_cast<size_t>(channel) * out_t + ot) * out_h + oh) * out_w + ow;
          const int token = ((ot / patch_t) * depth_h + oh / patch) * depth_w + ow / patch;
          const int feature =
              channel * patch_volume + ((ot % patch_t) * patch + oh % patch) * patch + ow % patch;
          const size_t source_index =
              static_cast<size_t>(token) * (depth_channels * patch_volume) + feature;
          CHECK(depth_vulkan[output_index] == depth_input[source_index]);
        }
      }
    }
  }
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_tensor_exact_vae_pointwise, "synthetic") {
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
  if (!vk.exact_vae_pointwise()) {
    bool rejected = false;
    try {
      vk.require_exact_vae_pointwise();
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    CHECK(rejected);
    return;
  }
  vk.require_exact_vae_pointwise();

  auto from_bits = [](uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  constexpr int rows = 3, columns = 67, inner = 67;
  constexpr int channels = 3, voxels = 67;
  const size_t matrix_count = static_cast<size_t>(rows) * columns;
  const size_t swiglu_input_count = static_cast<size_t>(rows) * 2 * inner;
  const size_t swiglu_output_count = static_cast<size_t>(rows) * inner;
  const size_t latent_count = static_cast<size_t>(channels) * voxels;
  std::vector<float> x(matrix_count), y(matrix_count), residual_bias(columns), scale(columns),
      swiglu_input(swiglu_input_count), swiglu_bias(2 * inner), latent(latent_count),
      mean(channels), std_dev(channels);
  for (size_t i = 0; i < matrix_count; ++i) {
    x[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f;
    y[i] = static_cast<float>(static_cast<int>(i % 37) - 18) / 32.0f;
  }
  for (int i = 0; i < columns; ++i) {
    residual_bias[i] = static_cast<float>((i % 11) - 5) / 64.0f;
    scale[i] = 0.25f + static_cast<float>(i % 7) / 16.0f;
  }
  for (size_t i = 0; i < swiglu_input.size(); ++i)
    swiglu_input[i] = static_cast<float>(static_cast<int>(i % 101) - 50) / 16.0f;
  for (size_t i = 0; i < swiglu_bias.size(); ++i)
    swiglu_bias[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32.0f;
  for (size_t i = 0; i < latent.size(); ++i)
    latent[i] = static_cast<float>(static_cast<int>(i % 47) - 23) / 16.0f;
  mean = {-0.25f, 0.0f, 0.375f};
  std_dev = {0.5f, -0.75f, 1.25f};

  // Each operation has an input-subnormal case, a normal cancellation whose
  // correctly rounded intermediate is subnormal, an output-underflow case,
  // signed zeros and exceptional values. The policy is asserted below rather
  // than merely comparing two backends with the same bug.
  x[0] = 0.0f;
  y[0] = from_bits(0x00800001u);
  residual_bias[0] = from_bits(0x80800000u);
  scale[0] = from_bits(0x7e800000u);
  x[1] = -0.0f;
  y[1] = from_bits(0x00800000u);
  residual_bias[1] = 0.0f;
  scale[1] = 0.5f;
  x[2] = from_bits(0x00000001u);
  y[2] = -0.0f;
  residual_bias[2] = 0.0f;
  scale[2] = 1.0f;
  x[3] = 1.0f;
  y[3] = from_bits(0x7fc12345u);
  residual_bias[3] = 0.0f;
  scale[3] = 1.0f;
  x[4] = -0.0f;
  y[4] = -0.0f;
  residual_bias[4] = -0.0f;
  scale[4] = 1.0f;
  x[5] = std::numeric_limits<float>::infinity();
  y[5] = 1.0f;
  residual_bias[5] = 0.0f;
  scale[5] = 1.0f;
  x[6] = std::numeric_limits<float>::infinity();
  y[6] = -std::numeric_limits<float>::infinity();
  residual_bias[6] = 0.0f;
  scale[6] = 1.0f;

  swiglu_input[0] = from_bits(0x00800001u);
  swiglu_bias[0] = from_bits(0x80800000u);
  swiglu_input[inner] = from_bits(0x7e800000u);
  swiglu_bias[inner] = 0.0f;
  swiglu_input[1] = 1.0f;
  swiglu_bias[1] = 0.0f;
  swiglu_input[inner + 1] = from_bits(0x00800000u);
  swiglu_bias[inner + 1] = 0.0f;
  swiglu_input[2] = from_bits(0x7fc12345u);
  swiglu_bias[2] = 0.0f;
  swiglu_input[inner + 2] = 1.0f;
  swiglu_bias[inner + 2] = 0.0f;
  swiglu_input[3] = -0.0f;
  swiglu_bias[3] = -0.0f;
  swiglu_input[inner + 3] = 2.0f;
  swiglu_bias[inner + 3] = 0.0f;
  swiglu_input[4] = std::numeric_limits<float>::infinity();
  swiglu_input[inner + 4] = 1.0f;
  swiglu_input[5] = -std::numeric_limits<float>::infinity();
  swiglu_input[inner + 5] = 1.0f;
  swiglu_input[6] = std::numeric_limits<float>::infinity();
  swiglu_input[inner + 6] = 0.0f;
  swiglu_input[7] = -std::numeric_limits<float>::infinity();
  swiglu_input[inner + 7] = std::numeric_limits<float>::infinity();
  for (int index = 4; index <= 10; ++index) {
    swiglu_bias[index] = 0.0f;
    swiglu_bias[inner + index] = 0.0f;
  }
  swiglu_input[8] = std::nextafter(-87.0f, -std::numeric_limits<float>::infinity());
  swiglu_input[9] = -87.0f;
  swiglu_input[10] = std::nextafter(-87.0f, std::numeric_limits<float>::infinity());
  swiglu_input[inner + 8] = 1.0f;
  swiglu_input[inner + 9] = 1.0f;
  swiglu_input[inner + 10] = 1.0f;

  latent[0] = from_bits(0x00800000u);
  std_dev[0] = 0.5f;
  mean[0] = 0.0f;
  latent[voxels] = from_bits(0x00000001u);
  std_dev[1] = 1.0f;
  mean[1] = -0.0f;
  latent[5] = std::numeric_limits<float>::infinity();
  latent[6] = -std::numeric_limits<float>::infinity();
  latent[2 * voxels] = from_bits(0x7fc01234u);
  latent[2 * voxels + 1] = std::numeric_limits<float>::infinity();
  std_dev[2] = 0.0f;
  mean[2] = 1.0f;

  cuda::DeviceBuffer<float> cx(matrix_count), cy(matrix_count), crb(columns), cs(columns),
      csi(swiglu_input_count), csb(2 * inner), cso(swiglu_output_count), cl(latent_count),
      cm(channels), csd(channels), clo(latent_count);
  cx.copy_from_host(x.data(), x.size());
  cy.copy_from_host(y.data(), y.size());
  crb.copy_from_host(residual_bias.data(), residual_bias.size());
  cs.copy_from_host(scale.data(), scale.size());
  csi.copy_from_host(swiglu_input.data(), swiglu_input.size());
  csb.copy_from_host(swiglu_bias.data(), swiglu_bias.size());
  cl.copy_from_host(latent.data(), latent.size());
  cm.copy_from_host(mean.data(), mean.size());
  csd.copy_from_host(std_dev.data(), std_dev.size());
  cuda::launch_layerscale_residual(cx.get(), cy.get(), crb.get(), cs.get(), rows, columns, nullptr);
  cuda::launch_swiglu(csi.get(), csb.get(), cso.get(), rows, inner, nullptr);
  cuda::launch_latent_denorm(cl.get(), cm.get(), csd.get(), clo.get(), channels, voxels, nullptr);

  // The legacy CUDA ABI permits nullable biases. Its null branch must remain
  // exactly equivalent to a present all-zero bias after the semantic rebase.
  cuda::DeviceBuffer<float> c_zero_residual_bias(columns), c_null_residual(matrix_count),
      c_zero_residual(matrix_count), c_nullable_y(matrix_count), c_zero_swiglu_bias(2 * inner),
      c_nullable_swiglu_input(swiglu_input_count), c_null_swiglu(swiglu_output_count),
      c_zero_swiglu(swiglu_output_count);
  std::vector<float> zero_residual_bias(columns, 0.0f), zero_swiglu_bias(2 * inner, 0.0f),
      nullable_x(matrix_count), nullable_y(matrix_count), nullable_swiglu_input(swiglu_input_count);
  for (size_t index = 0; index < matrix_count; ++index) {
    nullable_x[index] = 0.25f + static_cast<float>(index % 7) / 16.0f;
    nullable_y[index] = -0.5f + static_cast<float>(index % 11) / 32.0f;
  }
  for (size_t index = 0; index < swiglu_input_count; ++index)
    nullable_swiglu_input[index] = -1.0f + static_cast<float>(index % 23) / 16.0f;
  c_zero_residual_bias.copy_from_host(zero_residual_bias.data(), columns);
  c_zero_swiglu_bias.copy_from_host(zero_swiglu_bias.data(), 2 * inner);
  c_null_residual.copy_from_host(nullable_x.data(), nullable_x.size());
  c_zero_residual.copy_from_host(nullable_x.data(), nullable_x.size());
  c_nullable_y.copy_from_host(nullable_y.data(), nullable_y.size());
  c_nullable_swiglu_input.copy_from_host(nullable_swiglu_input.data(),
                                         nullable_swiglu_input.size());
  cuda::launch_layerscale_residual(c_null_residual.get(), c_nullable_y.get(), nullptr, cs.get(),
                                   rows, columns, nullptr);
  cuda::launch_layerscale_residual(c_zero_residual.get(), c_nullable_y.get(),
                                   c_zero_residual_bias.get(), cs.get(), rows, columns, nullptr);
  cuda::launch_swiglu(c_nullable_swiglu_input.get(), nullptr, c_null_swiglu.get(), rows, inner,
                      nullptr);
  cuda::launch_swiglu(c_nullable_swiglu_input.get(), c_zero_swiglu_bias.get(), c_zero_swiglu.get(),
                      rows, inner, nullptr);

  const uint64_t matrix_shape[] = {rows, columns};
  const uint64_t column_shape = columns;
  const uint64_t swiglu_in_shape[] = {rows, 2 * inner};
  const uint64_t swiglu_out_shape[] = {rows, inner};
  const uint64_t swiglu_bias_shape = 2 * inner;
  const uint64_t latent_shape[] = {channels, voxels};
  const uint64_t channel_shape = channels;
  DeviceTensor vx = vk.allocate(TensorLayout::contiguous(matrix_shape, 2));
  DeviceTensor vy = vk.allocate(TensorLayout::contiguous(matrix_shape, 2));
  DeviceTensor vrb = vk.allocate(TensorLayout::contiguous(&column_shape, 1));
  DeviceTensor vs = vk.allocate(TensorLayout::contiguous(&column_shape, 1));
  DeviceTensor vsi = vk.allocate(TensorLayout::contiguous(swiglu_in_shape, 2));
  DeviceTensor vsb = vk.allocate(TensorLayout::contiguous(&swiglu_bias_shape, 1));
  DeviceTensor vso = vk.allocate(TensorLayout::contiguous(swiglu_out_shape, 2));
  DeviceTensor vl = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
  DeviceTensor vm = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
  DeviceTensor vsd = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
  DeviceTensor vlo = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
  vk.upload(vx, x.data(), x.size());
  vk.upload(vy, y.data(), y.size());
  vk.upload(vrb, residual_bias.data(), residual_bias.size());
  vk.upload(vs, scale.data(), scale.size());
  vk.upload(vsi, swiglu_input.data(), swiglu_input.size());
  vk.upload(vsb, swiglu_bias.data(), swiglu_bias.size());
  vk.upload(vl, latent.data(), latent.size());
  vk.upload(vm, mean.data(), mean.size());
  vk.upload(vsd, std_dev.data(), std_dev.size());
  TensorBatch batch = vk.begin_batch();
  batch.layer_scale_residual_f32(vx, vy, vrb, vs);
  batch.swiglu_bias_f32(vsi, vsb, vso);
  batch.latent_denorm_f32(vl, vm, vsd, vlo);
  batch.submit().wait();

  auto compare = [&](auto& cuda_buffer, DeviceTensor& vulkan_tensor, size_t count,
                     const char* label) {
    std::vector<float> cuda_host(count), vulkan_host(count);
    cuda_buffer.copy_to_host(cuda_host.data(), count);
    vk.download(vulkan_tensor, vulkan_host.data(), count);
    size_t mismatch = count;
    for (size_t i = 0; i < count; ++i) {
      if (std::memcmp(&cuda_host[i], &vulkan_host[i], sizeof(float)) != 0) {
        mismatch = i;
        break;
      }
    }
    uint32_t cb = 0, vb = 0;
    if (mismatch != count) {
      std::memcpy(&cb, &cuda_host[mismatch], 4);
      std::memcpy(&vb, &vulkan_host[mismatch], 4);
    }
    CHECK_MSG(mismatch == count, "%s mismatch at %zu: %08x != %08x", label, mismatch, cb, vb);
    return cuda_host;
  };
  const auto residual = compare(cx, vx, matrix_count, "VAE residual");
  const auto swiglu = compare(cso, vso, swiglu_output_count, "VAE SwiGLU");
  const auto denorm = compare(clo, vlo, latent_count, "VAE latent denorm");
  auto bits_of = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    return bits;
  };
  CHECK(bits_of(residual[0]) == 0x00000000u);
  CHECK((bits_of(residual[1]) & 0x7fffffffu) == 0u);
  CHECK((bits_of(residual[2]) & 0x7fffffffu) == 0u);
  CHECK(bits_of(residual[3]) == 0x7fc00000u);
  CHECK(bits_of(residual[4]) == 0x80000000u);
  CHECK(bits_of(residual[5]) == 0x7f800000u);
  CHECK(bits_of(residual[6]) == 0x7fc00000u);
  CHECK(bits_of(swiglu[0]) == 0x00000000u);
  CHECK((bits_of(swiglu[1]) & 0x7fffffffu) == 0u);
  CHECK(bits_of(swiglu[2]) == 0x7fc00000u);
  CHECK(bits_of(swiglu[3]) == 0x80000000u);
  CHECK(bits_of(swiglu[4]) == 0x7f800000u);
  CHECK(bits_of(swiglu[5]) == 0x80000000u);
  CHECK(bits_of(swiglu[6]) == 0x7fc00000u);
  CHECK(bits_of(swiglu[7]) == 0x7fc00000u);
  CHECK(bits_of(swiglu[8]) == 0x80000000u);
  CHECK(bits_of(swiglu[9]) == 0x80000000u);
  CHECK((bits_of(swiglu[10]) & 0x7fffffffu) != 0u);
  CHECK((bits_of(denorm[0]) & 0x7fffffffu) == 0u);
  CHECK(bits_of(denorm[5]) == 0x7f800000u);
  CHECK(bits_of(denorm[6]) == 0xff800000u);
  CHECK(bits_of(denorm[voxels]) == 0x00000000u);
  CHECK(bits_of(denorm[2 * voxels]) == 0x7fc00000u);
  CHECK(bits_of(denorm[2 * voxels + 1]) == 0x7fc00000u);

  std::vector<float> null_residual(matrix_count), zero_residual(matrix_count),
      null_swiglu(swiglu_output_count), zero_swiglu(swiglu_output_count);
  c_null_residual.copy_to_host(null_residual.data(), null_residual.size());
  c_zero_residual.copy_to_host(zero_residual.data(), zero_residual.size());
  c_null_swiglu.copy_to_host(null_swiglu.data(), null_swiglu.size());
  c_zero_swiglu.copy_to_host(zero_swiglu.data(), zero_swiglu.size());
  CHECK(std::memcmp(null_residual.data(), zero_residual.data(),
                    null_residual.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(null_swiglu.data(), zero_swiglu.data(), null_swiglu.size() * sizeof(float)) ==
        0);

  // Every rejection below happens before access tracking/command mutation, so
  // the same batch remains usable and proves transactional validation.
  const uint64_t wrong_output_shape[] = {rows, inner + 1};
  DeviceTensor wrong_output = vk.allocate(TensorLayout::contiguous(wrong_output_shape, 2));
  {
    TensorBatch valid_after_rejection = vk.begin_batch();
    bool alias_rejected = false, shape_rejected = false;
    try {
      valid_after_rejection.layer_scale_residual_f32(vx, vx, vrb, vs);
    } catch (const std::invalid_argument&) {
      alias_rejected = true;
    }
    try {
      valid_after_rejection.swiglu_bias_f32(vsi, vsb, wrong_output);
    } catch (const std::invalid_argument&) {
      shape_rejected = true;
    }
    CHECK(alias_rejected && shape_rejected);
    valid_after_rejection.latent_denorm_f32(vl, vm, vsd, vlo);
    valid_after_rejection.submit().wait();
  }

  auto record_mixed = [&] {
    TensorBatch mixed = vk.begin_batch();
    for (int operation = 0; operation < 32; ++operation) {
      switch (operation % 3) {
      case 0:
        mixed.layer_scale_residual_f32(vx, vy, vrb, vs);
        break;
      case 1:
        mixed.swiglu_bias_f32(vsi, vsb, vso);
        break;
      default:
        mixed.latent_denorm_f32(vl, vm, vsd, vlo);
        break;
      }
    }
    return mixed.submit();
  };
  Submission warm_first = record_mixed(), warm_second = record_mixed();
  warm_first.wait();
  warm_second.wait();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 6; ++repeat) {
    Submission first = record_mixed(), second = record_mixed();
    Submission third = record_mixed();
    CHECK(first.value() < second.value() && second.value() < third.value());
    first.wait();
    second.wait();
    third.wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  }
  {
    TensorBatch too_many = vk.begin_batch();
    bool rejected = false;
    try {
      for (int operation = 0; operation < 33; ++operation)
        too_many.latent_denorm_f32(vl, vm, vsd, vlo);
    } catch (const std::logic_error&) {
      rejected = true;
    }
    CHECK(rejected);
    bool poisoned_submit_rejected = false;
    try {
      (void)too_many.submit();
    } catch (const std::logic_error&) {
      poisoned_submit_rejected = true;
    }
    CHECK(poisoned_submit_rejected);
  }

  // Submitted jobs retain all four resources even when every public wrapper
  // is dropped before completion, and release them after the token is done.
  const uint64_t used_before_drop = vk.pooled_used_bytes();
  Submission dropped;
  {
    DeviceTensor ti = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
    DeviceTensor tm = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
    DeviceTensor ts = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
    DeviceTensor to = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
    vk.upload(ti, latent.data(), latent.size());
    vk.upload(tm, mean.data(), mean.size());
    vk.upload(ts, std_dev.data(), std_dev.size());
    TensorBatch retained = vk.begin_batch();
    retained.latent_denorm_f32(ti, tm, ts, to);
    dropped = retained.submit();
  }
  dropped.wait();
  dropped = Submission{};
  Submission collected = record_mixed();
  collected.wait();
  CHECK(vk.pooled_used_bytes() == used_before_drop);
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_tensor_exact_bf16_rope, "synthetic") {
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

  auto run = [&](uint32_t mode, uint32_t rows, uint32_t heads, uint32_t head_dim) {
    const size_t count = static_cast<size_t>(rows) * heads * head_dim;
    std::vector<uint16_t> input(count);
    for (size_t i = 0; i < count; ++i) {
      if (i < 127)
        input[i] = static_cast<uint16_t>(i + 1);
      else if (i < 254)
        input[i] = static_cast<uint16_t>(0x8000u | (i - 126));
      else
        input[i] = static_cast<uint16_t>(((i * 977u + 0x3c00u) & 0x7fffu) |
                                         ((i & 7u) == 0u ? 0x8000u : 0u));
    }
    const uint16_t boundaries[] = {0x0000u, 0x8000u, 0x007fu, 0x807fu, 0x0080u, 0x8080u,
                                   0x0081u, 0x8081u, 0x7f80u, 0xff80u, 0x7fc1u};
    for (size_t i = 0; i < std::size(boundaries); ++i)
      input[254 + i] = boundaries[i];
    const uint32_t table_width = mode == 0 ? 96u : head_dim;
    std::vector<float> cosine, sine;
    if (mode == 0) {
      std::vector<double> positions(static_cast<size_t>(rows) * 3);
      for (size_t i = 0; i < positions.size(); ++i)
        positions[i] = static_cast<double>(static_cast<int>(i * 17u % 101u) - 50) / 8.0;
      auto tables = dit::build_h3_rope_tables(positions, 10000.0f, 16);
      cosine = std::move(tables.cosine);
      sine = std::move(tables.sine);
    } else {
      cosine.resize(static_cast<size_t>(rows) * head_dim);
      sine.resize(cosine.size());
      const uint32_t half = head_dim / 2;
      for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t j = 0; j < half; ++j) {
          const float angle = static_cast<float>((row + 1) * (j + 1)) / 37.0f;
          const float c = std::cos(angle), s = std::sin(angle);
          cosine[static_cast<size_t>(row) * head_dim + j] =
              cosine[static_cast<size_t>(row) * head_dim + j + half] = c;
          sine[static_cast<size_t>(row) * head_dim + j] =
              sine[static_cast<size_t>(row) * head_dim + j + half] = s;
        }
      }
      // Exact cancellation and an intermediate that underflows fp32 are part
      // of the backend-stable signed-zero policy.
      input[0] = input[head_dim / 2] = 0x3f80u;
      cosine[0] = cosine[head_dim / 2] = 0.5f;
      sine[0] = sine[head_dim / 2] = 0.5f;
      if (rows > 1) {
        const size_t data_row = static_cast<size_t>(heads) * head_dim;
        input[data_row + 1] = 0x0080u;
        input[data_row + 1 + head_dim / 2] = 0x0000u;
        const size_t table_row = head_dim;
        cosine[table_row + 1] = cosine[table_row + 1 + head_dim / 2] =
            std::numeric_limits<float>::min();
        sine[table_row + 1] = sine[table_row + 1 + head_dim / 2] = 0.0f;
        // The high-half FMA has a nonzero product that rounds into the fp32
        // subnormal range. Both backends canonicalize that internal product
        // before it can contribute to an otherwise normal result.
        input[data_row + 2] = 0x0080u;
        input[data_row + 2 + head_dim / 2] = 0x3f80u;
        cosine[table_row + 2 + head_dim / 2] = 0.75f;
        sine[table_row + 2 + head_dim / 2] = std::numeric_limits<float>::min();
      }
    }
    cuda::DeviceBuffer<__nv_bfloat16> cuda_data(count);
    cuda::DeviceBuffer<float> cuda_cos(cosine.size()), cuda_sin(sine.size());
    cuda_data.copy_from_host(reinterpret_cast<const __nv_bfloat16*>(input.data()), count);
    cuda_cos.copy_from_host(cosine.data(), cosine.size());
    cuda_sin.copy_from_host(sine.data(), sine.size());
    if (mode == 0)
      cuda::launch_rope_h3(cuda_data.get(), cuda_cos.get(), cuda_sin.get(), rows, heads, head_dim,
                           nullptr);
    else
      cuda::launch_rope_neox(cuda_data.get(), cuda_cos.get(), cuda_sin.get(), rows, heads, head_dim,
                             nullptr);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> expected(count), actual(count);
    cuda_data.copy_to_host(reinterpret_cast<__nv_bfloat16*>(expected.data()), count);

    const uint64_t data_shape[] = {rows, heads, head_dim};
    const uint64_t table_shape[] = {rows, table_width};
    DeviceTensor vk_data =
        vk.allocate(TensorLayout::contiguous(data_shape, 3), ScalarType::kBFloat16);
    DeviceTensor vk_cos = vk.allocate(TensorLayout::contiguous(table_shape, 2));
    DeviceTensor vk_sin = vk.allocate(TensorLayout::contiguous(table_shape, 2));
    vk.upload_bytes(vk_data, input.data(), input.size() * sizeof(uint16_t));
    vk.upload(vk_cos, cosine.data(), cosine.size());
    vk.upload(vk_sin, sine.data(), sine.size());
    TensorBatch batch = vk.begin_batch();
    if (mode == 0)
      batch.rope_h3_bf16(vk_data, vk_cos, vk_sin);
    else
      batch.rope_neox_bf16(vk_data, vk_cos, vk_sin);
    batch.submit().wait();
    vk.download_bytes(vk_data, actual.data(), actual.size() * sizeof(uint16_t));
    size_t mismatch = count;
    for (size_t i = 0; i < count; ++i) {
      if (expected[i] != actual[i]) {
        mismatch = i;
        break;
      }
    }
    CHECK_MSG(mismatch == count,
              "CUDA/Vulkan BF16 RoPE mode%u %ux%ux%u mismatch at %zu: %04x != %04x", mode, rows,
              heads, head_dim, mismatch, mismatch == count ? 0 : expected[mismatch],
              mismatch == count ? 0 : actual[mismatch]);
    if (mode == 0) {
      bool tail_preserved = true;
      for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t head = 0; head < heads; ++head)
          for (uint32_t d = 96; d < head_dim; ++d) {
            const size_t at = (static_cast<size_t>(row) * heads + head) * head_dim + d;
            tail_preserved &= actual[at] == input[at];
          }
      CHECK(tail_preserved);
    } else {
      CHECK(actual[0] == 0x0000u);
      CHECK(actual[static_cast<size_t>(heads) * head_dim + 1] == 0x0000u);
      CHECK(actual[static_cast<size_t>(heads) * head_dim + 2 + head_dim / 2] == 0x3f40u);
    }
  };
  run(0, 5, 7, 128);
  run(1, 5, 3, 72);
  run(1, 3, 5, 128);

  const uint64_t used_before_reuse = vk.pooled_used_bytes();
  {
    const uint64_t data_shape[] = {2, 3, 128}, table_shape[] = {2, 96};
    DeviceTensor first =
        vk.allocate(TensorLayout::contiguous(data_shape, 3), ScalarType::kBFloat16);
    DeviceTensor second =
        vk.allocate(TensorLayout::contiguous(data_shape, 3), ScalarType::kBFloat16);
    DeviceTensor cosine = vk.allocate(TensorLayout::contiguous(table_shape, 2));
    DeviceTensor sine = vk.allocate(TensorLayout::contiguous(table_shape, 2));
    std::vector<uint16_t> zeros(2 * 3 * 128);
    std::vector<float> ones(2 * 96, 1.0f), table_zeros(2 * 96);
    vk.upload_bytes(first, zeros.data(), zeros.size() * 2);
    vk.upload_bytes(second, zeros.data(), zeros.size() * 2);
    vk.upload(cosine, ones.data(), ones.size());
    vk.upload(sine, table_zeros.data(), table_zeros.size());
    auto submit = [&](DeviceTensor& data) {
      TensorBatch batch = vk.begin_batch();
      batch.rope_h3_bf16(data, cosine, sine);
      return batch.submit();
    };
    Submission warm_a = submit(first), warm_b = submit(second);
    warm_a.wait();
    warm_b.wait();
    const uint64_t stable_reserved = vk.reserved_bytes();
    const uint64_t stable_descriptors = vk.descriptor_set_allocations();
    for (int repeat = 0; repeat < 4; ++repeat) {
      Submission a = submit(first), b = submit(second), c = submit(first);
      CHECK(b.value() > a.value() && c.value() > b.value());
      a.wait();
      b.wait();
      c.wait();
      CHECK(vk.reserved_bytes() == stable_reserved);
      CHECK(vk.descriptor_set_allocations() == stable_descriptors);
    }
    const uint64_t odd_shape[] = {2, 3, 127};
    const uint64_t short_table_shape[] = {2, 95};
    DeviceTensor odd = vk.allocate(TensorLayout::contiguous(odd_shape, 3), ScalarType::kBFloat16);
    DeviceTensor short_table = vk.allocate(TensorLayout::contiguous(short_table_shape, 2));
    DeviceTensor wrong_type =
        vk.allocate(TensorLayout::contiguous(table_shape, 2), ScalarType::kBFloat16);
    auto valid_after_rejection = [&](auto&& invalid) {
      TensorBatch batch = vk.begin_batch();
      bool rejected = false;
      try {
        invalid(batch);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      CHECK(rejected);
      batch.rope_h3_bf16(first, cosine, sine);
      batch.submit().wait();
    };
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rope_h3_bf16(odd, cosine, sine);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rope_h3_bf16(first, short_table, sine);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rope_h3_bf16(first, wrong_type, sine);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rope_h3_bf16(first, cosine, cosine);
    });
  }
  {
    TensorBatch collect_completed_slots = vk.begin_batch();
  }
  CHECK(vk.pooled_used_bytes() == used_before_reuse);
}
