#include "harness.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nn_kernels.cuh"
#include "vidfab/cuda/vae_kernels.cuh"
#include "vidfab/vulkan/tensor.h"


VIDFAB_TEST(cuda_vulkan_tensor_exact_copy_and_add) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  bool full_exact_rejected = false;
  try {
    vk.require_full_fp32_arithmetic_exactness();
  } catch (const std::runtime_error&) {
    full_exact_rejected = true;
  }
  CHECK(full_exact_rejected == !physical.front().info().fp32_denorm_preserve);
  CHECK(vk.full_fp32_arithmetic_exactness() ==
        physical.front().info().fp32_denorm_preserve);
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
  const uint32_t copy_special[] = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x00800000u,
      0x3f800000u, 0x3f800000u, 0x7f800000u, 0xff800000u,
      0x7fc12345u, 0x7fa54321u};
  const uint32_t add_rhs[] = {
      0x80000000u, 0x80000000u, 0x00000001u, 0x807fffffu,
      0x33800000u, 0x33800001u, 0x3f800000u, 0xbf800000u,
      0x00000000u, 0x00000000u};
  std::memcpy(a.data(), copy_special, sizeof(copy_special));
  std::memcpy(b.data(), add_rhs, sizeof(add_rhs));

  cuda::DeviceBuffer<float> cuda_a(count), cuda_b(count), cuda_copy(count), cuda_sum(count);
  cuda_a.copy_from_host(a.data(), count);
  cuda_b.copy_from_host(b.data(), count);
  VIDFAB_CUDA_CHECK(cudaMemcpy(cuda_copy.get(), cuda_a.get(), count * sizeof(float),
                               cudaMemcpyDeviceToDevice));
  cuda::launch_add(cuda_a.get(), cuda_b.get(), cuda_sum.get(), count, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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

  CHECK(std::memcmp(cuda_copy_host.data(), vk_copy_host.data(),
                    count * sizeof(float)) == 0);
  // NaN results are excluded from the arithmetic exactness claim; every
  // finite/infinite result, including signed-zero/subnormal inputs, is exact.
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits = 0;
    std::memcpy(&bits, &a[i], sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) continue;
    // Indices 2/3 require denormal-preserving arithmetic. On devices without
    // that Vulkan mode, the explicit capability gate above fails rather than
    // claiming those results are CUDA-exact.
    if (!vk.full_fp32_arithmetic_exactness() && (i == 2 || i == 3)) continue;
    uint32_t cuda_bits = 0, vk_bits = 0;
    std::memcpy(&cuda_bits, &cuda_sum_host[i], sizeof(cuda_bits));
    std::memcpy(&vk_bits, &vk_sum_host[i], sizeof(vk_bits));
    CHECK_MSG(cuda_bits == vk_bits,
              "CUDA/Vulkan fp32 add mismatch at %zu: %08x != %08x", i,
              cuda_bits, vk_bits);
  }
}

VIDFAB_TEST(cuda_vulkan_tensor_exact_conversion_and_layout_ops) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
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
      0x00000000u, 0x80000000u, 0x00000001u, 0x007fffffu,
      0x00800000u, 0x33800000u, 0x33800001u, 0x387fffffu,
      0x38800000u, 0x3f800000u, 0x477fe000u, 0x7f800000u,
      0xff800000u, 0x7fc12345u, 0x7fa54321u, 0xffc12345u,
      0x33000000u, 0x33000001u, 0x3f808000u, 0x3f808001u,
      0x00800000u};
  std::memcpy(input.data(), special, sizeof(special));
  for (int i = 0; i < cols; ++i) bias[i] = static_cast<float>((i % 13) - 6) / 32.0f;
  for (int i = 0; i < 4; ++i) bias[i] = 0.0f;
  // Two normal operands whose exact sum is the smallest subnormal.
  const uint32_t normal_above_min = 0x00800001u;
  const uint32_t negative_min_normal = 0x80800000u;
  std::memcpy(&input[4], &normal_above_min, sizeof(float));
  std::memcpy(&bias[4], &negative_min_normal, sizeof(float));

  cuda::DeviceBuffer<float> c_input(count), c_bf16_back(count), c_f16_back(count);
  cuda::DeviceBuffer<uint16_t> c_bf16(count), c_f16(count);
  c_input.copy_from_host(input.data(), count);
  cuda::launch_narrow_to_bf16(
      c_input.get(), reinterpret_cast<__nv_bfloat16*>(c_bf16.get()), count, nullptr);
  cuda::launch_widen_bf16(reinterpret_cast<const __nv_bfloat16*>(c_bf16.get()),
                          c_bf16_back.get(), count, nullptr);
  cuda::launch_narrow_f16(c_input.get(), c_f16.get(), count, nullptr);
  cuda::launch_widen_f16(c_f16.get(), c_f16_back.get(), count, nullptr);

  cuda::DeviceBuffer<float> c_transpose(count), c_biased(count);
  cuda::DeviceBuffer<float> c_bias(cols);
  c_bias.copy_from_host(bias.data(), bias.size());
  VIDFAB_CUDA_CHECK(cudaMemcpy(c_biased.get(), c_input.get(), count * sizeof(float),
                               cudaMemcpyDeviceToDevice));
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
  cuda::launch_gather_rows_f32(c_matrix.get(), c_indices.get(), c_gathered.get(),
                               selected_rows, matrix_cols, nullptr);
  cuda::launch_scatter_rows_f32(c_gathered.get(), c_indices.get(), c_scattered.get(),
                                selected_rows, matrix_cols, nullptr);

  constexpr int heads = 3, sequence = 5, head_dim = 7;
  constexpr size_t heads_count = static_cast<size_t>(heads) * sequence * head_dim;
  std::vector<float> heads_input(heads_count);
  for (size_t i = 0; i < heads_count; ++i) heads_input[i] = input[i];
  cuda::DeviceBuffer<float> c_heads(heads_count);
  cuda::DeviceBuffer<uint16_t> c_tokens(heads_count);
  c_heads.copy_from_host(heads_input.data(), heads_input.size());
  cuda::launch_heads_to_tokens_bf16(
      c_heads.get(), reinterpret_cast<__nv_bfloat16*>(c_tokens.get()), sequence,
      heads, head_dim, nullptr);

  // Every axis is nontrivial, including temporal patching. 288 output values
  // leave a 32-invocation workgroup tail and catch an output-plane expression
  // that accidentally omits patch_t.
  constexpr int depth_t = 2, depth_h = 2, depth_w = 3, depth_channels = 3;
  constexpr int patch_t = 2, patch = 2;
  constexpr size_t depth_count = static_cast<size_t>(depth_t) * depth_h * depth_w *
                                 depth_channels * patch_t * patch * patch;
  std::vector<float> depth_input(depth_count);
  for (size_t i = 0; i < depth_input.size(); ++i)
    depth_input[i] = static_cast<float>(i) + 0.25f;
  cuda::DeviceBuffer<float> c_depth_input(depth_count), c_depth_output(depth_count);
  c_depth_input.copy_from_host(depth_input.data(), depth_input.size());
  cuda::launch_depth_to_space(c_depth_input.get(), c_depth_output.get(), depth_t,
                              depth_h, depth_w, depth_channels, patch_t, patch, nullptr);

  // Widening also has an explicit CUDA NaN policy for arbitrary checkpoint
  // half bits, independently of values produced by the narrowing kernel.
  constexpr size_t raw_half_count = 1u << 16;
  std::vector<uint16_t> raw_half_host(raw_half_count);
  for (size_t i = 0; i < raw_half_count; ++i)
    raw_half_host[i] = static_cast<uint16_t>(i);
  cuda::DeviceBuffer<uint16_t> c_raw_half(raw_half_count);
  cuda::DeviceBuffer<float> c_raw_half_wide(raw_half_count);
  c_raw_half.copy_from_host(raw_half_host.data(), raw_half_count);
  cuda::launch_widen_f16(c_raw_half.get(), c_raw_half_wide.get(), raw_half_count,
                         nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

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
  DeviceTensor v_indices = vk.allocate(TensorLayout::contiguous(&index_extent, 1),
                                       ScalarType::kInt32);
  DeviceTensor v_gathered = vk.allocate(TensorLayout::contiguous(selected_extents, 2));
  DeviceTensor v_scattered = vk.allocate(TensorLayout::contiguous(matrix_extents, 2));
  vk.upload(v_matrix, matrix.data(), matrix.size());
  vk.upload_bytes(v_indices, indices_host, sizeof(indices_host));
  vk.upload(v_scattered, scatter_initial.data(), scatter_initial.size());

  const uint64_t heads_extent = heads_count;
  DeviceTensor v_heads = vk.allocate(TensorLayout::contiguous(&heads_extent, 1));
  DeviceTensor v_tokens = vk.allocate(TensorLayout::contiguous(&heads_extent, 1),
                                      ScalarType::kBFloat16);
  vk.upload(v_heads, heads_input.data(), heads_input.size());
  const uint64_t depth_extent = depth_count;
  DeviceTensor v_depth_input = vk.allocate(TensorLayout::contiguous(&depth_extent, 1));
  DeviceTensor v_depth_output = vk.allocate(TensorLayout::contiguous(&depth_extent, 1));
  vk.upload(v_depth_input, depth_input.data(), depth_input.size());
  const uint64_t raw_half_extent = raw_half_count;
  DeviceTensor v_raw_half = vk.allocate(TensorLayout::contiguous(&raw_half_extent, 1),
                                        ScalarType::kFloat16);
  DeviceTensor v_raw_half_wide = vk.allocate(TensorLayout::contiguous(&raw_half_extent, 1));
  vk.upload_bytes(v_raw_half, raw_half_host.data(),
                  raw_half_host.size() * sizeof(uint16_t));

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
  batch.depth_to_space(v_depth_input, v_depth_output, depth_t, depth_h,
                       depth_w, depth_channels, patch_t, patch);
  batch.convert(v_raw_half, v_raw_half_wide);
  batch.submit().wait();

  auto compare_bytes = [&](auto& cuda_buffer, DeviceTensor& vulkan_tensor,
                           size_t elements, const char* label) {
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
    CHECK_MSG(mismatch == elements,
              "CUDA/Vulkan %s mismatch at %zu: %llx != %llx", label,
              mismatch, static_cast<unsigned long long>(cuda_bits),
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
    return (bits & 0x7f800000u) == 0x7f800000u &&
           (bits & 0x007fffffu) != 0;
  };
  auto is_subnormal = [](uint32_t bits) {
    return (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0;
  };
  for (size_t i = 0; i < count; ++i) {
    const uint32_t input_bits = bits_of(input[i]);
    const uint32_t bias_bits = bits_of(bias[i % cols]);
    const uint32_t cuda_bits = bits_of(biased_cuda[i]);
    if (is_nan(input_bits) || is_nan(bias_bits) || is_nan(cuda_bits)) continue;
    const bool uses_subnormal = is_subnormal(input_bits) || is_subnormal(bias_bits) ||
                                is_subnormal(cuda_bits);
    if (uses_subnormal) ++subnormal_cases;
    if (uses_subnormal && !vk.full_fp32_arithmetic_exactness()) continue;
    CHECK_MSG(cuda_bits == bits_of(biased_vulkan[i]),
              "CUDA/Vulkan add-bias mismatch at %zu: %08x != %08x", i,
              cuda_bits, bits_of(biased_vulkan[i]));
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
          const int feature = channel * patch_volume +
              ((ot % patch_t) * patch + oh % patch) * patch + ow % patch;
          const size_t source_index = static_cast<size_t>(token) *
                                          (depth_channels * patch_volume) +
                                      feature;
          CHECK(depth_vulkan[output_index] == depth_input[source_index]);
        }
      }
    }
  }
}

VIDFAB_TEST(cuda_vulkan_tensor_exact_vae_norms) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_fp32_vae_normalization()) {
    bool rejected = false;
    try { vk.require_exact_fp32_vae_normalization(); }
    catch (const std::runtime_error&) { rejected = true; }
    CHECK(rejected);
    return;
  }
  vk.require_exact_fp32_vae_normalization();

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
      for (int col = 0; col < dim; ++col) input[col] = 2.0f;
      for (int col = 0; col < dim; ++col)
        input[dim + col] = (col & 1) ? -0.0f : 0.0f;
    } else if (pattern == 2) {
      // Large values whose squares and reduction totals remain finite.
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = (i & 1) ? -1.0e16f : 1.0e16f;
    } else if (pattern == 3) {
      std::fill(input.begin(), input.end(), 0.0f);
    }

    cuda::DeviceBuffer<float> c_input(count), c_weight(dim), c_bias(dim),
        c_rms(count), c_layer(count);
    c_input.copy_from_host(input.data(), input.size());
    c_weight.copy_from_host(weight.data(), weight.size());
    c_bias.copy_from_host(bias.data(), bias.size());
    cuda::launch_rmsnorm(c_input.get(), c_weight.get(), c_rms.get(), rows, dim,
                         epsilon, nullptr);
    cuda::launch_layernorm(c_input.get(), c_weight.get(), c_bias.get(), c_layer.get(),
                           rows, dim, epsilon, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

    const uint64_t shape_extents[] = {static_cast<uint64_t>(rows),
                                      static_cast<uint64_t>(dim)};
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

    auto compare = [&](auto& cuda_buffer, DeviceTensor& vulkan_tensor,
                       const char* label) {
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
      CHECK_MSG(mismatch == count,
                "CUDA/Vulkan %s %dx%d mismatch at %zu: %08x != %08x", label,
                rows, dim, mismatch, cuda_bits, vulkan_bits);
    };
    compare(c_rms, v_rms, "VAE RMSNorm");
    compare(c_layer, v_layer, "VAE LayerNorm");
  };

  run_case(3, 513, 1.0e-6f, 0);   // odd scalar and workgroup tail
  run_case(2, 128, 1.0e-5f, 1);   // test-model width, signed zero/variance zero
  run_case(2, 2048, 1.0e-6f, 2);  // shipped video-VAE width
  run_case(2, 7, 1.0e-6f, 1);     // dimension below a warp
  run_case(1, 9, std::numeric_limits<float>::min(), 3);

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
    recoverable.rms_norm(input, weight, rms_a,
                         std::numeric_limits<float>::denorm_min());
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

VIDFAB_TEST(cuda_vulkan_tensor_exact_shared_norms) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options; options.enable_timeline_semaphore = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_fp32_vae_normalization()) return;

  constexpr int rows = 9;
  constexpr int dim = 128;
  constexpr size_t count = static_cast<size_t>(rows) * dim;
  std::vector<uint16_t> input(count), weight(dim);
  auto exact_bf16 = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
  };
  for (size_t i = 0; i < count; ++i)
    input[i] = exact_bf16(static_cast<float>(static_cast<int>(i % 31) - 15) / 8.0f);
  for (int i = 0; i < dim; ++i)
    weight[i] = exact_bf16(0.5f + static_cast<float>(i % 8) / 8.0f);
  cuda::DeviceBuffer<uint16_t> c_input(count), c_weight(dim), c_output(count);
  c_input.copy_from_host(input.data(), count);
  c_weight.copy_from_host(weight.data(), dim);
  cuda::launch_rmsnorm(reinterpret_cast<const __nv_bfloat16*>(c_input.get()),
                       reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
                       reinterpret_cast<__nv_bfloat16*>(c_output.get()), rows, dim,
                       1.0e-5f, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> cuda_output(count), vulkan_output(count);
  c_output.copy_to_host(cuda_output.data(), count);

  const uint64_t extents[] = {rows, dim};
  const uint64_t feature = dim;
  DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(extents, 2),
                                     ScalarType::kBFloat16);
  DeviceTensor v_weight = vk.allocate(TensorLayout::contiguous(&feature, 1),
                                      ScalarType::kBFloat16);
  DeviceTensor v_output = vk.allocate(TensorLayout::contiguous(extents, 2),
                                      ScalarType::kBFloat16);
  vk.upload_bytes(v_input, input.data(), input.size() * sizeof(uint16_t));
  vk.upload_bytes(v_weight, weight.data(), weight.size() * sizeof(uint16_t));
  TensorBatch batch = vk.begin_batch();
  batch.rms_norm_bf16(v_input, v_weight, v_output, 1.0e-5f);
  batch.submit().wait();
  vk.download_bytes(v_output, vulkan_output.data(),
                    vulkan_output.size() * sizeof(uint16_t));
  size_t mismatch = count;
  for (size_t i = 0; i < count; ++i) {
    if (cuda_output[i] != vulkan_output[i]) { mismatch = i; break; }
  }
  CHECK_MSG(mismatch == count,
            "CUDA/Vulkan BF16 head RMS mismatch at %zu: %04x != %04x", mismatch,
            mismatch == count ? 0 : cuda_output[mismatch],
            mismatch == count ? 0 : vulkan_output[mismatch]);
}

int main() { return ::vidfab::test::run_all(); }
