#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_linear_weight_f8_i8_exact, "synthetic") {
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

  auto exact_case = [&](const LinearWeightUpload& upload, const std::vector<uint16_t>& expected) {
    LinearWeight weight = LinearWeight::upload(vk, upload);
    const uint64_t shape[] = {upload.out_features, upload.in_features};
    DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);
    TensorBatch batch = vk.begin_batch();
    weight.materialize_bf16(batch, dense);
    batch.submit().wait();
    std::vector<uint16_t> actual(expected.size());
    vk.download_bytes(dense, actual.data(), actual.size() * sizeof(uint16_t));
    size_t mismatch = expected.size();
    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != actual[i]) {
        mismatch = i;
        break;
      }
    }
    CHECK_MSG(mismatch == expected.size(),
              "CUDA/Vulkan linear weight mismatch at %zu: %04x != %04x", mismatch,
              mismatch == expected.size() ? 0u : expected[mismatch],
              mismatch == expected.size() ? 0u : actual[mismatch]);
  };

  std::vector<uint8_t> f8(256);
  for (uint32_t i = 0; i < 256; ++i)
    f8[i] = static_cast<uint8_t>(i);
  const float f8_scale = 0.75f;
  cuda::DeviceBuffer<uint8_t> cuda_f8(f8.size());
  cuda::DeviceBuffer<float> cuda_f8_scale(1);
  cuda::DeviceBuffer<__nv_bfloat16> cuda_f8_output(f8.size());
  cuda_f8.copy_from_host(f8.data(), f8.size());
  cuda_f8_scale.copy_from_host(&f8_scale, 1);
  cuda::launch_dequant_f8e4m3(cuda_f8.get(), cuda_f8_scale.get(), cuda_f8_output.get(), f8.size(),
                              nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> f8_expected(f8.size());
  SLOPFAB_CUDA_CHECK(cudaMemcpy(f8_expected.data(), cuda_f8_output.get(),
                                f8_expected.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  LinearWeightUpload f8_upload;
  f8_upload.format = LinearWeightFormat::kFloat8E4M3;
  f8_upload.out_features = 1;
  f8_upload.in_features = static_cast<uint32_t>(f8.size());
  f8_upload.data = f8.data();
  f8_upload.data_bytes = f8.size();
  f8_upload.weight_scale = &f8_scale;
  f8_upload.weight_scale_count = 1;
  f8_upload.has_fp8_input_scale = true;
  f8_upload.fp8_input_scale = 0.125f;
  f8_upload.full_precision_matrix_mult = true;
  exact_case(f8_upload, f8_expected);
  {
    LinearWeight metadata = LinearWeight::upload(vk, f8_upload);
    CHECK(metadata.has_fp8_input_scale());
    CHECK(metadata.fp8_input_scale() == f8_upload.fp8_input_scale);
    CHECK(metadata.full_precision_matrix_mult());
  }

  constexpr uint32_t i8_rows = 3, i8_columns = 131;
  std::vector<int8_t> i8(static_cast<size_t>(i8_rows) * i8_columns);
  for (size_t i = 0; i < i8.size(); ++i) {
    i8[i] = static_cast<int8_t>((i * 73u + 128u) & 0xffu);
  }
  const float i8_scale[] = {0.5f, -0.25f, 1.5f};
  cuda::DeviceBuffer<int8_t> cuda_i8(i8.size());
  cuda::DeviceBuffer<float> cuda_i8_scale(i8_rows);
  cuda::DeviceBuffer<__nv_bfloat16> cuda_i8_output(i8.size());
  cuda_i8.copy_from_host(i8.data(), i8.size());
  cuda_i8_scale.copy_from_host(i8_scale, i8_rows);
  cuda::launch_dequant_i8_per_channel(cuda_i8.get(), cuda_i8_scale.get(), cuda_i8_output.get(),
                                      i8_rows, i8_columns, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> i8_expected(i8.size());
  SLOPFAB_CUDA_CHECK(cudaMemcpy(i8_expected.data(), cuda_i8_output.get(),
                                i8_expected.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  LinearWeightUpload i8_upload;
  i8_upload.format = LinearWeightFormat::kInt8;
  i8_upload.out_features = i8_rows;
  i8_upload.in_features = i8_columns;
  i8_upload.data = i8.data();
  i8_upload.data_bytes = i8.size();
  i8_upload.weight_scale = i8_scale;
  i8_upload.weight_scale_count = i8_rows;
  exact_case(i8_upload, i8_expected);

  // Native dense formats are immutable checkpoint payloads. Every possible
  // 16-bit pattern, including all NaN payloads and subnormals, must survive a
  // same-format materialization without a floating-point round trip.
  auto raw_dense_case = [&](LinearWeightFormat format, ScalarType type) {
    std::vector<uint16_t> patterns(65536);
    for (uint32_t i = 0; i < patterns.size(); ++i)
      patterns[i] = static_cast<uint16_t>(i);
    LinearWeightUpload upload;
    upload.format = format;
    upload.out_features = 256;
    upload.in_features = 256;
    upload.data = patterns.data();
    upload.data_bytes = patterns.size() * sizeof(uint16_t);
    LinearWeight weight = LinearWeight::upload(vk, upload);
    const uint64_t shape[] = {256, 256};
    DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2), type);
    TensorBatch batch = vk.begin_batch();
    if (type == ScalarType::kFloat16)
      weight.materialize_f16(batch, dense);
    else
      weight.materialize_bf16(batch, dense);
    batch.submit().wait();
    std::vector<uint16_t> actual(patterns.size());
    vk.download_bytes(dense, actual.data(), actual.size() * sizeof(uint16_t));
    CHECK(std::memcmp(patterns.data(), actual.data(), patterns.size() * sizeof(uint16_t)) == 0);
  };
  raw_dense_case(LinearWeightFormat::kFloat16, ScalarType::kFloat16);
  raw_dense_case(LinearWeightFormat::kBFloat16, ScalarType::kBFloat16);

  auto cross_dense_case = [&](LinearWeightFormat format, ScalarType source_type, const void* input,
                              uint64_t count, bool output_fp16) {
    const uint64_t input_bytes = count * (source_type == ScalarType::kFloat32 ? 4ull : 2ull);
    cuda::DeviceBuffer<uint8_t> cuda_input(input_bytes);
    cuda::DeviceBuffer<uint16_t> cuda_output(count);
    cuda_input.copy_from_host(static_cast<const uint8_t*>(input), input_bytes);
    const int source_op = source_type == ScalarType::kFloat32   ? 0
                          : source_type == ScalarType::kFloat16 ? 1
                                                                : 2;
    dense_weight_convert_probe<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        cuda_input.get(), cuda_output.get(), static_cast<int>(count), source_op, output_fp16);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> expected(count), actual(count);
    cuda_output.copy_to_host(expected.data(), count);
    LinearWeightUpload upload;
    upload.format = format;
    upload.out_features = 1;
    upload.in_features = static_cast<uint32_t>(count);
    upload.data = input;
    upload.data_bytes = input_bytes;
    LinearWeight weight = LinearWeight::upload(vk, upload);
    const uint64_t shape[] = {1, count};
    DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2),
                                     output_fp16 ? ScalarType::kFloat16 : ScalarType::kBFloat16);
    TensorBatch batch = vk.begin_batch();
    if (output_fp16)
      weight.materialize_f16(batch, dense);
    else
      weight.materialize_bf16(batch, dense);
    batch.submit().wait();
    vk.download_bytes(dense, actual.data(), count * sizeof(uint16_t));
    CHECK(std::memcmp(expected.data(), actual.data(), count * 2) == 0);
  };
  std::vector<uint16_t> all_half(65536), all_bf16(65536);
  for (uint32_t i = 0; i < 65536; ++i) {
    all_half[i] = static_cast<uint16_t>(i);
    all_bf16[i] = static_cast<uint16_t>(i);
  }
  cross_dense_case(LinearWeightFormat::kFloat16, ScalarType::kFloat16, all_half.data(),
                   all_half.size(), false);
  cross_dense_case(LinearWeightFormat::kBFloat16, ScalarType::kBFloat16, all_bf16.data(),
                   all_bf16.size(), true);
  std::vector<uint32_t> f32_bits(4099);
  uint32_t state = 0x31415926u;
  for (size_t i = 0; i < f32_bits.size(); ++i) {
    state = state * 1664525u + 1013904223u;
    f32_bits[i] = state;
  }
  const uint32_t special_f32[] = {0u,          0x80000000u, 1u,          0x007fffffu, 0x00800000u,
                                  0x7f7fffffu, 0x7f800000u, 0xff800000u, 0x7fc12345u};
  std::copy(std::begin(special_f32), std::end(special_f32), f32_bits.begin());
  cross_dense_case(LinearWeightFormat::kFloat32, ScalarType::kFloat32, f32_bits.data(),
                   f32_bits.size(), false);
  cross_dense_case(LinearWeightFormat::kFloat32, ScalarType::kFloat32, f32_bits.data(),
                   f32_bits.size(), true);
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_linear_weight_nvfp4_nf4_exact, "synthetic") {
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

  auto materialize = [&](const LinearWeightUpload& upload, bool fp16) {
    LinearWeight weight = LinearWeight::upload(vk, upload);
    const uint64_t shape[] = {upload.out_features, upload.in_features};
    DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2),
                                     fp16 ? ScalarType::kFloat16 : ScalarType::kBFloat16);
    TensorBatch batch = vk.begin_batch();
    if (fp16)
      weight.materialize_f16(batch, dense);
    else
      weight.materialize_bf16(batch, dense);
    batch.submit().wait();
    std::vector<uint16_t> result(static_cast<size_t>(upload.out_features) * upload.in_features);
    vk.download_bytes(dense, result.data(), result.size() * sizeof(uint16_t));
    return result;
  };

  // Three row tiles by five contraction tiles catches both physical scale-tile
  // strides; the previous one-tile shape could not distinguish a flat layout.
  constexpr uint32_t nv_out = 384, nv_in = 320;
  constexpr size_t nv_count = static_cast<size_t>(nv_out) * nv_in;
  std::vector<uint8_t> nv_packed((nv_count + 1) / 2);
  for (size_t i = 0; i < nv_packed.size(); ++i) {
    nv_packed[i] = static_cast<uint8_t>(((2 * i & 15u) << 4u) | ((2 * i + 1u) & 15u));
  }
  std::vector<uint8_t> nv_scales(nv_count / 16);
  for (size_t i = 0; i < nv_scales.size(); ++i) {
    // Covers every E4M3 bit pattern twice in the checkpoint's already-swizzled
    // physical scale array.
    nv_scales[i] = static_cast<uint8_t>(i);
  }
  const float nv_global = 1.3580322e-3f;
  cuda::DeviceBuffer<uint8_t> d_nv(nv_packed.size()), d_nv_scales(nv_scales.size());
  cuda::DeviceBuffer<__nv_bfloat16> d_nv_output(nv_count);
  d_nv.copy_from_host(nv_packed.data(), nv_packed.size());
  d_nv_scales.copy_from_host(nv_scales.data(), nv_scales.size());
  cuda::launch_dequant_nvfp4(d_nv.get(), d_nv_scales.get(), nv_global, d_nv_output.get(), nv_out,
                             nv_in, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> nv_expected(nv_count);
  SLOPFAB_CUDA_CHECK(cudaMemcpy(nv_expected.data(), d_nv_output.get(), nv_count * sizeof(uint16_t),
                                cudaMemcpyDeviceToHost));
  LinearWeightUpload nv_upload;
  nv_upload.format = LinearWeightFormat::kNVFloat4;
  nv_upload.out_features = nv_out;
  nv_upload.in_features = nv_in;
  nv_upload.data = nv_packed.data();
  nv_upload.data_bytes = nv_packed.size();
  nv_upload.block_scale = nv_scales.data();
  nv_upload.block_scale_count = nv_scales.size();
  nv_upload.global_scale = nv_global;
  const auto nv_actual = materialize(nv_upload, false);
  CHECK(std::memcmp(nv_expected.data(), nv_actual.data(), nv_count * sizeof(uint16_t)) == 0);

  constexpr uint32_t nf_out = 129, nf_in = 129;
  constexpr size_t nf_count = static_cast<size_t>(nf_out) * nf_in;
  constexpr uint32_t block = 64, nested_block = 256;
  const size_t nf_blocks = 1 + (nf_count - 1) / block;
  const size_t nf_nested_blocks = 1 + (nf_blocks - 1) / nested_block;
  std::vector<uint8_t> nf_packed((nf_count + 1) / 2), nf_absmax(nf_blocks);
  for (size_t i = 0; i < nf_packed.size(); ++i) {
    nf_packed[i] = static_cast<uint8_t>((((i * 5 + 3) & 15) << 4) | ((i * 11 + 9) & 15));
  }
  for (size_t i = 0; i < nf_absmax.size(); ++i)
    nf_absmax[i] = static_cast<uint8_t>((i * 73 + 19) & 255);
  const std::array<float, 16> nf_map = {-1.0f,        -0.6961928f,  -0.52507305f, -0.39491749f,
                                        -0.28444138f, -0.18477343f, -0.09105004f, 0.0f,
                                        0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
                                        0.44070983f,  0.56261700f,  0.72295684f,  1.0f};
  std::array<float, 256> nf_nested_map{};
  for (size_t i = 0; i < nf_nested_map.size(); ++i)
    nf_nested_map[i] = (static_cast<float>(i) - 127.0f) / 128.0f;
  std::vector<float> nf_nested_absmax(nf_nested_blocks);
  for (size_t i = 0; i < nf_nested_absmax.size(); ++i)
    nf_nested_absmax[i] = 0.75f + static_cast<float>(i) * 1.25f;
  const float nf_offset = 0.21360844373703003f;
  cuda::DeviceBuffer<uint8_t> d_nf(nf_packed.size()), d_nf_absmax(nf_absmax.size());
  cuda::DeviceBuffer<float> d_nf_map(nf_map.size()), d_nf_nested_map(nf_nested_map.size()),
      d_nf_nested_absmax(nf_nested_absmax.size());
  cuda::DeviceBuffer<uint16_t> d_nf_bf16(nf_count), d_nf_f16(nf_count);
  d_nf.copy_from_host(nf_packed.data(), nf_packed.size());
  d_nf_absmax.copy_from_host(nf_absmax.data(), nf_absmax.size());
  d_nf_map.copy_from_host(nf_map.data(), nf_map.size());
  d_nf_nested_map.copy_from_host(nf_nested_map.data(), nf_nested_map.size());
  d_nf_nested_absmax.copy_from_host(nf_nested_absmax.data(), nf_nested_absmax.size());
  cuda::launch_dequant_nf4(d_nf.get(), d_nf_absmax.get(), d_nf_map.get(), d_nf_nested_map.get(),
                           d_nf_nested_absmax.get(), block, nested_block, nf_offset,
                           reinterpret_cast<__nv_bfloat16*>(d_nf_bf16.get()), nf_out, nf_in,
                           nullptr);
  cuda::launch_dequant_nf4_f16(d_nf.get(), d_nf_absmax.get(), d_nf_map.get(), d_nf_nested_map.get(),
                               d_nf_nested_absmax.get(), block, nested_block, nf_offset,
                               reinterpret_cast<__half*>(d_nf_f16.get()), nf_count, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> nf_bf16_expected(nf_count), nf_f16_expected(nf_count);
  d_nf_bf16.copy_to_host(nf_bf16_expected.data(), nf_count);
  d_nf_f16.copy_to_host(nf_f16_expected.data(), nf_count);
  LinearWeightUpload nf_upload;
  nf_upload.format = LinearWeightFormat::kNF4;
  nf_upload.out_features = nf_out;
  nf_upload.in_features = nf_in;
  nf_upload.data = nf_packed.data();
  nf_upload.data_bytes = nf_packed.size();
  nf_upload.nf4_absmax = nf_absmax.data();
  nf_upload.nf4_absmax_count = nf_absmax.size();
  nf_upload.nf4_quant_map = nf_map.data();
  nf_upload.nf4_quant_map_count = nf_map.size();
  nf_upload.nf4_nested_quant_map = nf_nested_map.data();
  nf_upload.nf4_nested_quant_map_count = nf_nested_map.size();
  nf_upload.nf4_nested_absmax = nf_nested_absmax.data();
  nf_upload.nf4_nested_absmax_count = nf_nested_absmax.size();
  nf_upload.nf4_block_size = block;
  nf_upload.nf4_nested_block_size = nested_block;
  nf_upload.nf4_nested_offset = nf_offset;
  const uint64_t before_invalid = vk.pooled_used_bytes();
  {
    LinearWeightUpload invalid = nf_upload;
    invalid.nf4_quant_map_count = 15;
    bool rejected = false;
    try {
      (void)LinearWeight::upload(vk, invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    CHECK(rejected);
    CHECK(vk.pooled_used_bytes() == before_invalid);
  }
  {
    LinearWeightUpload invalid = nf_upload;
    invalid.nf4_block_size = 63;
    bool rejected = false;
    try {
      (void)LinearWeight::upload(vk, invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    CHECK(rejected);
    CHECK(vk.pooled_used_bytes() == before_invalid);
  }
  {
    LinearWeightUpload invalid = nf_upload;
    invalid.nf4_nested_quant_map_count = 255;
    bool rejected = false;
    try {
      (void)LinearWeight::upload(vk, invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    CHECK(rejected);
    CHECK(vk.pooled_used_bytes() == before_invalid);
  }
  const auto nf_bf16_actual = materialize(nf_upload, false);
  const auto nf_f16_actual = materialize(nf_upload, true);
  CHECK(std::memcmp(nf_bf16_expected.data(), nf_bf16_actual.data(), nf_count * sizeof(uint16_t)) ==
        0);
  CHECK(std::memcmp(nf_f16_expected.data(), nf_f16_actual.data(), nf_count * sizeof(uint16_t)) ==
        0);
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_linear_weight_activation_transforms_exact, "synthetic") {
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

  constexpr uint32_t rows = 3, dim = 256;
  constexpr size_t count = static_cast<size_t>(rows) * dim;
  std::vector<float> values(count), scales(dim);
  for (size_t i = 0; i < count; ++i)
    values[i] = static_cast<float>(static_cast<int>(i * 37 % 257) - 128) / 64.0f;
  for (size_t i = 0; i < dim; ++i)
    scales[i] = static_cast<float>(static_cast<int>(i * 19 % 61) - 30) / 32.0f;
  std::vector<uint16_t> bf_values(count), bf_scales(dim), zero_weight(dim);
  for (size_t i = 0; i < count; ++i)
    bf_values[i] = f32_to_bf16(values[i]);
  for (size_t i = 0; i < dim; ++i)
    bf_scales[i] = f32_to_bf16(scales[i]);

  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kBFloat16;
  upload.out_features = 1;
  upload.in_features = dim;
  upload.data = zero_weight.data();
  upload.data_bytes = zero_weight.size() * sizeof(uint16_t);
  upload.pre_quant_scale_bf16 = bf_scales.data();
  upload.pre_quant_scale_count = bf_scales.size();
  upload.convrot = true;
  upload.convrot_group = dim;
  LinearWeight weight = LinearWeight::upload(vk, upload);

  cuda::DeviceBuffer<uint16_t> d_bf_input(count), d_scale(dim), d_bf_scaled(count),
      d_bf_rotated(count);
  cuda::DeviceBuffer<float> d_f32_input(count), d_f32_scaled(count), d_f32_rotated(count);
  d_bf_input.copy_from_host(bf_values.data(), count);
  d_scale.copy_from_host(bf_scales.data(), dim);
  d_f32_input.copy_from_host(values.data(), count);
  cuda::launch_pre_quant_scale(reinterpret_cast<const __nv_bfloat16*>(d_bf_input.get()),
                               reinterpret_cast<const __nv_bfloat16*>(d_scale.get()),
                               reinterpret_cast<__nv_bfloat16*>(d_bf_scaled.get()), rows, dim,
                               nullptr);
  fp32_pre_quant_scale_probe<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
      d_f32_input.get(), reinterpret_cast<const __nv_bfloat16*>(d_scale.get()), d_f32_scaled.get(),
      static_cast<int>(count), dim);
  cuda::launch_convrot(reinterpret_cast<const __nv_bfloat16*>(d_bf_input.get()),
                       reinterpret_cast<__nv_bfloat16*>(d_bf_rotated.get()), rows, dim, dim,
                       nullptr);
  cuda::launch_convrot_f32(d_f32_input.get(), d_f32_rotated.get(), rows, dim, dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> bf_scaled_expected(count), bf_rotated_expected(count);
  std::vector<float> f32_scaled_expected(count), f32_rotated_expected(count);
  d_bf_scaled.copy_to_host(bf_scaled_expected.data(), count);
  d_bf_rotated.copy_to_host(bf_rotated_expected.data(), count);
  d_f32_scaled.copy_to_host(f32_scaled_expected.data(), count);
  d_f32_rotated.copy_to_host(f32_rotated_expected.data(), count);

  const uint64_t shape[] = {rows, dim};
  DeviceTensor bf_input = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);
  DeviceTensor bf_scaled = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);
  DeviceTensor bf_rotated = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);
  DeviceTensor f32_input = vk.allocate(TensorLayout::contiguous(shape, 2));
  DeviceTensor f32_scaled = vk.allocate(TensorLayout::contiguous(shape, 2));
  DeviceTensor f32_rotated = vk.allocate(TensorLayout::contiguous(shape, 2));
  vk.upload_bytes(bf_input, bf_values.data(), bf_values.size() * 2);
  vk.upload(f32_input, values.data(), values.size());
  TensorBatch batch = vk.begin_batch();
  weight.apply_pre_quant_scale(batch, bf_input, bf_scaled);
  weight.apply_pre_quant_scale(batch, f32_input, f32_scaled);
  weight.apply_convrot(batch, bf_input, bf_rotated);
  weight.apply_convrot(batch, f32_input, f32_rotated);
  batch.submit().wait();
  std::vector<uint16_t> bf_scaled_actual(count), bf_rotated_actual(count);
  std::vector<float> f32_scaled_actual(count), f32_rotated_actual(count);
  vk.download_bytes(bf_scaled, bf_scaled_actual.data(), count * 2);
  vk.download_bytes(bf_rotated, bf_rotated_actual.data(), count * 2);
  vk.download(f32_scaled, f32_scaled_actual.data(), count);
  vk.download(f32_rotated, f32_rotated_actual.data(), count);
  CHECK(std::memcmp(bf_scaled_expected.data(), bf_scaled_actual.data(), count * 2) == 0);
  CHECK(std::memcmp(bf_rotated_expected.data(), bf_rotated_actual.data(), count * 2) == 0);
  CHECK(std::memcmp(f32_scaled_expected.data(), f32_scaled_actual.data(), count * 4) == 0);
  CHECK(std::memcmp(f32_rotated_expected.data(), f32_rotated_actual.data(), count * 4) == 0);
}

SLOPFAB_TEST_CATEGORY(vulkan_linear_weight_bounded_reuse_and_lifetime, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
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
  constexpr uint32_t out = 3, in = 131;
  constexpr size_t count = static_cast<size_t>(out) * in;
  std::vector<int8_t> codes(count);
  for (size_t i = 0; i < count; ++i)
    codes[i] = static_cast<int8_t>(i * 29u);
  const float scales[] = {0.5f, -0.25f, 1.5f};
  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kInt8;
  upload.out_features = out;
  upload.in_features = in;
  upload.data = codes.data();
  upload.data_bytes = codes.size();
  upload.weight_scale = scales;
  upload.weight_scale_count = out;
  const uint64_t shape[] = {out, in};

  LinearWeight weight = LinearWeight::upload(vk, upload);
  DeviceTensor first = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);
  DeviceTensor second = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);
  auto submit = [&](DeviceTensor& output) {
    TensorBatch batch = vk.begin_batch();
    weight.materialize_bf16(batch, output);
    return batch.submit();
  };
  Submission warm_a = submit(first), warm_b = submit(second);
  warm_a.wait();
  warm_b.wait();
  {
    TensorBatch collect = vk.begin_batch();
  }
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 6; ++repeat) {
    Submission a = submit(first), b = submit(second), c = submit(first);
    CHECK(b.value() > a.value() && c.value() > b.value());
    a.wait();
    b.wait();
    c.wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  }
  {
    TensorBatch full = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      weight.materialize_bf16(full, first);
    full.submit().wait();
  }
  {
    TensorBatch overflow = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      weight.materialize_bf16(overflow, first);
    bool rejected = false;
    try {
      weight.materialize_bf16(overflow, first);
    } catch (const std::logic_error&) {
      rejected = true;
    }
    CHECK(rejected);
    bool submit_rejected = false;
    try {
      (void)overflow.submit();
    } catch (const std::logic_error&) {
      submit_rejected = true;
    }
    CHECK(submit_rejected);
  }
  CHECK(vk.reserved_bytes() == stable_reserved);
  const uint64_t saturated_descriptors = vk.descriptor_set_allocations();
  CHECK(saturated_descriptors >= stable_descriptors);
  for (int repeat = 0; repeat < 2; ++repeat) {
    TensorBatch full = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      weight.materialize_bf16(full, first);
    full.submit().wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == saturated_descriptors);
  }

  auto valid_after_rejection = [&](auto&& invalid) {
    TensorBatch batch = vk.begin_batch();
    bool rejected = false;
    try {
      invalid(batch);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    CHECK(rejected);
    weight.materialize_bf16(batch, first);
    batch.submit().wait();
  };
  DeviceTensor wrong_type = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kFloat32);
  const uint64_t short_shape[] = {out, in - 1};
  DeviceTensor wrong_shape =
      vk.allocate(TensorLayout::contiguous(short_shape, 2), ScalarType::kBFloat16);
  valid_after_rejection([&](TensorBatch& batch) {
    weight.materialize_bf16(batch, wrong_type);
  });
  valid_after_rejection([&](TensorBatch& batch) {
    weight.materialize_bf16(batch, wrong_shape);
  });
  Device other_device = physical.front().create_device(options);
  TensorContext other(other_device);
  DeviceTensor foreign = other.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);
  valid_after_rejection([&](TensorBatch& batch) {
    weight.materialize_bf16(batch, foreign);
  });

  // Discarded recording releases its speculative references immediately.
  const uint64_t used_before_discard = vk.pooled_used_bytes();
  {
    LinearWeight temporary = LinearWeight::upload(vk, upload);
    DeviceTensor output = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);
    {
      TensorBatch discarded = vk.begin_batch();
      temporary.materialize_bf16(discarded, output);
    }
  }
  CHECK(vk.pooled_used_bytes() == used_before_discard);

  // A submitted job retains compressed storage, scales, and dense output even
  // after every public wrapper drops, then releases them after its exact token
  // and slot collection.
  const uint64_t used_before_submit = vk.pooled_used_bytes();
  Submission retained;
  {
    LinearWeight temporary = LinearWeight::upload(vk, upload);
    DeviceTensor output = vk.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kBFloat16);
    TensorBatch batch = vk.begin_batch();
    temporary.materialize_bf16(batch, output);
    retained = batch.submit();
  }
  CHECK(vk.pooled_used_bytes() > used_before_submit);
  retained.wait();
  retained = Submission{};
  {
    TensorBatch collect = vk.begin_batch();
  }
  CHECK(vk.pooled_used_bytes() == used_before_submit);
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_cooperative_bf16_gemm_exact, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  constexpr uint32_t m = 64, n = 16, k = 5376;
  constexpr uint32_t input_rows = 128, output_rows = 128;
  constexpr uint32_t input_offset = 64, output_offset = 17;
  std::vector<uint16_t> input(size_t(input_rows) * k), weight(size_t(n) * k);
  std::vector<float> bias(n);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = f32_to_bf16(float(int(i % 31) - 15) / 32.0f);
  for (size_t i = 0; i < weight.size(); ++i)
    weight[i] = f32_to_bf16(float(int(i % 23) - 11) / 16.0f);
  for (uint32_t i = 0; i < n; ++i)
    bias[i] = float(int(i) - 7) / 64.0f;
  // Long-K cancellation, signed zero, and infinity rows stay inside the
  // documented non-NaN arithmetic domain.
  for (uint32_t inner = 0; inner < k; ++inner) {
    input[size_t(input_offset + 0) * k + inner] = f32_to_bf16((inner & 1) ? -1.0f : 1.0f);
    input[size_t(input_offset + 1) * k + inner] = 0x8000u;
    input[size_t(input_offset + 2) * k + inner] = 0;
  }
  input[size_t(input_offset + 2) * k] = 0x7f80u;
  for (uint32_t column = 0; column < n; ++column)
    weight[size_t(column) * k] = f32_to_bf16(1.0f);

  cuda::DeviceBuffer<uint16_t> ci(input.size()), cw(weight.size()), co(size_t(output_rows) * n);
  cuda::DeviceBuffer<float> cb(bias.size());
  ci.copy_from_host(input.data(), input.size());
  cw.copy_from_host(weight.data(), weight.size());
  cb.copy_from_host(bias.data(), bias.size());
  cuda::launch_deterministic_bf16_gemm_nt(reinterpret_cast<const __nv_bfloat16*>(ci.get()) +
                                              size_t(input_offset) * k,
                                          reinterpret_cast<const __nv_bfloat16*>(cw.get()),
                                          cb.get(), reinterpret_cast<__nv_bfloat16*>(co.get()), m,
                                          n, k, DenseGemmBias::kFloat32, 0, output_offset);
  std::vector<uint16_t> cuda_output(size_t(output_rows) * n);
  co.copy_to_host(cuda_output.data(), cuda_output.size());

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = true;
  options.enable_storage_buffer_16bit = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  const uint64_t is[] = {input_rows, k}, ws[] = {n, k};
  const uint64_t os[] = {output_rows, n}, bs[] = {n};
  DeviceTensor vi = context.allocate(TensorLayout::contiguous(is, 2), ScalarType::kBFloat16);
  DeviceTensor vw = context.allocate(TensorLayout::contiguous(ws, 2), ScalarType::kBFloat16);
  DeviceTensor vo = context.allocate(TensorLayout::contiguous(os, 2), ScalarType::kBFloat16);
  DeviceTensor vb = context.allocate(TensorLayout::contiguous(bs, 1), ScalarType::kFloat32);
  context.upload_bytes(vi, input.data(), input.size() * 2);
  context.upload_bytes(vw, weight.data(), weight.size() * 2);
  context.upload(vb, bias.data(), bias.size());
  DenseGemmPlanDesc desc{m, n, k, DenseGemmMode::kBFloat16, DenseGemmBias::kFloat32};
  DenseGemmPlan plan = DenseGemmPlan::create(context, desc);
  TensorBatch batch = context.begin_batch();
  plan.record(batch, vi, vw, vo, m, input_offset, output_offset, &vb);
  batch.submit().wait();
  std::vector<uint16_t> vulkan_output(cuda_output.size());
  context.download_bytes(vo, vulkan_output.data(), vulkan_output.size() * 2);
  size_t differences = 0;
  for (uint32_t row = 0; row < m; ++row) {
    for (uint32_t column = 0; column < n; ++column) {
      const size_t i = size_t(output_offset + row) * n + column;
      differences += cuda_output[i] != vulkan_output[i];
    }
  }
  CHECK_MSG(differences == 0, "cooperative BF16 GEMM differs in %zu/%zu values", differences,
            cuda_output.size());
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_cooperative_f16_gemm_exact, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  constexpr uint32_t m = 64, n = 64, k = 64;
  constexpr uint32_t input_rows = 128, output_rows = 128;
  constexpr uint32_t input_offset = 64, output_offset = 17;
  std::vector<float> input_f32(size_t(input_rows) * k);
  std::vector<uint16_t> input_f16(input_f32.size()), weight(size_t(n) * k);
  for (size_t i = 0; i < input_f32.size(); ++i) {
    input_f32[i] = float(int(i % 37) - 18) / 29.0f;
    input_f16[i] = f32_to_f16(input_f32[i]);
  }
  for (size_t i = 0; i < weight.size(); ++i)
    weight[i] = f32_to_f16(float(int(i % 29) - 14) / 23.0f);

  cuda::DeviceBuffer<uint16_t> ci(input_f16.size()), cw(weight.size());
  cuda::DeviceBuffer<float> co(size_t(output_rows) * n);
  ci.copy_from_host(input_f16.data(), input_f16.size());
  cw.copy_from_host(weight.data(), weight.size());
  // Point the CUDA reference at the same prepared source-row range.
  cuda::launch_deterministic_f16_gemm_nt(
      reinterpret_cast<const __half*>(ci.get()) + size_t(input_offset) * k,
      reinterpret_cast<const __half*>(cw.get()), co.get(), m, n, k, output_offset);
  std::vector<float> cuda_output(size_t(output_rows) * n);
  co.copy_to_host(cuda_output.data(), cuda_output.size());

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
  const uint64_t is[] = {input_rows, k}, ws[] = {n, k};
  const uint64_t os[] = {output_rows, n};
  DeviceTensor vi = context.allocate(TensorLayout::contiguous(is, 2), ScalarType::kFloat32);
  DeviceTensor vw = context.allocate(TensorLayout::contiguous(ws, 2), ScalarType::kFloat16);
  DeviceTensor vo = context.allocate(TensorLayout::contiguous(os, 2), ScalarType::kFloat32);
  context.upload(vi, input_f32.data(), input_f32.size());
  context.upload_bytes(vw, weight.data(), weight.size() * 2);
  PreparedF16Activation slot = PreparedF16Activation::create(context, m, k);
  DenseGemmPlanDesc desc{m, n, k, DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone};
  DenseGemmPlan plan = DenseGemmPlan::create(context, desc);
  TensorBatch batch = context.begin_batch();
  PreparedF16ActivationView prepared = slot.prepare(batch, vi, m, input_offset);
  plan.record(batch, prepared, vw, vo, output_offset);
  // A second projection consumes the same batch-scoped conversion.
  plan.record(batch, prepared, vw, vo, output_offset);
  batch.submit().wait();
  std::vector<float> vulkan_output(cuda_output.size());
  context.download(vo, vulkan_output.data(), vulkan_output.size());
  size_t differences = 0;
  for (uint32_t row = 0; row < m; ++row) {
    for (uint32_t column = 0; column < n; ++column) {
      const size_t i = size_t(output_offset + row) * n + column;
      differences += std::memcmp(&cuda_output[i], &vulkan_output[i], sizeof(float)) != 0;
    }
  }
  CHECK_MSG(differences == 0, "cooperative F16 GEMM differs in %zu/%zu values", differences,
            cuda_output.size());
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_scalar_gemm_modes_exact, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  constexpr uint32_t m = 3, n = 11, k = 19;
  constexpr uint32_t input_rows = 5, output_rows = 6;
  constexpr uint32_t input_offset = 1, output_offset = 2;
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
  const uint64_t as[] = {input_rows, k}, ws[] = {n, k};
  const uint64_t os[] = {output_rows, n}, bs[] = {n};

  auto run = [&](DenseGemmMode mode, DenseGemmBias bias_type) {
    std::vector<float> af(size_t(input_rows) * k), wf(size_t(n) * k);
    std::vector<float> biasf(n);
    for (size_t i = 0; i < af.size(); ++i)
      af[i] = float(int(i % 31) - 15) / 23.0f;
    for (size_t i = 0; i < wf.size(); ++i)
      wf[i] = float(int(i % 27) - 13) / 19.0f;
    for (uint32_t i = 0; i < n; ++i)
      biasf[i] = float(int(i) - 5) / 37.0f;

    const ScalarType at =
        mode == DenseGemmMode::kBFloat16 ? ScalarType::kBFloat16 : ScalarType::kFloat32;
    const ScalarType wt = mode == DenseGemmMode::kBFloat16     ? ScalarType::kBFloat16
                          : mode == DenseGemmMode::kFloat16Vae ? ScalarType::kFloat16
                                                               : ScalarType::kFloat32;
    const ScalarType ot =
        mode == DenseGemmMode::kBFloat16 ? ScalarType::kBFloat16 : ScalarType::kFloat32;
    DeviceTensor va = context.allocate(TensorLayout::contiguous(as, 2), at);
    DeviceTensor vw = context.allocate(TensorLayout::contiguous(ws, 2), wt);
    DeviceTensor vo = context.allocate(TensorLayout::contiguous(os, 2), ot);
    DeviceTensor vb;
    if (bias_type != DenseGemmBias::kNone) {
      vb = context.allocate(TensorLayout::contiguous(bs, 1), bias_type == DenseGemmBias::kBFloat16
                                                                 ? ScalarType::kBFloat16
                                                                 : ScalarType::kFloat32);
    }

    DenseGemmPlan plan = DenseGemmPlan::create(context, {m, n, k, mode, bias_type});
    if (mode == DenseGemmMode::kBFloat16) {
      std::vector<uint16_t> ah(af.size()), wh(wf.size()), biash(n),
          sentinel(size_t(output_rows) * n, 0x3f00u);
      for (size_t i = 0; i < ah.size(); ++i)
        ah[i] = f32_to_bf16(af[i]);
      for (size_t i = 0; i < wh.size(); ++i)
        wh[i] = f32_to_bf16(wf[i]);
      for (uint32_t i = 0; i < n; ++i)
        biash[i] = f32_to_bf16(biasf[i]);
      cuda::DeviceBuffer<uint16_t> ca(ah.size()), cw(wh.size()), cbh(biash.size()),
          co(sentinel.size());
      cuda::DeviceBuffer<float> cbf(biasf.size());
      ca.copy_from_host(ah.data(), ah.size());
      cw.copy_from_host(wh.data(), wh.size());
      co.copy_from_host(sentinel.data(), sentinel.size());
      const void* cb = nullptr;
      if (bias_type == DenseGemmBias::kFloat32) {
        cbf.copy_from_host(biasf.data(), biasf.size());
        cb = cbf.get();
      } else if (bias_type == DenseGemmBias::kBFloat16) {
        cbh.copy_from_host(biash.data(), biash.size());
        cb = cbh.get();
      }
      cuda::launch_deterministic_scalar_gemm_nt(ca.get(), cw.get(), cb, co.get(), m, n, k, mode,
                                                bias_type, input_offset, output_offset);
      std::vector<uint16_t> cuda_out(sentinel.size()), vk_out(sentinel.size());
      co.copy_to_host(cuda_out.data(), cuda_out.size());
      context.upload_bytes(va, ah.data(), ah.size() * 2);
      context.upload_bytes(vw, wh.data(), wh.size() * 2);
      context.upload_bytes(vo, sentinel.data(), sentinel.size() * 2);
      if (bias_type == DenseGemmBias::kFloat32)
        context.upload(vb, biasf.data(), biasf.size());
      else if (bias_type == DenseGemmBias::kBFloat16)
        context.upload_bytes(vb, biash.data(), biash.size() * 2);
      TensorBatch batch = context.begin_batch();
      plan.record(batch, va, vw, vo, m, input_offset, output_offset,
                  bias_type == DenseGemmBias::kNone ? nullptr : &vb);
      batch.submit().wait();
      context.download_bytes(vo, vk_out.data(), vk_out.size() * 2);
      CHECK(std::memcmp(cuda_out.data(), vk_out.data(), vk_out.size() * 2) == 0);
    } else {
      std::vector<float> sentinel(size_t(output_rows) * n, 0.375f);
      std::vector<uint16_t> ah16, wh16;
      cuda::DeviceBuffer<float> caf(af.size()), cwf(wf.size()), cbf(biasf.size()),
          co(sentinel.size());
      cuda::DeviceBuffer<uint16_t> cah, cwh;
      const void* ca = nullptr;
      const void* cw = nullptr;
      if (mode == DenseGemmMode::kFloat16Vae) {
        ah16.resize(af.size());
        wh16.resize(wf.size());
        for (size_t i = 0; i < af.size(); ++i)
          ah16[i] = f32_to_f16(af[i]);
        for (size_t i = 0; i < wf.size(); ++i)
          wh16[i] = f32_to_f16(wf[i]);
        cah.allocate(ah16.size());
        cwh.allocate(wh16.size());
        cah.copy_from_host(ah16.data(), ah16.size());
        cwh.copy_from_host(wh16.data(), wh16.size());
        ca = cah.get();
        cw = cwh.get();
        context.upload(va, af.data(), af.size());
        context.upload_bytes(vw, wh16.data(), wh16.size() * 2);
      } else {
        caf.copy_from_host(af.data(), af.size());
        cwf.copy_from_host(wf.data(), wf.size());
        ca = caf.get();
        cw = cwf.get();
        context.upload(va, af.data(), af.size());
        context.upload(vw, wf.data(), wf.size());
      }
      if (bias_type == DenseGemmBias::kFloat32) {
        cbf.copy_from_host(biasf.data(), biasf.size());
        context.upload(vb, biasf.data(), biasf.size());
      }
      co.copy_from_host(sentinel.data(), sentinel.size());
      cuda::launch_deterministic_scalar_gemm_nt(
          ca, cw, bias_type == DenseGemmBias::kNone ? nullptr : cbf.get(), co.get(), m, n, k, mode,
          bias_type, input_offset, output_offset);
      context.upload(vo, sentinel.data(), sentinel.size());
      PreparedF16Activation slot;
      if (mode == DenseGemmMode::kFloat16Vae)
        slot = PreparedF16Activation::create(context, m, k);
      TensorBatch batch = context.begin_batch();
      if (mode == DenseGemmMode::kFloat16Vae) {
        PreparedF16ActivationView prepared = slot.prepare(batch, va, m, input_offset);
        plan.record(batch, prepared, vw, vo, output_offset);
      } else {
        plan.record(batch, va, vw, vo, m, input_offset, output_offset,
                    bias_type == DenseGemmBias::kNone ? nullptr : &vb);
      }
      batch.submit().wait();
      std::vector<float> cuda_out(sentinel.size()), vk_out(sentinel.size());
      co.copy_to_host(cuda_out.data(), cuda_out.size());
      context.download(vo, vk_out.data(), vk_out.size());
      CHECK(std::memcmp(cuda_out.data(), vk_out.data(), vk_out.size() * 4) == 0);
    }
  };
  run(DenseGemmMode::kBFloat16, DenseGemmBias::kNone);
  run(DenseGemmMode::kBFloat16, DenseGemmBias::kFloat32);
  run(DenseGemmMode::kBFloat16, DenseGemmBias::kBFloat16);
  run(DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone);
  run(DenseGemmMode::kFloat32, DenseGemmBias::kNone);
  run(DenseGemmMode::kFloat32, DenseGemmBias::kFloat32);
}
