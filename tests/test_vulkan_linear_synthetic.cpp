#include "detail/vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(vulkan_linear_weight_cpu_reference, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  auto run = [&](const char* label, const LinearWeightUpload& upload,
                 const std::vector<uint16_t>& expected) {
    LinearWeight weight = LinearWeight::upload(context, upload);
    const uint64_t shape[] = {upload.out_features, upload.in_features};
    DeviceTensor output = context.allocate(TensorLayout::contiguous(shape, 2),
                                           ScalarType::kBFloat16);
    TensorBatch batch = context.begin_batch();
    weight.materialize_bf16(batch, output);
    batch.submit().wait();
    std::vector<uint16_t> actual(expected.size());
    context.download_bytes(output, actual.data(), actual.size() * 2);
    size_t mismatch = expected.size();
    for (size_t i = 0; i < expected.size(); ++i)
      if (actual[i] != expected[i]) { mismatch = i; break; }
    CHECK_MSG(mismatch == expected.size(), "%s mismatch at %zu: %04x != %04x",
              label, mismatch, mismatch == expected.size() ? 0u : expected[mismatch],
              mismatch == expected.size() ? 0u : actual[mismatch]);
  };

  const std::vector<float> dense_f32 = {-3.5f, -0.0f, 0.125f, 1.0f, 17.25f};
  std::vector<uint16_t> dense_expected(dense_f32.size());
  for (size_t i = 0; i < dense_f32.size(); ++i)
    dense_expected[i] = reference_bf16(dense_f32[i]);
  LinearWeightUpload dense;
  dense.format = LinearWeightFormat::kFloat32;
  dense.out_features = 1;
  dense.in_features = static_cast<uint32_t>(dense_f32.size());
  dense.data = dense_f32.data();
  dense.data_bytes = dense_f32.size() * 4;
  run("f32", dense, dense_expected);
  const std::vector<uint16_t> raw_bf16 = {0x0000, 0x8000, 0x0001, 0x7f80, 0x7fc1};
  dense.format = LinearWeightFormat::kBFloat16;
  dense.data = raw_bf16.data();
  dense.data_bytes = raw_bf16.size() * 2;
  run("bf16", dense, raw_bf16);

  std::vector<uint8_t> f8(256);
  std::vector<uint16_t> f8_expected(256);
  const float f8_scale = 0.75f;
  for (uint32_t i = 0; i < 256; ++i) {
    f8[i] = static_cast<uint8_t>(i);
    f8_expected[i] = reference_bf16(f8_e4m3_to_f32(f8[i]) * f8_scale);
  }
  LinearWeightUpload f8_upload;
  f8_upload.format = LinearWeightFormat::kFloat8E4M3;
  f8_upload.out_features = 1;
  f8_upload.in_features = 256;
  f8_upload.data = f8.data();
  f8_upload.data_bytes = f8.size();
  f8_upload.weight_scale = &f8_scale;
  f8_upload.weight_scale_count = 1;
  run("f8", f8_upload, f8_expected);

  constexpr uint32_t i8_out = 2, i8_in = 67;
  std::vector<int8_t> i8(i8_out * i8_in);
  const float i8_scales[] = {-0.25f, 1.5f};
  std::vector<uint16_t> i8_expected(i8.size());
  for (size_t i = 0; i < i8.size(); ++i) {
    i8[i] = static_cast<int8_t>(i * 71u);
    i8_expected[i] = reference_bf16(static_cast<float>(i8[i]) *
                                  i8_scales[i / i8_in]);
  }
  LinearWeightUpload i8_upload;
  i8_upload.format = LinearWeightFormat::kInt8;
  i8_upload.out_features = i8_out;
  i8_upload.in_features = i8_in;
  i8_upload.data = i8.data();
  i8_upload.data_bytes = i8.size();
  i8_upload.weight_scale = i8_scales;
  i8_upload.weight_scale_count = i8_out;
  run("i8", i8_upload, i8_expected);

  constexpr uint32_t nv_out = 128, nv_in = 64;
  const size_t nv_count = static_cast<size_t>(nv_out) * nv_in;
  std::vector<uint8_t> nv_codes(nv_count / 2), nv_scales(nv_count / 16, 0x38);
  std::vector<uint16_t> nv_expected(nv_count);
  for (size_t byte = 0; byte < nv_codes.size(); ++byte)
    nv_codes[byte] = static_cast<uint8_t>(((2 * byte & 15) << 4) |
                                          ((2 * byte + 1) & 15));
  const float nv_global = 0.25f;
  auto scale_slot = [&](uint32_t row, uint32_t block) {
    const uint32_t blocks_per_row = nv_in / 16;
    const uint32_t tile = (row >> 7) * (blocks_per_row >> 2) + (block >> 2);
    return static_cast<size_t>(tile) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4 + (block & 3);
  };
  for (size_t i = 0; i < nv_count; ++i) {
    const uint8_t packed = nv_codes[i / 2];
    const uint8_t code = (i & 1) == 0 ? packed >> 4 : packed & 15;
    const uint32_t row = static_cast<uint32_t>(i / nv_in);
    const float scale_value = f8_e4m3_to_f32(
        nv_scales[scale_slot(row, static_cast<uint32_t>((i % nv_in) / 16))]) *
        nv_global;
    nv_expected[i] = reference_bf16(f4_e2m1_to_f32(code) * scale_value);
  }
  LinearWeightUpload nv_upload;
  nv_upload.format = LinearWeightFormat::kNVFloat4;
  nv_upload.out_features = nv_out;
  nv_upload.in_features = nv_in;
  nv_upload.data = nv_codes.data();
  nv_upload.data_bytes = nv_codes.size();
  nv_upload.block_scale = nv_scales.data();
  nv_upload.block_scale_count = nv_scales.size();
  nv_upload.global_scale = nv_global;
  run("nvfp4", nv_upload, nv_expected);

  constexpr uint32_t nf_out = 3, nf_in = 45;
  const size_t nf_count = static_cast<size_t>(nf_out) * nf_in;
  std::vector<uint8_t> nf_codes((nf_count + 1) / 2), nf_absmax(3);
  for (size_t i = 0; i < nf_codes.size(); ++i)
    nf_codes[i] = static_cast<uint8_t>((((i + 3) & 15) << 4) | ((i + 9) & 15));
  for (size_t i = 0; i < nf_absmax.size(); ++i) nf_absmax[i] = static_cast<uint8_t>(i * 97);
  std::array<float, 16> nf_map{};
  std::array<float, 256> nested_map{};
  for (size_t i = 0; i < nf_map.size(); ++i) nf_map[i] = (float(i) - 7.0f) / 8.0f;
  for (size_t i = 0; i < nested_map.size(); ++i)
    nested_map[i] = (float(i) - 127.0f) / 128.0f;
  const float nested_absmax[] = {0.75f};
  const float offset = 0.125f;
  std::vector<uint16_t> nf_expected(nf_count);
  for (size_t i = 0; i < nf_count; ++i) {
    const size_t scale_index = i / 64;
    const float scale_value = nested_map[nf_absmax[scale_index]] *
                                  nested_absmax[scale_index / 256] + offset;
    const uint8_t packed = nf_codes[i / 2];
    const uint8_t code = (i & 1) == 0 ? packed >> 4 : packed & 15;
    nf_expected[i] = reference_bf16(nf_map[code] * scale_value);
  }
  LinearWeightUpload nf_upload;
  nf_upload.format = LinearWeightFormat::kNF4;
  nf_upload.out_features = nf_out;
  nf_upload.in_features = nf_in;
  nf_upload.data = nf_codes.data();
  nf_upload.data_bytes = nf_codes.size();
  nf_upload.nf4_absmax = nf_absmax.data();
  nf_upload.nf4_absmax_count = nf_absmax.size();
  nf_upload.nf4_quant_map = nf_map.data();
  nf_upload.nf4_quant_map_count = nf_map.size();
  nf_upload.nf4_nested_quant_map = nested_map.data();
  nf_upload.nf4_nested_quant_map_count = nested_map.size();
  nf_upload.nf4_nested_absmax = nested_absmax;
  nf_upload.nf4_nested_absmax_count = 1;
  nf_upload.nf4_nested_offset = offset;
  run("nf4", nf_upload, nf_expected);
}

SLOPFAB_TEST_CATEGORY(vulkan_streamed_nvfp4_gemm_cache, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = physical.front().info().cooperative_matrix;
  options.enable_storage_buffer_16bit = options.enable_cooperative_matrix;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  CHECK(!context.native_nvfp4_gemm_available());
  bool native_threw = false;
  try { context.require_native_nvfp4_gemm(); }
  catch (const std::runtime_error&) { native_threw = true; }
  CHECK(native_threw);

  constexpr uint32_t rows = 6, n = 128, k = 64;
  const size_t weight_elements = static_cast<size_t>(n) * k;
  std::vector<uint8_t> positive(weight_elements / 2, 0x22);
  std::vector<uint8_t> negative(weight_elements / 2, 0xaa);
  std::vector<uint8_t> scales(weight_elements / 16, 0x38);
  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kNVFloat4;
  upload.out_features = n;
  upload.in_features = k;
  upload.data_bytes = positive.size();
  upload.block_scale = scales.data();
  upload.block_scale_count = scales.size();
  upload.global_scale = 1.0f;
  upload.data = positive.data();
  LinearWeight w_positive = LinearWeight::upload(context, upload);
  upload.data = negative.data();
  upload.full_precision_matrix_mult = true;
  LinearWeight w_negative = LinearWeight::upload(context, upload);
  std::vector<uint16_t> pre_scale(k, reference_bf16(1.0f));
  upload.data = positive.data();
  upload.pre_quant_scale_bf16 = pre_scale.data();
  upload.pre_quant_scale_count = pre_scale.size();
  LinearWeight awq_weight = LinearWeight::upload(context, upload);

  const uint64_t input_shape[] = {rows, k};
  const uint64_t output_shape[] = {rows, n};
  DeviceTensor input = context.allocate(
      TensorLayout::contiguous(input_shape, 2), ScalarType::kBFloat16);
  DeviceTensor output = context.allocate(
      TensorLayout::contiguous(output_shape, 2), ScalarType::kBFloat16);
  std::vector<uint16_t> input_bits(static_cast<size_t>(rows) * k,
                                   reference_bf16(1.0f));
  std::vector<uint16_t> sentinel(static_cast<size_t>(rows) * n, 0x7fc1);
  context.upload_bytes(input, input_bits.data(), input_bits.size() * 2);
  context.upload_bytes(output, sentinel.data(), sentinel.size() * 2);
  DenseGemmPlan plan = DenseGemmPlan::create(
      context, {rows, n, k, DenseGemmMode::kBFloat16,
                DenseGemmBias::kNone});
  StreamedNVFP4WeightCache cache =
      StreamedNVFP4WeightCache::create(context, weight_elements * 2);
  CHECK(cache.capacity_elements() == weight_elements * 2);
  CHECK(cache.dense_bytes() == weight_elements * 4);

  // A foreign weight with a different valid shape must fail before it changes
  // the shared slot metadata or invalidates the current W1 generation.
  TensorContext foreign_context(device);
  std::vector<uint8_t> foreign_codes(weight_elements, 0x22);
  std::vector<uint8_t> foreign_scales(weight_elements / 8, 0x38);
  LinearWeightUpload foreign_upload = upload;
  foreign_upload.in_features = 128;
  foreign_upload.pre_quant_scale_bf16 = nullptr;
  foreign_upload.pre_quant_scale_count = 0;
  foreign_upload.data = foreign_codes.data();
  foreign_upload.data_bytes = foreign_codes.size();
  foreign_upload.block_scale = foreign_scales.data();
  foreign_upload.block_scale_count = foreign_scales.size();
  LinearWeight foreign_weight = LinearWeight::upload(foreign_context, foreign_upload);
  DenseGemmPlan foreign_shape_plan = DenseGemmPlan::create(
      context, {rows, n, 128, DenseGemmMode::kBFloat16,
                DenseGemmBias::kNone});

  TensorBatch first = context.begin_batch();
  PreparedNVFP4WeightView p = cache.prepare(first, w_positive, plan);
  const uint32_t capacity_after_prepare = first.remaining_operator_capacity();
  bool foreign_threw = false;
  try { (void)cache.prepare(first, foreign_weight, foreign_shape_plan); }
  catch (const std::invalid_argument&) { foreign_threw = true; }
  CHECK(foreign_threw);
  CHECK(first.remaining_operator_capacity() == capacity_after_prepare);
  bool shape_threw = false;
  try { (void)cache.prepare(first, w_positive, foreign_shape_plan); }
  catch (const std::invalid_argument&) { shape_threw = true; }
  CHECK(shape_threw);
  CHECK(first.remaining_operator_capacity() == capacity_after_prepare);
  // Both failures leave the prior generation, dense layout and access state
  // intact: it remains immediately recordable in this same batch.
  plan.record(first, input, p, output, 2, 0, 0);
  // AWQ transforms the activation before GEMM; it does not alter NVFP4
  // materialization. Preparing such a weight is therefore valid and, like
  // every successful prepare, supersedes the preceding cache generation.
  (void)cache.prepare(first, awq_weight, plan);
  bool awq_superseded_p = false;
  try { plan.record(first, input, p, output, 1, 0, 0); }
  catch (const std::invalid_argument&) { awq_superseded_p = true; }
  CHECK(awq_superseded_p);
  p = cache.prepare(first, w_positive, plan);
  plan.record(first, input, p, output, 2, 2, 2);
  PreparedNVFP4WeightView m = cache.prepare(first, w_negative, plan);
  CHECK(m.full_precision_matrix_mult());
  bool stale_threw = false;
  try { plan.record(first, input, p, output, 1, 0, 0); }
  catch (const std::invalid_argument&) { stale_threw = true; }
  CHECK(stale_threw);
  plan.record(first, input, m, output, 2, 4, 4);
  Submission first_token = first.submit();

  // Overwrite the same cache in a second queued submission. The queue-ordered
  // R->W barrier protects the first job without a CPU/device-wide wait.
  DeviceTensor second_output = context.allocate(
      TensorLayout::contiguous(output_shape, 2), ScalarType::kBFloat16);
  context.upload_bytes(second_output, sentinel.data(), sentinel.size() * 2);
  TensorBatch second = context.begin_batch();
  PreparedNVFP4WeightView again = cache.prepare(second, w_positive, plan);
  plan.record(second, input, again, second_output, rows);
  Submission second_token = second.submit();
  first_token.wait();
  second_token.wait();

  std::vector<uint16_t> got(sentinel.size()), got_second(sentinel.size());
  context.download_bytes(output, got.data(), got.size() * 2);
  context.download_bytes(second_output, got_second.data(), got_second.size() * 2);
  const uint16_t plus = reference_bf16(64.0f);
  const uint16_t minus = reference_bf16(-64.0f);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < n; ++col) {
      CHECK(got[static_cast<size_t>(row) * n + col] ==
            (row < 4 ? plus : minus));
      CHECK(got_second[static_cast<size_t>(row) * n + col] == plus);
    }
  }

  // A materialize plus 31 chunk/fanout reads exactly fills the bounded
  // 32-operation schedule. The 33rd operation is rejected, and discarding the
  // poisoned recording leaves the following batch usable.
  Submission warm_flights[2];
  for (int flight = 0; flight < 2; ++flight) {
    TensorBatch full = context.begin_batch();
    PreparedNVFP4WeightView prepared = cache.prepare(full, w_positive, plan);
    for (int i = 0; i < 31; ++i)
      plan.record(full, input, prepared, output, rows);
    warm_flights[flight] = full.submit();
  }
  warm_flights[0].wait(); warm_flights[1].wait();
  const uint64_t high_reserved = context.reserved_bytes();
  const uint64_t high_descriptors = context.descriptor_set_allocations();
  {
    TensorBatch overflow = context.begin_batch();
    PreparedNVFP4WeightView prepared = cache.prepare(overflow, w_positive, plan);
    for (int i = 0; i < 31; ++i)
      plan.record(overflow, input, prepared, output, rows);
    bool threw = false;
    try { plan.record(overflow, input, prepared, output, rows); }
    catch (const std::logic_error&) { threw = true; }
    CHECK(threw);
  }
  Submission tail;
  for (int i = 0; i < 50; ++i) {
    TensorBatch repeat = context.begin_batch();
    PreparedNVFP4WeightView prepared = cache.prepare(repeat, w_positive, plan);
    plan.record(repeat, input, prepared, output, rows);
    tail = repeat.submit();
  }
  tail.wait();
  CHECK(context.reserved_bytes() == high_reserved);
  CHECK(context.descriptor_set_allocations() == high_descriptors);
}

SLOPFAB_TEST_CATEGORY(vulkan_streamed_nvfp4_wrapper_drop, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = physical.front().info().cooperative_matrix;
  options.enable_storage_buffer_16bit = options.enable_cooperative_matrix;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  {
    const uint64_t warm_shape[] = {2, 64};
    DeviceTensor warm = context.allocate(
        TensorLayout::contiguous(warm_shape, 2), ScalarType::kBFloat16);
    std::vector<uint16_t> zeros(128);
    context.upload_bytes(warm, zeros.data(), zeros.size() * 2);
  }
  { TensorBatch collect = context.begin_batch(); }
  const uint64_t baseline = context.pooled_used_bytes();
  Submission token;
  {
    constexpr uint32_t rows = 2, n = 128, k = 64;
    const size_t elements = size_t(n) * k;
    std::vector<uint8_t> codes(elements / 2, 0x22);
    std::vector<uint8_t> scales(elements / 16, 0x38);
    LinearWeightUpload upload;
    upload.format = LinearWeightFormat::kNVFloat4;
    upload.out_features = n; upload.in_features = k;
    upload.data = codes.data(); upload.data_bytes = codes.size();
    upload.block_scale = scales.data();
    upload.block_scale_count = scales.size();
    LinearWeight weight = LinearWeight::upload(context, upload);
    const uint64_t is[] = {rows, k}, os[] = {rows, n};
    DeviceTensor input = context.allocate(TensorLayout::contiguous(is, 2),
                                          ScalarType::kBFloat16);
    DeviceTensor output = context.allocate(TensorLayout::contiguous(os, 2),
                                           ScalarType::kBFloat16);
    std::vector<uint16_t> bits(size_t(rows) * k, reference_bf16(1.0f));
    context.upload_bytes(input, bits.data(), bits.size() * 2);
    DenseGemmPlan plan = DenseGemmPlan::create(
        context, {rows, n, k, DenseGemmMode::kBFloat16,
                  DenseGemmBias::kNone});
    StreamedNVFP4WeightCache cache =
        StreamedNVFP4WeightCache::create(context, elements);
    TensorBatch batch = context.begin_batch();
    PreparedNVFP4WeightView prepared = cache.prepare(batch, weight, plan);
    plan.record(batch, input, prepared, output, rows);
    token = batch.submit();
  }
  CHECK(context.pooled_used_bytes() > baseline);
  token.wait();
  token = Submission{};
  { TensorBatch collect = context.begin_batch(); }
  CHECK_MSG(context.pooled_used_bytes() == baseline,
            "streamed wrapper drop retained %llu bytes (baseline %llu)",
            static_cast<unsigned long long>(context.pooled_used_bytes()),
            static_cast<unsigned long long>(baseline));
}

SLOPFAB_TEST_CATEGORY(vulkan_dense_gemm_tail_reference, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = physical.front().info().cooperative_matrix;
  options.enable_storage_buffer_16bit = options.enable_cooperative_matrix;
  options.enable_shader_float16 = options.enable_cooperative_matrix &&
      physical.front().info().shader_float16;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  constexpr uint32_t rows = 3, total_input_rows = 5, total_output_rows = 6;
  constexpr uint32_t n = 11, k = 19;
  const uint64_t input_shape[] = {total_input_rows, k};
  const uint64_t weight_shape[] = {n, k};
  const uint64_t output_shape[] = {total_output_rows, n};
  const uint64_t bias_shape[] = {n};
  DeviceTensor input = context.allocate(TensorLayout::contiguous(input_shape, 2),
                                        ScalarType::kBFloat16);
  DeviceTensor weight = context.allocate(TensorLayout::contiguous(weight_shape, 2),
                                         ScalarType::kBFloat16);
  DeviceTensor output = context.allocate(TensorLayout::contiguous(output_shape, 2),
                                         ScalarType::kBFloat16);
  DeviceTensor bias = context.allocate(TensorLayout::contiguous(bias_shape, 1),
                                       ScalarType::kFloat32);
  std::vector<uint16_t> input_bits(total_input_rows * k);
  std::vector<uint16_t> weight_bits(n * k);
  std::vector<uint16_t> output_bits(total_output_rows * n, 0x3e80u);
  std::vector<float> bias_values(n);
  for (size_t i = 0; i < input_bits.size(); ++i)
    input_bits[i] = reference_bf16(static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f);
  for (size_t i = 0; i < weight_bits.size(); ++i)
    weight_bits[i] = reference_bf16(static_cast<float>(static_cast<int>(i % 13) - 6) / 8.0f);
  for (uint32_t i = 0; i < n; ++i)
    bias_values[i] = static_cast<float>(static_cast<int>(i) - 5) / 32.0f;
  context.upload_bytes(input, input_bits.data(), input_bits.size() * 2);
  context.upload_bytes(weight, weight_bits.data(), weight_bits.size() * 2);
  context.upload_bytes(output, output_bits.data(), output_bits.size() * 2);
  context.upload(bias, bias_values.data(), bias_values.size());

  DenseGemmPlanDesc desc;
  desc.max_rows = 4;
  desc.out_features = n;
  desc.in_features = k;
  desc.mode = DenseGemmMode::kBFloat16;
  desc.bias = DenseGemmBias::kFloat32;
  DenseGemmPlan plan = DenseGemmPlan::create(context, desc);
  TensorBatch batch = context.begin_batch();
  plan.record(batch, input, weight, output, rows, 1, 2, &bias);
  batch.submit().wait();
  std::vector<uint16_t> actual(output_bits.size());
  context.download_bytes(output, actual.data(), actual.size() * 2);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < n; ++column) {
      float sum = 0.0f;
      for (uint32_t inner = 0; inner < k; ++inner) {
        float a = 0.0f, w = 0.0f;
        const uint32_t ab = uint32_t(input_bits[(row + 1) * k + inner]) << 16;
        const uint32_t wb = uint32_t(weight_bits[column * k + inner]) << 16;
        std::memcpy(&a, &ab, 4); std::memcpy(&w, &wb, 4);
        sum = std::fma(a, w, sum);
      }
      const uint16_t rounded = reference_bf16(sum);
      const uint32_t rounded_bits = uint32_t(rounded) << 16;
      std::memcpy(&sum, &rounded_bits, 4);
      const uint16_t expected = reference_bf16(sum + bias_values[column]);
      const size_t index = size_t(row + 2) * n + column;
      CHECK_MSG(actual[index] == expected,
                "gemm tail [%u,%u]: %04x != %04x", row, column,
                actual[index], expected);
    }
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    const size_t row = i / n;
    if (row < 2 || row >= 5) CHECK(actual[i] == 0x3e80u);
  }

  auto run_float_mode = [&](DenseGemmMode mode, DenseGemmBias bias_mode) {
    const uint64_t fs[] = {rows, k}, fws[] = {n, k}, fos[] = {rows, n};
    DeviceTensor fi = context.allocate(TensorLayout::contiguous(fs, 2),
                                       ScalarType::kFloat32);
    DeviceTensor fw = context.allocate(
        TensorLayout::contiguous(fws, 2),
        mode == DenseGemmMode::kFloat16Vae ? ScalarType::kFloat16
                                           : ScalarType::kFloat32);
    DeviceTensor fo = context.allocate(TensorLayout::contiguous(fos, 2),
                                       ScalarType::kFloat32);
    std::vector<float> host_input(rows * k), host_output(rows * n);
    std::vector<float> host_weight_f32;
    std::vector<uint16_t> host_weight_f16;
    for (size_t i = 0; i < host_input.size(); ++i)
      host_input[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 17.0f;
    context.upload(fi, host_input.data(), host_input.size());
    if (mode == DenseGemmMode::kFloat16Vae) {
      host_weight_f16.resize(n * k);
      for (size_t i = 0; i < host_weight_f16.size(); ++i)
        host_weight_f16[i] = f32_to_f16(
            static_cast<float>(static_cast<int>(i % 23) - 11) / 19.0f);
      context.upload_bytes(fw, host_weight_f16.data(), host_weight_f16.size() * 2);
    } else {
      host_weight_f32.resize(n * k);
      for (size_t i = 0; i < host_weight_f32.size(); ++i)
        host_weight_f32[i] =
            static_cast<float>(static_cast<int>(i % 23) - 11) / 19.0f;
      context.upload(fw, host_weight_f32.data(), host_weight_f32.size());
    }
    DenseGemmPlanDesc float_desc{rows, n, k, mode, bias_mode};
    DenseGemmPlan float_plan = DenseGemmPlan::create(context, float_desc);
    PreparedF16Activation slot;
    if (mode == DenseGemmMode::kFloat16Vae)
      slot = PreparedF16Activation::create(context, rows, k);
    TensorBatch float_batch = context.begin_batch();
    if (mode == DenseGemmMode::kFloat16Vae) {
      PreparedF16ActivationView prepared = slot.prepare(float_batch, fi, rows);
      float_plan.record(float_batch, prepared, fw, fo, 0,
                        bias_mode == DenseGemmBias::kNone ? nullptr : &bias);
      // A single conversion is shared by multiple projections in the same
      // chunk; recording a second consumer must not re-run preparation.
      float_plan.record(float_batch, prepared, fw, fo, 0,
                        bias_mode == DenseGemmBias::kNone ? nullptr : &bias);
    } else {
      float_plan.record(float_batch, fi, fw, fo, rows, 0, 0,
                        bias_mode == DenseGemmBias::kNone ? nullptr : &bias);
    }
    float_batch.submit().wait();
    context.download(fo, host_output.data(), host_output.size());
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t column = 0; column < n; ++column) {
        float expected = 0.0f;
        for (uint32_t inner = 0; inner < k; ++inner) {
          const float a = mode == DenseGemmMode::kFloat16Vae
              ? f16_to_f32(f32_to_f16(host_input[row * k + inner]))
              : host_input[row * k + inner];
          const float w = mode == DenseGemmMode::kFloat16Vae
              ? f16_to_f32(host_weight_f16[column * k + inner])
              : host_weight_f32[column * k + inner];
          expected = std::fma(a, w, expected);
        }
        if (bias_mode == DenseGemmBias::kFloat32)
          expected += bias_values[column];
        CHECK_MSG(float_bits(host_output[row * n + column]) == float_bits(expected),
                  "gemm float mode %u [%u,%u] differs", unsigned(mode), row,
                  column);
      }
    }
  };
  run_float_mode(DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone);
  run_float_mode(DenseGemmMode::kFloat32, DenseGemmBias::kFloat32);

  // Explicit fp32->fp16 boundary values exercise ties, signed zero,
  // subnormal-half results, carry into infinity, and both signs.
  {
    const std::vector<float> edge = {
        0.0f, -0.0f, std::ldexp(1.0f, -24), std::ldexp(1.0f, -25),
        std::nextafter(std::ldexp(1.0f, -25), 1.0f), 65504.0f,
        65520.0f, -65520.0f};
    const uint64_t es[] = {edge.size(), 1}, ews[] = {1, 1};
    DeviceTensor ei = context.allocate(TensorLayout::contiguous(es, 2));
    DeviceTensor ew = context.allocate(TensorLayout::contiguous(ews, 2),
                                       ScalarType::kFloat16);
    DeviceTensor eo = context.allocate(TensorLayout::contiguous(es, 2));
    const uint16_t one = f32_to_f16(1.0f);
    context.upload(ei, edge.data(), edge.size());
    context.upload_bytes(ew, &one, sizeof(one));
    PreparedF16Activation slot = PreparedF16Activation::create(
        context, static_cast<uint32_t>(edge.size()), 1);
    DenseGemmPlan edge_plan = DenseGemmPlan::create(
        context, {static_cast<uint32_t>(edge.size()), 1, 1,
                  DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone});
    TensorBatch edge_batch = context.begin_batch();
    PreparedF16ActivationView prepared = slot.prepare(
        edge_batch, ei, static_cast<uint32_t>(edge.size()));
    edge_plan.record(edge_batch, prepared, ew, eo);
    edge_batch.submit().wait();
    std::vector<float> got(edge.size());
    context.download(eo, got.data(), got.size());
    for (size_t i = 0; i < edge.size(); ++i) {
      const float narrowed = f16_to_f32(f32_to_f16(edge[i]));
      const float expected = std::fma(narrowed, 1.0f, 0.0f);
      CHECK(std::memcmp(&got[i], &expected, sizeof(float)) == 0);
    }
  }

  bool invalid_mode_rejected = false;
  try {
    DenseGemmPlanDesc invalid{rows, n, k,
        static_cast<DenseGemmMode>(0xffffffffu), DenseGemmBias::kNone};
    (void)DenseGemmPlan::create(context, invalid);
  } catch (const std::invalid_argument&) {
    invalid_mode_rejected = true;
  }
  CHECK(invalid_mode_rejected);
  // Invalid plan construction is entirely pre-record and cannot poison a
  // subsequent valid batch.
  run_float_mode(DenseGemmMode::kFloat32, DenseGemmBias::kNone);
  bool invalid_bias_rejected = false;
  try {
    (void)DenseGemmPlan::create(
        context, {rows, n, k, DenseGemmMode::kFloat16Vae,
                  DenseGemmBias::kFloat32});
  } catch (const std::invalid_argument&) {
    invalid_bias_rejected = true;
  }
  CHECK(invalid_bias_rejected);
  TensorBatch validation_batch = context.begin_batch();
  bool alias_rejected = false, range_rejected = false, missing_bias_rejected = false;
  try {
    plan.record(validation_batch, input, weight, input, rows, 1, 0, &bias);
  } catch (const std::invalid_argument&) {
    alias_rejected = true;
  }
  try {
    plan.record(validation_batch, input, weight, output, desc.max_rows + 1,
                0, 0, &bias);
  } catch (const std::invalid_argument&) {
    range_rejected = true;
  }
  try {
    plan.record(validation_batch, input, weight, output, rows, 1, 0, nullptr);
  } catch (const std::invalid_argument&) {
    missing_bias_rejected = true;
  }
  CHECK(alias_rejected); CHECK(range_rejected); CHECK(missing_bias_rejected);
  plan.record(validation_batch, input, weight, output, rows, 1, 0, &bias);
  validation_batch.submit().wait();

  // Prepared fp16 activations are batch-scoped, shared by distinct
  // projections, and retained exactly through their submission token.
  const uint64_t staging_warm_shape[] = {64, 32};
  {
    DeviceTensor staging_warm = context.allocate(
        TensorLayout::contiguous(staging_warm_shape, 2));
    std::vector<float> staging_warm_values(64 * 32, 0.0f);
    context.upload(staging_warm, staging_warm_values.data(),
                   staging_warm_values.size());
  }
  context.upload_bytes(input, input_bits.data(), input_bits.size() * 2);
  const uint64_t gemm_lifetime_baseline = context.pooled_used_bytes();
  {
    constexpr uint32_t lm = 64, lk = 32, ln0 = 16, ln1 = 32;
    const uint64_t ais[] = {lm, lk}, w0s[] = {ln0, lk}, w1s[] = {ln1, lk};
    const uint64_t o0s[] = {lm, ln0}, o1s[] = {lm, ln1};
    DeviceTensor ai = context.allocate(TensorLayout::contiguous(ais, 2));
    DeviceTensor w0 = context.allocate(TensorLayout::contiguous(w0s, 2),
                                       ScalarType::kFloat16);
    DeviceTensor w1 = context.allocate(TensorLayout::contiguous(w1s, 2),
                                       ScalarType::kFloat16);
    DeviceTensor o0 = context.allocate(TensorLayout::contiguous(o0s, 2));
    DeviceTensor o1 = context.allocate(TensorLayout::contiguous(o1s, 2));
    std::vector<float> ah(size_t(lm) * lk, 0.25f);
    std::vector<uint16_t> w0h(size_t(ln0) * lk, f32_to_f16(0.5f));
    std::vector<uint16_t> w1h(size_t(ln1) * lk, f32_to_f16(-0.25f));
    context.upload(ai, ah.data(), ah.size());
    context.upload_bytes(w0, w0h.data(), w0h.size() * 2);
    context.upload_bytes(w1, w1h.data(), w1h.size() * 2);
    PreparedF16Activation slot0 = PreparedF16Activation::create(context, lm, lk);
    PreparedF16Activation slot1 = PreparedF16Activation::create(context, lm, lk);
    DenseGemmPlan p0 = DenseGemmPlan::create(
        context, {lm, ln0, lk, DenseGemmMode::kFloat16Vae,
                  DenseGemmBias::kNone});
    DenseGemmPlan p1 = DenseGemmPlan::create(
        context, {lm, ln1, lk, DenseGemmMode::kFloat16Vae,
                  DenseGemmBias::kNone});

    TensorBatch first = context.begin_batch();
    PreparedF16ActivationView first_view = slot0.prepare(first, ai, lm);
    TensorBatch moved = std::move(first);
    p0.record(moved, first_view, w0, o0);
    p1.record(moved, first_view, w1, o1);
    Submission first_token = moved.submit();
    TensorBatch second = context.begin_batch();
    PreparedF16ActivationView second_view = slot1.prepare(second, ai, lm);
    p0.record(second, second_view, w0, o0);
    p1.record(second, second_view, w1, o1);
    Submission second_token = second.submit();
    CHECK(second_token.value() > first_token.value());
    // The third begin waits for/reuses the oldest bounded command slot before
    // slot0 is overwritten; it does not allocate a third command arena.
    TensorBatch third = context.begin_batch();
    PreparedF16ActivationView third_view = slot0.prepare(third, ai, lm);
    p0.record(third, third_view, w0, o0);
    Submission third_token = third.submit();
    CHECK(third_token.value() > second_token.value());
    first_token.wait(); second_token.wait(); third_token.wait();
    std::vector<float> got0(size_t(lm) * ln0), got1(size_t(lm) * ln1);
    context.download(o0, got0.data(), got0.size());
    context.download(o1, got1.data(), got1.size());
    for (float value : got0) CHECK(value == 4.0f);
    for (float value : got1) CHECK(value == -2.0f);
    const uint64_t warm_reserved = context.reserved_bytes();
    const uint64_t warm_descriptors = context.descriptor_set_allocations();

    // A submitted view is permanently stale, even if allocator addresses are
    // reused. Its rejection is pre-record, so a fresh view continues in the
    // same batch.
    TensorBatch after_submit = context.begin_batch();
    bool submitted_stale_rejected = false;
    try {
      p0.record(after_submit, third_view, w0, o0);
    } catch (const std::invalid_argument&) {
      submitted_stale_rejected = true;
    }
    CHECK(submitted_stale_rejected);
    PreparedF16ActivationView fresh = slot1.prepare(after_submit, ai, lm);
    p0.record(after_submit, fresh, w0, o0);
    after_submit.submit().wait();

    for (int repeat = 0; repeat < 4; ++repeat) {
      TensorBatch stable = context.begin_batch();
      PreparedF16ActivationView stable_view = slot0.prepare(stable, ai, lm);
      p0.record(stable, stable_view, w0, o0);
      p1.record(stable, stable_view, w1, o1);
      stable.submit().wait();
      CHECK(context.reserved_bytes() == warm_reserved);
      CHECK(context.descriptor_set_allocations() == warm_descriptors);
    }

    // Preparation is one logical operator: 31 consumers reach the exact
    // 32-op bound and the 32nd consumer poisons/rejects submission.
    bool thirty_third_gemm_rejected = false;
    {
      TensorBatch bounded = context.begin_batch();
      PreparedF16ActivationView bounded_view = slot0.prepare(bounded, ai, lm);
      for (int i = 0; i < 31; ++i) p0.record(bounded, bounded_view, w0, o0);
      try {
        p0.record(bounded, bounded_view, w0, o0);
      } catch (const std::logic_error&) {
        thirty_third_gemm_rejected = true;
      }
    }
    CHECK(thirty_third_gemm_rejected);
  }
  // A boundary operation collects completed jobs. All wrappers above were
  // dropped while the context stayed alive; no GEMM-owned device allocation
  // remains pinned.
  context.upload_bytes(input, input_bits.data(), input_bits.size() * 2);
  CHECK(context.pooled_used_bytes() == gemm_lifetime_baseline);

  // Drop every caller wrapper immediately after submit. The job retains all
  // four tensors, the prepared slot and plan context until its exact token is
  // collected; afterward the pool returns to the pre-job live-byte baseline.
  Submission wrapper_drop_token;
  {
    constexpr uint32_t dm = 64, dn = 16, dk = 32;
    const uint64_t das[] = {dm, dk}, dws[] = {dn, dk}, dos[] = {dm, dn};
    DeviceTensor da = context.allocate(TensorLayout::contiguous(das, 2));
    DeviceTensor dw = context.allocate(TensorLayout::contiguous(dws, 2),
                                       ScalarType::kFloat16);
    DeviceTensor dout = context.allocate(TensorLayout::contiguous(dos, 2));
    PreparedF16Activation dslot =
        PreparedF16Activation::create(context, dm, dk);
    DenseGemmPlan dplan = DenseGemmPlan::create(
        context, {dm, dn, dk, DenseGemmMode::kFloat16Vae,
                  DenseGemmBias::kNone});
    TensorBatch drop_batch = context.begin_batch();
    PreparedF16ActivationView dview = dslot.prepare(drop_batch, da, dm);
    dplan.record(drop_batch, dview, dw, dout);
    wrapper_drop_token = drop_batch.submit();
  }
  CHECK(context.pooled_used_bytes() > gemm_lifetime_baseline);
  wrapper_drop_token.wait();
  context.upload_bytes(input, input_bits.data(), input_bits.size() * 2);
  CHECK(context.pooled_used_bytes() == gemm_lifetime_baseline);

  auto check_isolated_view_contract = [&](bool supersession) {
    TensorContext isolated(device);
    constexpr uint32_t im = 1, in = 1, ik = 1;
    const uint64_t is[] = {im, ik}, ws[] = {in, ik}, os[] = {im, in};
    DeviceTensor ii = isolated.allocate(TensorLayout::contiguous(is, 2));
    DeviceTensor iw = isolated.allocate(TensorLayout::contiguous(ws, 2),
                                        ScalarType::kFloat16);
    DeviceTensor io = isolated.allocate(TensorLayout::contiguous(os, 2));
    PreparedF16Activation slot =
        PreparedF16Activation::create(isolated, im, ik);
    DenseGemmPlan plan = DenseGemmPlan::create(
        isolated, {im, in, ik, DenseGemmMode::kFloat16Vae,
                   DenseGemmBias::kNone});
    if (supersession) {
      TensorBatch batch = isolated.begin_batch();
      PreparedF16ActivationView old = slot.prepare(batch, ii, im);
      PreparedF16ActivationView newest = slot.prepare(batch, ii, im);
      bool rejected = false;
      try {
        plan.record(batch, old, iw, io);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      CHECK(rejected);
      plan.record(batch, newest, iw, io);
      batch.submit().wait();
    } else {
      PreparedF16ActivationView discarded;
      {
        TensorBatch abandoned = isolated.begin_batch();
        discarded = slot.prepare(abandoned, ii, im);
      }
      TensorBatch next = isolated.begin_batch();
      bool rejected = false;
      try {
        plan.record(next, discarded, iw, io);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      CHECK(rejected);
      PreparedF16ActivationView fresh = slot.prepare(next, ii, im);
      plan.record(next, fresh, iw, io);
      next.submit().wait();
    }
  };
  check_isolated_view_contract(true);
  check_isolated_view_contract(false);

  // Temporary development measurement; retained as a visible performance
  // guard until the production-shape suite supplies the same metric.
  {
    constexpr uint32_t bm = 64, bn = 5376, bk = 5376;
    const uint64_t as[] = {bm, bk}, ws[] = {bn, bk}, os[] = {bm, bn};
    DeviceTensor ai = context.allocate(TensorLayout::contiguous(as, 2),
                                       ScalarType::kBFloat16);
    DeviceTensor wi = context.allocate(TensorLayout::contiguous(ws, 2),
                                       ScalarType::kBFloat16);
    DeviceTensor oi = context.allocate(TensorLayout::contiguous(os, 2),
                                       ScalarType::kBFloat16);
    std::vector<uint16_t> az(size_t(bm) * bk, reference_bf16(0.25f));
    std::vector<uint16_t> wz(size_t(bn) * bk, reference_bf16(0.001f));
    context.upload_bytes(ai, az.data(), az.size() * 2);
    context.upload_bytes(wi, wz.data(), wz.size() * 2);
    DenseGemmPlanDesc bd{bm, bn, bk, DenseGemmMode::kBFloat16,
                         DenseGemmBias::kNone};
    DenseGemmPlan bp = DenseGemmPlan::create(context, bd);
    for (int iteration = -1; iteration < 3; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      TensorBatch b = context.begin_batch();
      bp.record(b, ai, wi, oi, bm);
      b.submit().wait();
      if (iteration >= 0) {
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::printf("  deterministic Vulkan BF16 GEMM 64x5376x5376: %.3f ms\n", ms);
      }
    }
    const auto batched_start = std::chrono::steady_clock::now();
    TensorBatch repeated = context.begin_batch();
    for (int i = 0; i < 16; ++i) bp.record(repeated, ai, wi, oi, bm);
    repeated.submit().wait();
    const double batched_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - batched_start).count() / 16.0;
    std::printf("  deterministic Vulkan BF16 GEMM batched device time proxy: %.3f ms\n",
                batched_ms);
  }
}

SLOPFAB_TEST_CATEGORY(vulkan_gemm_dispatch_geometry, "synthetic") {
  using slopfab::vulkan::detail::GemmDispatchGeometry;
  using slopfab::vulkan::detail::gemm_dispatch_geometry;
  GemmDispatchGeometry geometry{99, 99};
  CHECK(!gemm_dispatch_geometry(0, 1, 16, 16, 8, 8, &geometry));
  CHECK(!gemm_dispatch_geometry(1, 0, 16, 16, 8, 8, &geometry));
  CHECK(gemm_dispatch_geometry(1, 1, 16, 16, 8, 8, &geometry));
  CHECK(geometry.x == 1 && geometry.y == 1);
  CHECK(gemm_dispatch_geometry(15, 15, 16, 16, 8, 8, &geometry));
  CHECK(geometry.x == 1 && geometry.y == 1);
  CHECK(gemm_dispatch_geometry(16, 16, 16, 16, 8, 8, &geometry));
  CHECK(geometry.x == 1 && geometry.y == 1);
  CHECK(gemm_dispatch_geometry(17, 17, 16, 16, 8, 8, &geometry));
  CHECK(geometry.x == 2 && geometry.y == 2);
  CHECK(gemm_dispatch_geometry(128, 128, 16, 16, 8, 8, &geometry));
  CHECK(geometry.x == 8 && geometry.y == 8);
  CHECK(!gemm_dispatch_geometry(129, 128, 16, 16, 8, 8, &geometry));
  CHECK(!gemm_dispatch_geometry(128, 129, 16, 16, 8, 8, &geometry));
  CHECK(!gemm_dispatch_geometry(UINT64_MAX, UINT64_MAX, 16, 16,
                                UINT32_MAX, UINT32_MAX, &geometry));
  CHECK(!gemm_dispatch_geometry(1, 1, 0, 16, 8, 8, &geometry));
  CHECK(!gemm_dispatch_geometry(1, 1, 16, 16, 8, 8, nullptr));
}
