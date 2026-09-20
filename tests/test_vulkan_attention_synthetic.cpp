#include "detail/vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(vulkan_exact_blocked_attention_single_key, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore || !physical.front().info().shader_int64");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  if (!context.exact_normalization()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !context.exact_normalization()");
    return;
  }

  constexpr uint32_t sequence = 1, heads = 2, dim = 64;
  const uint64_t extent[] = {sequence, heads, dim};
  const TensorLayout layout = TensorLayout::contiguous(extent, 3);
  DeviceTensor q = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor k = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor v = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor out = context.allocate(layout, ScalarType::kBFloat16);
  std::vector<uint16_t> zeros(heads * dim, 0);
  std::vector<uint16_t> values(heads * dim);
  for (size_t i = 0; i < values.size(); ++i) {
    const float value = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
    values[i] = reference_bf16(value);
  }
  const uint16_t conversion_edges[] = {
      0x0000u, 0x8000u, 0x0001u, 0x007fu, 0x0080u, 0x387fu,
      0x3880u, 0x7f7fu, 0x7f80u, 0xff80u, 0x3f80u, 0x3f81u};
  std::copy(std::begin(conversion_edges), std::end(conversion_edges),
            values.begin());
  std::vector<uint16_t> expected(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    expected[i] = f32_to_bf16(f16_to_f32(f32_to_f16(bf16_to_f32(values[i]))));
  }
  // The fixed PV FMA starts from +0, so (-0 * 1) + +0 is +0.
  expected[1] = 0;
  context.upload_bytes(q, zeros.data(), zeros.size() * sizeof(uint16_t));
  context.upload_bytes(k, zeros.data(), zeros.size() * sizeof(uint16_t));
  context.upload_bytes(v, values.data(), values.size() * sizeof(uint16_t));
  BlockedAttentionPlanDesc desc;
  desc.sequence = sequence;
  desc.heads = heads;
  desc.head_dim = dim;
  desc.scale = 0.125f;
  BlockedAttentionPlan plan = BlockedAttentionPlan::create(context, desc);
  PreparedAttentionInputs prepared = PreparedAttentionInputs::create(context, desc);
  TensorBatch batch = context.begin_batch();
  PreparedAttentionView inputs = prepared.prepare(batch, q, k, v);
  plan.record(batch, inputs, out);
  batch.submit().wait();
  std::vector<uint16_t> actual(values.size());
  context.download_bytes(out, actual.data(), actual.size() * sizeof(uint16_t));
  size_t edge_mismatch = expected.size();
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) { edge_mismatch = i; break; }
  }
  CHECK_MSG(edge_mismatch == expected.size(),
            "attention conversion edge %zu: %04x != %04x",
            edge_mismatch,
            edge_mismatch == expected.size() ? 0u : expected[edge_mismatch],
            edge_mismatch == expected.size() ? 0u : actual[edge_mismatch]);

  CHECK(prepared.reserved_bytes() == values.size() * sizeof(uint16_t) * 3);
  DeviceTensor out2 = context.allocate(layout, ScalarType::kBFloat16);
  PreparedAttentionInputs prepared2 = PreparedAttentionInputs::create(context, desc);
  TensorBatch first = context.begin_batch();
  PreparedAttentionView first_view = prepared.prepare(first, q, k, v);
  plan.record(first, first_view, out);
  Submission first_token = first.submit();
  TensorBatch second = context.begin_batch();
  PreparedAttentionView second_view = prepared2.prepare(second, q, k, v);
  plan.record(second, second_view, out2);
  Submission second_token = second.submit();
  TensorBatch third = context.begin_batch();
  PreparedAttentionView third_view = prepared.prepare(third, q, k, v);
  plan.record(third, third_view, out);
  Submission third_token = third.submit();
  CHECK(first_token.value() < second_token.value());
  CHECK(second_token.value() < third_token.value());
  first_token.wait(); second_token.wait(); third_token.wait();
  const uint64_t stable_reserved = context.reserved_bytes();
  const uint64_t stable_descriptors = context.descriptor_set_allocations();
  for (int repeat = 0; repeat < 3; ++repeat) {
    TensorBatch stable = context.begin_batch();
    PreparedAttentionView stable_view = prepared.prepare(stable, q, k, v);
    plan.record(stable, stable_view, out);
    stable.submit().wait();
    CHECK(context.reserved_bytes() == stable_reserved);
    CHECK(context.descriptor_set_allocations() == stable_descriptors);
  }

  TensorBatch after_submit = context.begin_batch();
  bool stale_rejected = false;
  try { plan.record(after_submit, third_view, out); }
  catch (const std::invalid_argument&) { stale_rejected = true; }
  CHECK(stale_rejected);
  PreparedAttentionView fresh = prepared.prepare(after_submit, q, k, v);
  plan.record(after_submit, fresh, out);
  after_submit.submit().wait();

  {
    TensorBatch superseded = context.begin_batch();
    PreparedAttentionView old = prepared.prepare(superseded, q, k, v);
    PreparedAttentionView newest = prepared.prepare(superseded, q, k, v);
    bool old_rejected = false;
    try { plan.record(superseded, old, out); }
    catch (const std::invalid_argument&) { old_rejected = true; }
    CHECK(old_rejected);
    plan.record(superseded, newest, out);
    superseded.submit().wait();
  }
  PreparedAttentionView discarded;
  {
    TensorBatch abandoned = context.begin_batch();
    discarded = prepared.prepare(abandoned, q, k, v);
  }
  TensorBatch recovery = context.begin_batch();
  bool discarded_rejected = false;
  try { plan.record(recovery, discarded, out); }
  catch (const std::invalid_argument&) { discarded_rejected = true; }
  CHECK(discarded_rejected);
  PreparedAttentionView recovery_view = prepared.prepare(recovery, q, k, v);
  bool alias_rejected = false;
  try { plan.record(recovery, recovery_view, q); }
  catch (const std::invalid_argument&) { alias_rejected = true; }
  CHECK(alias_rejected);
  plan.record(recovery, recovery_view, out);
  recovery.submit().wait();

  bool thirty_third_rejected = false;
  {
    TensorBatch bounded = context.begin_batch();
    PreparedAttentionView bounded_view = prepared.prepare(bounded, q, k, v);
    for (int i = 0; i < 31; ++i) plan.record(bounded, bounded_view, out);
    try { plan.record(bounded, bounded_view, out); }
    catch (const std::logic_error&) { thirty_third_rejected = true; }
  }
  CHECK(thirty_third_rejected);

  // Qwen image grids vary between requests. Recreating shape-specific plans
  // and slots must reuse the pool and the bounded descriptor arenas rather
  // than accumulating one allocation set per observed grid.
  auto run_shape = [&](uint32_t variable_sequence) {
    const uint64_t variable_extent[] = {variable_sequence, 1, dim};
    const TensorLayout variable_layout =
        TensorLayout::contiguous(variable_extent, 3);
    const size_t variable_count = size_t(variable_sequence) * dim;
    std::vector<uint16_t> data(variable_count, 0);
    DeviceTensor variable_q = context.allocate(variable_layout, ScalarType::kBFloat16);
    DeviceTensor variable_k = context.allocate(variable_layout, ScalarType::kBFloat16);
    DeviceTensor variable_v = context.allocate(variable_layout, ScalarType::kBFloat16);
    DeviceTensor variable_out = context.allocate(variable_layout, ScalarType::kBFloat16);
    context.upload_bytes(variable_q, data.data(), data.size() * 2);
    context.upload_bytes(variable_k, data.data(), data.size() * 2);
    context.upload_bytes(variable_v, data.data(), data.size() * 2);
    BlockedAttentionPlanDesc variable_desc{
        variable_sequence, 1, dim, exact_attention_scale(dim)};
    BlockedAttentionPlan variable_plan =
        BlockedAttentionPlan::create(context, variable_desc);
    PreparedAttentionInputs variable_prepared =
        PreparedAttentionInputs::create(context, variable_desc);
    TensorBatch variable_batch = context.begin_batch();
    PreparedAttentionView variable_inputs = variable_prepared.prepare(
        variable_batch, variable_q, variable_k, variable_v);
    variable_plan.record(variable_batch, variable_inputs, variable_out);
    variable_batch.submit().wait();
  };
  for (uint32_t variable_sequence : {3u, 17u, 5u, 33u}) run_shape(variable_sequence);
  context.upload_bytes(q, zeros.data(), zeros.size() * sizeof(uint16_t));
  const uint64_t varied_reserved = context.reserved_bytes();
  const uint64_t varied_descriptors = context.descriptor_set_allocations();
  for (uint32_t variable_sequence : {33u, 5u, 17u, 3u}) run_shape(variable_sequence);
  context.upload_bytes(q, zeros.data(), zeros.size() * sizeof(uint16_t));
  CHECK(context.reserved_bytes() == varied_reserved);
  CHECK(context.descriptor_set_allocations() == varied_descriptors);

  const uint64_t attention_live_baseline = context.pooled_used_bytes();
  Submission dropped_wrappers;
  {
    constexpr uint32_t drop_sequence = 3;
    const uint64_t drop_extent[] = {drop_sequence, 1, dim};
    const TensorLayout drop_layout = TensorLayout::contiguous(drop_extent, 3);
    DeviceTensor drop_q = context.allocate(drop_layout, ScalarType::kBFloat16);
    DeviceTensor drop_k = context.allocate(drop_layout, ScalarType::kBFloat16);
    DeviceTensor drop_v = context.allocate(drop_layout, ScalarType::kBFloat16);
    DeviceTensor drop_out = context.allocate(drop_layout, ScalarType::kBFloat16);
    BlockedAttentionPlanDesc drop_desc{
        drop_sequence, 1, dim, exact_attention_scale(dim)};
    PreparedAttentionInputs drop_prepared =
        PreparedAttentionInputs::create(context, drop_desc);
    BlockedAttentionPlan drop_plan = BlockedAttentionPlan::create(context, drop_desc);
    TensorBatch drop_batch = context.begin_batch();
    PreparedAttentionView drop_inputs = drop_prepared.prepare(
        drop_batch, drop_q, drop_k, drop_v);
    drop_plan.record(drop_batch, drop_inputs, drop_out);
    dropped_wrappers = drop_batch.submit();
  }
  CHECK(context.pooled_used_bytes() > attention_live_baseline);
  dropped_wrappers.wait();
  context.upload_bytes(q, zeros.data(), zeros.size() * sizeof(uint16_t));
  CHECK(context.pooled_used_bytes() == attention_live_baseline);
}

SLOPFAB_TEST_CATEGORY(vulkan_sage_device_and_memory_selection, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  DeviceInfo info;
  info.subgroup_size = 32;
  info.max_compute_workgroup_size[0] = info.max_compute_workgroup_invocations = 1024;
  info.max_compute_shared_memory_bytes = 32768;
  info.max_storage_buffer_bytes = info.max_allocation_bytes = 1ull << 30;
  H3AttentionPlanDesc d{4096, 8, 128, exact_attention_scale(128), AttentionMode::kSage2};
  auto select = [&](uint64_t budget) { return detail::select_sage_configuration(info, d, budget); };
  auto c = select(0);
  CHECK(c.kernel == 1 && c.shared_bytes == 29056 && c.local_size == 256);
  CHECK(!c.parallel_mean && !c.prepared_value && c.extra_workspace_bytes == 0);
  const uint64_t partials = 8ull * 128 * 16 * 4;
  CHECK(select(partials-1).extra_workspace_bytes == 0);
  c = select(partials);
  CHECK(c.parallel_mean && !c.prepared_value && c.extra_workspace_bytes == partials);
  const uint64_t prepared = 4096ull * 8 * 128 * 2;
  CHECK(!select(partials+prepared-1).prepared_value);
  c = select(partials+prepared);
  CHECK(c.prepared_value && c.extra_workspace_bytes == partials+prepared);
  CHECK(c.required_workspace_bytes == prepared + 8ull * (128 + 2 * 256) * 4);
  info.max_compute_shared_memory_bytes = 41344;
  CHECK(select(0).kernel == 1 && detail::sage_kernel_fits(info, 2));
  info.max_compute_shared_memory_bytes = 45824;
  CHECK(select(0).query_rows == 64);
  info.max_compute_workgroup_invocations = 256;
  CHECK(select(0).kernel == 1);
  info.subgroup_size = 64;
  CHECK(!detail::sage_kernel_fits(info, 1));
  info.max_compute_workgroup_invocations = 512;
  CHECK(select(0).local_size == 512 && select(0).kernel == 1);
  info.max_compute_workgroup_invocations = 1024;
  CHECK(select(0).query_rows == 32 && detail::sage_kernel_fits(info, 3));
  info.max_compute_shared_memory_bytes = 65536;
  d.sage_kernel = 4;
  CHECK(select(0).shared_bytes == 58112);
  info.max_compute_shared_memory_bytes = 58111;
  bool rejected = false;
  try { (void)select(0); } catch (const std::invalid_argument&) { rejected = true; }
  CHECK(rejected);
  d.sage_kernel = 1;
  info.max_storage_buffer_bytes = c.required_workspace_bytes - prepared;
  CHECK(!select(UINT64_MAX).parallel_mean && !select(UINT64_MAX).prepared_value);
  info.subgroup_size = 16;
  CHECK(!detail::sage_kernel_fits(info, 1));
}

SLOPFAB_TEST_CATEGORY(vulkan_sage_variants_and_bounded_preparation, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("No Vulkan loader");
    return;
  }
  auto instance = Instance::create(); auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("No Vulkan device");
    return;
  }
  const auto& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int8 || !info.shader_float16 ||
      !info.storage_buffer_16bit || !info.cooperative_matrix_i8_i32_16x16x32 ||
      !info.cooperative_matrix_f16_f32_16x16x16 || !detail::sage_kernel_fits(info, 1)) {
    SKIP_UNSUPPORTED_HARDWARE("Sage cooperative tuples unavailable"); return;
  }
  DeviceOptions opts;
  opts.enable_timeline_semaphore = opts.enable_shader_int8 = true;
  opts.enable_shader_float16 = opts.enable_storage_buffer_16bit = opts.enable_cooperative_matrix = true;
  auto device = physical.front().create_device(opts);
  for (uint32_t dim : {64u, 128u}) {
    constexpr uint32_t sequence = 513, heads = 2;
    const size_t n = size_t(sequence)*heads*dim;
    auto data = test::make_data(n*3, 931, 0.8f);
    for (size_t i = 0; i < n; ++i) data[n+i] += i % dim < dim/2 ? 12.0f : -8.0f;
    std::vector<uint16_t> host(n*3), baseline;
    for (size_t i = 0; i < host.size(); ++i) host[i] = reference_bf16(data[i]);
    for (uint64_t budget : {0ull, 4096ull, 64ull<<20}) {
      TensorContextOptions tc; tc.max_batch_operators = 1; tc.sage_extra_workspace_bytes = budget;
      TensorContext context(device, tc);
      const uint64_t shape[] = {sequence, heads, dim};
      const auto layout = TensorLayout::contiguous(shape, 3);
      auto q = context.allocate(layout, ScalarType::kBFloat16);
      auto k = context.allocate(layout, ScalarType::kBFloat16);
      auto v = context.allocate(layout, ScalarType::kBFloat16);
      auto out = context.allocate(layout, ScalarType::kBFloat16);
      context.upload_bytes(q, host.data(), n*2); context.upload_bytes(k, host.data()+n, n*2);
      context.upload_bytes(v, host.data()+n*2, n*2);
      std::vector<int32_t> bounds(5*4);
      for (int i = 0; i < 5; ++i) { bounds[i*4] = 0; bounds[i*4+1] = 64; bounds[i*4+2] = 384; bounds[i*4+3] = 576; }
      auto ranges = H3AttentionRanges::create(context, sequence, bounds.data(), uint32_t(bounds.size()));
      for (uint32_t kernel = 1; kernel <= 4; ++kernel) {
        if (!detail::sage_kernel_fits(info, kernel)) continue;
        auto plan = H3AttentionPlan::create(context, {sequence, heads, dim, exact_attention_scale(dim), AttentionMode::kSage2, kernel});
        const auto c = plan.sage_configuration();
        CHECK(c.extra_workspace_bytes <= budget);
        CHECK(plan.workspace_bytes() == c.required_workspace_bytes + c.extra_workspace_bytes);
        CHECK(c.parallel_mean == (budget >= uint64_t(heads)*dim*3*4));
        CHECK(c.prepared_value == (budget >= uint64_t(heads)*dim*3*4 + n*2));
        TimestampQuery timing;
        if (info.timestamp_valid_bits) timing = TimestampQuery::create(device, 4);
        auto batch = context.begin_batch();
        plan.record(batch, q, k, v, out, &ranges, 0, 0, 0, timing.count() ? &timing : nullptr);
        CHECK(batch.remaining_operator_capacity() == 0);
        batch.submit().wait();
        if (timing.count()) {
          for (uint32_t i = 0; i < 3; ++i) CHECK(timing.elapsed_milliseconds(i, i+1) >= 0);
          bool rejected = false;
          try { (void)timing.elapsed_milliseconds(0, 4); } catch (const std::invalid_argument&) { rejected = true; }
          CHECK(rejected);
        }
        std::vector<uint16_t> actual(n);
        context.download_bytes(out, actual.data(), n*2);
        if (baseline.empty()) baseline = actual;
        for (size_t i = 0; i < n; ++i) CHECK_NEAR(bf16_to_f32(actual[i]), bf16_to_f32(baseline[i]), 0.002);
        // Repeat after changing V, covering preparation freshness and pooled
        // allocation stability. All rows agree independently of Q/K quants.
        std::vector<uint16_t> constant(n, reference_bf16(0.5f));
        context.upload_bytes(v, constant.data(), n*2);
        const uint64_t used = context.pooled_used_bytes(), reserved = context.reserved_bytes();
        auto repeat = context.begin_batch(); plan.record(repeat, q, k, v, out, &ranges);
        auto token = repeat.submit(); plan = H3AttentionPlan(); token.wait();
        context.download_bytes(out, actual.data(), n*2);
        CHECK(actual == constant);
        CHECK(context.pooled_used_bytes() <= used && context.reserved_bytes() == reserved);
        context.upload_bytes(v, host.data()+n*2, n*2);
      }
    }
  }
}

SLOPFAB_TEST_CATEGORY(vulkan_fast_attention_accuracy_and_lifetime, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("Vulkan loader unavailable");
    return;
  }
  Instance instance = Instance::create();
  auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("No Vulkan device");
    return;
  }
  const auto& info = physical.front().info();
  DeviceOptions options;
  options.enable_timeline_semaphore = info.timeline_semaphore;
  Device disabled_device = physical.front().create_device(options);
  TensorContext disabled(disabled_device);
  for (auto mode : {AttentionMode::kFlash2, AttentionMode::kSage2, AttentionMode::kSol}) {
    CHECK(!disabled.h3_attention_supported(mode));
    bool rejected = false;
    try { (void)H3AttentionPlan::create(disabled, {1, 1, 64, 0.125f, mode}); }
    catch (const std::runtime_error&) { rejected = true; }
    CHECK(rejected);
  }
  if (!info.cooperative_matrix_bf16_f32_16x16x16 || !info.shader_float16 ||
      !info.storage_buffer_16bit || !info.timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("Fast attention cooperative matrices unavailable"); return;
  }
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  {
    Device no_int8_device = physical.front().create_device(options);
    TensorContext no_int8(no_int8_device);
    CHECK(!no_int8.h3_attention_supported(AttentionMode::kSage2));
  }
  options.enable_shader_int8 = info.shader_int8;
  // Fast attention does not require shaderInt64 or the exact driver allow-list.
  Device device = physical.front().create_device(options);
  TensorContext context(device, {2, 1});
  CHECK(!context.exact_h3_attention());
  for (auto mode : {AttentionMode::kFlash2, AttentionMode::kSage2}) {
    if (!context.h3_attention_supported(mode)) {
      SKIP_UNSUPPORTED_HARDWARE("%s unavailable on %s (INT8 tuple %d)",
          attention_mode_name(mode), info.name.c_str(), int(info.cooperative_matrix_i8_i32_16x16x32));
      continue;
    }
    double worst_l2 = 0.0, worst_abs = 0.0;
    for (float invalid_scale : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN()}) {
      bool rejected = false;
      try { (void)H3AttentionPlan::create(context, {1, 1, 64, invalid_scale, mode}); }
      catch (const std::invalid_argument&) { rejected = true; }
      CHECK(rejected);
    }
    for (uint32_t dim : {64u, 128u}) for (uint32_t sequence : {1u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 129u, 257u, 513u}) {
      constexpr uint32_t heads = 2;
      const uint64_t shape[] = {sequence, heads, dim};
      auto layout = TensorLayout::contiguous(shape, 3);
      size_t n = size_t(sequence) * heads * dim;
      auto qf = test::make_data(n, 73, 0.8f), kf = test::make_data(n, 321, 0.8f);
      auto vf = test::make_data(n, 999, 0.8f);
      std::vector<uint16_t> qh(n), kh(n), vh(n), actual(n), chunk(n);
      for (size_t i = 0; i < n; ++i) {
        // Large channel bias exercises K smoothing. A zero Q fixture also
        // tests all-zero quantization scales and uniform softmax.
        qh[i] = reference_bf16(sequence == 17 ? 0.0f : qf[i]);
        kh[i] = reference_bf16(kf[i] + (i % dim < dim/2 ? 12.0f : -8.0f));
        vh[i] = reference_bf16(vf[i]);
        qf[i] = bf16_to_f32(qh[i]); kf[i] = bf16_to_f32(kh[i]); vf[i] = bf16_to_f32(vh[i]);
      }
      auto q = context.allocate(layout, ScalarType::kBFloat16);
      auto k = context.allocate(layout, ScalarType::kBFloat16);
      auto v = context.allocate(layout, ScalarType::kBFloat16);
      auto out = context.allocate(layout, ScalarType::kBFloat16);
      auto chunks = context.allocate(layout, ScalarType::kBFloat16);
      context.upload_bytes(q, qh.data(), n*2); context.upload_bytes(k, kh.data(), n*2);
      context.upload_bytes(v, vh.data(), n*2);
      const float scale = exact_attention_scale(dim);
      auto plan = H3AttentionPlan::create(context, {sequence, heads, dim, scale, mode});
      CHECK((plan.workspace_bytes() != 0) == (mode == AttentionMode::kSage2));
      // Distinct ranges per global 128-row query tile; disjoint intervals
      // and padded last keys exercise both range traversal and masking.
      std::vector<int32_t> bounds;
      for (uint32_t tile = 0; tile < (sequence+127)/128; ++tile) {
        if (sequence >= 257 && tile % 2 == 0)
          bounds.insert(bounds.end(), {0, 64, 192, 320});
        else bounds.insert(bounds.end(), {0, int32_t((sequence+63)/64*64), 0, 0});
      }
      auto ranges = H3AttentionRanges::create(context, sequence, bounds.data(), uint32_t(bounds.size()));
      for (bool banded : {false, true}) {
        auto range_ptr = banded ? &ranges : nullptr;
        auto batch = context.begin_batch();
        bool rejected = false;
        try { plan.record(batch, q, k, v, q); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
        plan.record(batch, q, k, v, out, range_ptr);
        batch.submit().wait();
        context.download_bytes(out, actual.data(), n*2);
        std::vector<double> expected(n), scores(sequence);
        for (uint32_t row = 0; row < sequence; ++row) for (uint32_t head = 0; head < heads; ++head) {
          double maximum = -std::numeric_limits<double>::infinity();
          for (uint32_t key = 0; key < sequence; ++key) {
            auto b = bounds.data() + (row/128)*4;
            bool selected = !banded || (key >= uint32_t(b[0]) && key < uint32_t(b[1])) ||
                (key >= uint32_t(b[2]) && key < uint32_t(b[3]));
            double score = 0;
            for (uint32_t d = 0; d < dim; ++d)
              score += double(qf[(row*heads+head)*dim+d]) * kf[(key*heads+head)*dim+d];
            scores[key] = selected ? score*scale : -std::numeric_limits<double>::infinity();
            maximum = std::max(maximum, scores[key]);
          }
          double sum = 0;
          for (auto& s : scores) { s = std::exp(s-maximum); sum += s; }
          for (uint32_t key = 0; key < sequence; ++key) for (uint32_t d = 0; d < dim; ++d)
            expected[(row*heads+head)*dim+d] += scores[key]/sum * vf[(key*heads+head)*dim+d];
        }
        double err2 = 0, ref2 = 0, max_abs = 0;
        for (size_t i = 0; i < n; ++i) {
          double delta = bf16_to_f32(actual[i]) - expected[i];
          CHECK(std::isfinite(delta));
          err2 += delta*delta; ref2 += expected[i]*expected[i];
          max_abs = std::max(max_abs, std::abs(delta));
        }
        double rel = std::sqrt(err2 / std::max(ref2, 1e-30));
        worst_l2 = std::max(worst_l2, rel); worst_abs = std::max(worst_abs, max_abs);
        CHECK_MSG(rel < (mode == AttentionMode::kSage2 ? 0.025 : 0.005),
            "%s S%u D%u band%d relative L2 %.8f", attention_mode_name(mode), sequence, dim, int(banded), rel);
        CHECK_MSG(max_abs < (mode == AttentionMode::kSage2 ? 0.025 : 0.005),
            "%s S%u D%u max abs %.8f", attention_mode_name(mode), sequence, dim, max_abs);
        // Split through a query tile, writing to the same global row offsets.
        const uint32_t cut = std::min(7u, sequence);
        auto first = context.begin_batch();
        plan.record(first, q, k, v, chunks, range_ptr, 0, cut, 0);
        auto token1 = first.submit();
        if (sequence > cut) {
          auto second = context.begin_batch();
          plan.record(second, q, k, v, chunks, range_ptr, cut, sequence-cut, cut);
          second.submit().wait();
        }
        token1.wait();
        context.download_bytes(chunks, chunk.data(), n*2);
        CHECK(chunk == actual);
        if (sequence == 65 || sequence == 129) {
          std::fill(chunk.begin(), chunk.end(), uint16_t(0x4210));
          context.upload_bytes(chunks, chunk.data(), n*2);
          const auto used = context.pooled_used_bytes();
          const auto reserved = context.reserved_bytes();
          auto remap = context.begin_batch();
          plan.record(remap, q, k, v, chunks, range_ptr, 7, sequence-14, 3);
          CHECK(remap.remaining_operator_capacity() == 0);
          remap.submit().wait();
          CHECK(context.pooled_used_bytes() == used);
          CHECK(context.reserved_bytes() == reserved);
          context.download_bytes(chunks, chunk.data(), n*2);
          for (uint32_t row = 0; row < sequence; ++row) for (uint32_t d = 0; d < heads*dim; ++d) {
            const uint16_t expected_word = row >= 3 && row < sequence-11
                ? actual[size_t(row+4)*heads*dim+d] : uint16_t(0x4210);
            CHECK(chunk[size_t(row)*heads*dim+d] == expected_word);
          }
          // Exceeding the one-operator limit poisons this batch, then the
          // next batch recovers all input/workspace access state.
          auto over_capacity = context.begin_batch();
          plan.record(over_capacity, q, k, v, out, range_ptr);
          bool rejected = false;
          try { plan.record(over_capacity, q, k, v, out, range_ptr); }
          catch (const std::logic_error&) { rejected = true; }
          CHECK(rejected);
        }
      }
      // Re-recorded preparation must observe changed inputs, and submitted
      // work owns the plan's buffers even if its public wrapper is destroyed.
      std::fill(vh.begin(), vh.end(), reference_bf16(0.25f));
      std::fill(qh.begin(), qh.end(), uint16_t(0)); std::fill(kh.begin(), kh.end(), uint16_t(0));
      context.upload_bytes(q, qh.data(), n*2); context.upload_bytes(k, kh.data(), n*2);
      context.upload_bytes(v, vh.data(), n*2);
      auto final_batch = context.begin_batch();
      plan.record(final_batch, q, k, v, out);
      auto final_token = final_batch.submit();
      plan = H3AttentionPlan();
      final_token.wait();
      context.download_bytes(out, actual.data(), n*2);
      CHECK(actual == vh);
    }
    std::printf("  %s FP64 reference worst relative L2 %.8f max abs %.8f\n",
        attention_mode_name(mode), worst_l2, worst_abs);
  }
}

SLOPFAB_TEST_CATEGORY(vulkan_attention_prepare_exhaustive_bf16, "synthetic") {
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
  Device device = physical.front().create_device(options);
  ComputeContext context(device, {2, 6, 1});
  ComputePipelineOptions pipeline_options;
  pipeline_options.storage_binding_count = 6;
  pipeline_options.push_constant_bytes = sizeof(uint32_t);
  pipeline_options.local_size[0] = 64;
  ComputePipeline pipeline = ComputePipeline::create(
      device, load_spirv(SLOPFAB_TEST_ATTENTION_PREPARE_SPV_PATH),
      pipeline_options);

  constexpr uint32_t patterns = 1u << 16;
  constexpr uint64_t bytes = uint64_t{patterns} * sizeof(uint16_t);
  constexpr uint32_t words = patterns / 2;
  BufferPool pool(device, 1024 * 1024);
  Buffer upload = pool.allocate(bytes, BufferUsage::kTransferSource,
                                MemoryUsage::kUpload);
  Buffer source = pool.allocate(bytes, BufferUsage::kTransferDestination |
                                         BufferUsage::kStorage,
                                MemoryUsage::kDevice);
  Buffer output0 = pool.allocate(bytes, BufferUsage::kStorage |
                                          BufferUsage::kTransferSource,
                                 MemoryUsage::kDevice);
  Buffer output1 = pool.allocate(bytes, BufferUsage::kStorage,
                                 MemoryUsage::kDevice);
  Buffer output2 = pool.allocate(bytes, BufferUsage::kStorage,
                                 MemoryUsage::kDevice);
  Buffer readback = pool.allocate(bytes, BufferUsage::kTransferDestination,
                                  MemoryUsage::kReadback);
  std::vector<uint16_t> input(patterns), actual(patterns);
  for (uint32_t i = 0; i < patterns; ++i) input[i] = static_cast<uint16_t>(i);
  upload.write(0, input.data(), bytes);
  CommandList commands = context.begin();
  commands.barrier(upload, BufferAccess::kHostWrite, BufferAccess::kTransferRead);
  commands.copy_buffer(upload, source, bytes);
  commands.barrier(source, BufferAccess::kTransferWrite, BufferAccess::kComputeRead);
  commands.bind_compute(pipeline, {{0, &source, 0, bytes},
                                   {1, &source, 0, bytes},
                                   {2, &source, 0, bytes},
                                   {3, &output0, 0, bytes},
                                   {4, &output1, 0, bytes},
                                   {5, &output2, 0, bytes}});
  commands.push_constants(&words, sizeof(words));
  commands.dispatch((words + 63) / 64);
  commands.barrier(output0, BufferAccess::kComputeWrite,
                   BufferAccess::kTransferRead);
  commands.copy_buffer(output0, readback, bytes);
  commands.barrier(readback, BufferAccess::kTransferWrite,
                   BufferAccess::kHostRead);
  context.submit(std::move(commands)).wait();
  readback.read(0, actual.data(), bytes);
  for (uint32_t i = 0; i < patterns; ++i) {
    const uint16_t expected = (input[i] & 0x7fffu) > 0x7f80u
        ? 0x7fffu : f32_to_f16(bf16_to_f32(input[i]));
    CHECK(actual[i] == expected);
  }
}
