#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_exact_causal_gqa_attention, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
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
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_causal_gqa_attention()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_causal_gqa_attention()");
    return;
  }

  constexpr uint32_t sequence = 129;
  constexpr uint32_t query_heads = 64;
  constexpr uint32_t kv_heads = 8;
  constexpr uint32_t dim = 128;
  const size_t q_count = size_t(sequence) * query_heads * dim;
  const size_t kv_count = size_t(sequence) * kv_heads * dim;
  std::vector<uint16_t> hq(q_count), hk(kv_count), hv(kv_count);
  for (size_t i = 0; i < q_count; ++i)
    hq[i] = f32_to_bf16(float(int(i % 29) - 14) / 32.0f);
  for (size_t i = 0; i < kv_count; ++i) {
    hk[i] = f32_to_bf16(float(int(i % 31) - 15) / 32.0f);
    hv[i] = f32_to_bf16(float(int(i % 37) - 18) / 16.0f);
  }
  hq[0] = 0x0001u;
  hq[1] = 0x807fu;
  hk[0] = 0x007fu;
  hk[1] = 0x8001u;
  hv[0] = 0x0001u;
  hv[1] = 0x807fu;
  cuda::DeviceBuffer<uint16_t> cq(q_count), ck(kv_count), cv(kv_count),
      co(q_count), co_repeat(q_count);
  cq.copy_from_host(hq.data(), q_count);
  ck.copy_from_host(hk.data(), kv_count);
  cv.copy_from_host(hv.data(), kv_count);
  cuda::launch_deterministic_causal_gqa_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, query_heads,
      kv_heads, dim, exact_attention_scale(dim));
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> expected(q_count);
  co.copy_to_host(expected.data(), q_count);
  cuda::launch_deterministic_causal_gqa_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co_repeat.get()), sequence, query_heads,
      kv_heads, dim, exact_attention_scale(dim));
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> expected_repeat(q_count);
  co_repeat.copy_to_host(expected_repeat.data(), q_count);
  CHECK(expected_repeat == expected);

  const uint64_t q_shape[] = {sequence, query_heads, dim};
  const uint64_t kv_shape[] = {sequence, kv_heads, dim};
  DeviceTensor q = vk.allocate(TensorLayout::contiguous(q_shape, 3),
                               ScalarType::kBFloat16);
  DeviceTensor k = vk.allocate(TensorLayout::contiguous(kv_shape, 3),
                               ScalarType::kBFloat16);
  DeviceTensor v = vk.allocate(TensorLayout::contiguous(kv_shape, 3),
                               ScalarType::kBFloat16);
  DeviceTensor out = vk.allocate(TensorLayout::contiguous(q_shape, 3),
                                 ScalarType::kBFloat16);
  vk.upload_bytes(q, hq.data(), q_count * sizeof(uint16_t));
  vk.upload_bytes(k, hk.data(), kv_count * sizeof(uint16_t));
  vk.upload_bytes(v, hv.data(), kv_count * sizeof(uint16_t));
  CausalGQAAttentionPlanDesc desc{sequence, query_heads, kv_heads, dim,
                                  exact_attention_scale(dim)};
  CausalGQAAttentionPlan plan = CausalGQAAttentionPlan::create(vk, desc);
  TensorBatch batch = vk.begin_batch();
  plan.record(batch, q, k, v, out, 0, 65, 0);
  plan.record(batch, q, k, v, out, 65, sequence - 65, 65);
  batch.submit().wait();
  std::vector<uint16_t> got(q_count);
  vk.download_bytes(out, got.data(), got.size() * sizeof(uint16_t));
  if (got != expected) {
    for (size_t i = 0; i < got.size(); ++i) {
      if (got[i] != expected[i]) {
        std::printf("  causal GQA mismatch %zu: CUDA %04x Vulkan %04x\n",
                    i, expected[i], got[i]);
        break;
      }
    }
  }
  CHECK(got == expected);
  CHECK(got[0] == 0x0000u);
  // The negative subnormal is first canonicalized to -0; adding that product
  // to the +0 accumulator has the specified round-to-nearest result +0.
  CHECK(got[1] == 0x0000u);
  // Causal row zero is exactly V row zero from the mapped KV head.
  for (uint32_t h = 0; h < query_heads; ++h) {
    const uint32_t kv = h / (query_heads / kv_heads);
    for (uint32_t d = 0; d < dim; ++d) {
      uint16_t expected_value = hv[size_t(kv) * dim + d];
      if ((expected_value & 0x7f80u) == 0u)
        expected_value = 0u;
      CHECK(got[size_t(h) * dim + d] == expected_value);
    }
  }

  // Exercise both sides of the 128-key recurrence boundary. Future K/V rows
  // carry large sentinels; row zero must still be exactly V[0] for each mapped
  // KV head, proving that the causal kernel never reads the upper triangle.
  auto run_boundary = [&](uint32_t boundary_sequence, uint32_t first_rows) {
    const size_t boundary_q_count =
        size_t(boundary_sequence) * query_heads * dim;
    const size_t boundary_kv_count =
        size_t(boundary_sequence) * kv_heads * dim;
    std::vector<uint16_t> bq(boundary_q_count, f32_to_bf16(0.0f));
    std::vector<uint16_t> bk(boundary_kv_count);
    std::vector<uint16_t> bv(boundary_kv_count);
    for (uint32_t row = 0; row < boundary_sequence; ++row) {
      for (uint32_t head = 0; head < kv_heads; ++head) {
        for (uint32_t d = 0; d < dim; ++d) {
          const size_t index =
              (size_t(row) * kv_heads + head) * dim + d;
          bk[index] = f32_to_bf16(row == 0 ? 0.0f : 31.0f);
          bv[index] = f32_to_bf16(
              row == 0 ? float(int(head) - 4) / 8.0f
                       : float(int((row + head + d) % 15) - 7) / 4.0f);
        }
      }
    }
    cuda::DeviceBuffer<uint16_t> dcq(boundary_q_count), dck(boundary_kv_count),
        dcv(boundary_kv_count), dco(boundary_q_count);
    dcq.copy_from_host(bq.data(), bq.size());
    dck.copy_from_host(bk.data(), bk.size());
    dcv.copy_from_host(bv.data(), bv.size());
    cuda::launch_deterministic_causal_gqa_attention(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(dcq.get()),
        reinterpret_cast<const __nv_bfloat16*>(dck.get()),
        reinterpret_cast<const __nv_bfloat16*>(dcv.get()),
        reinterpret_cast<__nv_bfloat16*>(dco.get()), boundary_sequence,
        query_heads, kv_heads, dim, exact_attention_scale(dim));
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> boundary_expected(boundary_q_count);
    dco.copy_to_host(boundary_expected.data(), boundary_expected.size());

    const uint64_t boundary_q_shape[] = {boundary_sequence, query_heads, dim};
    const uint64_t boundary_kv_shape[] = {boundary_sequence, kv_heads, dim};
    DeviceTensor vq_boundary = vk.allocate(
        TensorLayout::contiguous(boundary_q_shape, 3), ScalarType::kBFloat16);
    DeviceTensor vk_boundary = vk.allocate(
        TensorLayout::contiguous(boundary_kv_shape, 3), ScalarType::kBFloat16);
    DeviceTensor vv_boundary = vk.allocate(
        TensorLayout::contiguous(boundary_kv_shape, 3), ScalarType::kBFloat16);
    DeviceTensor vo_boundary = vk.allocate(
        TensorLayout::contiguous(boundary_q_shape, 3), ScalarType::kBFloat16);
    vk.upload_bytes(vq_boundary, bq.data(), bq.size() * sizeof(uint16_t));
    vk.upload_bytes(vk_boundary, bk.data(), bk.size() * sizeof(uint16_t));
    vk.upload_bytes(vv_boundary, bv.data(), bv.size() * sizeof(uint16_t));
    CausalGQAAttentionPlan boundary_plan = CausalGQAAttentionPlan::create(
        vk, {boundary_sequence, query_heads, kv_heads, dim,
             exact_attention_scale(dim)});
    TensorBatch boundary_batch = vk.begin_batch();
    boundary_plan.record(boundary_batch, vq_boundary, vk_boundary, vv_boundary,
                         vo_boundary, 0, first_rows, 0);
    if (first_rows < boundary_sequence)
      boundary_plan.record(boundary_batch, vq_boundary, vk_boundary,
                           vv_boundary, vo_boundary, first_rows,
                           boundary_sequence - first_rows, first_rows);
    boundary_batch.submit().wait();
    std::vector<uint16_t> boundary_got(boundary_q_count);
    vk.download_bytes(vo_boundary, boundary_got.data(),
                      boundary_got.size() * sizeof(uint16_t));
    CHECK(boundary_got == boundary_expected);
    for (uint32_t head = 0; head < query_heads; ++head) {
      const uint32_t mapped = head / (query_heads / kv_heads);
      for (uint32_t d = 0; d < dim; ++d)
        CHECK(boundary_got[size_t(head) * dim + d] ==
              bv[size_t(mapped) * dim + d]);
    }
  };
  run_boundary(1, 1);
  run_boundary(127, 63);
  run_boundary(128, 64);
  run_boundary(257, 129);

  DeviceTensor out_second = vk.allocate(TensorLayout::contiguous(q_shape, 3),
                                        ScalarType::kBFloat16);
  auto submit_full = [&](DeviceTensor& selected_output) {
    TensorBatch selected = vk.begin_batch();
    plan.record(selected, q, k, v, selected_output);
    return selected.submit();
  };
  Submission first_job = submit_full(out);
  Submission second_job = submit_full(out_second);
  CHECK(second_job.value() > first_job.value());
  Submission third_job = submit_full(out);
  CHECK(third_job.value() > second_job.value());
  first_job.wait();
  second_job.wait();
  third_job.wait();
  std::vector<uint16_t> second_got(q_count);
  vk.download_bytes(out_second, second_got.data(), second_got.size() * 2);
  CHECK(second_got == expected);
  vk.download_bytes(out, got.data(), got.size() * 2);
  CHECK(got == expected);
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 4; ++repeat) {
    Submission a = submit_full(out), b = submit_full(out_second),
               c = submit_full(out);
    CHECK(b.value() > a.value() && c.value() > b.value());
    a.wait(); b.wait(); c.wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  }

  // Validation happens before recording and a rejected call leaves the batch
  // usable. By contrast, exceeding the bounded 32-op command list poisons it.
  DeviceTensor wrong_type =
      vk.allocate(TensorLayout::contiguous(q_shape, 3), ScalarType::kFloat32);
  {
    TensorBatch recover = vk.begin_batch();
    bool rejected = false;
    try { plan.record(recover, q, k, v, wrong_type); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { plan.record(recover, q, k, v, q); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    plan.record(recover, q, k, v, out);
    recover.submit().wait();
  }
  {
    TensorBatch full = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      plan.record(full, q, k, v, out, 0, 1, 0);
    full.submit().wait();
  }
  const uint64_t saturated_reserved = vk.reserved_bytes();
  const uint64_t saturated_descriptors = vk.descriptor_set_allocations();
  {
    TensorBatch overflow = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      plan.record(overflow, q, k, v, out, 0, 1, 0);
    bool rejected = false;
    try { plan.record(overflow, q, k, v, out, 0, 1, 0); }
    catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
    bool submit_rejected = false;
    try { (void)overflow.submit(); }
    catch (const std::logic_error&) { submit_rejected = true; }
    CHECK(submit_rejected);
  }
  CHECK(vk.reserved_bytes() == saturated_reserved);
  CHECK(vk.descriptor_set_allocations() == saturated_descriptors);
  {
    TensorBatch full = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      plan.record(full, q, k, v, out, 0, 1, 0);
    full.submit().wait();
  }
  CHECK(vk.reserved_bytes() == saturated_reserved);
  CHECK(vk.descriptor_set_allocations() == saturated_descriptors);

  // The submitted job retains input allocations after all caller wrappers are
  // dropped. The kept output remains readable after token completion.
  Submission retained = submit_full(out);
  q = DeviceTensor();
  k = DeviceTensor();
  v = DeviceTensor();
  retained.wait();
  vk.download_bytes(out, got.data(), got.size() * 2);
  CHECK(got == expected);
}
