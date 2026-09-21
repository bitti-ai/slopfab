#include "detail/vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(vulkan_qwen_extended_prompt_attention, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("Vulkan unavailable");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("Vulkan timeline semaphores unavailable");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  if (!context.exact_causal_gqa_attention()) {
    SKIP_UNSUPPORTED_HARDWARE("Exact causal GQA unavailable");
    return;
  }

  constexpr uint32_t sequence = text::kMaxPromptTokens;
  constexpr uint32_t query_heads = 64, kv_heads = 8, dim = 128;
  const uint64_t q_shape[] = {sequence, query_heads, dim};
  const uint64_t kv_shape[] = {sequence, kv_heads, dim};
  auto q = context.allocate(TensorLayout::contiguous(q_shape, 3), ScalarType::kBFloat16);
  auto k = context.allocate(TensorLayout::contiguous(kv_shape, 3), ScalarType::kBFloat16);
  auto v = context.allocate(TensorLayout::contiguous(kv_shape, 3), ScalarType::kBFloat16);
  auto out = context.allocate(TensorLayout::contiguous(q_shape, 3), ScalarType::kBFloat16);
  std::vector<uint16_t> zeros(size_t(sequence) * query_heads * dim, 0);
  context.upload_bytes(q, zeros.data(), zeros.size() * sizeof(uint16_t));
  context.upload_bytes(k, zeros.data(), size_t(sequence) * kv_heads * dim * sizeof(uint16_t));
  std::vector<uint16_t> values(size_t(sequence) * kv_heads * dim, reference_bf16(1.0f));
  std::fill(values.begin() + values.size() / 2, values.end(), reference_bf16(3.0f));
  context.upload_bytes(v, values.data(), values.size() * sizeof(uint16_t));
  auto plan = CausalGQAAttentionPlan::create(
      context, {sequence, query_heads, kv_heads, dim, exact_attention_scale(dim)});
  bool rejected = false;
  try {
    (void)CausalGQAAttentionPlan::create(
        context, {sequence + 1, query_heads, kv_heads, dim, exact_attention_scale(dim)});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);

  // Evaluate only two rows: one past the old ceiling and the last row at the
  // new ceiling. Uniform attention must include the second half's values.
  auto batch = context.begin_batch();
  plan.record(batch, q, k, v, out, 8192, 1, 0);
  plan.record(batch, q, k, v, out, sequence - 1, 1, 1);
  batch.submit().wait();
  std::vector<uint16_t> result = std::move(zeros);
  context.download_bytes(out, result.data(), result.size() * sizeof(uint16_t));
  CHECK(std::all_of(result.begin(), result.begin() + query_heads * dim, [](uint16_t x) {
    return x == reference_bf16(1.0f);
  }));
  CHECK(std::all_of(result.begin() + query_heads * dim, result.begin() + 2 * query_heads * dim,
                    [](uint16_t x) {
                      return x == reference_bf16(2.0f);
                    }));
}
