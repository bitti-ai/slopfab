#include "detail/veda_fixture.h"
#include "slopfab/vulkan/veda_attention.h"

SLOPFAB_TEST(veda_vulkan_matches_reference) {
  using namespace slopfab;
  if (!vulkan::Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("No Vulkan loader");
    return;
  }
  auto instance = vulkan::Instance::create();
  auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().cooperative_matrix_bf16_f32_16x16x16 ||
      !physical.front().info().cooperative_matrix_f16_f32_16x16x16) {
    SKIP_UNSUPPORTED_HARDWARE("VEDA requires BF16 and FP16 cooperative matrices");
    return;
  }
  vulkan::DeviceOptions options;
  options.enable_timeline_semaphore = options.enable_shader_int64 = true;
  options.enable_shader_float16 = options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  auto device = physical.front().create_device(options);
  vulkan::TensorContext context(device);
  if (!context.h3_attention_supported(AttentionMode::kVeda)) {
    SKIP_UNSUPPORTED_HARDWARE("VEDA unavailable");
    return;
  }
  bool missing_predictor_rejected = false;
  try {
    (void)vulkan::H3AttentionPlan::create(context, {128, 2, 128, 0.125f, AttentionMode::kVeda});
  } catch (const std::invalid_argument&) {
    missing_predictor_rejected = true;
  }
  CHECK(missing_predictor_rejected);
  for (int dim : {64, 128})
    for (const char* budget : {"2.3", "0%"}) {
      test::VedaFixture fixture(dim, budget, "vulkan", dim == 128);
      const size_t n = fixture.q.size();
      const uint64_t shape[] = {uint64_t(fixture.request->layout.total_rows()), 2, uint64_t(dim)};
      auto allocate = [&] {
        return context.allocate(TensorLayout::contiguous(shape, 3), ScalarType::kBFloat16);
      };
      auto q = allocate(), k = allocate(), v = allocate(), out = allocate();
      context.upload_bytes(q, fixture.q.data(), n * 2);
      context.upload_bytes(k, fixture.k.data(), n * 2);
      context.upload_bytes(v, fixture.v.data(), n * 2);
      auto attention = vulkan::VedaAttentionPlan::create(context, fixture.request);
      CHECK(attention.workspace_bytes() > 0);
      for (int layer : {0, 1}) {
        auto projection =
            vulkan::VedaAttentionPlan::upload_projection(context, fixture.request->bundle, layer);
        auto batch = context.begin_batch();
        attention.record(batch, layer, projection, q, k, v, out);
        batch.submit().wait();
        std::vector<uint16_t> got(n);
        context.download_bytes(out, got.data(), n * 2);
        std::vector<float> values(n);
        for (size_t i = 0; i < n; ++i)
          values[i] = bf16_to_f32(got[i]);
        CHECK_CLOSE(fixture.reference(layer), values, .004, "Vulkan VEDA vs FP64 reference");
      }
    }
}
