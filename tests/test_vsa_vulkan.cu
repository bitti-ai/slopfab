#include "harness.h"
#include "slopfab/cuda/vsa_attention.cuh"
#include "slopfab/vulkan/vsa_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

SLOPFAB_TEST(vulkan_vsa_matches_cuda) {
  using namespace slopfab;
  if (!vulkan::Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("No Vulkan loader");
    return;
  }
  auto instance = vulkan::Instance::create();
  auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().cooperative_matrix_bf16_f32_16x16x16 ||
      !physical.front().info().cooperative_matrix_f16_f32_16x16x16) {
    SKIP_UNSUPPORTED_HARDWARE("VSA requires BF16 and FP16 cooperative matrices");
    return;
  }
  vulkan::DeviceOptions options;
  options.enable_timeline_semaphore = options.enable_shader_int64 = true;
  options.enable_shader_float16 = options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  auto device = physical.front().create_device(options);
  vulkan::TensorContextOptions context_options;
  context_options.max_batch_operators = 4;
  vulkan::TensorContext context(device, context_options);
  if (!context.h3_attention_supported(AttentionMode::kFlash2)) {
    SKIP_UNSUPPORTED_HARDWARE("Vulkan VSA capabilities unavailable");
    return;
  }
  cuda::Stream stream;
  for (int dim : {64, 128})
    for (int scenario : {0, 1, 2}) {
      dit::SequenceLayout layout;
      layout.num_text = scenario == 2 ? 0 : 65;
      layout.num_audio_rows = scenario == 2 ? 0 : 3;
      layout.num_latent_frames = scenario == 1 ? 13 : 5;
      layout.latent_height = scenario == 1 ? 18 : 10;
      layout.latent_width = scenario == 1 ? 22 : 14;
      layout.num_video_rows =
          layout.num_latent_frames * layout.latent_height * layout.latent_width / 4;
      auto tiles = dit::build_vsa_tiles(layout);
      const int seq = layout.total_rows(), heads = 2, n = static_cast<int>(tiles.sizes.size());
      const size_t count = size_t(seq) * heads * dim;
      std::vector<__nv_bfloat16> q(count), k(count), v(count), gate(count);
      auto fill = [&](auto& values, unsigned seed) {
        auto data = test::make_data(count, seed, 0.8f);
        for (size_t i = 0; i < count; ++i)
          values[i] = __float2bfloat16(data[i]);
      };
      fill(q, 71);
      fill(k, 83);
      fill(v, 97);
      fill(gate, 113);
      // Equal routing scores must select lower tile indices deterministically.
      if (scenario == 2)
        std::fill(q.begin(), q.end(), __float2bfloat16(0.0f));
      const uint64_t shape[] = {uint64_t(seq), uint64_t(heads), uint64_t(dim)};
      const uint64_t flat[] = {uint64_t(seq), uint64_t(heads * dim)};
      auto allocate = [&] {
        return context.allocate(TensorLayout::contiguous(shape, 3), ScalarType::kBFloat16);
      };
      auto vq = allocate(), vk = allocate(), vv = allocate(), vo = allocate();
      auto vg = context.allocate(TensorLayout::contiguous(flat, 2), ScalarType::kBFloat16);
      context.upload_bytes(vq, q.data(), count * 2);
      context.upload_bytes(vk, k.data(), count * 2);
      context.upload_bytes(vv, v.data(), count * 2);
      context.upload_bytes(vg, gate.data(), count * 2);
      auto plan = vulkan::VsaAttentionPlan::create(context, tiles, heads, dim);
      const uint64_t workspace =
          (tiles.rows.size() + tiles.sizes.size() + tiles.row_tiles.size()) * 4 +
          uint64_t(n) * heads * dim * 14 + uint64_t(heads) * n * ((n + 31) / 32) * 4;
      CHECK(plan.workspace_bytes() == workspace);
      cuda::DeviceBuffer<int32_t> rows(tiles.rows.size()), sizes(n), row_tiles(seq);
      rows.copy_from_host(tiles.rows.data(), rows.size(), stream.get());
      sizes.copy_from_host(tiles.sizes.data(), sizes.size(), stream.get());
      row_tiles.copy_from_host(tiles.row_tiles.data(), row_tiles.size(), stream.get());
      cuda::DeviceBuffer<__nv_bfloat16> cq(count), ck(count), cv(count), cg(count), co(count),
          compressed(size_t(n) * heads * dim);
      cq.copy_from_host(q.data(), count, stream.get());
      ck.copy_from_host(k.data(), count, stream.get());
      cv.copy_from_host(v.data(), count, stream.get());
      cg.copy_from_host(gate.data(), count, stream.get());
      cuda::VsaConfig config{n,          tiles.prefix_tiles, heads,          dim,
                             rows.get(), sizes.get(),        row_tiles.get()};
      cuda::Workspace ws;
      ws.reserve(cuda::vsa_attention_workspace_bytes(n, heads, dim));
      cuda::vsa_attention_forward(stream.get(), cq.get(), ck.get(), cv.get(), co.get(),
                                  compressed.get(), config, ws);
      std::vector<__nv_bfloat16> expected(count), actual(count);
      co.copy_to_host(expected.data(), count, stream.get());
      stream.synchronize();
      auto batch = context.begin_batch();
      plan.record(batch, vq, vk, vv, vo);
      batch.submit().wait();
      context.download_bytes(vo, actual.data(), count * 2);
      auto compare = [&](const char* stage) {
        double maximum = 0, square = 0;
        bool finite = true;
        for (size_t i = 0; i < count; ++i) {
          const double delta = double(__bfloat162float(expected[i])) - __bfloat162float(actual[i]);
          finite = finite && std::isfinite(delta);
          maximum = std::max(maximum, std::abs(delta));
          square += delta * delta;
        }
        std::printf("  D=%d scenario=%d %s max %.7g RMS %.7g\n", dim, scenario, stage, maximum,
                    std::sqrt(square / count));
        CHECK(finite);
        CHECK_MSG(maximum <= 0.008, "Vulkan/CUDA VSA %s maximum %g", stage, maximum);
        CHECK(std::sqrt(square / count) < 0.001);
      };
      compare("sparse");
      cuda::vsa_add_compression(stream.get(), co.get(), cg.get(), compressed.get(), 0, seq, config);
      co.copy_to_host(expected.data(), count, stream.get());
      stream.synchronize();
      // Repeat the plan in a fresh batch, then add the learned branch. This also
      // checks write/read barriers when shared plan scratch is reused.
      auto gated = context.begin_batch();
      plan.record(gated, vq, vk, vv, vo);
      plan.add_compression(gated, vg, vo);
      gated.submit().wait();
      context.download_bytes(vo, actual.data(), count * 2);
      compare("gated");
      auto invalid = tiles;
      invalid.rows[0] = -1;
      bool rejected = false;
      try {
        auto bad = vulkan::VsaAttentionPlan::create(context, invalid, heads, dim);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      CHECK(rejected);
      auto invalid_batch = context.begin_batch();
      rejected = false;
      try {
        plan.record(invalid_batch, vq, vk, vv, vq);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      CHECK(rejected);
      rejected = false;
      try {
        plan.add_compression(invalid_batch, vg, vo);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      CHECK(rejected);
    }
}
