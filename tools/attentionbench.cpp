// Host-observed, warmed attention latency including recording/submission and
// completion, excluding allocation and upload/download. No checkpoint needed.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "slopfab/attention.h"
#include "slopfab/dtype.h"
#include "slopfab/vulkan/tensor.h"
#if SLOPFAB_WITH_CUDA
#include "slopfab/cuda/attention.cuh"
#include "slopfab/cuda/device.h"
#endif

template<class F> double latency(F run) {
  run();
  std::vector<double> ms;
  for (int i = 0; i < 7; ++i) {
    const auto start = std::chrono::steady_clock::now();
    run();
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count());
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size()/2];
}

int main(int argc, char** argv) try {
  using namespace slopfab;
  const int sequence = argc > 1 ? std::stoi(argv[1]) : 1024;
  const int heads = argc > 2 ? std::stoi(argv[2]) : 4;
  const uint64_t extra_mib = argc > 3 ? std::stoull(argv[3]) : 64;
  const uint32_t kernel = argc > 4 ? uint32_t(std::stoul(argv[4])) : 0;
  const uint32_t device_index = argc > 5 ? uint32_t(std::stoul(argv[5])) : 0;
  const uint32_t dim = argc > 6 ? uint32_t(std::stoul(argv[6])) : 128;
  if (extra_mib > 65536 || kernel > 4) throw std::invalid_argument("Invalid extra workspace MiB or Sage kernel (0..4)");
  if (sequence < 1 || sequence > 65536 || heads < 1 || heads > 128 || (dim != 64 && dim != 128) || argc > 7)
    throw std::invalid_argument("Usage: slopfab_attentionbench [sequence 1..65536] [heads 1..128] [extra_MiB 0..65536] [sage_kernel 0..4] [Vulkan_device_index] [head_dim 64|128]");
  const size_t n = size_t(sequence)*heads*dim;
  std::vector<uint16_t> q(n), k(n), v(n), out(n);
  uint32_t state = 0x12345678;
  auto random = [&] {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    float x = (float(state & 65535)/32768.0f - 1.0f) * 1.5f;
    uint32_t bits; std::memcpy(&bits, &x, 4);
    return uint16_t((bits + 32767u + ((bits>>16)&1))>>16);
  };
  for (size_t i = 0; i < n; ++i) { q[i] = random(); k[i] = random(); v[i] = random(); }
  auto instance = vulkan::Instance::create();
  auto physical = instance.enumerate_devices();
  if (physical.empty()) throw std::runtime_error("No Vulkan device");
  if (device_index >= physical.size()) throw std::out_of_range("Vulkan device index");
  const auto& info = physical[device_index].info();
  vulkan::DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  options.enable_shader_int8 = info.shader_int8;
  options.enable_shader_int64 = info.shader_int64;
  auto device = physical[device_index].create_device(options);
  vulkan::TensorContextOptions context_options;
  context_options.sage_extra_workspace_bytes = extra_mib << 20;
  vulkan::TensorContext context(device, context_options);
  const uint64_t shape[] = {uint64_t(sequence), uint64_t(heads), dim};
  auto layout = TensorLayout::contiguous(shape, 3);
  auto dq = context.allocate(layout, ScalarType::kBFloat16);
  auto dk = context.allocate(layout, ScalarType::kBFloat16);
  auto dv = context.allocate(layout, ScalarType::kBFloat16);
  auto output = context.allocate(layout, ScalarType::kBFloat16);
  context.upload_bytes(dq, q.data(), n*2); context.upload_bytes(dk, k.data(), n*2);
  context.upload_bytes(dv, v.data(), n*2);
  std::printf("%s S%d H%d D%d; median of 7 warm host-observed runs\n", info.name.c_str(), sequence, heads, dim);
  std::vector<uint16_t> flash;
  for (auto mode : {AttentionMode::kExact, AttentionMode::kFlash2, AttentionMode::kSage2}) {
    if (!context.h3_attention_supported(mode)) { std::printf("Vulkan %s unavailable\n", attention_mode_name(mode)); continue; }
    auto plan = vulkan::H3AttentionPlan::create(context,
        {uint32_t(sequence), uint32_t(heads), dim, exact_attention_scale(dim), mode,
         mode == AttentionMode::kSage2 ? kernel : 0});
    auto run = [&] {
      auto batch = context.begin_batch(); plan.record(batch, dq, dk, dv, output); batch.submit().wait();
    };
    const double ms = latency(run);
    context.download_bytes(output, out.data(), n*2);
    for (auto word : out) if (!std::isfinite(bf16_to_f32(word))) throw std::runtime_error("Nonfinite Vulkan output");
    if (mode == AttentionMode::kFlash2) flash = out;
    std::printf("Vulkan %-6s %9.3f ms; workspace %.3f MiB\n", attention_mode_name(mode), ms, double(plan.workspace_bytes())/1048576);
    if (mode == AttentionMode::kSage2) {
      const auto c = plan.sage_configuration();
      std::printf("  Sage kernel %u: Q%u/V%u, %u threads, %u shared bytes, parallel mean %d, prepared V %d, extra %.3f MiB\n",
          c.kernel, c.query_rows, c.value_rows, c.local_size, c.shared_bytes,
          int(c.parallel_mean), int(c.prepared_value), double(c.extra_workspace_bytes)/1048576);
      if (info.timestamp_valid_bits) {
        auto timing = vulkan::TimestampQuery::create(device, 4);
        std::vector<double> phases[3];
        for (int repeat = 0; repeat < 7; ++repeat) {
          auto batch = context.begin_batch();
          plan.record(batch, dq, dk, dv, output, nullptr, 0, 0, 0, &timing);
          batch.submit().wait();
          for (uint32_t phase = 0; phase < 3; ++phase)
            phases[phase].push_back(timing.elapsed_milliseconds(phase, phase+1));
        }
        for (auto& phase : phases) std::sort(phase.begin(), phase.end());
        std::printf("  GPU timestamps: smoothing %.3f, quantization/V %.3f, attention %.3f ms\n",
            phases[0][3], phases[1][3], phases[2][3]);
      } else std::printf("  GPU timestamps unavailable on this queue\n");
    }
  }
#if SLOPFAB_WITH_CUDA
  if (device_index != 0) return 0; // Vulkan and CUDA enumeration orders need not agree.
  cuda::set_device(0);
  std::printf("CUDA device 0: %s (Sage uses the shipped architecture-selected P/V path)\n", cuda::query_device(0).name.c_str());
  cuda::DeviceBuffer<__nv_bfloat16> cq(n), ck(n), cv(n), co(n);
  SLOPFAB_CUDA_CHECK(cudaMemcpy(cq.get(), q.data(), n*2, cudaMemcpyHostToDevice));
  SLOPFAB_CUDA_CHECK(cudaMemcpy(ck.get(), k.data(), n*2, cudaMemcpyHostToDevice));
  SLOPFAB_CUDA_CHECK(cudaMemcpy(cv.get(), v.data(), n*2, cudaMemcpyHostToDevice));
  cuda::Workspace ws;
  cuda::AttentionConfig config; config.seq_len = sequence; config.num_heads = heads;
  config.head_dim = dim; config.scale = exact_attention_scale(dim);
  for (auto backend : {cuda::AttentionBackend::kFused, cuda::AttentionBackend::kSage2}) {
    ws.reserve(cuda::attention_workspace_bytes(config, backend));
    const double ms = latency([&] {
      cuda::attention_forward(nullptr, nullptr, cq.get(), ck.get(), cv.get(), co.get(), config, backend, ws);
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    });
    SLOPFAB_CUDA_CHECK(cudaMemcpy(out.data(), co.get(), n*2, cudaMemcpyDeviceToHost));
    double e2 = 0, r2 = 0;
    for (size_t i = 0; i < flash.size(); ++i) {
      double r = bf16_to_f32(flash[i]), e = bf16_to_f32(out[i])-r;
      if (!std::isfinite(e)) throw std::runtime_error("Nonfinite CUDA output");
      e2 += e*e; r2 += r*r;
    }
    std::printf("CUDA   %-6s %9.3f ms; relative L2 vs Vulkan flash2 %.6f\n",
        backend == cuda::AttentionBackend::kFused ? "flash2" : "sage2", ms, std::sqrt(e2/std::max(r2, 1e-30)));
  }
#endif
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "attentionbench: %s\n", e.what()); return 1;
}
