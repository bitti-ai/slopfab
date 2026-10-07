// Run separately with SLOPFAB_SEEDVR2_MLP_ROWS=0/4096; output is raw float32.
#include "../src/seedvr2/runtime.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace slopfab;
using namespace slopfab::seedvr2;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: seedvr2_mlpbench DIT_CHECKPOINT OUTPUT_F32\n");
    return 2;
  }
  try {
    cuda::set_device(0);
    SafeTensors checkpoint;
    checkpoint.open(argv[1]);
    Runtime runtime;
    runtime.resident_budget = size_t(1) << 30;
    runtime.clear(checkpoint);
    const std::string prefix = "blocks.10.", branch = "all";
    for (const auto& item : checkpoint.tensors())
      if (item.first.rfind(prefix + "mlp." + branch + ".", 0) == 0 ||
          item.first.rfind(prefix + "ada." + branch + ".mlp_", 0) == 0)
        runtime.weight(item.first);
    constexpr int rows = 7200, channels = 2560;
    std::vector<float> input(size_t(rows) * channels), embedding(size_t(channels) * 6);
    uint32_t state = 13;
    for (auto* values : {&input, &embedding})
      for (float& value : *values) {
        state = state * 1664525u + 1013904223u;
        value = (float(state >> 8) / 16777216.0f - 0.5f) * 0.2f;
      }
    auto source = runtime.upload(input, 1, 1, rows, channels);
    auto emb = runtime.upload(embedding, 1, 1, 1, channels * 6);
    auto x = runtime.tensor(1, 1, rows, channels);
    cudaEvent_t start, stop;
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&start));
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> gpu_times, wall_times;
    for (int repeat = 0; repeat < 35; ++repeat) {
      SLOPFAB_CUDA_CHECK(cudaMemcpy(x.data.get(), source.data.get(), x.size() * sizeof(BFloat),
                                    cudaMemcpyDeviceToDevice));
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const auto wall_start = std::chrono::steady_clock::now();
      SLOPFAB_CUDA_CHECK(cudaEventRecord(start));
      runtime.mlp(x, emb, prefix, branch);
      SLOPFAB_CUDA_CHECK(cudaEventRecord(stop));
      SLOPFAB_CUDA_CHECK(cudaEventSynchronize(stop));
      const double wall_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - wall_start).count();
      float gpu_ms = 0;
      SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
      if (repeat >= 5) {
        gpu_times.push_back(gpu_ms);
        wall_times.push_back(wall_ms);
      }
    }
    auto median = [](std::vector<double> times) {
      std::sort(times.begin(), times.end());
      return (times[14] + times[15]) * 0.5;
    };
    const char* chunk = std::getenv("SLOPFAB_SEEDVR2_MLP_ROWS");
    std::printf("chunk=%s GPU_median_ms=%.5f wall_median_ms=%.5f activation_peak_mib=%.2f "
                "activation_reserved_mib=%.2f temporary_peak_mib=%.2f resident_weights_mib=%.2f "
                "activation_allocations=%zu\n", chunk ? chunk : "default",
                median(gpu_times), median(wall_times), runtime.activations->peak_bytes() / 1048576.0,
                runtime.activations->reserved_bytes() / 1048576.0,
                runtime.temporaries->peak_bytes() / 1048576.0,
                runtime.resident_bytes / 1048576.0, runtime.activations->allocations());
    // The pool peaks above exclude this diagnostic float download allocation.
    auto output = runtime.download(x);
    if (std::string(argv[2]) != "-") {
      std::ofstream file(argv[2], std::ios::binary);
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
    }
    SLOPFAB_CUDA_CHECK(cudaEventDestroy(start));
    SLOPFAB_CUDA_CHECK(cudaEventDestroy(stop));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
