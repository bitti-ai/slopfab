// Real checkpoint benchmark. Timings include tile preparation, transfers and assembly.
#include "slopfab/upscale.h"
#if SLOPFAB_WITH_CUDA
#include "slopfab/cuda/device.h"
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

int main(int argc, char** argv) {
  if (argc != 9) {
    std::fprintf(
        stderr,
        "usage: realesrganbench MODEL cuda|vulkan WIDTH HEIGHT FRAMES TILE REPEATS OUTPUT_PREFIX\n");
    return 2;
  }
  try {
    const std::string backend_name = argv[2];
    if (backend_name != "cuda" && backend_name != "vulkan")
      throw std::invalid_argument("backend must be cuda or vulkan");
    const auto backend =
        backend_name == "cuda" ? slopfab::DeviceBackend::kCuda : slopfab::DeviceBackend::kVulkan;
    const int w = std::stoi(argv[3]), h = std::stoi(argv[4]), frames = std::stoi(argv[5]);
    slopfab::UpscaleOptions options;
    options.tile_size = std::stoi(argv[6]);
    const int repeats = std::stoi(argv[7]);
    slopfab::validate_upscale_options(options);
    const auto output_count = slopfab::upscale_output_elements(frames, h, w);
    if (repeats < 1)
      throw std::invalid_argument("repeats must be positive");
    std::atomic<bool> stop{false};
    std::atomic<size_t> peak{0};
    size_t baseline = 0;
    std::thread sampler;

    struct Join {
      std::atomic<bool>& stop;
      std::thread& thread;

      ~Join() {
        stop = true;
        if (thread.joinable())
          thread.join();
      }
    } join{stop, sampler};
#if SLOPFAB_WITH_CUDA
    // CUDA global memory use, sampled every 10 ms, including other processes.
    // Leave Vulkan runs free of CUDA context allocations; report memory as unavailable.
    if (backend == slopfab::DeviceBackend::kCuda) {
      slopfab::cuda::set_device(0);
      size_t free = 0, total = 0;
      SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free, &total));
      baseline = total - free;
      peak = baseline;
      sampler = std::thread([&] {
        cudaSetDevice(0);
        while (!stop.load()) {
          size_t f = 0, t = 0;
          if (cudaMemGetInfo(&f, &t) == cudaSuccess)
            peak.store(std::max(peak.load(), t - f));
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      });
    }
#endif
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    slopfab::RealEsrgan model(argv[1], backend);
    std::printf("load_seconds=%.6f\n", std::chrono::duration<double>(Clock::now() - start).count());
    slopfab::PixelBuffer input(output_count / 16);
    for (int c = 0; c < 3; ++c)
      for (int t = 0; t < frames; ++t)
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x)
            input[((size_t(c) * frames + t) * h + y) * w + x] =
                float((x * (c + 1) + y * (3 - c) + t * 17) % 256) / 255;
    for (int run = 0; run < repeats; ++run) {
      const auto begin = Clock::now();
      auto output = model.upscale(input, frames, h, w, options);
      const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
      size_t resident = 0;
#if SLOPFAB_WITH_CUDA
      if (backend == slopfab::DeviceBackend::kCuda) {
        size_t f = 0, t = 0;
        SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&f, &t));
        resident = t - f;
        peak.store(std::max(peak.load(), resident));
      }
#endif
      std::printf("result run=%d seconds=%.6f baseline_mib=%.1f peak_mib=%.1f resident_mib=%.1f\n",
                  run, seconds, baseline / 1048576.0, peak.load() / 1048576.0,
                  resident / 1048576.0);
      std::fflush(stdout);
      if (std::string(argv[8]) != "-") {
        std::ofstream file(std::string(argv[8]) + "-" + std::to_string(run) + ".f32",
                           std::ios::binary);
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
