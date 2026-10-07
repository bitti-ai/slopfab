// Reproducible real-checkpoint benchmark; output files contain raw RGB float32.
#include "slopfab/seedvr2.h"
#include "slopfab/cuda/device.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

int main(int argc, char** argv) {
  if (argc != 9) {
    std::fprintf(stderr,
                 "usage: seedvr2bench DIT VAE WIDTH HEIGHT FRAMES TILE REPEATS OUTPUT_PREFIX\n");
    return 2;
  }
  try {
    slopfab::seedvr2::Options o;
    o.transformer = argv[1];
    o.vae = argv[2];
    o.width = std::stoi(argv[3]);
    o.height = std::stoi(argv[4]);
    o.segment_frames = std::stoi(argv[5]);
    o.vae_tile = std::stoi(argv[6]);
    const int repeats = std::stoi(argv[7]);
    slopfab::seedvr2::validate(o);
    slopfab::cuda::set_device(0);
    size_t free = 0, total = 0;
    SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free, &total));
    const size_t baseline = total - free;
    std::atomic<bool> stop{false};
    std::atomic<size_t> peak{baseline};
    std::thread sampler([&] {
      cudaSetDevice(0);
      while (!stop.load()) {
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess)
          peak.store(std::max(peak.load(), t - f));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    });

    struct Join {
      std::atomic<bool>& stop;
      std::thread& thread;

      ~Join() {
        stop = true;
        thread.join();
      }
    } join{stop, sampler};

    using Clock = std::chrono::steady_clock;
    slopfab::seedvr2::Restorer model(o);
    std::vector<slopfab::seedvr2::Frame> frames(
        o.segment_frames, slopfab::seedvr2::Frame(size_t(o.width) * o.height * 3));
    for (int t = 0; t < o.segment_frames; ++t)
      for (int y = 0; y < o.height; ++y)
        for (int x = 0; x < o.width; ++x)
          for (int c = 0; c < 3; ++c)
            frames[t][(size_t(y) * o.width + x) * 3 + c] =
                float((x * (c + 1) + y * (3 - c) + t * 17) % 256) / 255;
    for (int run = 0; run < repeats; ++run) {
      auto start = Clock::now(), phase = start;
      std::string stage;
      model.progress = [&](const std::string& message) {
        const std::string next = message.substr(0, message.find(" tile"));
        const std::string label = message.rfind("DiT", 0) == 0 ? "DiT" : next;
        if (label != stage) {
          auto now = Clock::now();
          if (!stage.empty())
            std::printf("stage %d %s %.6f\n", run, stage.c_str(),
                        std::chrono::duration<double>(now - phase).count());
          stage = label;
          phase = now;
        }
      };
      auto output = model.restore(frames, 0);
      const auto end = Clock::now();
      std::printf("stage %d %s+post %.6f\n", run, stage.c_str(),
                  std::chrono::duration<double>(end - phase).count());
      std::printf("result run=%d seconds=%.6f baseline_mib=%.1f peak_mib=%.1f\n", run,
                  std::chrono::duration<double>(end - start).count(), baseline / 1048576.0,
                  peak.load() / 1048576.0);
      std::fflush(stdout);
      if (std::string(argv[8]) != "-") {
        std::ofstream file(std::string(argv[8]) + "-" + std::to_string(run) + ".f32",
                           std::ios::binary);
        file.exceptions(std::ios::failbit | std::ios::badbit);
        for (const auto& frame : output)
          file.write(reinterpret_cast<const char*>(frame.data()), frame.size() * sizeof(float));
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
