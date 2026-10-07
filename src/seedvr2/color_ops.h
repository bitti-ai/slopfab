#pragma once
#include "slopfab/seedvr2.h"
#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>
#include <thread>

namespace slopfab::seedvr2 {
// Independent RGB accumulators retain the original per-channel addition order
// while each pass walks the interleaved image contiguously.
inline void finish_frame(Frame& out, const Frame& reference, bool color_match) {
  if (color_match) {
    double a[3] = {}, b[3] = {}, aa[3] = {}, bb[3] = {}, scale[3];
    const size_t n = out.size() / 3;
    for (size_t i = 0; i < out.size(); i += 3)
      for (int c = 0; c < 3; ++c) {
        a[c] += out[i + c];
        b[c] += reference[i + c];
      }
    for (int c = 0; c < 3; ++c) {
      a[c] /= n;
      b[c] /= n;
    }
    for (size_t i = 0; i < out.size(); i += 3)
      for (int c = 0; c < 3; ++c) {
        aa[c] += (out[i + c] - a[c]) * (out[i + c] - a[c]);
        bb[c] += (reference[i + c] - b[c]) * (reference[i + c] - b[c]);
      }
    for (int c = 0; c < 3; ++c)
      scale[c] = std::sqrt((bb[c] / n + 1e-6) / (aa[c] / n + 1e-6));
    for (size_t i = 0; i < out.size(); i += 3)
      for (int c = 0; c < 3; ++c)
        out[i + c] = float((out[i + c] - a[c]) * scale[c] + b[c]);
  }
  for (float& value : out) {
    if (!std::isfinite(value))
      throw std::runtime_error("SeedVR2: nonfinite restored pixel");
    value = std::clamp(value, 0.0f, 1.0f);
  }
}

inline void finish_frames(std::vector<Frame>& output, const std::vector<Frame>& reference,
                          bool color_match, unsigned workers = 8) {
  workers = std::min(workers, std::max(1u, std::thread::hardware_concurrency()));
  workers = std::max(1u, std::min(workers, unsigned(output.size())));
  if (output.empty() || output.front().size() < size_t(256) * 256 * 3)
    workers = 1;
  auto run = [&](unsigned worker) {
    for (size_t f = worker; f < output.size(); f += workers)
      finish_frame(output[f], reference[f], color_match);
  };
  std::vector<std::future<void>> tasks;
  for (unsigned worker = 1; worker < workers; ++worker)
    tasks.push_back(std::async(std::launch::async, run, worker));
  run(0);
  for (auto& task : tasks)
    task.get();
}
}
