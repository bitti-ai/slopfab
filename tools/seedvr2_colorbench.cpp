// CPU-only color correction benchmark and exact-output regression check.
#include "../src/seedvr2/color_ops.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace slopfab::seedvr2;

static void original(Frame& out, const Frame& reference) {
  for (int c = 0; c < 3; ++c) {
    double a = 0, b = 0, aa = 0, bb = 0;
    size_t n = out.size() / 3;
    for (size_t i = c; i < out.size(); i += 3) {
      a += out[i];
      b += reference[i];
    }
    a /= n;
    b /= n;
    for (size_t i = c; i < out.size(); i += 3) {
      aa += (out[i] - a) * (out[i] - a);
      bb += (reference[i] - b) * (reference[i] - b);
    }
    double scale = std::sqrt((bb / n + 1e-6) / (aa / n + 1e-6));
    for (size_t i = c; i < out.size(); i += 3)
      out[i] = float((out[i] - a) * scale + b);
  }
  for (auto& v : out) {
    if (!std::isfinite(v))
      throw std::runtime_error("SeedVR2: nonfinite restored pixel");
    v = std::clamp(v, 0.0f, 1.0f);
  }
}

int main() {
  std::vector<Frame> input(5, Frame(size_t(1280) * 720 * 3)), reference = input;
  uint32_t state = 123;
  for (size_t f = 0; f < input.size(); ++f)
    for (size_t i = 0; i < input[f].size(); ++i) {
      state = state * 1664525u + 1013904223u;
      input[f][i] = float(state >> 8) / 16777216.0f * 1.3f - 0.1f;
      state = state * 1664525u + 1013904223u;
      reference[f][i] = float(state >> 8) / 16777216.0f;
    }
  auto expected = input;
  for (size_t f = 0; f < expected.size(); ++f)
    original(expected[f], reference[f]);
  for (unsigned workers : {0u, 1u, 2u, 4u, 5u}) {
    std::vector<double> times;
    for (int repeat = 0; repeat < 9; ++repeat) {
      auto actual = input;
      auto start = std::chrono::steady_clock::now();
      if (!workers) {
        for (size_t f = 0; f < actual.size(); ++f)
          original(actual[f], reference[f]);
      } else
        finish_frames(actual, reference, true, workers);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start).count();
      if (repeat)
        times.push_back(ms);
      for (size_t f = 0; f < actual.size(); ++f)
        if (std::memcmp(actual[f].data(), expected[f].data(), actual[f].size() * sizeof(float))) {
          std::fprintf(stderr, "Output mismatch: workers=%u frame=%zu\n", workers, f);
          return 1;
        }
    }
    std::sort(times.begin(), times.end());
    std::printf("%s workers=%u: median %.3f ms (range %.3f-%.3f), bit-identical\n",
                workers ? "fused" : "original", workers,
                (times[3] + times[4]) * 0.5, times.front(), times.back());
  }
  for (auto& frame : input)
    std::fill(frame.begin(), frame.end(), 0.3f);
  for (auto& frame : reference)
    std::fill(frame.begin(), frame.end(), 0.7f);
  expected = input;
  for (size_t f = 0; f < expected.size(); ++f)
    original(expected[f], reference[f]);
  finish_frames(input, reference, true);
  if (input != expected)
    return 2;
  Frame clamp = {-0.5f, 0.5f, 1.5f};
  finish_frame(clamp, clamp, false);
  if (clamp != Frame({0, 0.5f, 1}))
    return 3;
  input.back()[0] = std::numeric_limits<float>::quiet_NaN();
  try {
    finish_frames(input, reference, false);
    return 4;
  } catch (const std::runtime_error&) {
  }
  std::puts("Flat-color, no-color-match clipping, and worker error propagation passed.");
}
