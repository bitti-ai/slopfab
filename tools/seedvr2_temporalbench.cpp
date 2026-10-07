// Controlled temporal probe. Raw dumps are interleaved RGB, frames in sequence.
#include "slopfab/seedvr2.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

int main(int argc, char** argv) {
  if (argc != 10) {
    std::fprintf(
        stderr,
        "usage: seedvr2_temporalbench DIT VAE WIDTH HEIGHT SEGMENT TILE static|pan FRAMES OUTPUT.f32\n");
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
    const std::string mode = argv[7];
    const int count = std::stoi(argv[8]);
    if ((mode != "static" && mode != "pan") || count < 2)
      throw std::invalid_argument("mode must be static or pan; at least two frames required");
    slopfab::seedvr2::validate(o);
    std::ofstream file(argv[9], std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    slopfab::seedvr2::Restorer model(o);
    int read = 0, written = 0;
    slopfab::seedvr2::Frame previous;
    const auto begin = std::chrono::steady_clock::now();
    slopfab::seedvr2::stream(
        o,
        [&](slopfab::seedvr2::Frame& frame) {
          if (read == count)
            return false;
          frame.resize(size_t(o.width) * o.height * 3);
          for (int y = 0; y < o.height; ++y)
            for (int x = 0; x < o.width; ++x)
              for (int c = 0; c < 3; ++c) {
                const int sx = x + (mode == "pan" ? read * 2 : 0);
                const float smooth = 0.45f + 0.18f * std::sin(float(sx) * .035f + c) +
                                     0.12f * std::cos(float(y) * .047f + c);
                const float detail = ((sx / 8 + y / 8) % 2 ? .06f : -.06f);
                frame[(size_t(y) * o.width + x) * 3 + c] = smooth + detail;
              }
          ++read;
          return true;
        },
        [&](const slopfab::seedvr2::Frame& frame) {
          if (!previous.empty()) {
            double squared = 0;
            for (size_t i = 0; i < frame.size(); ++i) {
              const double difference = double(frame[i]) - previous[i];
              squared += difference * difference;
            }
            std::printf("frame=%d adjacent_rmse=%.9f\n", written,
                        std::sqrt(squared / frame.size()));
          }
          previous = frame;
          file.write(reinterpret_cast<const char*>(frame.data()), frame.size() * sizeof(float));
          ++written;
        },
        [&](const std::vector<slopfab::seedvr2::Frame>& frames, uint64_t first) {
          return model.restore(frames, first);
        });
    std::printf("frames=%d seconds=%.6f\n", written,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
