// Controlled translation probe: output dumps are interleaved RGB float32.
#include "slopfab/upscale.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

int main(int argc, char** argv) {
  if (argc != 10) {
    std::fprintf(
        stderr,
        "usage: realesrgan_temporalbench MODEL cuda|vulkan WIDTH HEIGHT TILE PAD FRAMES static|pan OUTPUT.f32\n");
    return 2;
  }
  try {
    const std::string device = argv[2], mode = argv[8];
    const int w = std::stoi(argv[3]), h = std::stoi(argv[4]), frames = std::stoi(argv[7]);
    if ((device != "cuda" && device != "vulkan") || (mode != "static" && mode != "pan") ||
        frames < 2)
      throw std::invalid_argument("invalid backend, mode or frame count");
    slopfab::UpscaleOptions options;
    options.tile_size = std::stoi(argv[5]);
    options.tile_pad = std::stoi(argv[6]);
    const auto count =
        slopfab::upscale_output_elements(1, h, w, slopfab::UpscaleMethod::kRealEsrgan, options);
    slopfab::RealEsrgan model(argv[1], device == "cuda" ? slopfab::DeviceBackend::kCuda
                                                        : slopfab::DeviceBackend::kVulkan);
    slopfab::PixelBuffer input(count / 16);
    std::vector<float> packed(count);
    std::ofstream output(argv[9], std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    for (int f = 0; f < frames; ++f) {
      for (int c = 0; c < 3; ++c)
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x) {
            const int sx = x + (mode == "pan" ? 2 * f : 0);
            const float value = .45f + .18f * std::sin(float(sx) * .035f + c) +
                                .12f * std::cos(float(y) * .047f + c) +
                                ((sx / 8 + y / 8) % 2 ? .06f : -.06f);
            input[(size_t(c) * h + y) * w + x] = std::round(value * 255) / 255;
          }
      const auto start = std::chrono::steady_clock::now();
      const auto restored = model.upscale(input, 1, h, w, options);
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      for (size_t p = 0; p < count / 3; ++p)
        for (int c = 0; c < 3; ++c)
          packed[p * 3 + c] = restored[size_t(c) * (count / 3) + p];
      output.write(reinterpret_cast<const char*>(packed.data()), packed.size() * sizeof(float));
      std::printf("frame=%d seconds=%.6f\n", f, seconds);
      std::fflush(stdout);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
