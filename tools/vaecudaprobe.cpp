#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "slopfab/safetensors.h"
#include "slopfab/vae/keyframe_encoder.h"

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::fprintf(stderr, "usage: slopfab_vaecudaprobe <checkpoint> [--smoke]\n");
    return 2;
  }
  try {
    slopfab::SafeTensors checkpoint;
    checkpoint.open(argv[1]);
    slopfab::vae::KeyframeEncoder encoder(checkpoint);
    std::puts("H3 image encoder weights uploaded to CUDA");
    if (argc == 3) {
      if (std::string(argv[2]) != "--smoke") return 2;
      slopfab::RGBImage image;
      image.width = image.height = 32;
      image.pixels.assign(32 * 32 * 3, 127);
      std::vector<float> mean(24, 0.0f), stddev(24, 1.0f);
      const auto rows = encoder.encode_reference_image(image, mean, stddev);
      if (rows.size() != 96) throw std::runtime_error("unexpected smoke output shape");
      std::printf("32x32 forward complete: %zu condition values\n", rows.size());
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
