#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vidfab {

struct RGBImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

// Common media formats are demuxed and decoded through runtime-loaded FFmpeg;
// binary PPM remains available without FFmpeg. The first video frame is used.
RGBImage load_reference_image(const std::string& path);

// Scale-adaptive separable Lanczos-3 with half-pixel pixel-center mapping.
RGBImage resize_reference_lanczos(const RGBImage& image, int width, int height);

}  // namespace vidfab
