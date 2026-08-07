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

// Windows builds use the system WIC codecs; every build supports binary PPM.
RGBImage load_reference_image(const std::string& path);

}  // namespace vidfab
