#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "harness.h"
#include "vidfab/image.h"

namespace {

VIDFAB_TEST(reference_image_loads_binary_ppm) {
  const std::string path = "vidfab_test_reference.ppm";
  {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n# comment\n2 1\n255\n";
    const char pixels[] = {char(1), char(2), char(3), char(250), char(251), char(252)};
    out.write(pixels, sizeof(pixels));
  }
  const vidfab::RGBImage image = vidfab::load_reference_image(path);
  std::remove(path.c_str());
  CHECK(image.width == 2);
  CHECK(image.height == 1);
  CHECK(image.pixels.size() == 6);
  CHECK(image.pixels[0] == 1 && image.pixels[5] == 252);
}

VIDFAB_TEST(reference_image_rejects_truncated_ppm) {
  const std::string path = "vidfab_test_truncated_reference.ppm";
  {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n2 2\n255\nshort";
  }
  bool rejected = false;
  try {
    (void)vidfab::load_reference_image(path);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  std::remove(path.c_str());
  CHECK(rejected);
}

}  // namespace
