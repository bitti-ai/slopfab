#include <cstdio>
#include <cstdlib>
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

VIDFAB_TEST(reference_media_decodes_common_formats_with_ffmpeg) {
  const std::string source = "vidfab_test_media_source.ppm";
  {
    std::ofstream out(source, std::ios::binary);
    out << "P6\n3 2\n255\n";
    const char pixels[] = {char(255), 0, 0, 0, char(255), 0, 0, 0, char(255),
                           char(255), char(255), 0, 0, char(255), char(255),
                           char(255), 0, char(255)};
    out.write(pixels, sizeof(pixels));
  }
#ifdef _WIN32
  constexpr const char* quiet = " >NUL 2>&1";
#else
  constexpr const char* quiet = " >/dev/null 2>&1";
#endif
  int exercised = 0;
  for (const char* extension : {"png", "jpg", "bmp"}) {
    const std::string output = std::string("vidfab_test_media.") + extension;
    const std::string command = "ffmpeg -y -loglevel error -i " + source + " " + output + quiet;
    if (std::system(command.c_str()) != 0) continue;
    const vidfab::RGBImage image = vidfab::load_reference_image(output);
    CHECK(image.width == 3);
    CHECK(image.height == 2);
    CHECK(image.pixels.size() == 18);
    std::remove(output.c_str());
    ++exercised;
  }
  std::remove(source.c_str());
  if (exercised == 0) std::printf("  ffmpeg executable unavailable; skipping format fixtures\n");
  else CHECK(exercised == 3);
}

VIDFAB_TEST(reference_image_lanczos_golden) {
  vidfab::RGBImage image;
  image.width = 2;
  image.height = 1;
  image.pixels = {0, 10, 20, 255, 110, 20};
  const auto identity = vidfab::resize_reference_lanczos(image, 2, 1);
  CHECK(identity.pixels == image.pixels);

  const auto wide = vidfab::resize_reference_lanczos(image, 4, 1);
  CHECK(wide.width == 4 && wide.height == 1);
  // Scale-adaptive Lanczos-3, half-pixel centers, edge replication and
  // round-half-up byte conversion.
  const uint8_t golden[] = {0, 0, 20, 54, 31, 20, 201, 89, 20, 255, 120, 20};
  CHECK(wide.pixels == std::vector<uint8_t>(std::begin(golden), std::end(golden)));

  vidfab::RGBImage constant{3, 2, std::vector<uint8_t>(18, 73)};
  const auto scaled = vidfab::resize_reference_lanczos(constant, 7, 5);
  for (uint8_t value : scaled.pixels) CHECK(value == 73);
}

}  // namespace
