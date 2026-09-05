#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.h"
#include "slopfab/image.h"

namespace {

SLOPFAB_TEST(reference_image_loads_binary_ppm) {
  const std::string path = "slopfab_test_reference.ppm";
  {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n# comment\n2 1\n255\n";
    const char pixels[] = {char(1), char(2), char(3), char(250), char(251), char(252)};
    out.write(pixels, sizeof(pixels));
  }
  const slopfab::RGBImage image = slopfab::load_reference_image(path);
  std::remove(path.c_str());
  CHECK(image.width == 2);
  CHECK(image.height == 1);
  CHECK(image.pixels.size() == 6);
  CHECK(image.pixels[0] == 1 && image.pixels[5] == 252);
}

SLOPFAB_TEST(reference_image_rejects_truncated_ppm) {
  const std::string path = "slopfab_test_truncated_reference.ppm";
  {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n2 2\n255\nshort";
  }
  bool rejected = false;
  try {
    (void)slopfab::load_reference_image(path);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  std::remove(path.c_str());
  CHECK(rejected);
}

SLOPFAB_TEST(reference_media_decodes_common_formats_with_ffmpeg) {
  const std::string source = "slopfab_test_media_source.ppm";
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
    const std::string output = std::string("slopfab_test_media.") + extension;
    const std::string command = "ffmpeg -y -loglevel error -i " + source + " " + output + quiet;
    if (std::system(command.c_str()) != 0) continue;
    const slopfab::RGBImage image = slopfab::load_reference_image(output);
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

SLOPFAB_TEST(reference_image_lanczos_golden) {
  slopfab::RGBImage image;
  image.width = 2;
  image.height = 1;
  image.pixels = {0, 10, 20, 255, 110, 20};
  const auto identity = slopfab::resize_reference_lanczos(image, 2, 1);
  CHECK(identity.pixels == image.pixels);

  const auto wide = slopfab::resize_reference_lanczos(image, 4, 1);
  CHECK(wide.width == 4 && wide.height == 1);
  // Scale-adaptive Lanczos-3, half-pixel centers, edge replication and
  // round-half-up byte conversion.
  const uint8_t golden[] = {0, 0, 20, 54, 31, 20, 201, 89, 20, 255, 120, 20};
  CHECK(wide.pixels == std::vector<uint8_t>(std::begin(golden), std::end(golden)));

  slopfab::RGBImage constant{3, 2, std::vector<uint8_t>(18, 73)};
  const auto scaled = slopfab::resize_reference_lanczos(constant, 7, 5);
  for (uint8_t value : scaled.pixels) CHECK(value == 73);
}

// The Lanczos resample as it was before the weight tables: every weight
// recomputed inside the loop over the other axis. Copied verbatim from
// src/core/image.cpp so the tables can be held to exact equality against it.
//
// The trap this guards is the tempting extra step of folding 1/sum into the
// stored weights. That is algebraically the same and numerically is not: it
// rounds each weight once instead of rounding the accumulated sum, and it
// moved thousands of output bytes by one in the survey that found it. Exact
// equality is the only assertion that would catch it — a 1e-3 tolerance would
// not.
slopfab::RGBImage resize_reference_lanczos_unmemoised(const slopfab::RGBImage& image, int width,
                                                     int height) {
  auto kernel = [](double x) {
    x = std::abs(x);
    if (x == 0.0) return 1.0;
    if (x >= 3.0) return 0.0;
    constexpr double pi = 3.14159265358979323846;
    return std::sin(pi * x) * std::sin(pi * x / 3.0) / (pi * pi * x * x / 3.0);
  };
  const double sx = static_cast<double>(image.width) / width;
  const double sy = static_cast<double>(image.height) / height;
  const double fx = std::max(1.0, sx), fy = std::max(1.0, sy);
  std::vector<double> tmp(static_cast<size_t>(image.height) * width * 3);
  for (int y = 0; y < image.height; ++y) for (int x = 0; x < width; ++x) {
    const double center = (x + 0.5) * sx - 0.5;
    const int first = static_cast<int>(std::floor(center - 3.0 * fx + 1.0));
    const int last = static_cast<int>(std::floor(center + 3.0 * fx));
    double sum = 0.0, rgb[3] = {};
    for (int ix = first; ix <= last; ++ix) {
      const double w = kernel((ix - center) / fx);
      const int sample = std::clamp(ix, 0, image.width - 1);
      sum += w;
      for (int c = 0; c < 3; ++c)
        rgb[c] += w * image.pixels[(static_cast<size_t>(y) * image.width + sample) * 3 + c];
    }
    for (int c = 0; c < 3; ++c) tmp[(static_cast<size_t>(y) * width + x) * 3 + c] = rgb[c] / sum;
  }
  slopfab::RGBImage out{width, height, std::vector<uint8_t>(static_cast<size_t>(width) * height * 3)};
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
    const double center = (y + 0.5) * sy - 0.5;
    const int first = static_cast<int>(std::floor(center - 3.0 * fy + 1.0));
    const int last = static_cast<int>(std::floor(center + 3.0 * fy));
    double sum = 0.0, rgb[3] = {};
    for (int iy = first; iy <= last; ++iy) {
      const double w = kernel((iy - center) / fy);
      const int sample = std::clamp(iy, 0, image.height - 1);
      sum += w;
      for (int c = 0; c < 3; ++c)
        rgb[c] += w * tmp[(static_cast<size_t>(sample) * width + x) * 3 + c];
    }
    for (int c = 0; c < 3; ++c) out.pixels[(static_cast<size_t>(y) * width + x) * 3 + c] =
        static_cast<uint8_t>(std::clamp(std::floor(rgb[c] / sum + 0.5), 0.0, 255.0));
  }
  return out;
}

slopfab::RGBImage noise_image(int w, int h, uint32_t seed) {
  slopfab::RGBImage image{w, h, std::vector<uint8_t>(static_cast<size_t>(w) * h * 3)};
  uint32_t state = seed;
  for (uint8_t& p : image.pixels) {
    state = state * 1664525u + 1013904223u;
    p = static_cast<uint8_t>(state >> 24);
  }
  return image;
}

SLOPFAB_TEST(reference_image_lanczos_weight_tables_are_bit_identical) {
  struct Case {
    const char* name;
    int in_w, in_h, out_w, out_h;
  };
  // Downscale (the shipped direction, at the shipped aspect ratio), upscale,
  // mixed, and the two degenerate one-pixel axes. Sizes are kept modest
  // because the reference is the slow implementation.
  const Case cases[] = {
      {"downscale, 16:9 to 5:3", 320, 180, 213, 128},
      {"upscale both axes", 40, 30, 128, 96},
      {"down on x, up on y", 200, 40, 64, 120},
      {"identity", 64, 48, 64, 48},
      {"single row", 96, 1, 32, 1},
      {"single column", 1, 96, 1, 32},
      {"non-integer ratio", 101, 67, 45, 39},
  };
  for (const Case& c : cases) {
    const slopfab::RGBImage src = noise_image(c.in_w, c.in_h, 0x9E3779B9u ^ static_cast<uint32_t>(c.out_w));
    const slopfab::RGBImage expect = resize_reference_lanczos_unmemoised(src, c.out_w, c.out_h);
    const slopfab::RGBImage actual = slopfab::resize_reference_lanczos(src, c.out_w, c.out_h);
    CHECK(actual.width == expect.width && actual.height == expect.height);
    size_t differing = 0;
    for (size_t i = 0; i < expect.pixels.size() && i < actual.pixels.size(); ++i) {
      if (expect.pixels[i] != actual.pixels[i]) ++differing;
    }
    CHECK_MSG(differing == 0, "%s: %zu of %zu bytes differ from the unmemoised resample", c.name,
              differing, expect.pixels.size());
  }

  // A second call at the same extents takes the cached tables. It must return
  // the same bytes — a cache that handed back a table built for other extents
  // would be caught here and nowhere else.
  const slopfab::RGBImage src = noise_image(320, 180, 12345u);
  const slopfab::RGBImage first = slopfab::resize_reference_lanczos(src, 213, 128);
  const slopfab::RGBImage second = slopfab::resize_reference_lanczos(src, 213, 128);
  CHECK(first.pixels == second.pixels);
  // Interleaving a different size between the two must not disturb it either.
  (void)slopfab::resize_reference_lanczos(src, 64, 64);
  const slopfab::RGBImage third = slopfab::resize_reference_lanczos(src, 213, 128);
  CHECK(first.pixels == third.pixels);
}

}  // namespace
