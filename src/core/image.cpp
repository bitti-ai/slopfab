#include "slopfab/image.h"

#if SLOPFAB_WITH_FFMPEG
#include "slopfab/video/media.h"
#endif

#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <cmath>
#include <algorithm>

namespace slopfab {
namespace {

std::runtime_error image_error(const std::string& path, const std::string& reason) {
  return std::runtime_error("reference image '" + path + "': " + reason);
}

std::string ppm_token(std::istream& in) {
  std::string token;
  for (;;) {
    in >> std::ws;
    if (in.peek() != '#')
      break;
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  in >> token;
  return token;
}

RGBImage load_ppm(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw image_error(path, "cannot open file");
  if (ppm_token(in) != "P6")
    throw image_error(path, "not a binary PPM (P6)");
  RGBImage image;
  try {
    image.width = std::stoi(ppm_token(in));
    image.height = std::stoi(ppm_token(in));
    if (std::stoi(ppm_token(in)) != 255)
      throw image_error(path, "PPM max value must be 255");
  } catch (const std::invalid_argument&) {
    throw image_error(path, "invalid PPM header");
  } catch (const std::out_of_range&) {
    throw image_error(path, "PPM dimensions are out of range");
  }
  if (image.width <= 0 || image.height <= 0 ||
      static_cast<uint64_t>(image.width) * image.height > std::numeric_limits<size_t>::max() / 3) {
    throw image_error(path, "invalid image dimensions");
  }
  const int separator = in.get();
  if (separator == EOF || !std::isspace(static_cast<unsigned char>(separator))) {
    throw image_error(path, "missing whitespace after PPM header");
  }
  image.pixels.resize(static_cast<size_t>(image.width) * image.height * 3);
  in.read(reinterpret_cast<char*>(image.pixels.data()),
          static_cast<std::streamsize>(image.pixels.size()));
  if (in.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
    throw image_error(path, "truncated pixel data");
  }
  return image;
}

} // namespace

RGBImage load_reference_image(const std::string& path) {
  std::ifstream probe(path, std::ios::binary);
  char magic[2] = {};
  probe.read(magic, 2);
  if (!probe)
    throw image_error(path, "cannot open or read file");
  if (magic[0] == 'P' && magic[1] == '6')
    return load_ppm(path);
#if SLOPFAB_WITH_FFMPEG
  // A video file is a legal reference: its first frame is the image.
  const video::DecodedVideoFrame frame = video::decode_first_video_frame(path);
  return {frame.width, frame.height, frame.rgb24};
#else
  // No FFmpeg in this build, so no demuxer either — stills only, through the
  // platform decoder.
  return load_platform_image(path);
#endif
}

namespace {

// One axis of the separable Lanczos resample, resolved once.
//
// Every weight depends only on the *output* index along the axis being
// resampled — the two sin() calls and the division in `kernel` do not look at
// the other coordinate at all — yet the whole table used to be rebuilt inside
// the loop over that other coordinate: image.height times for the horizontal
// pass and `width` times for the vertical one. At 1920x1080 -> 1280x768 that is
// about 25 million redundant kernel evaluations.
//
// BIT-IDENTITY: the *raw* kernel values are stored and `sum` is kept beside
// them, so the pixel loop still divides at the end exactly as it did. Folding
// the normalisation into the stored weights instead — dividing once per output
// index rather than once per pixel — is the obvious next step and is wrong: it
// changes the rounding of the result, and on a 1920x1080 -> 1280x768 test it
// moved 11,400 bytes of 1.49 M by one.
struct AxisWeights {
  std::vector<size_t> offset; // into `weight` / `sample`, per output index
  std::vector<int> count;
  std::vector<double> weight;
  std::vector<int> sample; // input index, already edge-clamped
  std::vector<double> sum; // per output index, summed in ascending tap order
};

double lanczos3(double x) {
  x = std::abs(x);
  if (x == 0.0)
    return 1.0;
  if (x >= 3.0)
    return 0.0;
  constexpr double pi = 3.14159265358979323846;
  return std::sin(pi * x) * std::sin(pi * x / 3.0) / (pi * pi * x * x / 3.0);
}

AxisWeights build_axis_weights(int in_extent, int out_extent) {
  const double scale = static_cast<double>(in_extent) / out_extent;
  const double support = std::max(1.0, scale);
  AxisWeights a;
  a.offset.resize(static_cast<size_t>(out_extent));
  a.count.resize(static_cast<size_t>(out_extent));
  a.sum.resize(static_cast<size_t>(out_extent));
  for (int o = 0; o < out_extent; ++o) {
    const double center = (o + 0.5) * scale - 0.5;
    const int first = static_cast<int>(std::floor(center - 3.0 * support + 1.0));
    const int last = static_cast<int>(std::floor(center + 3.0 * support));
    a.offset[static_cast<size_t>(o)] = a.weight.size();
    a.count[static_cast<size_t>(o)] = last - first + 1;
    double sum = 0.0;
    for (int i = first; i <= last; ++i) {
      const double w = lanczos3((i - center) / support);
      sum += w;
      a.weight.push_back(w);
      a.sample.push_back(std::clamp(i, 0, in_extent - 1));
    }
    a.sum[static_cast<size_t>(o)] = sum;
  }
  return a;
}

// Keyed on the pair of extents, which is everything the table depends on.
// thread_local rather than a shared static with a lock: the tables are a few
// tens of kilobytes and this is not a path worth serialising over.
const AxisWeights& axis_weights(int in_extent, int out_extent) {
  static thread_local std::map<std::pair<int, int>, AxisWeights> cache;
  const std::pair<int, int> key{in_extent, out_extent};
  auto it = cache.find(key);
  if (it == cache.end())
    it = cache.emplace(key, build_axis_weights(in_extent, out_extent)).first;
  return it->second;
}

} // namespace

RGBImage resize_reference_lanczos(const RGBImage& image, int width, int height) {
  if (image.width <= 0 || image.height <= 0 || width <= 0 || height <= 0 ||
      image.pixels.size() != static_cast<size_t>(image.width) * image.height * 3)
    throw std::runtime_error("reference image: invalid Lanczos resize");

  const AxisWeights& hw = axis_weights(image.width, width);
  std::vector<double> tmp(static_cast<size_t>(image.height) * width * 3);
  for (int y = 0; y < image.height; ++y)
    for (int x = 0; x < width; ++x) {
      const size_t off = hw.offset[static_cast<size_t>(x)];
      const int taps = hw.count[static_cast<size_t>(x)];
      double rgb[3] = {};
      for (int k = 0; k < taps; ++k) {
        const double w = hw.weight[off + static_cast<size_t>(k)];
        const int sample = hw.sample[off + static_cast<size_t>(k)];
        for (int c = 0; c < 3; ++c)
          rgb[c] += w * image.pixels[(static_cast<size_t>(y) * image.width + sample) * 3 + c];
      }
      for (int c = 0; c < 3; ++c)
        tmp[(static_cast<size_t>(y) * width + x) * 3 + c] = rgb[c] / hw.sum[static_cast<size_t>(x)];
    }

  const AxisWeights& vw = axis_weights(image.height, height);
  RGBImage out{width, height, std::vector<uint8_t>(static_cast<size_t>(width) * height * 3)};
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      const size_t off = vw.offset[static_cast<size_t>(y)];
      const int taps = vw.count[static_cast<size_t>(y)];
      double rgb[3] = {};
      for (int k = 0; k < taps; ++k) {
        const double w = vw.weight[off + static_cast<size_t>(k)];
        const int sample = vw.sample[off + static_cast<size_t>(k)];
        for (int c = 0; c < 3; ++c)
          rgb[c] += w * tmp[(static_cast<size_t>(sample) * width + x) * 3 + c];
      }
      for (int c = 0; c < 3; ++c)
        out.pixels[(static_cast<size_t>(y) * width + x) * 3 + c] = static_cast<uint8_t>(
            std::clamp(std::floor(rgb[c] / vw.sum[static_cast<size_t>(y)] + 0.5), 0.0, 255.0));
    }
  return out;
}

} // namespace slopfab
