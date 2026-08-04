#include "vidfab/video/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace vidfab::video {
namespace {

inline uint8_t clamp_u8(float v) {
  return static_cast<uint8_t>(std::lround(std::min(255.0f, std::max(0.0f, v))));
}

}  // namespace

void write_y4m(const std::string& path, const std::vector<float>& planar_rgb, int frames,
               int height, int width, FrameRate fps) {
  if (frames <= 0 || height <= 0 || width <= 0) {
    throw std::runtime_error("y4m: frame count and dimensions must be positive");
  }
  // 4:2:0 subsampling halves both axes, so both must be even.
  if ((width % 2) != 0 || (height % 2) != 0) {
    throw std::runtime_error("y4m: 4:2:0 requires even width and height");
  }
  const size_t frame_pixels = static_cast<size_t>(height) * width;
  const size_t expected = static_cast<size_t>(3) * frames * frame_pixels;
  if (planar_rgb.size() != expected) {
    throw std::runtime_error("y4m: expected " + std::to_string(expected) + " samples, got " +
                             std::to_string(planar_rgb.size()));
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("y4m: cannot open " + path + " for writing");

  out << "YUV4MPEG2 W" << width << " H" << height << " F" << fps.numerator << ":"
      << fps.denominator << " Ip A1:1 C420jpeg\n";

  const size_t plane = static_cast<size_t>(frames) * frame_pixels;
  const float* r_plane = planar_rgb.data();
  const float* g_plane = planar_rgb.data() + plane;
  const float* b_plane = planar_rgb.data() + 2 * plane;

  std::vector<uint8_t> luma(frame_pixels);
  const size_t chroma_w = static_cast<size_t>(width) / 2;
  const size_t chroma_h = static_cast<size_t>(height) / 2;
  std::vector<uint8_t> cb(chroma_w * chroma_h);
  std::vector<uint8_t> cr(chroma_w * chroma_h);

  for (int f = 0; f < frames; ++f) {
    const size_t base = static_cast<size_t>(f) * frame_pixels;

    for (size_t i = 0; i < frame_pixels; ++i) {
      const float r = r_plane[base + i];
      const float g = g_plane[base + i];
      const float b = b_plane[base + i];
      // BT.709 limited range: Y spans 16..235.
      const float y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
      luma[i] = clamp_u8(16.0f + 219.0f * y);
    }

    // Average each 2x2 block before converting, matching standard 4:2:0
    // downsampling rather than point-sampling one corner.
    for (size_t cy = 0; cy < chroma_h; ++cy) {
      for (size_t cx = 0; cx < chroma_w; ++cx) {
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            const size_t i = (cy * 2 + dy) * static_cast<size_t>(width) + (cx * 2 + dx);
            r += r_plane[base + i];
            g += g_plane[base + i];
            b += b_plane[base + i];
          }
        }
        r *= 0.25f;
        g *= 0.25f;
        b *= 0.25f;
        const float y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        const float u = (b - y) / 1.8556f;
        const float v = (r - y) / 1.5748f;
        cb[cy * chroma_w + cx] = clamp_u8(128.0f + 224.0f * u);
        cr[cy * chroma_w + cx] = clamp_u8(128.0f + 224.0f * v);
      }
    }

    out << "FRAME\n";
    out.write(reinterpret_cast<const char*>(luma.data()),
              static_cast<std::streamsize>(luma.size()));
    out.write(reinterpret_cast<const char*>(cb.data()), static_cast<std::streamsize>(cb.size()));
    out.write(reinterpret_cast<const char*>(cr.data()), static_cast<std::streamsize>(cr.size()));
  }

  if (!out) throw std::runtime_error("y4m: write failed for " + path);
}

void write_ppm(const std::string& path, const std::vector<float>& planar_rgb, int frames,
               int height, int width, int frame_index) {
  if (frame_index < 0 || frame_index >= frames) {
    throw std::runtime_error("ppm: frame index out of range");
  }
  const size_t frame_pixels = static_cast<size_t>(height) * width;
  const size_t plane = static_cast<size_t>(frames) * frame_pixels;
  const size_t base = static_cast<size_t>(frame_index) * frame_pixels;

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("ppm: cannot open " + path + " for writing");
  out << "P6\n" << width << " " << height << "\n255\n";

  std::vector<uint8_t> row(static_cast<size_t>(width) * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t i = base + static_cast<size_t>(y) * width + x;
      row[static_cast<size_t>(x) * 3 + 0] = clamp_u8(planar_rgb[i] * 255.0f);
      row[static_cast<size_t>(x) * 3 + 1] = clamp_u8(planar_rgb[plane + i] * 255.0f);
      row[static_cast<size_t>(x) * 3 + 2] = clamp_u8(planar_rgb[2 * plane + i] * 255.0f);
    }
    out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
  }
  if (!out) throw std::runtime_error("ppm: write failed for " + path);
}

}  // namespace vidfab::video
