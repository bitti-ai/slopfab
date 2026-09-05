#include "slopfab/video/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace slopfab::video {
namespace {

inline uint8_t clamp_u8(float v) {
  return static_cast<uint8_t>(std::lround(std::min(255.0f, std::max(0.0f, v))));
}

// Runs `body(begin, end)` over [0, count) split into contiguous *static*
// ranges, the first of which runs on the calling thread.
//
// Static rather than dynamic on purpose, and the same shape as the VAE's pixel
// de-normalise (src/vae/decode_pipeline.cpp): every output element is computed
// by the same expression from the same inputs, in the same order within a
// range, and nothing is reduced or reordered across ranges — so the result is
// bit-identical to the serial loop and it is obvious by inspection that it is.
//
// The caps mirror that path too. Eight workers, because the work is
// DRAM-bandwidth-bound and the memory system saturates around eight streams on
// this class of machine, while SMT siblings actively contend. And a floor,
// because a create/join pair costs tens of microseconds: below it the spawn tax
// exceeds the saving and the serial path is simply faster.
unsigned choose_workers(size_t count, size_t work, size_t min_work) {
  constexpr unsigned kMaxWorkers = 8;
  unsigned workers = std::thread::hardware_concurrency();
  if (workers == 0) workers = 1;
  workers = std::min(workers, kMaxWorkers);
  workers = static_cast<unsigned>(std::min<size_t>(workers, std::max<size_t>(count, 1)));
  if (work < min_work) workers = 1;
  return workers;
}

// Joins in the destructor as well as on the happy path: if `emplace_back`
// throws part-way through, a plain vector of threads would run ~thread on
// still-joinable threads and call std::terminate.
struct JoiningPool {
  std::vector<std::thread> threads;
  ~JoiningPool() {
    for (std::thread& t : threads) {
      if (t.joinable()) t.join();
    }
  }
};

template <typename Body>
void static_ranges(size_t count, size_t work, size_t min_work, const Body& body) {
  const unsigned workers = choose_workers(count, work, min_work);
  if (workers <= 1) {
    body(static_cast<size_t>(0), count);
    return;
  }
  JoiningPool pool;
  pool.threads.reserve(workers - 1);
  const size_t share = (count + workers - 1) / workers;
  for (unsigned w = 1; w < workers; ++w) {
    const size_t begin = std::min(count, share * w);
    const size_t end = std::min(count, begin + share);
    if (begin == end) break;
    pool.threads.emplace_back(body, begin, end);
  }
  body(static_cast<size_t>(0), std::min(count, share));
}

// Roughly 1.5 lround per pixel plus three plane reads. The floor is in pixels
// rather than bytes because the arithmetic, not the traffic, is what dominates.
constexpr size_t kMinParallelPixels = 1u << 18;  // 256k pixels

// One frame's chroma rows [begin, end), serially, plus the odd tail luma row
// when `end` is the last range. Both the row-split entry point and the .y4m
// writer's frame-split call this; keeping it a plain function is what stops the
// two from nesting thread pools inside each other.
void yuv420_chroma_rows(size_t begin, size_t end, const float* r, const float* g, const float* b,
                        int height, int width, uint8_t* y_plane, int y_stride, uint8_t* u_plane,
                        int u_stride, uint8_t* v_plane, int v_stride) {
  const int chroma_w = width / 2;
  const int chroma_h = height / 2;

  const auto luma_row = [&](size_t y) {
    for (int x = 0; x < width; ++x) {
      const size_t i = y * static_cast<size_t>(width) + x;
      const float luma = 0.2126f * r[i] + 0.7152f * g[i] + 0.0722f * b[i];
      y_plane[y * static_cast<size_t>(y_stride) + x] = clamp_u8(16.0f + 219.0f * luma);
    }
  };

  for (size_t cy = begin; cy < end; ++cy) {
    luma_row(cy * 2);
    luma_row(cy * 2 + 1);
    for (int cx = 0; cx < chroma_w; ++cx) {
      float rs = 0.0f;
      float gs = 0.0f;
      float bs = 0.0f;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const size_t i =
              (cy * 2 + static_cast<size_t>(dy)) * static_cast<size_t>(width) + (cx * 2 + dx);
          rs += r[i];
          gs += g[i];
          bs += b[i];
        }
      }
      rs *= 0.25f;
      gs *= 0.25f;
      bs *= 0.25f;
      const float luma = 0.2126f * rs + 0.7152f * gs + 0.0722f * bs;
      const float u = (bs - luma) / 1.8556f;
      const float v = (rs - luma) / 1.5748f;
      u_plane[cy * static_cast<size_t>(u_stride) + cx] = clamp_u8(128.0f + 224.0f * u);
      v_plane[cy * static_cast<size_t>(v_stride) + cx] = clamp_u8(128.0f + 224.0f * v);
    }
  }
  // An odd height leaves a final luma row that no chroma row owns, and the
  // chroma-row split would skip it — leaving those bytes uninitialised. The
  // serial version this replaced wrote every luma row, so the last range writes
  // it here. Both container callers reject odd dimensions today, which is the
  // only reason this was latent rather than a live bug; it stopped being merely
  // internal the moment the function moved into a public header.
  if (end == static_cast<size_t>(chroma_h) && (height & 1) != 0) {
    luma_row(static_cast<size_t>(height) - 1);
  }
}

}  // namespace

void rgb_frame_to_yuv420(const float* r, const float* g, const float* b, int height, int width,
                         uint8_t* y_plane, int y_stride, uint8_t* u_plane, int u_stride,
                         uint8_t* v_plane, int v_stride) {
  // BT.709 limited range: Y spans 16..235, chroma 16..240 around 128. The 2x2
  // block is averaged before conversion, which is standard 4:2:0 downsampling
  // rather than point-sampling one corner.
  //
  // One frame at a time is all the muxer can offer — ffmpeg wants one AVFrame
  // filled and encoded before the next — so the split here is by *chroma* row:
  // worker `cy` owns luma rows 2cy and 2cy+1 and chroma row cy, which covers
  // both loops with one create/join rather than two. write_y4m has whole frames
  // available and splits those instead; see there.
  static_ranges(static_cast<size_t>(height / 2),
                static_cast<size_t>(height) * static_cast<size_t>(width), kMinParallelPixels,
                [&](size_t begin, size_t end) {
                  yuv420_chroma_rows(begin, end, r, g, b, height, width, y_plane, y_stride, u_plane,
                                     u_stride, v_plane, v_stride);
                });
}

void write_y4m(const std::string& path, const PixelBuffer& planar_rgb, int frames,
               int height, int width, FrameRate fps, FrameConverter* converter) {
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

  const size_t chroma_w = static_cast<size_t>(width) / 2;
  const size_t chroma_h = static_cast<size_t>(height) / 2;

  // Split by *frame*, not by row. Calling the row-split conversion once per
  // frame created and joined a pool per frame: at 1280x768 and 124 frames that
  // is 124 rounds of seven create/join pairs, tens of microseconds each, on a
  // stage measured in a few hundred milliseconds. Frames are independent, so
  // each worker takes a whole frame and its own scratch planes, and the pool is
  // built once per batch of `workers` frames instead.
  //
  // The writes stay on this thread, in frame order, after the batch has joined,
  // so the bytes on disk are exactly what the serial loop wrote. The cost is
  // `workers` copies of one frame's YUV rather than one — about 12 MiB at
  // 1280x768 with eight workers.
  const unsigned workers = converter != nullptr
                               ? 1u
                               : choose_workers(static_cast<size_t>(frames), frame_pixels,
                                                kMinParallelPixels);
  std::vector<std::vector<uint8_t>> luma(workers), cb(workers), cr(workers);
  for (unsigned w = 0; w < workers; ++w) {
    luma[w].resize(frame_pixels);
    cb[w].resize(chroma_w * chroma_h);
    cr[w].resize(chroma_w * chroma_h);
  }

  // The serial whole-frame conversion, so a frame worker never nests a pool
  // inside itself. Identical arithmetic to rgb_frame_to_yuv420, because it is
  // the body rgb_frame_to_yuv420 splits.
  const auto convert = [&](int f, unsigned w) {
    const size_t base = static_cast<size_t>(f) * frame_pixels;
    if (converter != nullptr) {
      try {
        converter->convert(r_plane + base, g_plane + base, b_plane + base, height, width,
                           luma[w].data(), width, cb[w].data(), static_cast<int>(chroma_w),
                           cr[w].data(), static_cast<int>(chroma_w));
      } catch (...) {
        out.close();
        std::remove(path.c_str());
        throw;
      }
    } else {
      yuv420_chroma_rows(0, chroma_h, r_plane + base, g_plane + base, b_plane + base,
                         height, width, luma[w].data(), width, cb[w].data(),
                         static_cast<int>(chroma_w), cr[w].data(),
                         static_cast<int>(chroma_w));
    }
  };

  for (int f0 = 0; f0 < frames; f0 += static_cast<int>(workers)) {
    const int n = std::min(static_cast<int>(workers), frames - f0);
    {
      JoiningPool pool;
      pool.threads.reserve(static_cast<size_t>(n - 1));
      for (int k = 1; k < n; ++k) {
        pool.threads.emplace_back(convert, f0 + k, static_cast<unsigned>(k));
      }
      convert(f0, 0);
    }  // every worker in the batch has joined before a byte of it is written
    for (int k = 0; k < n; ++k) {
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(luma[k].data()),
                static_cast<std::streamsize>(luma[k].size()));
      out.write(reinterpret_cast<const char*>(cb[k].data()),
                static_cast<std::streamsize>(cb[k].size()));
      out.write(reinterpret_cast<const char*>(cr[k].data()),
                static_cast<std::streamsize>(cr[k].size()));
    }
  }

  if (!out) throw std::runtime_error("y4m: write failed for " + path);
}

void write_ppm(const std::string& path, const PixelBuffer& planar_rgb, int frames,
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

}  // namespace slopfab::video
