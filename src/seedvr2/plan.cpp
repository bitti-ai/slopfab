#include "slopfab/seedvr2.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace slopfab::seedvr2 {

void validate(const Options& o) {
  if (o.width < 16 || o.height < 16 || o.width > 8192 || o.height > 8192)
    throw std::invalid_argument("SeedVR2: width and height must be between 16 and 8192");
  if (o.segment_frames < 1 || o.segment_frames > 129 || (o.segment_frames - 1) % 4)
    throw std::invalid_argument("SeedVR2: segment frames must be 1 or 4n+1, at most 129");
  if (o.vae_tile != 0 && (o.vae_tile < 128 || o.vae_tile > 2048 || o.vae_tile % 16))
    throw std::invalid_argument("SeedVR2: VAE tile must be 0 or a multiple of 16 in [128,2048]");
  if (o.device < 0)
    throw std::invalid_argument("SeedVR2: device must be nonnegative");
}

std::vector<Window> attention_windows(int t, int h, int w, bool shifted) {
  if (t < 1 || h < 1 || w < 1 || t > 1024 || h > 512 || w > 512)
    throw std::invalid_argument("SeedVR2: invalid attention geometry");
  const double scale = std::sqrt(3600.0 / (double(h) * w));
  // nearbyint matches Python's ties-to-even round at the reference window boundary.
  const int wh = std::max(1, int(std::ceil(std::nearbyint(h * scale) / 3)));
  const int ww = std::max(1, int(std::ceil(std::nearbyint(w * scale) / 3)));
  const int wt = (std::min(t, 30) + 3) / 4;
  auto ranges = [shifted](int n, int size) {
    std::vector<std::pair<int, int>> result;
    const double shift = shifted && size < n ? 0.5 : 0.0;
    const int count = shift ? int(std::ceil((n - shift) / size)) + 1 : (n + size - 1) / size;
    for (int i = 0; i < count; ++i) {
      int lo = std::max(0, int((i - shift) * size));
      int hi = std::min(n, int((i - shift + 1) * size));
      if (hi > lo)
        result.emplace_back(lo, hi);
    }
    return result;
  };
  std::vector<Window> result;
  for (auto x : ranges(w, ww))
    for (auto y : ranges(h, wh))
      for (auto z : ranges(t, wt))
        result.push_back({z.first, z.second, y.first, y.second, x.first, x.second});
  return result;
}

uint64_t stream(const Options& o, const ReadFrame& read, const WriteFrame& write,
                const RestoreSegment& restore) {
  validate(o);
  const size_t pixels = size_t(o.width) * o.height * 3;
  std::vector<Frame> input;
  Frame pending;
  uint64_t first = 0, written = 0;
  bool eof = false;
  for (;;) {
    while (input.size() < size_t(o.segment_frames) && !eof) {
      Frame frame;
      if (!read(frame)) {
        eof = true;
        break;
      }
      if (frame.size() != pixels)
        throw std::runtime_error("SeedVR2: invalid input frame size");
      input.push_back(std::move(frame));
    }
    // A retained overlap alone is already restored; do not run it again at EOF.
    if (input.empty() || (input.size() == 1 && !pending.empty() && eof)) {
      if (!pending.empty()) {
        write(pending);
        ++written;
      }
      break;
    }
    const size_t real = input.size();
    const size_t padded = real == 1 ? 1 : ((real - 1 + 3) / 4) * 4 + 1;
    auto batch = input;
    while (batch.size() < padded)
      batch.push_back(batch.back());
    auto output = restore(batch, first);
    if (output.size() != padded)
      throw std::runtime_error("SeedVR2: incorrect output frame count");
    for (const auto& frame : output)
      if (frame.size() != pixels)
        throw std::runtime_error("SeedVR2: invalid output frame size");
    if (!pending.empty()) {
      for (size_t p = 0; p < pixels; ++p)
        output[0][p] = 0.5f * (output[0][p] + pending[p]);
      pending.clear();
    }
    const bool retain = !eof && o.segment_frames > 1;
    for (size_t i = 0; i < real - size_t(retain); ++i) {
      write(output[i]);
      ++written;
    }
    if (!retain) {
      input.clear();
      first += real;
      if (eof)
        break;
    } else {
      pending = std::move(output[real - 1]);
      Frame last = std::move(input.back());
      input.clear();
      input.push_back(std::move(last));
      first += real - 1;
    }
  }
  return written;
}
} // namespace slopfab::seedvr2
