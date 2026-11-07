#include "network.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace slopfab::syncnet {
std::vector<float> mfcc(const std::vector<int16_t>& pcm) {
  if (pcm.empty())
    throw std::invalid_argument("SyncNet: empty audio");
  constexpr double pi = 3.14159265358979323846;
  constexpr double eps = std::numeric_limits<double>::epsilon();
  const size_t frames = pcm.size() <= 400 ? 1 : 1 + (pcm.size() - 400 + 159) / 160;
  std::vector<float> result(frames * 13);
  int bins[28];
  for (int i = 0; i < 28; ++i) {
    double mel = 2595 * std::log10(1 + 8000. / 700) * i / 27;
    bins[i] = int(std::floor(513 * 700 * (std::pow(10., mel / 2595) - 1) / 16000));
  }
  for (size_t f = 0; f < frames; ++f) {
    std::array<std::complex<double>, 512> fft{};
    for (size_t j = 0; j < 400; ++j) {
      size_t pos = f * 160 + j;
      if (pos < pcm.size())
        fft[j] = pcm[pos] - (pos ? .97 * pcm[pos - 1] : 0);
    }
    for (int i = 1, j = 0; i < 512; ++i) {
      int bit = 256;
      for (; j & bit; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j)
        std::swap(fft[i], fft[j]);
    }
    for (int len = 2; len <= 512; len *= 2) {
      auto root = std::polar(1., -2 * pi / len);
      for (int i = 0; i < 512; i += len) {
        std::complex<double> w = 1;
        for (int j = 0; j < len / 2; ++j) {
          auto u = fft[i + j], v = fft[i + j + len / 2] * w;
          fft[i + j] = u + v;
          fft[i + j + len / 2] = u - v;
          w *= root;
        }
      }
    }
    double power[257], energy = 0, filters[26]{};
    for (int i = 0; i < 257; ++i) {
      power[i] = std::norm(fft[i]) / 512;
      energy += power[i];
    }
    for (int m = 0; m < 26; ++m) {
      for (int k = bins[m]; k < bins[m + 1]; ++k)
        filters[m] += power[k] * (k - bins[m]) / (bins[m + 1] - bins[m]);
      for (int k = bins[m + 1]; k < bins[m + 2]; ++k)
        filters[m] += power[k] * (bins[m + 2] - k) / (bins[m + 2] - bins[m + 1]);
      filters[m] = std::log(filters[m] == 0 ? eps : filters[m]);
    }
    result[f * 13] = float(std::log(energy == 0 ? eps : energy));
    for (int k = 1; k < 13; ++k) {
      double dct = 0;
      for (int j = 0; j < 26; ++j)
        dct += filters[j] * std::cos(pi * k * (j + .5) / 26);
      result[f * 13 + k] = float(dct * std::sqrt(2. / 26) * (1 + 11 * std::sin(pi * k / 22)));
    }
  }
  return result;
}

Score score(const std::vector<std::vector<float>>& video,
            const std::vector<std::vector<float>>& audio, int shift) {
  if (video.empty() || video.size() != audio.size() || shift < 0 || shift > 250)
    throw std::invalid_argument("SyncNet: invalid embeddings or search range");
  for (size_t i = 0; i < video.size(); ++i) {
    if (video[i].size() != 1024 || audio[i].size() != 1024)
      throw std::invalid_argument("SyncNet: expected 1024-dimensional embeddings");
    for (int c = 0; c < 1024; ++c)
      if (!std::isfinite(video[i][c]) || !std::isfinite(audio[i][c]))
        throw std::runtime_error("SyncNet: non-finite embedding");
  }
  Score result{};
  for (int delta = -shift; delta <= shift; ++delta) {
    double sum = 0;
    for (int i = 0; i < int(video.size()); ++i) {
      int j = i + delta;
      double sq = 0;
      for (int c = 0; c < 1024; ++c) {
        // PyTorch pairwise_distance adds eps to each component, including padded windows.
        float d = video[i][c] - (j < 0 || j >= int(audio.size()) ? 0.f : audio[j][c]) + 1e-6f;
        sq += double(d) * d;
      }
      sum += std::sqrt(sq);
    }
    result.shift_distances.push_back(sum / video.size());
  }
  int best = int(std::min_element(result.shift_distances.begin(), result.shift_distances.end()) -
                 result.shift_distances.begin());
  auto sorted = result.shift_distances;
  std::sort(sorted.begin(), sorted.end());
  result.offset_frames = shift - best;
  result.distance = result.shift_distances[best];
  result.confidence = sorted[shift] - result.distance;
  result.zero_shift_distance = result.shift_distances[shift];
  result.search_boundary = shift > 0 && (best == 0 || best == 2 * shift);
  return result;
}

Image resize_crop(const Image& x, float left, float top, float width, float height, int out_h,
                  int out_w) {
  Image y(out_h, out_w, x.c);
  for (int r = 0; r < out_h; ++r) {
    float sy = top + (r + .5f) * height / out_h - .5f;
    // OpenCV resize replicates edge pixels of the integer crop before interpolation.
    sy = std::clamp(sy, top, top + height - 1);
    int y0 = int(std::floor(sy));
    float fy = sy - y0;
    for (int s = 0; s < out_w; ++s) {
      float sx = left + (s + .5f) * width / out_w - .5f;
      sx = std::clamp(sx, left, left + width - 1);
      int x0 = int(std::floor(sx));
      float fx = sx - x0;
      auto pixel = [&](int iy, int ix, int c) {
        return iy < 0 || ix < 0 || iy >= x.h || ix >= x.w
                   ? 110.f
                   : x.data[(size_t(iy) * x.w + ix) * x.c + c];
      };
      for (int c = 0; c < x.c; ++c)
        y.data[(size_t(r) * out_w + s) * x.c + c] =
            (1 - fy) * ((1 - fx) * pixel(y0, x0, c) + fx * pixel(y0, x0 + 1, c)) +
            fy * ((1 - fx) * pixel(y0 + 1, x0, c) + fx * pixel(y0 + 1, x0 + 1, c));
    }
  }
  return y;
}
} // namespace slopfab::syncnet
