#include "internal.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace slopfab::demucs {
void validate_options(const Options& o) {
  if (o.model.empty())
    throw std::invalid_argument("Demucs: a safetensors model is required");
  if (!std::isfinite(o.segment) || o.segment < 0.05 || o.segment > 7.8)
    throw std::invalid_argument("Demucs: segment must be between 0.05 and 7.8 seconds");
  if (!std::isfinite(o.overlap) || o.overlap < 0 || o.overlap > 0.9)
    throw std::invalid_argument("Demucs: overlap must be between 0 and 0.9");
}

Result separate(const std::vector<float>& stereo, const Options& o) {
  validate_options(o);
  if (stereo.empty() || stereo.size() % 2 || stereo.size() > (1ull << 28))
    throw std::invalid_argument("Demucs: expected nonempty stereo PCM (at most 1 GiB)");
  for (float v : stereo)
    if (!std::isfinite(v))
      throw std::invalid_argument("Demucs: non-finite input audio");
  auto progress = [&](double fraction, const std::string& stage) {
    if (o.progress && !o.progress(fraction, stage))
      throw std::runtime_error("Demucs: cancelled");
  };
  progress(0, "loading Demucs");
  Model model(o.model);
  const int length = int(stereo.size() / 2);
  Result result;
  result.sources = {"drums", "bass", "other", "vocals"};
  result.stems.resize(4, std::vector<float>(stereo.size()));
  // Match the upstream CLI's mono-reference normalization before apply_model.
  double mean = 0, variance = 0;
  bool silent = true;
  for (int t = 0; t < length; ++t) {
    mean += (double(stereo[2 * t]) + stereo[2 * t + 1]) / 2;
    silent &= stereo[2 * t] == 0 && stereo[2 * t + 1] == 0;
  }
  mean /= length;
  for (int t = 0; t < length; ++t) {
    double d = (double(stereo[2 * t]) + stereo[2 * t + 1]) / 2 - mean;
    variance += d * d;
  }
  float scale = float(std::sqrt(variance / std::max(1, length - 1)));
  // Constant/anti-phase stereo must not divide by zero.
  if (scale < 1e-8f)
    scale = 1;
  if (silent) {
    progress(1, "complete");
    return result;
  }
  const int segment = int(o.segment * 44100);
  const int stride = std::max(1, int((1 - o.overlap) * segment));
  std::vector<float> total(length), weight(segment);
  for (int t = 0; t < segment; ++t)
    weight[t] = float(std::min(t + 1, segment - t)) / ((segment + 1) / 2);
  for (int offset = 0; offset < length; offset += stride) {
    int count = std::min(segment, length - offset);
    int left = (segment - count) / 2;
    Tensor chunk(2, 1, segment);
    // Retain real context on either side of a short final chunk, just like
    // upstream TensorChunk.padded; zero-pad only beyond the whole recording.
    for (int t = 0; t < segment; ++t) {
      int source = offset + t - left;
      if (source >= 0 && source < length)
        for (int c = 0; c < 2; ++c)
          chunk.at(c, 0, t) = (stereo[size_t(source) * 2 + c] - float(mean)) / scale;
    }
    auto stems = model.forward(chunk, [&](const std::string& stage) {
      progress(double(offset) / length, stage);
    });
    for (int t = 0; t < count; ++t) {
      total[offset + t] += weight[t];
      for (int s = 0; s < 4; ++s)
        for (int c = 0; c < 2; ++c) {
          float v = stems.at(2 * s + c, 0, t + left);
          if (!std::isfinite(v))
            throw std::runtime_error("Demucs: non-finite model output");
          result.stems[s][size_t(offset + t) * 2 + c] += weight[t] * v;
        }
    }
    progress(double(std::min(offset + stride, length)) / length, "separating");
  }
  for (auto& stem : result.stems)
    for (int t = 0; t < length; ++t)
      for (int c = 0; c < 2; ++c)
        stem[size_t(t) * 2 + c] = stem[size_t(t) * 2 + c] / total[t] * scale + float(mean);
  progress(1, "complete");
  return result;
}
} // namespace slopfab::demucs
