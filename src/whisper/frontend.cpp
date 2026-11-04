#include "internal.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace slopfab::whisper {
const json::Value& assets() {
  static const auto value = json::parse(embedded_asset("profiles.json"));
  return value;
}

std::vector<float> mono_16k(const std::vector<float>& pcm, int rate, int channels) {
  if (rate < 1000 || rate > 384000 || channels < 1 || channels > 32 || pcm.empty() ||
      pcm.size() % channels)
    throw std::invalid_argument("Whisper: invalid PCM shape/sample rate");
  std::vector<float> mono(pcm.size() / channels);
  for (size_t i = 0; i < mono.size(); ++i) {
    double sum = 0;
    for (int c = 0; c < channels; ++c) {
      float v = pcm[i * channels + c];
      if (!std::isfinite(v))
        throw std::invalid_argument("Whisper: non-finite PCM");
      sum += v;
    }
    mono[i] = float(sum / channels);
  }
  if (rate == 16000)
    return mono;
  // Windowed sinc low-pass before decimation; linear interpolation aliases music.
  const double pi = std::acos(-1.0), cutoff = std::min(1.0, 16000.0 / rate) * 0.95;
  const int radius = int(std::ceil(32 / cutoff));
  std::vector<float> out(size_t(std::ceil(double(mono.size()) * 16000 / rate)));
  for (size_t i = 0; i < out.size(); ++i) {
    double center = double(i) * rate / 16000, sum = 0, norm = 0;
    int64_t first = int64_t(std::floor(center));
    for (int k = -radius; k <= radius; ++k) {
      double d = center - (first + k);
      if (std::abs(d) > radius)
        continue;
      double s = std::abs(d) < 1e-12 ? cutoff : std::sin(pi * cutoff * d) / (pi * d);
      s *= 0.5 + 0.5 * std::cos(pi * d / radius);
      int64_t at = std::clamp<int64_t>(first + k, 0, int64_t(mono.size()) - 1);
      sum += mono[size_t(at)] * s;
      norm += s;
    }
    out[i] = float(sum / norm);
  }
  return out;
}

std::vector<float> log_mel(const std::vector<float>& audio, int mels) {
  if (mels != 80 && mels != 128)
    throw std::invalid_argument("Whisper: unsupported mel count");
  constexpr int fft = 400, hop = 160;
  const size_t padded = audio.size() + 480000, frames = padded / hop;
  if (frames > size_t(INT32_MAX) / mels)
    throw std::invalid_argument("Whisper: audio too long");
  const auto& filters = assets().find(mels == 80 ? "mel_80" : "mel_128")->as_array();
  std::vector<std::vector<std::pair<int, float>>> sparse(mels);
  for (int m = 0; m < mels; ++m)
    for (int k = 0; k < 201; ++k) {
      float w = float(filters[size_t(m) * 201 + k].as_number());
      if (w != 0)
        sparse[m].emplace_back(k, w);
    }
  std::complex<float> twiddle[fft];
  float window[fft];
  const double pi = std::acos(-1.0);
  for (int i = 0; i < fft; ++i) {
    twiddle[i] = {float(std::cos(-2 * pi * i / fft)), float(std::sin(-2 * pi * i / fft))};
    window[i] = float(0.5 - 0.5 * std::cos(2 * pi * i / fft));
  }
  std::vector<float> result(frames * mels);
  float maximum = -10;
  for (size_t f = 0; f < frames; ++f) {
    float input[fft], power[201];
    for (int k = 0; k < fft; ++k) {
      int64_t at = int64_t(f * hop) + k - 200;
      if (at < 0)
        at = -at;
      if (at >= int64_t(padded))
        at = 2 * int64_t(padded - 1) - at;
      input[k] = (size_t(at) < audio.size() ? audio[size_t(at)] : 0) * window[k];
    }
    // Mixed-radix DFT, 400 = 20 * 20. Only the nonnegative spectrum is needed.
    std::complex<float> tmp[20][20];
    for (int n = 0; n < 20; ++n)
      for (int k = 0; k < 20; ++k) {
        std::complex<float> sum{};
        for (int j = 0; j < 20; ++j)
          sum += input[n + 20 * j] * twiddle[(20 * j * k) % 400];
        tmp[n][k] = sum;
      }
    for (int k = 0; k <= 200; ++k) {
      std::complex<float> sum{};
      for (int n = 0; n < 20; ++n)
        sum += tmp[n][k % 20] * twiddle[(n * k) % 400];
      power[k] = std::norm(sum);
    }
    for (int m = 0; m < mels; ++m) {
      float sum = 0;
      for (auto [k, w] : sparse[m])
        sum += power[k] * w;
      float value = std::log10(std::max(sum, 1e-10f));
      result[f * mels + m] = value;
      maximum = std::max(maximum, value);
    }
  }
  for (float& v : result)
    v = (std::max(v, maximum - 8) + 4) / 4;
  return result;
}
}
