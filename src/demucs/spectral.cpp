#include "internal.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace slopfab::demucs {
namespace {
constexpr int nfft = 4096, hop = 1024;
constexpr double pi = 3.14159265358979323846;
using Complex = std::complex<float>;

void fft(std::vector<Complex>& a, bool inverse) {
  const int n = int(a.size());
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(a[i], a[j]);
  }
  for (int len = 2; len <= n; len *= 2) {
    // Compute each twiddle independently to avoid accumulated phase error.
    std::vector<Complex> roots(len / 2);
    for (int j = 0; j < len / 2; ++j) {
      double angle = (inverse ? 2 : -2) * pi * j / len;
      roots[j] = Complex(float(std::cos(angle)), float(std::sin(angle)));
    }
    for (int i = 0; i < n; i += len)
      for (int j = 0; j < len / 2; ++j) {
        auto u = a[i + j], v = a[i + j + len / 2] * roots[j];
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
      }
  }
  for (auto& value : a)
    value /= std::sqrt(float(n));
}

std::vector<float> window() {
  std::vector<float> w(nfft);
  for (int i = 0; i < nfft; ++i)
    w[i] = float(0.5 - 0.5 * std::cos(2 * pi * i / nfft));
  return w;
}
} // namespace

Tensor::Tensor(int channels, int frequencies, int time) : c(channels), f(frequencies), t(time) {
  if (c <= 0 || f <= 0 || t <= 0 || uint64_t(c) * f * t > (1ull << 29))
    throw std::invalid_argument("Demucs: invalid or excessive activation shape");
  v.resize(size_t(c) * f * t);
}

Tensor spectrogram(const Tensor& wave) {
  const int frames = (wave.t + hop - 1) / hop;
  Tensor out(wave.c * 2, nfft / 2, frames);
  auto win = window();
  // Match pad1d's zero extension before reflect for very short inputs.
  int left = 3 * hop / 2, right = left + frames * hop - wave.t;
  int extra = std::max(0, std::max(left, right) - wave.t + 1);
  int extra_right = std::min(right, extra), extra_left = extra - extra_right;
  int extended = wave.t + extra;
  for (int ch = 0; ch < wave.c; ++ch)
    for (int t = 0; t < frames; ++t) {
      std::vector<Complex> buf(nfft);
      for (int k = 0; k < nfft; ++k) {
        int at = t * hop + k - left + extra_left;
        if (at < 0)
          at = -at;
        if (at >= extended)
          at = 2 * extended - 2 - at;
        at -= extra_left;
        buf[k] = (at >= 0 && at < wave.t ? wave.at(ch, 0, at) : 0) * win[k];
      }
      fft(buf, false);
      for (int f = 0; f < nfft / 2; ++f) {
        out.at(ch * 2, f, t) = buf[f].real();
        out.at(ch * 2 + 1, f, t) = buf[f].imag();
      }
    }
  return out;
}

Tensor inverse_spectrogram(const Tensor& spec, int length) {
  if (spec.c % 2 || spec.f != nfft / 2 || spec.t != (length + hop - 1) / hop)
    throw std::invalid_argument("Demucs: invalid inverse STFT shape");
  Tensor out(spec.c / 2, 1, length);
  auto win = window();
  std::vector<float> denominator(length);
  // Include the four zero STFT frames in the window envelope, as torch.istft does.
  for (int t = -2; t < spec.t + 2; ++t)
    for (int k = 0; k < nfft; ++k) {
      int at = t * hop + k - 3 * hop / 2;
      if (at >= 0 && at < length)
        denominator[at] += win[k] * win[k];
    }
  for (int ch = 0; ch < out.c; ++ch)
    for (int t = 0; t < spec.t; ++t) {
      std::vector<Complex> buf(nfft);
      for (int f = 0; f < nfft / 2; ++f) {
        buf[f] = Complex(spec.at(ch * 2, f, t), spec.at(ch * 2 + 1, f, t));
        if (f)
          buf[nfft - f] = std::conj(buf[f]);
      }
      fft(buf, true);
      for (int k = 0; k < nfft; ++k) {
        int at = t * hop + k - 3 * hop / 2;
        if (at >= 0 && at < length)
          out.at(ch, 0, at) += buf[k].real() * win[k];
      }
    }
  for (int ch = 0; ch < out.c; ++ch)
    for (int t = 0; t < length; ++t)
      out.at(ch, 0, t) /= denominator[t];
  return out;
}

std::pair<float, float> normalize(Tensor& x) {
  double mean = 0, variance = 0;
  for (float v : x.v)
    mean += v;
  mean /= x.v.size();
  for (float v : x.v)
    variance += (v - mean) * (v - mean);
  float stddev = float(std::sqrt(variance / std::max(size_t(1), x.v.size() - 1)));
  for (auto& v : x.v)
    v = (v - float(mean)) / (stddev + 1e-5f);
  return {float(mean), stddev};
}
} // namespace slopfab::demucs
