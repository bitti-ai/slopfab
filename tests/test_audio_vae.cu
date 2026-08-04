// Audio VAE decoder kernel tests against independent CPU references.
//
// Every reference below is transcribed from docs/audio_vae_spec.md — that is,
// from ref/FL2VA/audio_vae/*.py — and not from the kernel it checks. The
// resamplers in particular are written the long way (materialise the replicate
// padding, run the full transposed convolution, then crop) rather than in the
// closed form the kernel uses, so a mistake in the derivation shows up here
// instead of agreeing with itself.
//
// Two deliberate choices about the test data:
//
//   * Kernels and inputs are ASYMMETRIC. A symmetric kernel convolves the same
//     forwards and backwards, so a reversed-kernel bug — the classic
//     correlation-vs-convolution mixup — passes a symmetric test.
//   * Channel counts differ between input and output, and lengths are not
//     multiples of the block tile, so the tail-guard paths are exercised.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

#include "harness.h"
#include "vidfab/cuda/audio_vae_kernels.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/safetensors.h"
#include "vidfab/vae/audio_decoder.h"

namespace {

using vidfab::cuda::DeviceBuffer;

constexpr int kTaps = vidfab::cuda::kAudioAAKernel;  // 12

// Deterministic pseudo-random fill; avoids <random> so results are identical
// across standard library versions. Same generator as tests/test_kernels.cu.
std::vector<float> make_data(size_t n, uint32_t seed, float scale = 1.0f) {
  std::vector<float> v(n);
  uint32_t s = seed | 1u;
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    v[i] = ((static_cast<float>(s & 0xFFFFFFu) / 16777216.0f) - 0.5f) * 2.0f * scale;
  }
  return v;
}

DeviceBuffer<float> to_device(const std::vector<float>& host) {
  DeviceBuffer<float> d(host.size());
  d.copy_from_host(host.data(), host.size());
  return d;
}

std::vector<float> to_host(const DeviceBuffer<float>& d) {
  std::vector<float> h(d.size());
  d.copy_to_host(h.data(), h.size());
  return h;
}

// --- CPU references ---------------------------------------------------------

// torch.nn.functional.conv1d: cross-correlation with zero padding, stride 1.
// The kernel is NOT reversed.
std::vector<float> cpu_conv1d(const std::vector<float>& x, const std::vector<float>& w,
                              const std::vector<float>& bias, int batch, int in_ch, int out_ch,
                              int len_in, int kernel, int pad, int dilation) {
  const int len_out = len_in + 2 * pad - dilation * (kernel - 1);
  std::vector<float> y(static_cast<size_t>(batch) * out_ch * len_out);
  for (int b = 0; b < batch; ++b) {
    for (int co = 0; co < out_ch; ++co) {
      for (int n = 0; n < len_out; ++n) {
        double sum = bias.empty() ? 0.0 : bias[static_cast<size_t>(co)];
        for (int ci = 0; ci < in_ch; ++ci) {
          for (int k = 0; k < kernel; ++k) {
            const int t = n + k * dilation - pad;
            if (t < 0 || t >= len_in) continue;
            sum += static_cast<double>(x[(static_cast<size_t>(b) * in_ch + ci) * len_in + t]) *
                   w[(static_cast<size_t>(co) * in_ch + ci) * kernel + k];
          }
        }
        y[(static_cast<size_t>(b) * out_ch + co) * len_out + n] = static_cast<float>(sum);
      }
    }
  }
  return y;
}

// torch.nn.functional.conv_transpose1d in its defining scatter form. Weight is
// [Cin, Cout, K]; output_padding and dilation are 0 and 1.
std::vector<float> cpu_conv_transpose1d(const std::vector<float>& x, const std::vector<float>& w,
                                        const std::vector<float>& bias, int batch, int in_ch,
                                        int out_ch, int len_in, int kernel, int stride, int pad) {
  const int len_full = (len_in - 1) * stride + kernel;
  const int len_out = len_full - 2 * pad;
  std::vector<double> full(static_cast<size_t>(batch) * out_ch * len_full, 0.0);
  for (int b = 0; b < batch; ++b) {
    for (int ci = 0; ci < in_ch; ++ci) {
      for (int i = 0; i < len_in; ++i) {
        const double xv = x[(static_cast<size_t>(b) * in_ch + ci) * len_in + i];
        for (int co = 0; co < out_ch; ++co) {
          for (int k = 0; k < kernel; ++k) {
            full[(static_cast<size_t>(b) * out_ch + co) * len_full + i * stride + k] +=
                xv * w[(static_cast<size_t>(ci) * out_ch + co) * kernel + k];
          }
        }
      }
    }
  }
  std::vector<float> y(static_cast<size_t>(batch) * out_ch * len_out);
  for (int b = 0; b < batch; ++b) {
    for (int co = 0; co < out_ch; ++co) {
      for (int n = 0; n < len_out; ++n) {
        const double v = full[(static_cast<size_t>(b) * out_ch + co) * len_full + n + pad];
        y[(static_cast<size_t>(b) * out_ch + co) * len_out + n] =
            static_cast<float>(v + (bias.empty() ? 0.0 : bias[static_cast<size_t>(co)]));
      }
    }
  }
  return y;
}

// SnakeBeta with alpha_logscale=True: x + sin(exp(a)x)^2 / (exp(b) + 1e-9).
// Note the division by beta, and the guard applied after the exponential.
void cpu_snake(std::vector<float>& x, const std::vector<float>& log_alpha,
               const std::vector<float>& log_beta, int batch, int channels, int len) {
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      const double alpha = std::exp(static_cast<double>(log_alpha[static_cast<size_t>(c)]));
      const double beta = std::exp(static_cast<double>(log_beta[static_cast<size_t>(c)]));
      for (int n = 0; n < len; ++n) {
        const size_t i = (static_cast<size_t>(b) * channels + c) * len + n;
        const double v = x[i];
        const double s = std::sin(alpha * v);
        x[i] = static_cast<float>(v + s * s / (beta + 1e-9));
      }
    }
  }
}

std::vector<float> replicate_pad(const std::vector<float>& x, int batch, int channels, int len,
                                 int left, int right) {
  const int out_len = len + left + right;
  std::vector<float> y(static_cast<size_t>(batch) * channels * out_len);
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int n = 0; n < out_len; ++n) {
        int s = n - left;
        s = s < 0 ? 0 : (s >= len ? len - 1 : s);
        y[(static_cast<size_t>(b) * channels + c) * out_len + n] =
            x[(static_cast<size_t>(b) * channels + c) * len + s];
      }
    }
  }
  return y;
}

// UpSample1d(ratio=2, kernel_size=12), written the long way:
//   pad 5/5 replicate -> depthwise conv_transpose1d stride 2 -> scale by 2 ->
//   crop 15 from each end.
std::vector<float> cpu_aa_upsample(const std::vector<float>& x, const std::vector<float>& filter,
                                   int batch, int channels, int len) {
  constexpr int kPad = kTaps / 2 - 1;                   // 5
  constexpr int kCropLeft = kPad * 2 + (kTaps - 2) / 2; // 15
  constexpr int kCropRight = kPad * 2 + (kTaps - 2 + 1) / 2;
  const std::vector<float> xp = replicate_pad(x, batch, channels, len, kPad, kPad);
  const int lp = len + 2 * kPad;
  const int len_full = (lp - 1) * 2 + kTaps;
  std::vector<double> full(static_cast<size_t>(batch) * channels * len_full, 0.0);
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < lp; ++i) {
        const double v = xp[(static_cast<size_t>(b) * channels + c) * lp + i];
        for (int k = 0; k < kTaps; ++k) {
          full[(static_cast<size_t>(b) * channels + c) * len_full + i * 2 + k] +=
              v * filter[static_cast<size_t>(k)];
        }
      }
    }
  }
  const int len_out = len_full - kCropLeft - kCropRight;
  std::vector<float> y(static_cast<size_t>(batch) * channels * len_out);
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int n = 0; n < len_out; ++n) {
        y[(static_cast<size_t>(b) * channels + c) * len_out + n] = static_cast<float>(
            2.0 * full[(static_cast<size_t>(b) * channels + c) * len_full + n + kCropLeft]);
      }
    }
  }
  return y;
}

// DownSample1d(ratio=2, kernel_size=12): replicate pad 5 left / 6 right, then a
// stride-2 correlation. The asymmetry is the phase.
std::vector<float> cpu_aa_downsample(const std::vector<float>& x, const std::vector<float>& filter,
                                     int batch, int channels, int len) {
  const std::vector<float> xp = replicate_pad(x, batch, channels, len, kTaps / 2 - 1, kTaps / 2);
  const int lp = len + kTaps - 1;
  const int len_out = (lp - kTaps) / 2 + 1;
  std::vector<float> y(static_cast<size_t>(batch) * channels * len_out);
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int n = 0; n < len_out; ++n) {
        double sum = 0.0;
        for (int k = 0; k < kTaps; ++k) {
          sum += static_cast<double>(xp[(static_cast<size_t>(b) * channels + c) * lp + 2 * n + k]) *
                 filter[static_cast<size_t>(k)];
        }
        y[(static_cast<size_t>(b) * channels + c) * len_out + n] = static_cast<float>(sum);
      }
    }
  }
  return y;
}

// The shipped 12-tap Kaiser-windowed sinc, recomputed rather than loaded so the
// unit tests do not need the checkpoint. Verified against the checkpoint's
// buffers to 2.6e-8 (docs/audio_vae_spec.md §7.1).
std::vector<float> kaiser_sinc12() {
  const double cutoff = 0.25;
  const double half_width = 0.3;
  const int half = kTaps / 2;
  const double a = 2.285 * (half - 1) * 3.14159265358979323846 * (4.0 * half_width) + 7.95;
  const double beta = 0.1102 * (a - 8.7);  // a = 51.02 > 50
  // Zeroth-order modified Bessel of the first kind, series form.
  auto bessel_i0 = [](double v) {
    double sum = 1.0;
    double term = 1.0;
    for (int k = 1; k < 64; ++k) {
      term *= (v / (2.0 * k)) * (v / (2.0 * k));
      sum += term;
    }
    return sum;
  };
  std::vector<float> f(kTaps);
  double total = 0.0;
  std::vector<double> raw(kTaps);
  for (int i = 0; i < kTaps; ++i) {
    const double r = 2.0 * i / (kTaps - 1) - 1.0;  // symmetric window, periodic=False
    const double win = bessel_i0(beta * std::sqrt(1.0 - r * r)) / bessel_i0(beta);
    const double t = (i - half) + 0.5;
    const double arg = 2.0 * cutoff * t;
    const double sinc = arg == 0.0 ? 1.0 : std::sin(3.14159265358979323846 * arg) /
                                               (3.14159265358979323846 * arg);
    raw[i] = 2.0 * cutoff * win * sinc;
    total += raw[i];
  }
  for (int i = 0; i < kTaps; ++i) f[static_cast<size_t>(i)] = static_cast<float>(raw[i] / total);
  return f;
}

// --- tests ------------------------------------------------------------------

VIDFAB_TEST(audio_snake_beta) {
  const int batch = 2;
  const int channels = 5;
  const int len = 37;  // not a multiple of the block width

  // Values chosen so the wrong formulas are visibly different rather than
  // marginally so: alpha and beta both positive and negative in log space,
  // which is what the real checkpoint carries (34% of its alphas are negative).
  const std::vector<float> log_alpha = {0.0f, 0.6931472f, -0.6931472f, 1.2f, -1.1f};
  const std::vector<float> log_beta = {-0.9f, 0.4f, 1.5f, -1.4f, 0.25f};
  std::vector<float> x = make_data(static_cast<size_t>(batch) * channels * len, 7u, 2.5f);

  std::vector<float> want = x;
  cpu_snake(want, log_alpha, log_beta, batch, channels, len);

  DeviceBuffer<float> dx = to_device(x);
  DeviceBuffer<float> da = to_device(log_alpha);
  DeviceBuffer<float> db = to_device(log_beta);
  vidfab::cuda::launch_snake_beta(dx.get(), da.get(), db.get(), batch, channels, len, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want, to_host(dx), 2e-5, "snake_beta");

  // Hand-computed anchor. Channel 1 has log_alpha = ln 2, log_beta = 0.4, so
  //   y = x + sin(2x)^2 / (e^0.4 + 1e-9).
  // Checked independently of the reference loop above so a shared mistake in
  // both cannot hide.
  {
    std::vector<float> one(1, 0.75f);
    const std::vector<float> a1 = {0.6931472f};
    const std::vector<float> b1 = {0.4f};
    DeviceBuffer<float> d1 = to_device(one);
    DeviceBuffer<float> da1 = to_device(a1);
    DeviceBuffer<float> db1 = to_device(b1);
    vidfab::cuda::launch_snake_beta(d1.get(), da1.get(), db1.get(), 1, 1, 1, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const double s = std::sin(2.0 * 0.75);
    const double expect = 0.75 + s * s / std::exp(0.4);
    CHECK_NEAR(to_host(d1)[0], expect, 1e-5);
    // The two wrong readings of the same parameters must not coincide with it.
    const double divide_by_alpha = 0.75 + s * s / 2.0;
    const double no_exp = 0.75 + std::sin(0.6931472 * 0.75) * std::sin(0.6931472 * 0.75) / 0.4;
    CHECK(std::abs(expect - divide_by_alpha) > 0.05);
    CHECK(std::abs(expect - no_exp) > 0.05);
  }
}

VIDFAB_TEST(audio_conv1d) {
  // Asymmetric everywhere: 3 in / 5 out channels, a length that straddles the
  // 512-wide block tile, and random (hence non-palindromic) weights.
  struct Case {
    int kernel;
    int dilation;
  };
  const Case cases[] = {{1, 1}, {3, 1}, {7, 1}, {3, 3}, {7, 3}, {11, 5}, {3, 5}};

  const int batch = 2;
  const int in_ch = 3;
  const int out_ch = 5;
  for (const int len : {40, 600}) {
    const std::vector<float> x = make_data(static_cast<size_t>(batch) * in_ch * len, 11u, 1.5f);
    DeviceBuffer<float> dx = to_device(x);
    for (const Case& c : cases) {
      const int pad = (c.kernel * c.dilation - c.dilation) / 2;  // dac_utils.py:11-12
      const int len_out = len + 2 * pad - c.dilation * (c.kernel - 1);
      const std::vector<float> w =
          make_data(static_cast<size_t>(out_ch) * in_ch * c.kernel, 23u + c.kernel, 0.9f);
      const std::vector<float> bias = make_data(static_cast<size_t>(out_ch), 31u, 0.4f);

      const std::vector<float> want =
          cpu_conv1d(x, w, bias, batch, in_ch, out_ch, len, c.kernel, pad, c.dilation);
      CHECK(want.size() == static_cast<size_t>(batch) * out_ch * len_out);

      DeviceBuffer<float> dw = to_device(w);
      DeviceBuffer<float> dbias = to_device(bias);
      DeviceBuffer<float> dy(want.size());
      vidfab::cuda::launch_conv1d(dx.get(), dw.get(), dbias.get(), dy.get(), batch, in_ch, out_ch,
                                  len, len_out, c.kernel, pad, c.dilation, nullptr);
      VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const std::string what = "conv1d k" + std::to_string(c.kernel) + " d" +
                               std::to_string(c.dilation) + " L" + std::to_string(len);
      CHECK_CLOSE(want, to_host(dy), 2e-5, what.c_str());
    }
  }

  // A null bias is the conv_post case, not an error.
  {
    const int len = 20;
    const std::vector<float> x = make_data(static_cast<size_t>(in_ch) * len, 5u);
    const std::vector<float> w = make_data(static_cast<size_t>(in_ch) * 7, 9u);
    const std::vector<float> want = cpu_conv1d(x, w, {}, 1, in_ch, 1, len, 7, 3, 1);
    DeviceBuffer<float> dx2 = to_device(x);
    DeviceBuffer<float> dw = to_device(w);
    DeviceBuffer<float> dy(want.size());
    vidfab::cuda::launch_conv1d(dx2.get(), dw.get(), nullptr, dy.get(), 1, in_ch, 1, len, len, 7, 3,
                                1, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE(want, to_host(dy), 2e-5, "conv1d no bias");
  }

  // An impulse input recovers the kernel in order. This is what actually pins
  // down correlation-vs-convolution: with x = delta at position p and SAME
  // padding, y[p - (k - (K-1)/2)] picks out w[k], so a reversed kernel writes
  // the taps backwards and this comparison fails.
  {
    const int len = 15;
    const int kernel = 5;
    std::vector<float> x(static_cast<size_t>(len), 0.0f);
    x[7] = 1.0f;
    std::vector<float> w(static_cast<size_t>(kernel));
    for (int k = 0; k < kernel; ++k) w[static_cast<size_t>(k)] = static_cast<float>(k + 1);
    DeviceBuffer<float> dx3 = to_device(x);
    DeviceBuffer<float> dw = to_device(w);
    DeviceBuffer<float> dy(static_cast<size_t>(len));
    vidfab::cuda::launch_conv1d(dx3.get(), dw.get(), nullptr, dy.get(), 1, 1, 1, len, len, kernel, 2,
                                1, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> got = to_host(dy);
    // y[n] = sum_k x[n + k - 2] w[k]; x[7]=1 => y[9-k] = w[k].
    for (int k = 0; k < kernel; ++k) {
      CHECK_NEAR(got[static_cast<size_t>(9 - k)], static_cast<double>(k + 1), 1e-6);
    }
  }
}

VIDFAB_TEST(audio_conv_transpose1d) {
  // The two (kernel, stride) pairs the decoder actually uses, plus a case whose
  // kernel is not a whole number of strides so the phase logic is exercised.
  struct Case {
    int kernel;
    int stride;
    int pad;
  };
  const Case cases[] = {{9, 5, 2}, {4, 2, 1}, {6, 3, 1}, {5, 2, 1}};

  const int batch = 2;
  const int in_ch = 4;
  const int out_ch = 3;  // deliberately different from in_ch and not a multiple of 8
  const int len = 37;
  const std::vector<float> x = make_data(static_cast<size_t>(batch) * in_ch * len, 41u, 1.2f);
  DeviceBuffer<float> dx = to_device(x);

  for (const Case& c : cases) {
    const std::vector<float> w =
        make_data(static_cast<size_t>(in_ch) * out_ch * c.kernel, 53u + c.kernel, 0.8f);
    const std::vector<float> bias = make_data(static_cast<size_t>(out_ch), 61u, 0.3f);
    const std::vector<float> want =
        cpu_conv_transpose1d(x, w, bias, batch, in_ch, out_ch, len, c.kernel, c.stride, c.pad);
    const int len_out = (len - 1) * c.stride - 2 * c.pad + c.kernel;
    CHECK(want.size() == static_cast<size_t>(batch) * out_ch * len_out);

    DeviceBuffer<float> dw = to_device(w);
    DeviceBuffer<float> dbias = to_device(bias);
    DeviceBuffer<float> dy(want.size());
    vidfab::cuda::launch_conv_transpose1d(dx.get(), dw.get(), dbias.get(), dy.get(), batch, in_ch,
                                          out_ch, len, len_out, c.kernel, c.stride, c.pad, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::string what =
        "conv_transpose1d k" + std::to_string(c.kernel) + " s" + std::to_string(c.stride);
    CHECK_CLOSE(want, to_host(dy), 2e-5, what.c_str());
  }

  // The upsamplers' own padding rule, (k - u) // 2, is what makes every stage
  // multiply the length by exactly u — and therefore what makes the decoder's
  // output exactly 800 samples per latent.
  const int rates[7] = {5, 5, 2, 2, 2, 2, 2};
  const int kernels[7] = {9, 9, 4, 4, 4, 4, 4};
  int length = 13;
  for (int i = 0; i < 7; ++i) {
    const int pad = (kernels[i] - rates[i]) / 2;
    const int out = (length - 1) * rates[i] - 2 * pad + kernels[i];
    CHECK(out == length * rates[i]);
    length = out;
  }
  CHECK(length == 13 * 800);

  // The odd rates pair with 2u-1, not 2u. Getting this "obvious" simplification
  // wrong changes the output length, which is why it is asserted here.
  for (int i = 0; i < 7; ++i) {
    CHECK(kernels[i] == 2 * rates[i] - (rates[i] % 2));
  }
}

VIDFAB_TEST(audio_aa_activation) {
  const std::vector<float> filter = kaiser_sinc12();

  // The recomputed filter must be normalised and symmetric; both properties are
  // load-bearing (see below) and both are cheap to assert.
  double total = 0.0;
  for (int k = 0; k < kTaps; ++k) total += filter[static_cast<size_t>(k)];
  CHECK_NEAR(total, 1.0, 1e-6);
  for (int k = 0; k < kTaps / 2; ++k) {
    CHECK_NEAR(filter[static_cast<size_t>(k)], filter[static_cast<size_t>(kTaps - 1 - k)], 1e-6);
  }

  const int batch = 2;
  const int channels = 3;
  for (const int len : {9, 400}) {
    const std::vector<float> x =
        make_data(static_cast<size_t>(batch) * channels * len, 71u + len, 1.7f);
    const std::vector<float> log_alpha = {0.3f, -0.8f, 1.1f};
    const std::vector<float> log_beta = {-0.5f, 0.7f, -0.2f};

    // Reference: upsample, snake, downsample — each written out longhand.
    std::vector<float> mid = cpu_aa_upsample(x, filter, batch, channels, len);
    CHECK(mid.size() == static_cast<size_t>(batch) * channels * 2 * len);
    cpu_snake(mid, log_alpha, log_beta, batch, channels, 2 * len);
    const std::vector<float> want = cpu_aa_downsample(mid, filter, batch, channels, 2 * len);
    // Activation1d preserves length exactly: T -> 2T -> 2T -> T.
    CHECK(want.size() == x.size());

    DeviceBuffer<float> dx = to_device(x);
    DeviceBuffer<float> df = to_device(filter);
    DeviceBuffer<float> da = to_device(log_alpha);
    DeviceBuffer<float> db = to_device(log_beta);
    DeviceBuffer<float> dmid(mid.size());
    DeviceBuffer<float> dy(want.size());

    vidfab::cuda::launch_aa_upsample_snake(dx.get(), df.get(), da.get(), db.get(), dmid.get(),
                                           batch, channels, len, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::string up_what = "aa_upsample_snake L" + std::to_string(len);
    CHECK_CLOSE(mid, to_host(dmid), 2e-5, up_what.c_str());

    vidfab::cuda::launch_aa_downsample(dmid.get(), df.get(), dy.get(), batch, channels, 2 * len,
                                       len, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::string what = "aa_activation L" + std::to_string(len);
    CHECK_CLOSE(want, to_host(dy), 2e-5, what.c_str());
  }

  // A constant signal is a fixed point of the resampler pair, because the
  // filter sums to 1 and — being symmetric — splits that evenly between its odd
  // and even taps, which the `ratio *` factor then undoes. Replicate padding
  // makes this exact at the edges too, so any phase or padding-mode mistake
  // shows up as an edge transient rather than a rounding difference.
  {
    const int len = 24;
    const std::vector<float> ones(static_cast<size_t>(len), 0.375f);
    const std::vector<float> zero_alpha(1, -20.0f);  // exp(-20) ~ 0: sin term vanishes
    const std::vector<float> big_beta(1, 20.0f);
    DeviceBuffer<float> dx = to_device(ones);
    DeviceBuffer<float> df = to_device(filter);
    DeviceBuffer<float> da = to_device(zero_alpha);
    DeviceBuffer<float> db = to_device(big_beta);
    DeviceBuffer<float> dmid(static_cast<size_t>(2 * len));
    DeviceBuffer<float> dy(static_cast<size_t>(len));
    vidfab::cuda::launch_aa_upsample_snake(dx.get(), df.get(), da.get(), db.get(), dmid.get(), 1, 1,
                                           len, nullptr);
    vidfab::cuda::launch_aa_downsample(dmid.get(), df.get(), dy.get(), 1, 1, 2 * len, len, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> mid = to_host(dmid);
    const std::vector<float> got = to_host(dy);
    for (size_t i = 0; i < mid.size(); ++i) CHECK_NEAR(mid[i], 0.375, 1e-5);
    for (size_t i = 0; i < got.size(); ++i) CHECK_NEAR(got[i], 0.375, 1e-5);
  }

  // Length arithmetic for an odd input: the decimation keeps ceil(L/2).
  CHECK(((7 - 1) / 2 + 1) == 4);
}

VIDFAB_TEST(audio_elementwise) {
  const size_t n = 5000;
  std::vector<float> a = make_data(n, 3u, 2.0f);
  const std::vector<float> b = make_data(n, 4u, 2.0f);

  std::vector<float> want_add(n);
  for (size_t i = 0; i < n; ++i) want_add[i] = a[i] + b[i];
  DeviceBuffer<float> da = to_device(a);
  DeviceBuffer<float> db = to_device(b);
  vidfab::cuda::launch_add_inplace(da.get(), db.get(), n, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_add, to_host(da), 1e-6, "add_inplace");

  std::vector<float> want_scale(n);
  // Multiply by the reciprocal, exactly as the kernel does: `x / 3.0f` differs
  // in the last bit and the clamp check below runs at zero tolerance.
  for (size_t i = 0; i < n; ++i) want_scale[i] = want_add[i] * (1.0f / 3.0f);
  vidfab::cuda::launch_scale_inplace(da.get(), 1.0f / 3.0f, n, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_scale, to_host(da), 1e-6, "scale_inplace");

  std::vector<float> want_clamp(n);
  for (size_t i = 0; i < n; ++i) {
    want_clamp[i] = want_scale[i] < -1.0f ? -1.0f : (want_scale[i] > 1.0f ? 1.0f : want_scale[i]);
  }
  vidfab::cuda::launch_clamp_inplace(da.get(), -1.0f, 1.0f, n, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_clamp, to_host(da), 0.0, "clamp_inplace");

  // [batch, 1, frames] planar -> interleaved. Left is batch item 0
  // (docs/audio_vae_spec.md U1).
  const int frames = 11;
  std::vector<float> planar(static_cast<size_t>(2 * frames));
  for (int t = 0; t < frames; ++t) {
    planar[static_cast<size_t>(t)] = static_cast<float>(t);
    planar[static_cast<size_t>(frames + t)] = static_cast<float>(100 + t);
  }
  DeviceBuffer<float> dp = to_device(planar);
  DeviceBuffer<float> di(planar.size());
  vidfab::cuda::launch_interleave(dp.get(), di.get(), 2, frames, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = to_host(di);
  for (int t = 0; t < frames; ++t) {
    CHECK_NEAR(got[static_cast<size_t>(2 * t)], static_cast<double>(t), 0.0);
    CHECK_NEAR(got[static_cast<size_t>(2 * t + 1)], static_cast<double>(100 + t), 0.0);
  }
}

// --- end to end against the real checkpoint ---------------------------------

// Latents matching the NumPy reference's generator exactly, so the golden
// values below correspond to a known input.
std::vector<float> reference_latents(int batch, int channels, int len, uint32_t seed) {
  std::vector<float> v(static_cast<size_t>(batch) * channels * len);
  uint32_t s = seed;
  for (size_t i = 0; i < v.size(); ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<float>((s >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
  }
  return v;
}

std::string checkpoint_path() {
  if (const char* env = std::getenv("VIDFAB_AUDIO_VAE")) return env;
  return "weights/vae/minimax_h3_audio_vae_fp32.safetensors";
}

// Golden output for reference_latents(2, 32, 3, 12345), produced by a NumPy
// float64 transcription of the reference decode path — written from
// ref/FL2VA/audio_vae/*.py, not from this port. Sample index is into the
// interleaved stream, so even indices are batch item 0.
const int kGoldenIndex[] = {0,    1,    2,    3,    4,    5,    6,    7,
                            400,  401,  800,  801,  1200, 1201, 2000, 2001,
                            3000, 3001, 4000, 4001, 4796, 4797, 4798, 4799};
const float kGolden[] = {
    +0.06559666f, +0.00993082f, -0.06596538f, -0.01808799f, -0.09343581f, -0.04275811f,
    -0.06899163f, -0.07203731f, -0.09627534f, +0.10140902f, -0.08329765f, -0.00430612f,
    +0.02938782f, +0.01965548f, -0.00378015f, -0.15679068f, +0.06335449f, -0.01373321f,
    -0.03484774f, +0.03697970f, +0.14894537f, +0.04332416f, +0.14315795f, -0.00255305f,
};

VIDFAB_TEST(audio_decoder_checkpoint) {
  const std::string path = checkpoint_path();
  if (!std::filesystem::exists(path)) {
    std::fprintf(stderr, "  (no %s; skipping the end-to-end audio decode)\n", path.c_str());
    return;
  }

  size_t free_at_rest = 0;
  size_t total_memory = 0;
  VIDFAB_CUDA_CHECK(cudaMemGetInfo(&free_at_rest, &total_memory));

  vidfab::SafeTensors ckpt;
  ckpt.open(path);
  CHECK(ckpt.tensor_count() == 917);

  vidfab::vae::AudioDecoder decoder;
  decoder.load(ckpt);
  // 779 of the 917 tensors are on the decode path; the other 138 are the
  // encoder, its attention projection, and the two latent-statistics vectors.
  CHECK(decoder.weight_bytes() == 259672032u);
  CHECK(decoder.latents_mean().size() == 32);
  CHECK(decoder.latents_std().size() == 32);
  CHECK_NEAR(decoder.latents_mean()[0], -0.020211687, 1e-7);
  CHECK_NEAR(decoder.latents_std()[0], 1.6895524, 1e-6);

  const int latents = 3;
  const std::vector<float> z = reference_latents(2, 32, latents, 12345u);
  const vidfab::vae::DecodedAudio audio = decoder.decode(z.data(), latents);

  CHECK(audio.channels == 2);
  CHECK(audio.sample_rate == 32000);
  // 800 samples per latent, exactly: every upsample stage multiplies the length
  // by its rate and nothing trims.
  CHECK(audio.num_frames() == latents * 800);
  CHECK(audio.samples.size() == static_cast<size_t>(latents) * 800 * 2);

  std::vector<float> want(std::size(kGolden));
  std::vector<float> got(std::size(kGolden));
  for (size_t i = 0; i < std::size(kGolden); ++i) {
    want[i] = kGolden[i];
    got[i] = audio.samples[static_cast<size_t>(kGoldenIndex[i])];
  }
  CHECK_CLOSE_REL(want, got, 1e-3, 1e-2, "audio decode vs reference");
  {
    double worst_abs = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
      worst_abs = std::max(worst_abs, std::abs(static_cast<double>(want[i] - got[i])));
    }
    std::fprintf(stderr, "  worst golden deviation: %.3e absolute\n", worst_abs);
  }

  double sum_sq = 0.0;
  double peak = 0.0;
  bool all_finite = true;
  for (float v : audio.samples) {
    if (!std::isfinite(v)) all_finite = false;
    sum_sq += static_cast<double>(v) * v;
    peak = std::max(peak, std::abs(static_cast<double>(v)));
  }
  CHECK_MSG(all_finite, "audio decode produced a non-finite sample");
  CHECK_NEAR(std::sqrt(sum_sq / audio.samples.size()), 0.10619, 2e-3);
  CHECK_NEAR(peak, 0.4025816, 2e-3);

  // A constant latent is the cheapest guard against a padding bug: replicate
  // padding keeps every anti-alias window well defined, so nothing may go
  // non-finite and nothing may run away.
  {
    const std::vector<float> flat(static_cast<size_t>(2) * 32 * 5, 0.25f);
    const vidfab::vae::DecodedAudio quiet = decoder.decode(flat.data(), 5);
    CHECK(quiet.num_frames() == 5 * 800);
    bool finite = true;
    double worst = 0.0;
    for (float v : quiet.samples) {
      if (!std::isfinite(v)) finite = false;
      worst = std::max(worst, std::abs(static_cast<double>(v)));
    }
    CHECK_MSG(finite, "constant latent decoded to a non-finite sample");
    CHECK_MSG(worst <= 1.0, "constant latent decoded outside [-1, 1]: peak %.6f", worst);
  }

  // A ten-second decode, timed. This is the shape the pipeline actually runs.
  {
    const int ten_seconds = 405;
    const std::vector<float> big = reference_latents(2, 32, ten_seconds, 999u);

    const vidfab::vae::DecodedAudio warm = decoder.decode(big.data(), ten_seconds);
    CHECK(warm.num_frames() == ten_seconds * 800);

    // Measured against the free memory recorded before the checkpoint was
    // opened, so this covers weights and the activation pool together. The pool
    // grows to its final size on the first ten-second decode and is reused
    // afterwards, which is why the reading is taken after the warm-up run.
    size_t free_now = 0;
    VIDFAB_CUDA_CHECK(cudaMemGetInfo(&free_now, &total_memory));
    const double peak_mib = static_cast<double>(free_at_rest - free_now) / 1048576.0;

    cudaEvent_t start;
    cudaEvent_t stop;
    VIDFAB_CUDA_CHECK(cudaEventCreate(&start));
    VIDFAB_CUDA_CHECK(cudaEventCreate(&stop));
    VIDFAB_CUDA_CHECK(cudaEventRecord(start));
    const vidfab::vae::DecodedAudio timed = decoder.decode(big.data(), ten_seconds);
    VIDFAB_CUDA_CHECK(cudaEventRecord(stop));
    VIDFAB_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    VIDFAB_CUDA_CHECK(cudaEventDestroy(start));
    VIDFAB_CUDA_CHECK(cudaEventDestroy(stop));

    double peak10 = 0.0;
    bool finite = true;
    for (float v : timed.samples) {
      if (!std::isfinite(v)) finite = false;
      peak10 = std::max(peak10, std::abs(static_cast<double>(v)));
    }
    CHECK_MSG(finite, "10 s decode produced a non-finite sample");
    CHECK(peak10 <= 1.0);
    std::fprintf(stderr,
                 "  10 s decode (A=405, 324000 samples/channel): %.1f ms, %.2f MiB weights, "
                 "%.2f MiB peak device\n",
                 ms, static_cast<double>(decoder.weight_bytes()) / 1048576.0, peak_mib);
  }
}

}  // namespace
// `main` lives in tests/test_kernels.cu; this translation unit only registers.
