#include <cmath>
#include <vector>
#include "harness.h"
#include "slopfab/cuda/reference_encoder.cuh"

namespace {
int reflect(int p, int size) {
  while (p < 0 || p >= size) p = p < 0 ? -p : 2 * size - p - 2;
  return p;
}
}

SLOPFAB_TEST(reference_convolution_precision_and_pool_reuse) {
  using namespace slopfab::cuda;
  Stream stream;
  ReferenceEncoderOps ops(stream.get());
  constexpr int ci = 2, co = 3, frames = 5, height = 4, width = 6;
  std::vector<float> input(ci * frames * height * width);
  std::vector<__half> half_input(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    half_input[i] = __float2half_rn(std::sin(float(i) * .13f));
    input[i] = __half2float(half_input[i]);
  }
  auto x = ops.allocate<float>(input.size());
  auto hx = ops.allocate<__half>(input.size());
  x.copy_from_host(input.data(), input.size(), stream.get());
  hx.copy_from_host(half_input.data(), half_input.size(), stream.get());
  // Exercise both strides, causal left padding, reflected spatial boundaries,
  // channel changes and repeated leases while both inputs remain live.
  for (int repeat = 0; repeat < 2; ++repeat) for (int k : {1, 3})
    for (int ss : {1, 2}) for (int ts : {1, 2}) for (bool down : {false, true}) {
      std::vector<__half> weight(co * ci * k * k * k), bias(co);
      for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = __float2half_rn(std::cos(float(i) * .17f) * .03f);
      for (int c = 0; c < co; ++c) bias[c] = __float2half_rn(.01f * c);
      DeviceBuffer<__half> w(weight.size()), b(bias.size());
      w.copy_from_host(weight.data(), weight.size(), stream.get());
      b.copy_from_host(bias.data(), bias.size(), stream.get());
      auto y = ops.conv3d(x.get(), w.get(), b.get(), ci, co, frames, height, width, k, ss, ts, down);
      auto hy = ops.conv3d(hx.get(), w.get(), b.get(), ci, co, frames, height, width, k, ss, ts, down);
      std::vector<float> actual(y.size());
      std::vector<__half> half_actual(hy.size());
      y.copy_to_host(actual.data(), actual.size(), stream.get());
      hy.copy_to_host(half_actual.data(), half_actual.size(), stream.get());
      stream.synchronize();
      const int oh = height / ss, ow = width / ss, ot = (frames - 1) / ts + 1;
      for (int c = 0; c < co; ++c) for (int t = 0; t < ot; ++t)
        for (int yy = 0; yy < oh; ++yy) for (int xx = 0; xx < ow; ++xx) {
          double expected = 0;
          for (int ic = 0; ic < ci; ++ic) for (int kt = 0; kt < k; ++kt)
            for (int ky = 0; ky < k; ++ky) for (int kx = 0; kx < k; ++kx) {
              const int it = t * ts + kt - k + 1;
              if (it < 0) continue;
              const int iy = reflect(yy * ss + ky - (down ? 0 : k / 2), height);
              const int ix = reflect(xx * ss + kx - (down ? 0 : k / 2), width);
              const int wi = (((c * ci + ic) * k + kt) * k + ky) * k + kx;
              expected += double(input[((ic * frames + it) * height + iy) * width + ix]) * __half2float(weight[wi]);
            }
          expected += __half2float(bias[c]);
          const int i = ((c * ot + t) * oh + yy) * ow + xx;
          CHECK_NEAR(actual[i], expected, 2e-6);
          CHECK_NEAR(__half2float(half_actual[i]), expected, 5e-4);
        }
    }
}

SLOPFAB_TEST(reference_half_groupnorm_and_residual) {
  using namespace slopfab::cuda;
  Stream stream;
  ReferenceEncoderOps ops(stream.get());
  constexpr int channels = 32, frames = 2, height = 4, width = 6;
  std::vector<__half> input(channels * frames * height * width), weight(channels), bias(channels);
  for (size_t i = 0; i < input.size(); ++i) input[i] = __float2half_rn(std::sin(float(i) * .11f));
  for (int c = 0; c < channels; ++c) { weight[c] = __float2half_rn(.8f); bias[c] = __float2half_rn(.1f); }
  auto x = ops.allocate<__half>(input.size());
  auto y = ops.allocate<__half>(input.size());
  DeviceBuffer<__half> w(channels), b(channels);
  x.copy_from_host(input.data(), input.size(), stream.get());
  w.copy_from_host(weight.data(), weight.size(), stream.get());
  b.copy_from_host(bias.data(), bias.size(), stream.get());
  reference_groupnorm(x.get(), w.get(), b.get(), y.get(), channels, frames, height, width, stream.get());
  ops.add(y.get(), x.get(), y.size());
  std::vector<__half> actual(y.size());
  y.copy_to_host(actual.data(), actual.size(), stream.get());
  stream.synchronize();
  constexpr int plane = height * width;
  for (int c = 0; c < channels; ++c) for (int t = 0; t < frames; ++t) {
    double mean = 0, variance = 0;
    const int offset = (c * frames + t) * plane;
    for (int i = 0; i < plane; ++i) mean += __half2float(input[offset + i]) / double(plane);
    for (int i = 0; i < plane; ++i) variance += std::pow(__half2float(input[offset + i]) - mean, 2) / plane;
    for (int i = 0; i < plane; ++i) {
      const double v = (__half2float(input[offset + i]) - mean) / std::sqrt(variance + 1e-6) * __half2float(weight[c]) + __half2float(bias[c]);
      CHECK_NEAR(__half2float(actual[offset + i]), v / (1 + std::exp(-v)) + __half2float(input[offset + i]), .002);
    }
  }
}
