#include <cmath>
#include <vector>

#include "harness.h"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/keyframe_encoder.cuh"

using vidfab::cuda::DeviceBuffer;

VIDFAB_TEST(keyframe_conv3d_single_frame_causal_slice) {
  // Cin=1,Cout=1,K=3. Only the last temporal plane may contribute.
  std::vector<float> x = {1, 2, 3, 4, 5, 6};
  std::vector<__half> w(27, __float2half(100.0f));
  for (int i = 18; i < 27; ++i) w[i] = __float2half(1.0f);
  std::vector<__half> bias = {__float2half(0.5f)};
  DeviceBuffer<float> dx(x.size()), dy(x.size());
  DeviceBuffer<__half> dw(w.size()), db(1);
  dx.copy_from_host(x.data(), x.size());
  dw.copy_from_host(w.data(), w.size());
  db.copy_from_host(bias.data(), 1);
  vidfab::cuda::launch_keyframe_conv3d(dx.get(), dw.get(), db.get(), dy.get(), 1, 1, 2, 3,
                                       3, 1, true, false, nullptr);
  std::vector<float> y(6);
  dy.copy_to_host(y.data(), y.size());
  // Reflect-padded 3x3 neighborhood at top-left:
  // 5,4,5 / 2,1,2 / 5,4,5.
  CHECK_NEAR(y[0], 33.5, 1e-5);
}

VIDFAB_TEST(keyframe_groupnorm_silu_matches_cpu) {
  std::vector<float> x = {1, 2, 3, 4, 10, 12, 14, 16};
  std::vector<__half> weight(2, __float2half(1.0f)), bias(2, __float2half(0.0f));
  DeviceBuffer<float> dx(8), dy(8);
  DeviceBuffer<__half> dw(2), db(2);
  dx.copy_from_host(x.data(), 8);
  dw.copy_from_host(weight.data(), 2);
  db.copy_from_host(bias.data(), 2);
  vidfab::cuda::launch_keyframe_groupnorm_silu(dx.get(), dw.get(), db.get(), dy.get(), 2, 2, 2,
                                               1, 1e-6f, nullptr);
  std::vector<float> y(8);
  dy.copy_to_host(y.data(), 8);
  float mean = 0.0f;
  for (float v : x) mean += v;
  mean /= 8.0f;
  float var = 0.0f;
  for (float v : x) var += (v - mean) * (v - mean);
  var /= 8.0f;
  for (int i = 0; i < 8; ++i) {
    const float n = (x[i] - mean) / std::sqrt(var + 1e-6f);
    CHECK_NEAR(y[i], n / (1.0f + std::exp(-n)), 2e-5);
  }
}
