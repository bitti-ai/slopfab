#include "harness.h"
#include "../src/seedvr2/image_ops.cuh"
#include <algorithm>

using namespace slopfab;
using namespace slopfab::seedvr2;

SLOPFAB_TEST(seedvr2_tile_preparation) {
  if (cuda::device_count() == 0) {
    SKIP_UNSUPPORTED_HARDWARE("CUDA required");
    return;
  }
  Runtime runtime;
  for (bool image : {false, true}) {
    const int channels = image ? 3 : 16;
    auto input = test::make_data(size_t(2) * 3 * 5 * channels, 51, 0.8f);
    cuda::DeviceBuffer<float> source(input.size());
    source.copy_from_host(input.data(), input.size());
    Tensor tile(2, 4, 4, channels);
    if (image)
      prepare_image_tile(source.get(), tile, 3, 5, 1, 3);
    else
      prepare_latent_tile(source.get(), tile, 3, 5, 1, 3);
    std::vector<float> expected(tile.size());
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
          for (int c = 0; c < channels; ++c) {
            float value =
                input[((size_t(z) * 3 + std::min(y + 1, 2)) * 5 + std::min(x + 3, 4)) * channels +
                      c];
            value = image ? 2.0f * value - 1.0f : value / 0.9152f;
            expected[((size_t(z) * 4 + y) * 4 + x) * channels + c] =
                __bfloat162float(__float2bfloat16(value));
          }
    CHECK_CLOSE(expected, runtime.download(tile), 0, "GPU tile extraction and conversion");
  }
}

SLOPFAB_TEST(seedvr2_tile_feathering) {
  if (cuda::device_count() == 0) {
    SKIP_UNSUPPORTED_HARDWARE("CUDA required");
    return;
  }
  Runtime runtime;
  // Four overlapping tiles, two frames, cropped right/bottom padding, and
  // different values in every tile exercise both moments and RGB output.
  for (int channels : {3, 32}) {
    const size_t count = size_t(2) * 5 * 5 * channels;
    cuda::DeviceBuffer<float> sum(count), coverage(25);
    sum.zero();
    coverage.zero();
    std::vector<float> expected(count, 0), weights(25, 0);
    for (int y0 : {0, 2})
      for (int x0 : {0, 2}) {
        auto values =
            test::make_data(size_t(2) * 4 * 4 * channels, uint32_t(5 + y0 * 2 + x0), 1.1f);
        auto tile = runtime.upload(values, 2, 4, 4, channels);
        accumulate_image_tile(tile, sum.get(), coverage.get(), 5, 5, 6, 6, y0, x0, 2);
        auto feather = [](int p, int lo) {
          float value = 1;
          if (lo > 0)
            value = std::min(value, float(p - lo + 1) / 2);
          if (lo + 4 < 6)
            value = std::min(value, float(lo + 4 - p) / 2);
          return value;
        };
        for (int y = y0; y < std::min(y0 + 4, 5); ++y)
          for (int x = x0; x < std::min(x0 + 4, 5); ++x) {
            const float weight = feather(y, y0) * feather(x, x0);
            weights[size_t(y) * 5 + x] += weight;
            for (int z = 0; z < 2; ++z)
              for (int c = 0; c < channels; ++c) {
                const size_t in = ((size_t(z) * 4 + y - y0) * 4 + x - x0) * channels + c;
                const size_t out = ((size_t(z) * 5 + y) * 5 + x) * channels + c;
                expected[out] += weight * __bfloat162float(__float2bfloat16(values[in]));
              }
          }
      }
    normalize_image_tiles(sum.get(), coverage.get(), count, 5, 5, channels, channels == 3);
    for (size_t i = 0; i < count; ++i) {
      expected[i] /= weights[(i / channels) % 25];
      if (channels == 3)
        expected[i] = expected[i] * 0.5f + 0.5f;
    }
    std::vector<float> actual(count);
    sum.copy_to_host(actual.data(), actual.size());
    CHECK_CLOSE(expected, actual, 0, "GPU feathering preserves CPU tile order and rounding");
  }
}
