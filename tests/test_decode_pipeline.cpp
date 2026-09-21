// Host-only contract tests for the dedicated still-image VAE scheduler.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "harness.h"
#include "slopfab/vae/vit_decoder.h"

namespace {

class StillBackend final : public slopfab::vae::VideoVaeWindowBackend {
public:
  StillBackend() {
    config_.in_channels = 1;
    config_.out_channels = 3;
    config_.patch = 2;
    config_.patch_t = 4;
  }

  const slopfab::vae::ViTConfig& config() const override {
    return config_;
  }

  void forward_windows(const float* z, int batch, int T, int H, int W,
                       std::vector<std::vector<float>>& out, const size_t* slots) override {
    ++calls;
    seen_t = T;
    first_latent = z[0];
    seen_latents.assign(z, z + static_cast<size_t>(batch) * config_.in_channels * T * H * W);
    const int out_h = H * config_.patch;
    const int out_w = W * config_.patch;
    const size_t plane_pixels = static_cast<size_t>(out_h) * out_w;
    const int planes = config_.out_channels * T * config_.patch_t;
    for (int b = 0; b < batch; ++b) {
      std::vector<float>& dst = out[slots[static_cast<size_t>(b)]];
      dst.resize(static_cast<size_t>(planes) * plane_pixels);
      for (int p = 0; p < planes; ++p) {
        const float value =
            static_cast<float>(p - 6) * 0.01f +
            (spatial_values ? static_cast<float>(slots[b]) * 0.1f + calls * 0.02f : 0.0f);
        std::fill_n(dst.data() + static_cast<size_t>(p) * plane_pixels, plane_pixels, value);
      }
    }
  }

  void denormalize_latents(const float* normalized, int channels, uint64_t voxels,
                           const std::vector<float>& mean, const std::vector<float>& std_dev,
                           std::vector<float>& output) override {
    output.resize(static_cast<size_t>(channels) * voxels);
    for (int c = 0; c < channels; ++c) {
      for (uint64_t i = 0; i < voxels; ++i) {
        const size_t at = static_cast<size_t>(c) * voxels + static_cast<size_t>(i);
        output[at] =
            normalized[at] * std_dev[static_cast<size_t>(c)] + mean[static_cast<size_t>(c)];
      }
    }
  }

  void release_host_registrations() override {
    released = true;
  }

  slopfab::vae::ViTConfig config_;
  int calls = 0;
  int seen_t = 0;
  float first_latent = 0.0f;
  std::vector<float> seen_latents;
  bool released = false;
  bool spatial_values = false;
};

} // namespace

SLOPFAB_TEST(still_decode_uses_seven_tokens_and_first_retained_phase) {
  StillBackend backend;
  slopfab::vae::DecodeSchedule schedule;
  schedule.tile_size = 4;
  schedule.tile_overlap_min = 2;
  const int h = 2, w = 3;
  std::vector<float> latent(static_cast<size_t>(h) * w, 0.25f);

  const slopfab::vae::DecodedVideo image =
      slopfab::vae::decode_still_image(backend, latent.data(), h, w, {1.0f}, {2.0f}, schedule);

  CHECK(backend.calls == 1);
  CHECK(backend.seen_t == 7);
  CHECK_NEAR(backend.first_latent, 1.5, 0.0);
  CHECK(backend.released);
  CHECK(image.channels == 3);
  CHECK(image.frames == 1);
  CHECK(image.height == 4);
  CHECK(image.width == 6);
  CHECK(image.data.size() == static_cast<size_t>(3 * 4 * 6));

  const float expected[3] = {
      -0.03f * 0.229f + 0.485f,
      0.25f * 0.224f + 0.456f,
      0.53f * 0.225f + 0.406f,
  };
  const size_t plane = static_cast<size_t>(image.height) * image.width;
  for (int c = 0; c < 3; ++c) {
    for (size_t i = 0; i < plane; ++i) {
      CHECK_NEAR(image.data[static_cast<size_t>(c) * plane + i], expected[c], 1e-7);
    }
  }
}

SLOPFAB_TEST(still_decode_repeats_spatial_tiles_in_channel_time_order) {
  StillBackend backend;
  backend.config_.in_channels = 2;
  slopfab::vae::DecodeSchedule schedule;
  schedule.tile_size = 4;
  schedule.tile_overlap_min = 2;
  const std::vector<float> latent = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const std::vector<float> mean = {1, -2};
  const std::vector<float> stddev = {2, 3};
  const auto image =
      slopfab::vae::decode_still_image(backend, latent.data(), 2, 3, mean, stddev, schedule);
  CHECK(image.frames == 1);
  CHECK(backend.seen_t == 7);
  CHECK(backend.seen_latents.size() == 2 * 2 * 7 * 2 * 2);
  for (int tile = 0; tile < 2; ++tile) {
    for (int channel = 0; channel < 2; ++channel) {
      for (int frame = 0; frame < 7; ++frame) {
        for (int row = 0; row < 2; ++row) {
          for (int column = 0; column < 2; ++column) {
            const size_t source = (channel * 2 + row) * 3 + tile + column;
            const size_t target = (((tile * 2 + channel) * 7 + frame) * 2 + row) * 2 + column;
            CHECK(backend.seen_latents[target] == latent[source] * stddev[channel] + mean[channel]);
          }
        }
      }
    }
  }
}

SLOPFAB_TEST(still_decode_rejects_an_invalid_phase_contract) {
  StillBackend backend;
  slopfab::vae::DecodeSchedule schedule;
  schedule.frame_pre_padding = 4;
  const std::vector<float> latent(4, 0.0f);
  bool rejected = false;
  try {
    (void)slopfab::vae::decode_still_image(backend, latent.data(), 2, 2, {0.0f}, {1.0f}, schedule);
  } catch (...) {
    rejected = true;
  }
  CHECK(rejected);
}

SLOPFAB_TEST(decode_composes_diagonals_before_temporal_crossfade) {
  slopfab::vae::DecodeSchedule schedule;
  schedule.tile_size = 8;
  schedule.tile_overlap_min = 4;
  const float mean[] = {0.485f, 0.456f, 0.406f};
  const float stddev[] = {0.229f, 0.224f, 0.225f};
  const std::vector<float> latent(12 * 6 * 6, 0.0f);
  StillBackend backend;
  backend.spatial_values = true;
  const auto image =
      slopfab::vae::decode_still_image(backend, latent.data(), 6, 6, {0.0f}, {1.0f}, schedule);
  CHECK(backend.calls == 1 && backend.seen_t == 7 && backend.released);
  // Four tile constants at the overlap midpoint have equal weight. The old
  // above/left merge yielded 0.2 here instead of the correct 0.15.
  for (int c = 0; c < 3; ++c) {
    const float raw = (c * 28 + 3 - 6) * 0.01f + 0.15f + 0.02f;
    CHECK_NEAR(image.data[c * 144 + 6 * 12 + 6], raw * stddev[c] + mean[c], 1e-7);
  }
  backend.calls = 0;
  backend.released = false;
  const auto video =
      slopfab::vae::decode_video(backend, latent.data(), 12, 6, 6, {0.0f}, {1.0f}, schedule);
  CHECK(video.frames == 39 && video.height == 12 && video.width == 12);
  CHECK(backend.calls == 2 && backend.seen_t == 7 && backend.released);
  for (int c = 0; c < 3; ++c) {
    for (int f = 0; f < video.frames; ++f) {
      float temporal;
      if (f < 17) {
        temporal = (c * 28 + f + 3 - 6) * 0.01f + 0.02f;
      } else if (f < 22) {
        const int k = f - 17;
        const float previous = (c * 28 + 23 + k - 6) * 0.01f + 0.02f;
        const float current = (c * 28 + 3 + k - 6) * 0.01f + 0.04f;
        temporal = previous * (1.0f - k / 5.0f) + current * (k / 5.0f);
      } else {
        const int phase = f < 34 ? f - 17 + 3 : f - 34 + 23;
        temporal = (c * 28 + phase - 6) * 0.01f + 0.04f;
      }
      const size_t plane = (static_cast<size_t>(c) * video.frames + f) * 144;
      CHECK_NEAR(video.data[plane + 6 * 12 + 6], (temporal + 0.15f) * stddev[c] + mean[c], 2e-7);
      CHECK_NEAR(video.data[plane], temporal * stddev[c] + mean[c], 2e-7);
      CHECK_NEAR(video.data[plane + 143], (temporal + 0.3f) * stddev[c] + mean[c], 2e-7);
    }
  }
}
