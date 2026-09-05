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

  const slopfab::vae::ViTConfig& config() const override { return config_; }

  void forward_windows(const float* z, int batch, int T, int H, int W,
                       std::vector<std::vector<float>>& out,
                       const size_t* slots) override {
    ++calls;
    seen_t = T;
    first_latent = z[0];
    const int out_h = H * config_.patch;
    const int out_w = W * config_.patch;
    const size_t plane_pixels = static_cast<size_t>(out_h) * out_w;
    const int planes = config_.out_channels * T * config_.patch_t;
    for (int b = 0; b < batch; ++b) {
      std::vector<float>& dst = out[slots[static_cast<size_t>(b)]];
      dst.resize(static_cast<size_t>(planes) * plane_pixels);
      for (int p = 0; p < planes; ++p) {
        const float value = static_cast<float>(p - 6) * 0.01f;
        std::fill_n(dst.data() + static_cast<size_t>(p) * plane_pixels,
                    plane_pixels, value);
      }
    }
  }

  void denormalize_latents(const float* normalized, int channels,
                           uint64_t voxels, const std::vector<float>& mean,
                           const std::vector<float>& std_dev,
                           std::vector<float>& output) override {
    output.resize(static_cast<size_t>(channels) * voxels);
    for (int c = 0; c < channels; ++c) {
      for (uint64_t i = 0; i < voxels; ++i) {
        const size_t at = static_cast<size_t>(c) * voxels + static_cast<size_t>(i);
        output[at] = normalized[at] * std_dev[static_cast<size_t>(c)] +
                     mean[static_cast<size_t>(c)];
      }
    }
  }

  void release_host_registrations() override { released = true; }

  slopfab::vae::ViTConfig config_;
  int calls = 0;
  int seen_t = 0;
  float first_latent = 0.0f;
  bool released = false;
};

}  // namespace

SLOPFAB_TEST(still_decode_uses_one_token_and_first_retained_phase) {
  StillBackend backend;
  slopfab::vae::DecodeSchedule schedule;
  schedule.tile_size = 4;
  schedule.tile_overlap_min = 2;
  const int h = 2, w = 3;
  std::vector<float> latent(static_cast<size_t>(h) * w, 0.25f);

  const slopfab::vae::DecodedVideo image = slopfab::vae::decode_still_image(
      backend, latent.data(), h, w, {1.0f}, {2.0f}, schedule);

  CHECK(backend.calls == 1);
  CHECK(backend.seen_t == 1);
  CHECK_NEAR(backend.first_latent, 1.5, 0.0);
  CHECK(backend.released);
  CHECK(image.channels == 3);
  CHECK(image.frames == 1);
  CHECK(image.height == 4);
  CHECK(image.width == 6);
  CHECK(image.data.size() == static_cast<size_t>(3 * 4 * 6));

  // Phase 3 is selected from [channel][four temporal phases]. The fake emits
  // -0.03, +0.01 and +0.05 for that phase in channels R, G and B.
  const float expected[3] = {
      -0.03f * 0.229f + 0.485f,
       0.01f * 0.224f + 0.456f,
       0.05f * 0.225f + 0.406f,
  };
  const size_t plane = static_cast<size_t>(image.height) * image.width;
  for (int c = 0; c < 3; ++c) {
    for (size_t i = 0; i < plane; ++i) {
      CHECK_NEAR(image.data[static_cast<size_t>(c) * plane + i], expected[c], 1e-7);
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
    (void)slopfab::vae::decode_still_image(backend, latent.data(), 2, 2,
                                          {0.0f}, {1.0f}, schedule);
  } catch (...) {
    rejected = true;
  }
  CHECK(rejected);
}
