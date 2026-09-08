#include <vector>

#include "harness.h"
#include "../tools/stillprobe_helpers.h"

namespace {

class TemporalInvariantBackend final : public slopfab::vae::VideoVaeWindowBackend {
 public:
  TemporalInvariantBackend() {
    config_.in_channels = 3;
    config_.patch = 2;
  }

  const slopfab::vae::ViTConfig& config() const override { return config_; }

  void forward_windows(const float* latent, int batch, int frames, int height, int width,
                       std::vector<std::vector<float>>& output, const size_t* slots) override {
    seen_frames = frames;
    const size_t pixels = static_cast<size_t>(height) * width;
    const size_t output_pixels = pixels * 4;
    for (int tile = 0; tile < batch; ++tile) {
      auto& values = output[slots[tile]];
      values.resize(3 * frames * 4 * output_pixels);
      for (int channel = 0; channel < 3; ++channel) {
        for (int frame = 0; frame < frames; ++frame) {
          for (size_t pixel = 0; pixel < pixels; ++pixel) {
            CHECK(latent[((tile * 3 + channel) * frames + frame) * pixels + pixel] ==
                  static_cast<float>(channel + 1));
          }
        }
        for (int frame = 0; frame < frames * 4; ++frame) {
          for (size_t pixel = 0; pixel < output_pixels; ++pixel) {
            values[(static_cast<size_t>(channel) * frames * 4 + frame) * output_pixels + pixel] =
                channel * 0.1f + (frame % 4) * 0.02f + static_cast<float>(pixel) * 0.001f;
          }
        }
      }
    }
  }

  void denormalize_latents(const float* input, int channels, uint64_t voxels,
                           const std::vector<float>& mean, const std::vector<float>& stddev,
                           std::vector<float>& output) override {
    output.resize(channels * voxels);
    for (int channel = 0; channel < channels; ++channel) {
      for (uint64_t pixel = 0; pixel < voxels; ++pixel) {
        const size_t index = channel * voxels + pixel;
        output[index] = input[index] * stddev[channel] + mean[channel];
      }
    }
  }

  void release_host_registrations() override {}

  int seen_frames = 0;
  slopfab::vae::ViTConfig config_;
};

}

SLOPFAB_TEST(stillprobe_repeats_each_channel_without_interleaving) {
  const std::vector<float> latent = {1, 2, 3, 4, 5, 6};
  const auto repeated = slopfab::probe::repeat_still_latent(latent, 3, 7);
  CHECK(repeated.size() == 42);
  for (int channel = 0; channel < 3; ++channel) {
    for (int frame = 0; frame < 7; ++frame) {
      for (int pixel = 0; pixel < 2; ++pixel) {
        CHECK(repeated[(channel * 7 + frame) * 2 + pixel] == latent[channel * 2 + pixel]);
      }
    }
  }
  CHECK(slopfab::test::throws([] { slopfab::probe::repeat_still_latent({1, 2}, 0, 7); }));
  CHECK(slopfab::test::throws([] { slopfab::probe::repeat_still_latent({1, 2}, 3, 7); }));
  CHECK(slopfab::test::throws([] { slopfab::probe::repeat_still_latent({1, 2}, 2, 0); }));
}

SLOPFAB_TEST(stillprobe_extracts_planar_first_frame) {
  slopfab::vae::DecodedVideo video;
  video.frames = 22;
  video.height = 2;
  video.width = 3;
  video.data.resize(3 * 22 * 6);
  for (size_t index = 0; index < video.data.size(); ++index)
    video.data[index] = static_cast<float>(index);
  const auto image = slopfab::probe::first_frame(video);
  CHECK(image.frames == 1);
  CHECK(image.height == 2);
  CHECK(image.width == 3);
  CHECK(image.data.size() == 18);
  for (int channel = 0; channel < 3; ++channel) {
    for (int pixel = 0; pixel < 6; ++pixel)
      CHECK(image.data[channel * 6 + pixel] == video.data[channel * 22 * 6 + pixel]);
  }
  CHECK(slopfab::test::throws([] {
    slopfab::vae::DecodedVideo invalid;
    invalid.frames = 1;
    invalid.height = 2;
    invalid.width = 3;
    invalid.data.resize(17);
    slopfab::probe::first_frame(invalid);
  }));
}

SLOPFAB_TEST(stillprobe_matches_video_phase_channels_and_spatial_blending) {
  TemporalInvariantBackend backend;
  slopfab::vae::DecodeSchedule schedule;
  schedule.tile_size = 4;
  schedule.tile_overlap_min = 2;
  const std::vector<float> mean = {0, 0, 0};
  const std::vector<float> stddev = {1, 1, 1};
  std::vector<float> latent(3 * 4 * 4);
  for (int channel = 0; channel < 3; ++channel)
    std::fill_n(latent.data() + channel * 16, 16, static_cast<float>(channel + 1));
  const auto original = slopfab::vae::decode_still_image(
      backend, latent.data(), 4, 4, mean, stddev, schedule);
  CHECK(backend.seen_frames == 7);
  const auto repeated = slopfab::probe::repeat_still_latent(latent, 3, 7);
  const auto video = slopfab::vae::decode_video(
      backend, repeated.data(), 7, 4, 4, mean, stddev, schedule);
  CHECK(backend.seen_frames == 7);
  CHECK(video.frames == 22);
  const auto extracted = slopfab::probe::first_frame(video);
  CHECK(original.data == extracted.data);
}
