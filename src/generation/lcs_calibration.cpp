#include "slopfab/lcs.h"
#include "slopfab/vae/keyframe_encoder.h"
#include "helpers.h"
#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/keyframe_encoder.h"
#endif
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace slopfab {
namespace {
RGBImage solid(int size, float h, float saturation, float value) {
  const float x = h * 6;
  const int sector = int(x) % 6;
  const float f = x - std::floor(x);
  const float p = value * (1 - saturation), q = value * (1 - saturation * f),
              t = value * (1 - saturation * (1 - f));
  const float rgb[6][3] = {{value, t, p}, {q, value, p}, {p, value, t},
                           {p, q, value}, {t, p, value}, {value, p, q}};
  RGBImage image;
  image.width = image.height = size;
  image.pixels.resize(size_t(size) * size * 3);
  for (size_t i = 0; i < image.pixels.size(); ++i)
    image.pixels[i] = uint8_t(std::lround(rgb[sector][i % 3] * 255));
  return image;
}

RGBImage blurred(const RGBImage& image) {
  RGBImage out = image;
  for (int y = 0; y < image.height; ++y)
    for (int x = 0; x < image.width; ++x) {
      int sum = 0;
      for (int dy = -3; dy <= 3; ++dy)
        for (int dx = -3; dx <= 3; ++dx)
          sum += image.pixels[(size_t(std::clamp(y + dy, 0, image.height - 1)) * image.width +
                               std::clamp(x + dx, 0, image.width - 1)) *
                              3];
      for (int c = 0; c < 3; ++c)
        out.pixels[(size_t(y) * image.width + x) * 3 + c] = uint8_t(sum / 49);
    }
  return out;
}
}

bool calibrate_lcs(const std::string& vae_path, DeviceBackend backend, bool portable,
                   const std::string& output, int colors, int size,
                   const std::function<bool(int, int)>& progress) {
  if (vae_path.empty() || output.empty() || colors < 8 || colors > 4096 || size < 64 ||
      size > 512 || size % 32)
    throw std::invalid_argument(
        "LCS calibration needs a VAE, output, 8..4096 colors and 64..512 image size divisible by 32");
  if (std::filesystem::weakly_canonical(std::filesystem::u8path(output)) ==
      std::filesystem::weakly_canonical(std::filesystem::u8path(vae_path)))
    throw std::invalid_argument("LCS calibration cannot overwrite its VAE");
  if (backend != DeviceBackend::kCuda && backend != DeviceBackend::kVulkan)
    throw std::invalid_argument("LCS calibration: invalid backend");
  const int total = colors + 8 + 8;
  if (progress && !progress(0, total))
    return false;
  SafeTensors checkpoint;
  checkpoint.open(vae_path);
  vae::validate_keyframe_encoder_weights(checkpoint);
  const auto fingerprint = lcs_vae_fingerprint(vae_path);
  const auto mean = generation::read_stat(checkpoint, "latents_mean", 24);
  const auto stddev = generation::read_stat(checkpoint, "latents_std", 24);
  const auto run = [&](auto& encoder) {
    int done = 0;
    const auto encode = [&](const RGBImage& image) {
      const auto pixels = vae::prepare_keyframe_pixels(image);
      const auto moments = encoder.encode_moments(pixels.data(), size, size);
      // Posterior mean only: deterministic calibration without encoder sampling noise.
      return vae::patchify_keyframe_latents(moments.data(), size / 16, size / 16);
    };
    const auto average = [](const std::vector<float>& patches, std::vector<float>& out) {
      const size_t rows = patches.size() / LcsData::dim;
      for (int i = 0; i < LcsData::dim; ++i) {
        double sum = 0;
        for (size_t r = 0; r < rows; ++r)
          sum += patches[r * LcsData::dim + i];
        out.push_back(float(sum / double(rows)));
      }
    };
    std::vector<float> samples, anchors, differences;
    for (int i = 0; i < colors; ++i) {
      const float h = std::fmod(float(i) * .61803398875f, 1.f);
      average(encode(solid(size, h, .3f + .7f * float((i * 73) % 100) / 100,
                           .3f + .7f * float((i * 47) % 100) / 100)),
              samples);
      if (progress && !progress(++done, total))
        return false;
    }
    constexpr float hues[]{0, 4.f / 6, 2.f / 6, 5.f / 6, 3.f / 6, 1.f / 6, 0, 0};
    for (int i = 0; i < 8; ++i) {
      average(encode(solid(size, hues[i], i < 6 ? 1.f : 0.f, i == 6 ? 0.f : 1.f)), anchors);
      if (progress && !progress(++done, total))
        return false;
    }
    LcsData data = fit_lcs(samples, anchors);
    data.latent_mean = mean;
    data.latent_std = stddev;
    data.vae_sha256 = fingerprint;
    // Four deterministic grayscale stimuli, each paired with a blurred version.
    for (int stimulus = 0; stimulus < 4; ++stimulus) {
      RGBImage image = solid(size, 0, 0, .5f);
      uint32_t random = uint32_t(17 + stimulus);
      for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
          random = 1664525u * random + 1013904223u;
          const float v =
              stimulus == 0
                  ? float(random >> 24)
                  : 127.5f + 100.f * std::sin(float(x + y * .37f) * (1.f + stimulus) * .17f);
          for (int c = 0; c < 3; ++c)
            image.pixels[(size_t(y) * size + x) * 3 + c] = uint8_t(std::clamp(v, 0.f, 255.f));
        }
      auto sharp = encode(image);
      if (progress && !progress(++done, total))
        return false;
      auto blur = encode(blurred(image));
      if (progress && !progress(++done, total))
        return false;
      for (size_t i = 0; i < sharp.size(); ++i)
        differences.push_back(sharp[i] - blur[i]);
    }
    fit_lcs_sharpness(data, differences);
    data.save(output);
    return true;
  };
  if (backend == DeviceBackend::kCuda) {
#if SLOPFAB_WITH_CUDA
    vae::KeyframeEncoder encoder(checkpoint);
    return run(encoder);
#else
    throw std::invalid_argument("LCS calibration: CUDA not compiled in");
#endif
  }
#if SLOPFAB_WITH_VULKAN
  auto device = generation::create_vulkan_inference_device(portable);
  auto encoder = vulkan::KeyframeEncoder::create(device);
  encoder.load(checkpoint);
  return run(encoder);
#else
  (void)portable;
  throw std::invalid_argument("LCS calibration: Vulkan not compiled in");
#endif
}
} // namespace slopfab
