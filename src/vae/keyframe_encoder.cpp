#include "vidfab/vae/keyframe_encoder.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "vidfab/dtype.h"

namespace vidfab::vae {

std::vector<float> prepare_keyframe_pixels(const RGBImage& image) {
  if (image.width <= 0 || image.height <= 0 ||
      image.pixels.size() != static_cast<size_t>(image.width) * image.height * 3) {
    throw std::runtime_error("keyframe encoder: invalid RGB image");
  }
  constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
  constexpr float std_dev[3] = {0.229f, 0.224f, 0.225f};
  const size_t plane = static_cast<size_t>(image.width) * image.height;
  std::vector<float> out(3 * plane);
  for (size_t i = 0; i < plane; ++i) {
    for (int c = 0; c < 3; ++c) {
      out[static_cast<size_t>(c) * plane + i] =
          (static_cast<float>(image.pixels[3 * i + c]) / 255.0f - mean[c]) / std_dev[c];
    }
  }
  return out;
}

std::vector<float> sample_keyframe_latents(const float* moments, const float* normal,
                                           int height, int width,
                                           const std::vector<float>& latents_mean,
                                           const std::vector<float>& latents_std) {
  constexpr int channels = 24;
  if (!moments || !normal || height <= 0 || width <= 0 ||
      latents_mean.size() != channels || latents_std.size() != channels) {
    throw std::runtime_error("keyframe encoder: invalid posterior inputs");
  }
  const size_t voxels = static_cast<size_t>(height) * width;
  std::vector<float> out(static_cast<size_t>(channels) * voxels);
  for (int c = 0; c < channels; ++c) {
    if (!(latents_std[c] > 0.0f)) throw std::runtime_error("keyframe encoder: invalid latent std");
    for (size_t i = 0; i < voxels; ++i) {
      const size_t j = static_cast<size_t>(c) * voxels + i;
      // DiagonalGaussianDistribution clamps logvar to [-30,20]. The reference
      // then casts the sampled posterior through fp16 before normalization.
      const float logvar = std::clamp(moments[static_cast<size_t>(channels) * voxels + j],
                                      -30.0f, 20.0f);
      const float sampled = moments[j] + std::exp(0.5f * logvar) * normal[j];
      const float rounded = f16_to_f32(f32_to_f16(sampled));
      out[j] = (rounded - latents_mean[c]) / latents_std[c];
    }
  }
  return out;
}

}  // namespace vidfab::vae
