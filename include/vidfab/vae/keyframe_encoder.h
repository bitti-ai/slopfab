#pragma once

#include <cstdint>
#include <vector>

#include "vidfab/image.h"

namespace vidfab::vae {

// Reference-compatible input to the H3 video VAE encoder, planar [3,H,W].
std::vector<float> prepare_keyframe_pixels(const RGBImage& image);

// Converts encoder moments [mean(24), logvar(24)] at every latent voxel into
// normalized H3 conditioning latents. `normal` is the standard-normal draw
// (the production caller uses torch-compatible seed 42 generation).
std::vector<float> sample_keyframe_latents(const float* moments, const float* normal,
                                           int height, int width,
                                           const std::vector<float>& latents_mean,
                                           const std::vector<float>& latents_std);

}  // namespace vidfab::vae
