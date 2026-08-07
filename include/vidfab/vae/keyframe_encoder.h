#pragma once

#include <cstdint>
#include <vector>

#include "vidfab/image.h"
#include "vidfab/safetensors.h"

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

struct EncoderWeightSummary {
  size_t tensors = 0;
  size_t bytes = 0;
};

// Validates the exact six-level H3 image encoder and quant_conv checkpoint
// graph. Decoder tensors may coexist in the archive and are ignored.
EncoderWeightSummary validate_keyframe_encoder_weights(const SafeTensors& checkpoint);

// Patchifies one normalized [24,H,W] image latent into Ref2VA rows
// [H/2*W/2, 96]. This is the single-frame specialization of the transformer's
// 1x2x2 video patch embedding layout.
std::vector<float> patchify_keyframe_latents(const float* latents, int height, int width);

}  // namespace vidfab::vae
