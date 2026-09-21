#pragma once

#include <cstdint>
#include <vector>
#include <memory>

#include "slopfab/image.h"
#include "slopfab/safetensors.h"

namespace slopfab::vae {

// Reference-compatible input to the H3 video VAE encoder, planar [3,H,W].
std::vector<float> prepare_keyframe_pixels(const RGBImage& image);

// Converts encoder moments [mean(24), logvar(24)] at every latent voxel into
// normalized H3 conditioning latents. `normal` is the standard-normal draw
// (the production caller uses torch-compatible seed 42 generation).
std::vector<float> sample_keyframe_latents(const float* moments, const float* normal, int height,
                                           int width, const std::vector<float>& latents_mean,
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

// PyTorch CPUGeneratorImpl MT19937 + contiguous-float normal_fill, seeded 42.
std::vector<float> torch_cpu_normal_seed42(size_t count);

#ifdef SLOPFAB_WITH_CUDA
class KeyframeEncoder {
public:
  explicit KeyframeEncoder(const SafeTensors& checkpoint);
  ~KeyframeEncoder();
  KeyframeEncoder(KeyframeEncoder&&) noexcept;
  KeyframeEncoder& operator=(KeyframeEncoder&&) noexcept;
  KeyframeEncoder(const KeyframeEncoder&) = delete;
  KeyframeEncoder& operator=(const KeyframeEncoder&) = delete;

  // Runs the genuine single-frame encoder and quant_conv graph. Input is
  // ImageNet-normalized planar [3,H,W], output [48,H/16,W/16].
  std::vector<float> encode_moments(const float* pixels, int height, int width);

  // One causal temporal clip, [3,T,H,W] -> [48,ceil(T/4),H/16,W/16].
  // Used by the 17-frame chunked reference video path and parity probes.
  // The low-level API defaults to the FP32 authority. Generation explicitly
  // selects mixed precision: FP16 activations with FP32 accumulation/norms.
  std::vector<float> encode_temporal_moments(const float* pixels, int frames, int height, int width,
                                             bool mixed_precision = false);
  // Normalized reference frames at 24 fps, already snapped to 17*n+5.
  std::vector<float> encode_reference_video(const std::vector<RGBImage>& frames,
                                            int encoding_frames,
                                            const std::vector<float>& latents_mean,
                                            const std::vector<float>& latents_std,
                                            bool mixed_precision = false);

  // Complete Ref2VA conditioning path. `normal` contains 24*(H/16)*(W/16)
  // standard-normal values in channel-major order.
  std::vector<float> encode_condition_rows(const RGBImage& image, const float* normal,
                                           const std::vector<float>& latents_mean,
                                           const std::vector<float>& latents_std);

  // Production seed-42 form used by Ref2VA.
  std::vector<float> encode_reference_image(const RGBImage& image,
                                            const std::vector<float>& latents_mean,
                                            const std::vector<float>& latents_std);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif

} // namespace slopfab::vae
