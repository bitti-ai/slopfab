// The MiniMax H3 video VAE decoder.
//
// Despite the encoder being a conv pyramid, the decoder that shipped is pure
// transformer: 36 pre-norm blocks at width 2048, then a single Linear(2048 ->
// 3072) plus depth-to-space that performs all 16x spatial and 4x temporal
// upsampling at once. See docs/vae_decoder_spec.md.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vidfab/safetensors.h"

namespace vidfab::vae {

const std::vector<float>& default_video_latents_mean();
const std::vector<float>& default_video_latents_std();

struct ViTConfig {
  int num_layers = 36;
  int dim = 2048;
  int heads = 32;
  int head_dim = 64;      // heads * head_dim == dim
  int ffn_inner = 8192;   // w1 emits 2 * ffn_inner because the FFN is gated
  int in_channels = 24;   // latent channels
  int out_channels = 3;   // RGB
  int patch = 16;         // spatial upsample factor
  int patch_t = 4;        // temporal upsample factor
  int num_register = 4;   // learned register tokens
  int num_suffix = 5;     // register tokens + one literal zero token
  int rope_dim = 48;      // of head_dim; the remaining 16 dims pass through
  float rope_theta = 100.0f;
  float eps = 1e-5f;

  // Flat width of proj_out: out_channels * patch_t * patch * patch.
  int patch_dim() const { return out_channels * patch_t * patch * patch; }
};

// Temporal chunking and spatial tiling constants, derived in the reference
// from clip_length=17 / token_drop=3 / vae_ratio_t=4. See spec section 0.
struct DecodeSchedule {
  int tokens_chunk_size = 5;
  int token_overlap = 2;
  int frame_pre_padding = 3;
  int frame_overlap = 5;
  int chunk_dec = 20;  // tokens_chunk_size * 4
  int tile_size = 256;
  int tile_overlap_min = 64;
  bool tiling_enabled = true;

  int tokens_per_window() const { return tokens_chunk_size + token_overlap; }  // 7
};

// Decoded output in planar float RGB, values already de-normalised to [0,1].
struct DecodedVideo {
  int channels = 3;
  int frames = 0;
  int height = 0;
  int width = 0;
  std::vector<float> data;  // [3][frames][height][width], contiguous

  size_t frame_stride() const { return static_cast<size_t>(height) * width; }
  size_t plane_stride() const { return frame_stride() * frames; }
};

class ViTDecoder {
 public:
  ViTDecoder();
  ~ViTDecoder();
  ViTDecoder(const ViTDecoder&) = delete;
  ViTDecoder& operator=(const ViTDecoder&) = delete;

  // Uploads linear matrices in checkpoint-native fp16 for tensor-core GEMM
  // with fp32 accumulation. Norm, bias, scale, residual, attention and output
  // buffers stay fp32.
  void load(const SafeTensors& checkpoint, const ViTConfig& config = {});

  const ViTConfig& config() const;

  // Device memory currently held by weights, in bytes.
  size_t weight_bytes() const;

  // Runs the transformer on one already-de-normalised latent window.
  // `z` is [24, T, H, W] contiguous fp32 on the host; `out` receives
  // [3, T*4, H*16, W*16] contiguous fp32 in ImageNet-normalised space.
  void forward_window(const float* z, int T, int H, int W, std::vector<float>& out);

  // Runs equal-shape independent windows together through the token-wise
  // projections. Attention remains document-local: suffix tokens, RoPE and
  // softmax denominators are never shared between batch items. `z` is
  // [batch, C, T, H, W]; `out` receives one pixel tensor per batch item.
  void forward_windows(const float* z, int batch, int T, int H, int W,
                       std::vector<std::vector<float>>& out);

  // Full decode: latent de-normalisation, temporal chunking, spatial tiling,
  // cross-fade stitching and pixel de-normalisation.
  // `z_norm` is [24, T_lat, H_lat, W_lat] as produced by the diffusion model.
  DecodedVideo decode(const float* z_norm, int T_lat, int H_lat, int W_lat,
                      const std::vector<float>& latents_mean,
                      const std::vector<float>& latents_std,
                      const DecodeSchedule& schedule = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vidfab::vae
