#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "slopfab/device_tensor.h"
#include "slopfab/safetensors.h"

namespace slopfab {
struct LatentClip;
struct LatentUpscaleOptions {
  float scale = 2.0f;
  // Upstream's 32-frame segments with replicated context and overlap blending.
  // GroupNorm makes chunked inference an approximation to whole-clip inference.
  bool temporal_chunking = true;
  // Exact decoded pixel dimensions; both zero retains scale-based sizing.
  int target_width = 0;
  int target_height = 0;
};

inline constexpr const char* kDefaultLatentUpscaleModel =
    "weights/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors";

void validate_latent_upscale_options(const LatentUpscaleOptions& options);
void validate_latent_upscale_checkpoint(const SafeTensors& checkpoint);
// Latent dimensions, rounded to even sizes (32 decoded pixels), ties to even.
std::pair<int, int> latent_upscale_dimensions(int height, int width,
                                              const LatentUpscaleOptions& options = {});

// Normalized H3 sampler latents [24,T,H,W], as produced by unpatchify_video.
// Internally applies/reverses the companion node's additional normalization;
// callers must not pass raw VAE latents. Time is preserved. Loads/releases
// weights within the call; false progress callbacks throw UpscaleCancelled.
std::vector<float> upscale_latents(const std::vector<float>& input, int frames, int height,
                                   int width, const std::string& checkpoint, DeviceBackend backend,
                                   const LatentUpscaleOptions& options = {},
                                   const std::function<bool(int, int)>& progress = {});

// Upscale a completed archive/snapshot without decoding. Returns a new owning
// clip with resized packed video and canvas metadata; preserves audio, time,
// normalization and model provenance. The input is never modified.
std::shared_ptr<const LatentClip> upscale_latent_clip(
    const LatentClip& input, const std::string& checkpoint, DeviceBackend backend,
    const LatentUpscaleOptions& options = {},
    const std::function<bool(int, int)>& progress = {});
} // namespace slopfab
