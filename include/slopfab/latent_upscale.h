#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "slopfab/device_tensor.h"
#include "slopfab/safetensors.h"

namespace slopfab {
struct LatentUpscaleOptions {
  float scale = 2.0f;
  // Upstream's 32-frame segments with replicated context and overlap blending.
  // GroupNorm makes chunked inference an approximation to whole-clip inference.
  bool temporal_chunking = true;
};

inline constexpr const char* kDefaultLatentUpscaleModel =
    "weights/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors";

void validate_latent_upscale_options(const LatentUpscaleOptions& options);
void validate_latent_upscale_checkpoint(const SafeTensors& checkpoint);
// Latent dimensions, rounded to even sizes (32 decoded pixels), ties to even.
std::pair<int, int> latent_upscale_dimensions(int height, int width,
                                              const LatentUpscaleOptions& options = {});

// Normalized H3 latents [24,T,H,W], as produced by unpatchify_video. No extra
// normalization or denormalization is needed. Time is preserved. Loads/releases
// weights within the call; false progress callbacks throw UpscaleCancelled.
std::vector<float> upscale_latents(const std::vector<float>& input, int frames, int height,
                                   int width, const std::string& checkpoint, DeviceBackend backend,
                                   const LatentUpscaleOptions& options = {},
                                   const std::function<bool(int, int)>& progress = {});
} // namespace slopfab
