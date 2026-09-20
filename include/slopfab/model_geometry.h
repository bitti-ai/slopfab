#pragma once
#include <string>
#include <string_view>
#include <vector>
#include "slopfab/safetensors.h"

namespace slopfab {
// Data shared by packing, planning and archive compatibility. A different codec
// or position algorithm needs an implementation, not just different dimensions.
struct LatentGeometry {
  int version = 1;
  int fps = 24;
  int spatial_compression = 16;
  int video_channels = 24;
  int audio_features = 32;
  int audio_channels = 2;
  int audio_latents_per_second = 40;
  int audio_sample_rate = 32000;
  int patch_height = 2, patch_width = 2;
  int frames_per_chunk = 17, latents_per_chunk = 5;
  int frame_offset = 5, latent_frame_offset = 2;
  int canvas_multiple = 32;
  int trained_max_pixels = 768 * 1344;
  double min_aspect = 0.25, max_aspect = 4.0;
  double rope_frame_rescale = 5.0 / 3.0;
  double rope_spatial_scale = 32.0;
  std::vector<int> rope_frames_per_latent = {1,4,4,4,4};
  int video_patch_dim() const { return video_channels * patch_height * patch_width; }
  std::string fingerprint() const;
};
const LatentGeometry& h3_latent_geometry();
void validate_latent_geometry(const LatentGeometry& geometry);
// Rejects contracts not implemented by the current H3 transformer/codec kernels.
void require_h3_latent_geometry(const LatentGeometry& geometry);
LatentGeometry parse_model_geometry(std::string_view text);
LatentGeometry read_model_geometry(const SafeTensors& checkpoint);
std::string geometry_json(const LatentGeometry& geometry);
} // namespace slopfab
