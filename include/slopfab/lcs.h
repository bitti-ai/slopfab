#pragma once
#include <array>
#include <memory>
#include <string>
#include <vector>
#include "slopfab/sampling_effect.h"
#include "slopfab/device_tensor.h"

namespace slopfab {
struct LcsData {
  static constexpr int dim = 96;
  std::vector<float> basis;                   // raw VAE space [96,3], orthonormal columns
  std::vector<float> mean;                    // [96]
  std::vector<float> anchors;                 // [8,3]: R B G M C Y black white
  std::vector<float> latent_mean, latent_std; // [24], model normalization
  std::vector<float> sharpness; // optional unit direction [96], orthogonal to color and DC
  std::string vae_sha256;
  float explained_variance = 0;
  void validate() const;
  void save(const std::string& path) const;
  static std::shared_ptr<const LcsData> load(const std::string& path);
};

struct LcsSettings {
  int mode = 0; // interpolated=0, type_i=1, type_ii=2
  std::array<float, 3> color{1, 0, 0};
  float strength = 0;
  float contrast = 1, brightness = 0, saturation = 1, temperature = 0;
  float sharpness = 0;
  int start_step = 0, end_step = 2147483647; // inclusive, video updates
  int anchor = 0;                            // none=0, self=1, smooth=2
  float anchor_strength = 0;
  bool active() const;
  void validate() const;
};

LcsSettings parse_lcs_settings(const std::string& json);
// Fit raw VAE color samples [N,96] and anchor patch means [8,96].
LcsData fit_lcs(const std::vector<float>& colors, const std::vector<float>& anchors);
// Fit the leading direction of sharp-minus-blurred patch samples [N,96].
void fit_lcs_sharpness(LcsData& data, const std::vector<float>& differences);
// One fresh instance per generation. Mask is empty, one spatial patch plane, or
// a full frame-major patch volume; values in [0,1]. Color edits preserve the
// orthogonal residual. Native arithmetic runs on CPU for both GPU backends.
SamplingEffect make_lcs_effect(std::shared_ptr<const LcsData> data, LcsSettings settings,
                               std::vector<float> mask = {});
std::string lcs_vae_fingerprint(const std::string& path);
// Synchronous native calibration; false means cancelled, no output is written.
bool calibrate_lcs(const std::string& vae_path, DeviceBackend backend, bool portable,
                   const std::string& output, int colors, int image_size,
                   const std::function<bool(int, int)>& progress = {});
} // namespace slopfab
