#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "slopfab/device_tensor.h"
#include "slopfab/pixel_buffer.h"
#include "slopfab/safetensors.h"
#include "slopfab/seedvr2.h"

namespace slopfab {

// Stable identifiers, mirrored by SLOPFAB_UPSCALE_* in the C API.
enum class UpscaleMethod { kRealEsrgan = 1, kSeedVr2 = 2 };
UpscaleMethod parse_upscale_method(std::string_view name);
const char* upscale_method_name(UpscaleMethod method);
const char* default_upscale_model(UpscaleMethod method);
int upscale_scale_factor(UpscaleMethod method);

struct UpscaleOptions {
  // Input-pixel units. Zero tile_size runs the complete padded frame at once.
  int tile_size = 128;
  int tile_pad = 10;
  int pre_pad = 10; // reflect at the right and bottom edges, as in RealESRGANer
  // SeedVR2: tile_size is the VAE tile in output pixels; padding above is unused.
  std::string vae_path = "weights/seedvr2/ema_vae_fp16.safetensors";
  int width = 0, height = 0; // SeedVR2 target size; both zero selects 4x
  int segment_frames = 5;
  uint64_t seed = 666;
  int device = 0;
  bool color_match = true;
};

void validate_upscale_options(const UpscaleOptions& options,
                              UpscaleMethod method = UpscaleMethod::kRealEsrgan);
std::pair<int, int> upscale_dimensions(int height, int width, UpscaleMethod method,
                                       const UpscaleOptions& options = {});
seedvr2::Options seedvr2_options(const UpscaleOptions& options, int height, int width);
// Requires the x4plus RRDBNet: RGB, 64 features, 32 growth channels, 23 blocks.
void validate_realesrgan_checkpoint(const SafeTensors& checkpoint);
void validate_upscale_checkpoint(const SafeTensors& checkpoint, UpscaleMethod method);
size_t upscale_output_elements(int frames, int height, int width,
                               UpscaleMethod method = UpscaleMethod::kRealEsrgan,
                               const UpscaleOptions& options = {});

class UpscaleCancelled : public std::runtime_error {
public:
  UpscaleCancelled() : std::runtime_error("upscaling cancelled") {
  }
};

// Shared interface for upscaling implementations. Input/output layout is
// [3,frames,height,width], RGB in [0,1]. The method defines the spatial scale.
// Progress is called before each tile and on completion; false cancels.
class Upscaler {
public:
  virtual ~Upscaler() = default;
  virtual PixelBuffer upscale(const PixelBuffer& input, int frames, int height, int width,
                              const UpscaleOptions& options = {},
                              const std::function<bool(int, int)>& progress = {}) = 0;
};

std::unique_ptr<Upscaler> make_upscaler(UpscaleMethod method, const std::string& checkpoint,
                                        DeviceBackend backend);

// Native FP32 RealESRGAN_x4plus inference; owns uploaded weights for reuse.
// Not thread safe. CUDA uses the current device; Vulkan uses device index zero.
class RealEsrgan final : public Upscaler {
public:
  RealEsrgan(const std::string& checkpoint, DeviceBackend backend);
  ~RealEsrgan() override;
  RealEsrgan(const RealEsrgan&) = delete;
  RealEsrgan& operator=(const RealEsrgan&) = delete;

  // Input/output layout is [3,frames,height,width], RGB in [0,1]. Output is
  // clamped to [0,1] and has four times the width and height. Rejects NaN/Inf.
  // Progress is called before each tile and on completion; false cancels.
  PixelBuffer upscale(const PixelBuffer& input, int frames, int height, int width,
                      const UpscaleOptions& options = {},
                      const std::function<bool(int, int)>& progress = {}) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace slopfab
