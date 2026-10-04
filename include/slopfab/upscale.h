#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "slopfab/device_tensor.h"
#include "slopfab/pixel_buffer.h"
#include "slopfab/safetensors.h"

namespace slopfab {

struct UpscaleOptions {
  // Input-pixel units. Zero tile_size runs the complete padded frame at once.
  int tile_size = 128;
  int tile_pad = 10;
  int pre_pad = 10; // reflect at the right and bottom edges, as in RealESRGANer
};

void validate_upscale_options(const UpscaleOptions& options);
// Requires the x4plus RRDBNet: RGB, 64 features, 32 growth channels, 23 blocks.
void validate_realesrgan_checkpoint(const SafeTensors& checkpoint);
size_t upscale_output_elements(int frames, int height, int width);

class UpscaleCancelled : public std::runtime_error {
public:
  UpscaleCancelled() : std::runtime_error("upscaling cancelled") {
  }
};

// Native FP32 inference; owns uploaded weights and can be reused across clips.
// Not thread safe. CUDA uses the current device; Vulkan uses device index zero.
class RealEsrgan {
public:
  RealEsrgan(const std::string& checkpoint, DeviceBackend backend);
  ~RealEsrgan();
  RealEsrgan(const RealEsrgan&) = delete;
  RealEsrgan& operator=(const RealEsrgan&) = delete;

  // Input/output layout is [3,frames,height,width], RGB in [0,1]. Output is
  // clamped to [0,1] and has four times the width and height. Rejects NaN/Inf.
  // Progress is called before each tile and on completion; false cancels.
  PixelBuffer upscale(const PixelBuffer& input, int frames, int height, int width,
                      const UpscaleOptions& options = {},
                      const std::function<bool(int, int)>& progress = {});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace slopfab
