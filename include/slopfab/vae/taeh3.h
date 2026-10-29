#pragma once
#include "slopfab/device_tensor.h"
#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace slopfab::vae {
// TAEH3's base decoder, with spatial and temporal upsampling disabled as
// supported upstream. Pixel shuffle remains enabled: output is 2H x 2W.
// Input is diffusion-space CTHW, without the full VAE's scale/shift.
class TaeH3 {
public:
  TaeH3(const std::string& path, DeviceBackend backend);
  ~TaeH3();
  std::vector<uint8_t> decode(const std::vector<float>& latents, int frames, int height,
                            int width, const std::function<bool()>& cancelled = {});
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
