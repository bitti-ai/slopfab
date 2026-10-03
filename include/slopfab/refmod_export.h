#pragma once

#include "slopfab/device_tensor.h"
#include "slopfab/reference_media.h"
#include <string>

namespace slopfab {

struct RefModExportRequest {
  std::vector<std::string> image_paths;
  std::vector<std::shared_ptr<const ReferenceMedia>> media;
  std::string video_vae_path, audio_vae_path;
  std::string name, description;
  int short_edge = 768; // Multiple of 32, 32..768; area capped at 768*1344.
#if !SLOPFAB_WITH_CUDA && SLOPFAB_WITH_VULKAN
  DeviceBackend backend = DeviceBackend::kVulkan;
#else
  DeviceBackend backend = DeviceBackend::kCuda;
#endif
#ifdef __linux__
  bool vulkan_portable_arithmetic = true;
#else
  bool vulkan_portable_arithmetic = false;
#endif
};

// Only VAEs are loaded. Encodes images followed by media in attachment order;
// each video's soundtrack becomes an audio member immediately after its video.
// One member is saved as v4, multiple members as v5. Atomic F32 safetensors.
// Returns the number of members. Caller serializes GPU work with generation.
int export_refmod(const RefModExportRequest& request, const std::string& output);

} // namespace slopfab
