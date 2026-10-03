#pragma once

#include "slopfab/device_tensor.h"
#include "slopfab/text/encoder.h"

namespace slopfab::text {

struct TextExportRequest {
  std::string prompt;
  std::string text_encoder_path;
  std::string tokenizer_path; // Empty selects the embedded tokenizer.
#if !SLOPFAB_WITH_CUDA && SLOPFAB_WITH_VULKAN
  DeviceBackend backend = DeviceBackend::kVulkan;
#else
  DeviceBackend backend = DeviceBackend::kCuda;
#endif
  Residency residency = Residency::kStreaming;
  EncoderArithmetic arithmetic = EncoderArithmetic::kShipped;
#ifdef __linux__
  bool vulkan_portable_arithmetic = true;
#else
  bool vulkan_portable_arithmetic = false;
#endif
};

// Encodes only text, without a diffusion transformer or either VAE. Saves an
// atomic F32 prompt archive with integer modality tags and provenance metadata.
// Returns the token count. Caller serializes GPU work with generation.
int export_prompt_embedding(const TextExportRequest& request, const std::string& output);

} // namespace slopfab::text
