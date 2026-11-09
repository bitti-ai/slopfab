#pragma once
#include <string>
#include <vector>
#include "slopfab/safetensors.h"

namespace slopfab {
struct GenerateRequest;
// Open-H3-Regenerate-2K-IC: a Ref2VA refinement pass, not a pixel upscaler.
struct RegenerateSettings {
  std::string mode; // empty (off), up3, up4, hf8
  std::string silence_path;
  std::string upsampler_path;
};
int regenerate_steps(const std::string& mode);
bool is_regenerate_lora(const SafeTensors& checkpoint);
void validate_regenerate_request(const GenerateRequest& request);
std::string regenerate_prompt(const std::string& caption);
std::vector<float> regenerate_silence(const std::string& path);
} // namespace slopfab
