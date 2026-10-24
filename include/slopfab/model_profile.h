#pragma once
#include <string>
#include "slopfab/safetensors.h"

namespace slopfab {
// Family-level policy belongs here; architecture-specific arithmetic stays in
// the corresponding runner. Auto selection uses tensors, never filenames.
struct ModelProfile {
  const char* family;
  const char* name;
  const char* transformer_hint;
  const char* conditioner_hint;
  const char* vae_hint;
  bool image_only;
  int canvas_multiple;
  int conditioner_width;
};
const ModelProfile& model_profile(const std::string& family);
const ModelProfile& checkpoint_model_profile(const SafeTensors& checkpoint);
struct GenerateRequest;
struct GeneratePlan;
std::string request_model_family(const GenerateRequest& request);
GeneratePlan resolve_moss_plan(const GenerateRequest& request);
std::vector<float> moss_sigma_schedule(int steps, float shift);
} // namespace slopfab
