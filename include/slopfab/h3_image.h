#pragma once
#include "slopfab/safetensors.h"
#include "slopfab/vae/vit_decoder.h"
#include <vector>

namespace slopfab {
struct GenerateRequest;
struct GeneratePlan;
bool is_h3_image_transformer(const SafeTensors& checkpoint);
void validate_h3_image_transformer(const SafeTensors& checkpoint);
const std::vector<float>& h3_image_sigmas();
GeneratePlan resolve_h3_image_plan(const GenerateRequest& request);
vae::ViTConfig h3_image_vae_config();
} // namespace slopfab
