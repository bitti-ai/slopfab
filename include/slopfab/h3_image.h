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
// Rounded canvas-grid reference sizes: VAE short edge [1024,2048], Qwen 1024.
void h3_image_reference_size(int width, int height, int* out_height, int* out_width,
                             bool vision = false);
std::string h3_image_picture_label(size_t index);
} // namespace slopfab
