#pragma once
#include "slopfab/safetensors.h"
#include "slopfab/vae/vit_decoder.h"
#include <array>

namespace slopfab::dit {
// Lakonik/pi-H3 DWD adapters: FL2VA replacements, rank-64 updates and an
// eight-layer pixel head. These are not standalone transformer checkpoints.
bool is_pi_h3(const SafeTensors& adapter);
void validate_pi_h3(const SafeTensors& adapter);
void validate_pi_h3_base(const SafeTensors& base);
int pi_h3_steps(const std::string& path); // 0 if the filename is ambiguous
std::vector<float> pi_h3_base_sigmas(int steps);
SafeTensors pi_h3_overlay(const SafeTensors& base, const SafeTensors& adapter);
// Materialize only pixel-head matrices; the backbone keeps native quantization.
std::vector<float> pi_h3_weight(const SafeTensors&, const std::string& name);
void pi_h3_unrotate(float* values, size_t count);
std::vector<float> pi_h3_modulate(const SafeTensors&, const float* hidden, size_t rows,
                                 const std::array<float, 8>& code);
void pi_h3_expand_rows(const float* expanded, float* rows, int begin, int count, int H, int W);
void pi_h3_pixels(const float* patches, int begin, int count, int H, int W,
                  bool first_frame, vae::DecodedVideo& output);
} // namespace slopfab::dit
