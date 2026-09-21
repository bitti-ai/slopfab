#pragma once

#include <string>

#include "slopfab/safetensors.h"

namespace slopfab {

// ComfyUI asymmetric W4A8 descriptor stored beside a packed linear weight.
// The format implies ConvRot on the contraction axis; only the published
// group-16 / ConvRot-256 contract is currently accepted.
struct W4A8State {
  int group_size = 0;
  int convrot_group_size = 0;
};

bool is_w4a8_weight(const SafeTensors& checkpoint, const std::string& weight_name);
W4A8State read_w4a8_state(const SafeTensors& checkpoint, const std::string& weight_name,
                          const char* consumer);

} // namespace slopfab
