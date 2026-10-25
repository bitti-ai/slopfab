#pragma once

#include <string>

#include "slopfab/safetensors.h"
#include "slopfab/json.h"

namespace slopfab {

// ComfyUI packed integer descriptor. Both formats use ConvRot-256.
// W4A8 uses group-16 scales; W6A8 accepts groups divisible by 16.
struct W4A8State {
  int bits = 4;
  int group_size = 0;
  int convrot_group_size = 0;
};

// Shared H3 packed integer contract. W6A8 stores low nibbles followed by
// two-bit high planes per row, with uniform levels code-32 and no codebook.
bool is_w6a8_weight(const SafeTensors& checkpoint, const std::string& weight_name);
bool is_packed_int8_weight(const SafeTensors& checkpoint, const std::string& weight_name);
W4A8State read_packed_int8_state(const SafeTensors& checkpoint, const std::string& weight_name,
                                 const char* consumer);
void validate_packed_int8_weight(const SafeTensors& checkpoint, const std::string& weight_name,
                                 int out, int in);

// Accept per-layer .comfy_quant tensors and checkpoint-wide Comfy metadata.
json::Value quantization_state(const SafeTensors& checkpoint, const std::string& weight_name);
void validate_w4a8_weight(const SafeTensors& checkpoint, const std::string& weight_name, int out, int in);
bool is_w4a8_weight(const SafeTensors& checkpoint, const std::string& weight_name);
W4A8State read_w4a8_state(const SafeTensors& checkpoint, const std::string& weight_name,
                          const char* consumer);

} // namespace slopfab
