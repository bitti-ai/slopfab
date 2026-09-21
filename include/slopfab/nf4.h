#pragma once

#include <string>
#include <vector>

#include "slopfab/safetensors.h"

namespace slopfab {

struct NF4State {
  std::string source_dtype;
  int block_size = 0;
  int nested_block_size = 0;
  float nested_offset = 0.0f;
  std::vector<int64_t> shape;
};

bool is_nf4_weight(const SafeTensors& checkpoint, const std::string& weight_name);
NF4State read_nf4_state(const SafeTensors& checkpoint, const std::string& weight_name,
                        const char* consumer, bool require_bfloat16 = false);

} // namespace slopfab
