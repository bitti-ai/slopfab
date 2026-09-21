#pragma once

#include <vector>

#include "slopfab/safetensors.h"

namespace slopfab {

// Comfy INT8 matrices store one scale per output row. ConvRot weights are in
// rotated coordinates; applying the symmetric regular Hadamard again restores
// ordinary matrix coordinates before multiplication.
struct Int8WeightState {
  int rows = 0;
  int columns = 0;
  int rotation_group = 1; // 1 means no rotation
  const int8_t* codes = nullptr;
  const float* scales = nullptr;
};

Int8WeightState read_int8_weight(const SafeTensors& checkpoint, const std::string& name,
                                 const char* consumer);
std::vector<uint16_t> unpack_int8_weight(const Int8WeightState& state,
                                         bool canonicalize_subnormals);

} // namespace slopfab
