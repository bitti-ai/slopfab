#pragma once
#include "slopfab/safetensors.h"

namespace slopfab::dit::detail {
struct QuantTag {
  std::string format;             // empty when the tensor carries no blob
  bool full_precision = false;
  bool convrot = false;
  int convrot_group = 256;
};
QuantTag read_comfy_quant(const SafeTensors& st, const std::string& name);
}
