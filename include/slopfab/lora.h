#pragma once

#include <map>
#include <string>
#include <vector>

#include "slopfab/safetensors.h"

namespace slopfab {

struct LoraSpec {
  std::string path;
  float strength = 1.0f;
};

// Host factors in original (unrotated, unscaled) model coordinates. Each
// update is strength * alpha/rank * B @ A. Scale is folded into B at upload.
struct LoraFactors {
  int rank = 0, in = 0, out = 0;
  std::vector<float> a, b;
};

class LoraAdapters {
 public:
  // Accepts H3 block/refiner attention and MLP adapters. Unknown keys, missing
  // partners, non-finite values and incompatible base shapes are errors.
  void load(const std::vector<LoraSpec>& specs, const SafeTensors& base);
  // Multiple adapters for one target are concatenated into one factor pair.
  const std::vector<LoraFactors>* find(const std::string& projection) const;
  size_t projection_count() const { return factors_.size(); }
 private:
  std::map<std::string, std::vector<LoraFactors>> factors_;
};

}  // namespace slopfab
