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
// update is strength * alpha/rank * B @ A. Scale is folded into B at load.
struct LoraFactors {
  int rank = 0, in = 0, out = 0;
  std::vector<float> a, b;
};

// Explicit asset preparation, outside inference. Embeds a validated local grid;
// allow_download opts into the pinned legacy FL2VA asset acquisition.
void prepare_lora_grid(const std::string& adapter_path, int width, bool allow_download = false);

class LoraAdapters {
 public:
  // Accepts H3 block/refiner attention and MLP adapters, including Diffusers
  // separate Q/K/V, video input/output and pruned AdaLN projections. Full-width
  // AdaLN factors use an embedded timestep grid and are fitted to the base
  // checkpoint's table with a checked residual. Inference reads an embedded or
  // local companion grid and never downloads assets or rewrites adapters.
  // Unknown keys, missing
  // partners, non-finite values and incompatible base shapes are errors.
  void load(const std::vector<LoraSpec>& specs, const SafeTensors& base);
  // Multiple adapters for one target are concatenated into one factor pair.
  const std::vector<LoraFactors>* find(const std::string& projection) const;
  size_t projection_count() const { return factors_.size(); }
  // F32 video endpoints are merged at load time; biases remain unchanged.
  std::vector<float> merged_endpoint_weight(const SafeTensors& base,
                                            const std::string& projection) const;
  // Pruned AdaLN weights and biases are merged in F32 on both backends.
  bool has_adaln(const std::string& projection) const;
  std::vector<float> merged_adaln_weight(const SafeTensors& base,
                                         const std::string& projection) const;
  std::vector<float> merged_adaln_bias(const SafeTensors& base,
                                       const std::string& projection) const;
 private:
  std::map<std::string, std::vector<LoraFactors>> factors_;
  std::map<std::string, std::vector<float>> adaln_bias_;
};

}  // namespace slopfab
