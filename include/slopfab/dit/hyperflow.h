#pragma once

#include <memory>
#include <optional>
#include "slopfab/dit/packing.h"
#include "slopfab/sampling_settings.h"
#include "slopfab/safetensors.h"

namespace slopfab::dit {

struct HyperflowMetadata {
  SamplingSettings sampling;
  float gate = 0, alpha = 0;
  int rank = 0;
  std::string revision;
};
std::optional<HyperflowMetadata>
hyperflow_metadata(const std::map<std::string, std::string>& metadata);

// Exact, pre-evaluated two-time modulation. Preparation uses the original
// time MLP and full AdaLN weights; inference retains the quantized base.
class Hyperflow {
public:
  static std::shared_ptr<Hyperflow> load(const SafeTensors& adapter,
                                        const SafeTensors& base, float strength);
  int rank() const { return static_cast<int>(pairs_.size() / 2); }
  std::vector<float> code(const RowTimesteps& times, size_t rows = 0) const;
  std::vector<float> weight(const std::string& projection) const;
  size_t weight_bytes() const;

private:
  SafeTensors archive_;
  std::vector<float> pairs_;
};

RowTimesteps build_hyperflow_timesteps(const SequenceLayout& layout, const PackedIndices& indices,
                                       float video_t, float video_r, float audio_t, float audio_r);
} // namespace slopfab::dit
