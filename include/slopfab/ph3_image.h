#pragma once

#include "slopfab/safetensors.h"
#include <vector>

namespace slopfab {
struct GenerateRequest;
struct GeneratePlan;

// The published P-H3 Image transformer uses a distinct rank-16 / 4097-row
// table. Inspect tensors rather than relying on distribution filenames.
bool is_ph3_image_transformer(const SafeTensors& checkpoint);
void validate_ph3_image_transformer(const SafeTensors& checkpoint);
const std::vector<float>& ph3_image_sigmas();
GeneratePlan resolve_ph3_image_plan(const GenerateRequest& request);

inline constexpr const char* kPH3ImageRuntimePending =
    "P-H3 Image supports checkpoint inspection and text-to-image planning only; "
    "GPU rank-16 modulation, the text connector and image VAE are not wired into generation yet";
} // namespace slopfab
