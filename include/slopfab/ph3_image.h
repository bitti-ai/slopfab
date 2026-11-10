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

} // namespace slopfab
