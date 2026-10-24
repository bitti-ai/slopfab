#pragma once
#include "slopfab/safetensors.h"
#include <array>
#include <vector>

namespace slopfab::dit {
// Released OmniRef: eight fused blocks on the 32-interval [0, .999] clock.
constexpr int kOmniRefHeads = 32, kOmniRefSteps = 8;
std::vector<float> omniref_sigmas(float shift);
int omniref_step(float video_timestep);
std::array<float, kOmniRefHeads> omniref_head_weights(int step, float shift);
// Comfy exports store a full head zero and residuals relative to it thereafter.
// Return eight ordinary projections, with the output feature order unchanged.
std::vector<float> fuse_omniref_heads(const TensorView& tensor, float shift);
}
