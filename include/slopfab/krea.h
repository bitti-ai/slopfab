#pragma once
#include "slopfab/pipeline.h"

namespace slopfab {
GeneratePlan resolve_krea_plan(const GenerateRequest& request);
std::vector<float> krea_sigma_schedule(int steps, float shift);
float krea_guidance(const GenerateRequest& request);
}
