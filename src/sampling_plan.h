#pragma once

namespace slopfab {
struct GenerateRequest;
struct GeneratePlan;
// Resolve recipe data once; both backend runners consume the resulting grids.
void resolve_sampling_plan(const GenerateRequest& request, GeneratePlan& plan);
}
