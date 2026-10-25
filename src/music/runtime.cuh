#pragma once
#include "../moss/runtime.cuh"
#include "slopfab/generate.h"
#include <functional>

namespace slopfab::music {
using moss::Tensor;
using moss::Runtime;
using moss::BFloat;
using Hook = std::function<void()>;
Tensor embedding(Runtime& rt, const std::string& name, const std::vector<int32_t>& ids);
Tensor yue_norm(Runtime& rt, const Tensor& x, const std::string& name, int width = 0);
Tensor rope(int count, int offset, int dim = 128, float theta = 1000000.f);
std::vector<float> decode_yue(Runtime& rt, const std::vector<float>& latent, Hook hook = {});
RunResult run_yue(const GenerateRequest&, const GeneratePlan&, const RunOptions&);
}
