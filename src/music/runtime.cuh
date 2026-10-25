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
Tensor probe_yue(Runtime& rt, const std::vector<int32_t>& ids, const std::vector<float>& state = {},
                 const std::string& trace = "");
std::vector<float> decode_yue(Runtime& rt, const std::vector<float>& latent, Hook hook = {});
std::vector<float> decode_minimax(Runtime& rt, const std::vector<float>& latent, Hook hook = {});
std::vector<float> minimax_hidden(Runtime&, const std::vector<int32_t>&, const MusicSettings&,
                                  int maximum, uint64_t seed, Hook hook = {});
Tensor minimax_condition(Runtime&, const std::vector<float>&);
Tensor minimax_prediction(Runtime&, const Tensor&, const Tensor&, float time, Hook hook = {},
                          const std::string& trace = "");
Tensor probe_minimax_global(Runtime&, const std::vector<int32_t>&);
Tensor probe_minimax_depth(Runtime&, const Tensor&);
RunResult run_yue(const GenerateRequest&, const GeneratePlan&, const RunOptions&);
}
