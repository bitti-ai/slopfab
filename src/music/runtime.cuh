#pragma once
#ifdef SLOPFAB_MUSIC_VULKAN
#include "vulkan_runtime.h"
#define SLOPFAB_MUSIC_NS music_vulkan
#else
#include "../moss/runtime.cuh"
#define SLOPFAB_MUSIC_NS music
#endif
#include "slopfab/generate.h"
#include <functional>

namespace slopfab::SLOPFAB_MUSIC_NS {
#ifdef SLOPFAB_MUSIC_VULKAN
namespace moss = music_gpu;
using namespace slopfab::music;
#endif
using moss::Tensor;
using moss::Runtime;
#ifndef SLOPFAB_MUSIC_VULKAN
using moss::BFloat;
#endif
float gpu_bf(float);
void gpu_copy_rows(const Tensor&, Tensor&, int, int, int);
void gpu_copy_columns(const Tensor&, Tensor&, int);
void gpu_zero(Tensor&);
void gpu_condition(Runtime&, Tensor&, const Tensor&, const std::string&);
Tensor gpu_yue_norm(Runtime&, const Tensor&, const std::string&, int);
Tensor gpu_ln(Runtime&, const Tensor&, const std::string&);
Tensor gpu_causal_attention(Runtime&, const Tensor&, const Tensor&, const Tensor&);
using Hook = std::function<void()>;
Tensor embedding(Runtime& rt, const std::string& name, const std::vector<int32_t>& ids);
Tensor yue_norm(Runtime& rt, const Tensor& x, const std::string& name, int width = 0);
Tensor rope(int count, int offset, int dim = 128, float theta = 1000000.f);
Tensor probe_yue(Runtime& rt, const std::vector<int32_t>& ids, const std::vector<float>& state = {},
                 const std::string& trace = "");
std::vector<float> decode_yue(Runtime& rt, const std::vector<float>& latent, Hook hook = {});
std::vector<float> decode_minimax(Runtime& rt, const std::vector<float>& latent, Hook hook = {});
std::vector<float> minimax_hidden(Runtime&, const std::vector<int32_t>&, const MusicSettings&,
                                  int maximum, uint64_t seed, Hook hook = {},
                                  const std::string& trace = "", Hook frame_completed = {});
Tensor minimax_condition(Runtime&, const std::vector<float>&);
Tensor minimax_prediction(Runtime&, const Tensor&, const Tensor&, float time, Hook hook = {},
                          const std::string& trace = "");
Tensor probe_minimax_global(Runtime&, const std::vector<int32_t>&);
Tensor probe_minimax_depth(Runtime&, const Tensor&);
RunResult run_yue(const GenerateRequest&, const GeneratePlan&, const RunOptions&);
// Developer replay of captured AR output through the production flow/decoder.
RunResult run_minimax_from_hidden(const GenerateRequest&, const GeneratePlan&, const RunOptions&,
                                  std::vector<float> hidden);
}
