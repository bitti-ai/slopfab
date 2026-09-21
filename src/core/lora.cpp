#include "slopfab/lora.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "slopfab/json.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/dit/checkpoint.h"
#include "lora_grid.h"
#include "../dit/h3_lora.h"

namespace slopfab {
namespace {
bool ends(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string normalize(std::string name) {
  for (const char* p : {"base_model.model.", "model.diffusion_model.", "diffusion_model."}) {
    const std::string prefix(p);
    if (name.compare(0, prefix.size(), prefix) == 0)
      name.erase(0, prefix.size());
  }
  return name;
}

using detail::h3::projection_name;
using detail::h3::adaln_target;
using detail::h3::supported;
using detail::h3::AdaLNBasis;

std::vector<float> values(const TensorView& tensor) {
  if (tensor.dtype != DType::kF32 && tensor.dtype != DType::kF16 && tensor.dtype != DType::kBF16)
    throw std::runtime_error("LoRA: expected F32/F16/BF16 for " + tensor.name);
  auto result = to_f32(tensor);
  for (float v : result)
    if (!std::isfinite(v))
      throw std::runtime_error("LoRA: non-finite value in " + tensor.name);
  return result;
}

} // namespace

const std::vector<LoraFactors>* LoraAdapters::find(const std::string& name) const {
  const auto it = factors_.find(name);
  return it == factors_.end() ? nullptr : &it->second;
}

void LoraAdapters::load(const std::vector<LoraSpec>& specs, const SafeTensors& base) {
  decltype(factors_) next;
  decltype(adaln_bias_) next_bias;
  for (const LoraSpec& spec : specs) {
    if (spec.path.empty() || !std::isfinite(spec.strength))
      throw std::runtime_error("LoRA: path must be nonempty and strength finite");
    SafeTensors adapter;
    adapter.open(spec.path);
    AdaLNBasis adaln_basis;
    double worst_adaln_error = 0;
    int rebased = 0;
    float metadata_alpha = 0;
    bool has_metadata_alpha = false;
    if (const auto it = adapter.metadata().find("lora_adapter_metadata");
        it != adapter.metadata().end()) {
      const auto config = json::parse(it->second);
      if (!config.is_object())
        throw std::runtime_error("LoRA: adapter metadata must be an object");
      for (const char* flag : {"use_dora", "use_rslora", "fan_in_fan_out", "lora_bias"})
        if (const auto* value = config.find(flag); value && !value->is_null() && value->as_bool())
          throw std::runtime_error(std::string("LoRA: unsupported adapter setting ") + flag);
      if (const auto* pattern = config.find("alpha_pattern");
          pattern && !pattern->is_null() && !pattern->as_object().empty())
        throw std::runtime_error("LoRA: per-target alpha_pattern is unsupported");
      if (const auto* alpha = config.find("lora_alpha")) {
        metadata_alpha = static_cast<float>(alpha->as_number());
        if (!std::isfinite(metadata_alpha))
          throw std::runtime_error("LoRA: non-finite metadata alpha");
        has_metadata_alpha = true;
      }
    }

    struct Pair {
      const TensorView *a = nullptr, *b = nullptr, *alpha = nullptr;
      bool swap_ffn = false;
    };

    std::map<std::string, Pair> pairs;
    for (const auto& item : adapter.tensors()) {
      if (item.first == detail::kLoraGridTensor)
        continue;
      const std::string key = normalize(item.first);
      bool matched = false;
      for (const auto& suffix :
           std::vector<std::pair<std::string, int>>{{".lora_A.weight", 0},
                                                    {".lora_B.weight", 1},
                                                    {".lora_A.default.weight", 0},
                                                    {".lora_B.default.weight", 1},
                                                    {".lora_down.weight", 0},
                                                    {".lora_up.weight", 1},
                                                    {".alpha", 2}}) {
        if (!ends(key, suffix.first))
          continue;
        const std::string name = projection_name(key.substr(0, key.size() - suffix.first.size()));
        if (!supported(name))
          throw std::runtime_error("LoRA: unsupported target " + name);
        Pair& pair = pairs[name];
        if (detail::h3::needs_output_basis_conversion(key))
          pair.swap_ffn = true;
        const TensorView*& slot = suffix.second == 0   ? pair.a
                                  : suffix.second == 1 ? pair.b
                                                       : pair.alpha;
        if (slot)
          throw std::runtime_error("LoRA: duplicate tensor for " + name);
        slot = &item.second;
        matched = true;
        break;
      }
      if (!matched)
        throw std::runtime_error("LoRA: unsupported tensor " + item.first);
    }
    if (pairs.empty())
      throw std::runtime_error("LoRA: adapter contains no projections: " + spec.path);
    for (const auto& item : pairs) {
      const std::string& name = item.first;
      const Pair& p = item.second;
      if (!p.a || !p.b)
        throw std::runtime_error("LoRA: missing A/B partner for " + name);
      if (p.a->shape.size() != 2 || p.b->shape.size() != 2 || p.a->shape[0] <= 0 ||
          p.a->shape[1] <= 0 || p.b->shape[0] <= 0 || p.a->shape[0] != p.b->shape[1] ||
          p.a->shape[0] > INT32_MAX || p.a->shape[1] > INT32_MAX || p.b->shape[0] > INT32_MAX)
        throw std::runtime_error("LoRA: invalid matrix shapes for " + name);
      LoraFactors f;
      f.rank = static_cast<int>(p.a->shape[0]);
      f.in = static_cast<int>(p.a->shape[1]);
      f.out = static_cast<int>(p.b->shape[0]);
      const bool rebase = detail::h3::validate_target(base, name, f);
      float alpha = has_metadata_alpha ? metadata_alpha : static_cast<float>(f.rank);
      if (p.alpha) {
        if (p.alpha->numel() != 1)
          throw std::runtime_error("LoRA: alpha must be scalar for " + name);
        alpha = values(*p.alpha)[0];
      }
      f.a = values(*p.a);
      f.b = values(*p.b);
      detail::h3::convert_output_basis(f, p.swap_ffn);
      const float scale = spec.strength * (alpha / static_cast<float>(f.rank));
      if (!std::isfinite(scale))
        throw std::runtime_error("LoRA: non-finite scale for " + name);
      for (float& v : f.b) {
        v *= scale;
        if (!std::isfinite(v) || !std::isfinite(bf16_to_f32(f32_to_bf16(v))))
          throw std::runtime_error("LoRA: scaled B overflows BF16 for " + name);
      }
      for (float v : f.a)
        if (!std::isfinite(bf16_to_f32(f32_to_bf16(v))))
          throw std::runtime_error("LoRA: A overflows BF16 for " + name);
      if (scale != 0.0f) {
        if (rebase) {
          if (adaln_basis.map.empty())
            adaln_basis.load(base, adapter, f.in);
          std::vector<float> bias;
          worst_adaln_error = std::max(worst_adaln_error, adaln_basis.project(f, bias));
          auto& combined = next_bias[name];
          if (combined.empty())
            combined.resize(bias.size());
          for (size_t i = 0; i < bias.size(); ++i) {
            combined[i] += bias[i];
            if (!std::isfinite(combined[i]))
              throw std::runtime_error("LoRA: combined AdaLN bias overflow");
          }
          ++rebased;
        }
        next[name].push_back(std::move(f));
      }
    }
    if (rebased)
      std::fprintf(
          stderr,
          "LoRA: converted %d AdaLN targets to pruned coordinates; worst activation fit error %.4f%% (%s)\n",
          rebased, worst_adaln_error * 100, spec.path.c_str());
  }
  // Concatenation represents sum_i B_i A_i as one low-rank product. This
  // keeps dispatch count constant even with several adapters on Vulkan.
  for (auto& entry : next) {
    auto& list = entry.second;
    if (list.size() < 2)
      continue;
    LoraFactors combined;
    combined.in = list.front().in;
    combined.out = list.front().out;
    for (const auto& f : list) {
      if (f.rank > INT32_MAX - combined.rank)
        throw std::runtime_error("LoRA: combined rank exceeds supported dimensions");
      combined.rank += f.rank;
      combined.a.insert(combined.a.end(), f.a.begin(), f.a.end());
    }
    combined.b.reserve(static_cast<size_t>(combined.out) * combined.rank);
    for (int row = 0; row < combined.out; ++row)
      for (const auto& f : list)
        combined.b.insert(combined.b.end(), f.b.begin() + static_cast<size_t>(row) * f.rank,
                          f.b.begin() + static_cast<size_t>(row + 1) * f.rank);
    list.clear();
    list.push_back(std::move(combined));
  }
  factors_ = std::move(next);
  adaln_bias_ = std::move(next_bias);
}

bool LoraAdapters::has_adaln(const std::string& name) const {
  return adaln_target(name) && find(name);
}

std::vector<float> LoraAdapters::merged_adaln_weight(const SafeTensors& base,
                                                     const std::string& name) const {
  if (!adaln_target(name))
    throw std::runtime_error("LoRA: unsupported AdaLN merge " + name);
  const auto& weight = base.at(name + ".weight");
  auto result = values(weight);
  if (const auto* updates = find(name))
    for (const auto& f : *updates) {
      if (weight.shape != std::vector<int64_t>{f.out, f.in})
        throw std::runtime_error("LoRA: AdaLN merge dimensions do not match " + name);
      for (int row = 0; row < f.out; ++row)
        for (int col = 0; col < f.in; ++col) {
          double delta = 0;
          for (int r = 0; r < f.rank; ++r)
            delta += double(f.b[size_t(row) * f.rank + r]) * f.a[size_t(r) * f.in + col];
          float& value = result[size_t(row) * f.in + col];
          value = static_cast<float>(double(value) + delta);
          if (!std::isfinite(value))
            throw std::runtime_error("LoRA: merged AdaLN weight overflows " + name);
        }
    }
  return result;
}

std::vector<float> LoraAdapters::merged_adaln_bias(const SafeTensors& base,
                                                   const std::string& name) const {
  if (!adaln_target(name))
    throw std::runtime_error("LoRA: unsupported AdaLN bias merge " + name);
  auto result = values(base.at(name + ".bias"));
  if (auto it = adaln_bias_.find(name); it != adaln_bias_.end()) {
    if (result.size() != it->second.size())
      throw std::runtime_error("LoRA: AdaLN bias shape mismatch " + name);
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] += it->second[i];
      if (!std::isfinite(result[i]))
        throw std::runtime_error("LoRA: merged AdaLN bias overflows " + name);
    }
  }
  return result;
}

std::vector<float> LoraAdapters::merged_endpoint_weight(const SafeTensors& base,
                                                        const std::string& name) const {
  if (name != "video_patch_proj" && name != "final_layer.video_out")
    throw std::runtime_error("LoRA: unsupported endpoint merge " + name);
  const auto& weight = base.at(name + ".weight");
  auto result = values(weight);
  if (const auto* updates = find(name))
    for (const auto& f : *updates) {
      if (weight.shape != std::vector<int64_t>{f.out, f.in})
        throw std::runtime_error("LoRA: endpoint merge dimensions do not match " + name);
      // Endpoints are small, unquantized F32 matrices. Merge once on the host;
      // both backends then use their existing F32 input/output GEMMs.
      for (int row = 0; row < f.out; ++row)
        for (int col = 0; col < f.in; ++col) {
          double delta = 0;
          for (int r = 0; r < f.rank; ++r)
            delta += double(f.b[size_t(row) * f.rank + r]) * f.a[size_t(r) * f.in + col];
          float& value = result[size_t(row) * f.in + col];
          value = static_cast<float>(double(value) + delta);
          if (!std::isfinite(value))
            throw std::runtime_error("LoRA: merged endpoint overflows " + name);
        }
    }
  return result;
}
} // namespace slopfab
