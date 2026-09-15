#include "slopfab/lora.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "slopfab/nf4.h"
#include "slopfab/json.h"
#include "slopfab/tensor_convert.h"

namespace slopfab {
namespace {
bool ends(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size()-suffix.size(), suffix.size(), suffix) == 0;
}
std::string normalize(std::string name) {
  for (const char* p : {"base_model.model.", "model.diffusion_model.", "diffusion_model."}) {
    const std::string prefix(p);
    if (name.compare(0, prefix.size(), prefix) == 0) name.erase(0, prefix.size());
  }
  return name;
}
std::string projection_name(std::string name) {
  if (name == "proj_in") return "video_patch_proj";
  if (name == "proj_out") return "final_layer.video_out";
  const std::string prefix = "transformer_blocks.";
  if (name.compare(0, prefix.size(), prefix) != 0) return name;
  name.replace(0, prefix.size(), "blocks.");
  for (const auto& mapping : std::vector<std::pair<std::string, std::string>>{
      {".attn.to_out.0", ".attn.out_proj"},
      {".ff.net.0.proj", ".mlp.fc1"}, {".ff.net.2", ".mlp.fc2"}}) {
    if (ends(name, mapping.first)) {
      name.replace(name.size() - mapping.first.size(), mapping.first.size(), mapping.second);
      break;
    }
  }
  return name;
}
bool split_qkv(const std::string& name) {
  return ends(name, ".attn.to_q") || ends(name, ".attn.to_k") || ends(name, ".attn.to_v");
}
bool supported(const std::string& name) {
  if (name == "video_patch_proj" || name == "final_layer.video_out") return true;
  size_t start = 0;
  if (name.compare(0, 7, "blocks.") == 0) start = 7;
  else if (name.compare(0, 21, "token_refiner.blocks.") == 0) start = 21;
  else return false;
  const size_t dot = name.find('.', start);
  if (dot == std::string::npos || dot == start) return false;
  for (size_t i = start; i < dot; ++i) if (name[i] < '0' || name[i] > '9') return false;
  const std::string tail = name.substr(dot + 1);
  return tail == "attn.qkv_proj" || tail == "attn.out_proj" || tail == "mlp.fc1" ||
         tail == "mlp.fc2" || split_qkv(name);
}
std::vector<float> values(const TensorView& tensor) {
  if (tensor.dtype != DType::kF32 && tensor.dtype != DType::kF16 && tensor.dtype != DType::kBF16)
    throw std::runtime_error("LoRA: expected F32/F16/BF16 for " + tensor.name);
  auto result = to_f32(tensor);
  for (float v : result) if (!std::isfinite(v))
    throw std::runtime_error("LoRA: non-finite value in " + tensor.name);
  return result;
}
}  // namespace

const std::vector<LoraFactors>* LoraAdapters::find(const std::string& name) const {
  const auto it = factors_.find(name);
  return it == factors_.end() ? nullptr : &it->second;
}

void LoraAdapters::load(const std::vector<LoraSpec>& specs, const SafeTensors& base) {
  decltype(factors_) next;
  for (const LoraSpec& spec : specs) {
    if (spec.path.empty() || !std::isfinite(spec.strength))
      throw std::runtime_error("LoRA: path must be nonempty and strength finite");
    SafeTensors adapter;
    adapter.open(spec.path);
    float metadata_alpha = 0;
    bool has_metadata_alpha = false;
    if (const auto it = adapter.metadata().find("lora_adapter_metadata");
        it != adapter.metadata().end()) {
      const auto config = json::parse(it->second);
      if (!config.is_object()) throw std::runtime_error("LoRA: adapter metadata must be an object");
      for (const char* flag : {"use_dora", "use_rslora", "fan_in_fan_out", "lora_bias"})
        if (const auto* value = config.find(flag); value && !value->is_null() && value->as_bool())
          throw std::runtime_error(std::string("LoRA: unsupported adapter setting ") + flag);
      if (const auto* pattern = config.find("alpha_pattern"); pattern && !pattern->is_null() &&
          !pattern->as_object().empty())
        throw std::runtime_error("LoRA: per-target alpha_pattern is unsupported");
      if (const auto* alpha = config.find("lora_alpha")) {
        metadata_alpha = static_cast<float>(alpha->as_number());
        if (!std::isfinite(metadata_alpha)) throw std::runtime_error("LoRA: non-finite metadata alpha");
        has_metadata_alpha = true;
      }
    }
    struct Pair {
      const TensorView *a = nullptr, *b = nullptr, *alpha = nullptr;
      bool swap_ffn = false;
    };
    std::map<std::string, Pair> pairs;
    for (const auto& item : adapter.tensors()) {
      const std::string key = normalize(item.first);
      bool matched = false;
      for (const auto& suffix : std::vector<std::pair<std::string, int>>{
          {".lora_A.weight", 0}, {".lora_B.weight", 1},
          {".lora_A.default.weight", 0}, {".lora_B.default.weight", 1},
          {".lora_down.weight", 0}, {".lora_up.weight", 1}, {".alpha", 2}}) {
        if (!ends(key, suffix.first)) continue;
        const std::string name = projection_name(key.substr(0, key.size() - suffix.first.size()));
        if (!supported(name)) throw std::runtime_error("LoRA: unsupported target " + name);
        Pair& pair = pairs[name];
        if (key.compare(0, 19, "transformer_blocks.") == 0 &&
            key.find(".ff.net.0.proj.") != std::string::npos)
          pair.swap_ffn = true;
        const TensorView*& slot = suffix.second == 0 ? pair.a : suffix.second == 1 ? pair.b : pair.alpha;
        if (slot) throw std::runtime_error("LoRA: duplicate tensor for " + name);
        slot = &item.second;
        matched = true;
        break;
      }
      if (!matched) throw std::runtime_error("LoRA: unsupported tensor " + item.first);
    }
    if (pairs.empty()) throw std::runtime_error("LoRA: adapter contains no projections: " + spec.path);
    for (const auto& item : pairs) {
      const std::string& name = item.first;
      const Pair& p = item.second;
      if (!p.a || !p.b) throw std::runtime_error("LoRA: missing A/B partner for " + name);
      if (p.a->shape.size() != 2 || p.b->shape.size() != 2 ||
          p.a->shape[0] <= 0 || p.a->shape[1] <= 0 || p.b->shape[0] <= 0 ||
          p.a->shape[0] != p.b->shape[1] ||
          p.a->shape[0] > INT32_MAX || p.a->shape[1] > INT32_MAX || p.b->shape[0] > INT32_MAX)
        throw std::runtime_error("LoRA: invalid matrix shapes for " + name);
      LoraFactors f;
      f.rank = static_cast<int>(p.a->shape[0]);
      f.in = static_cast<int>(p.a->shape[1]);
      f.out = static_cast<int>(p.b->shape[0]);
      const bool split = split_qkv(name);
      const std::string base_name = split ? name.substr(0, name.size() - 4) + "qkv_proj" : name;
      const TensorView& w = base.at(base_name + ".weight");
      const bool packed = w.dtype == DType::kU8 && base.find(base_name + ".weight_scale");
      const bool nf4 = is_nf4_weight(base, base_name + ".weight");
      const auto shape = nf4 ? read_nf4_state(base, base_name + ".weight", "LoRA").shape : w.shape;
      if (shape != std::vector<int64_t>{int64_t(f.out) * (split ? 3 : 1), packed ? f.in / 2 : f.in} || (packed && f.in % 2))
        throw std::runtime_error("LoRA: base model dimensions do not match " + name);
      if ((name == "video_patch_proj" || name == "final_layer.video_out") &&
          (w.dtype != DType::kF32 && w.dtype != DType::kF16 && w.dtype != DType::kBF16))
        throw std::runtime_error("LoRA: video endpoint must have floating-point weights: " + name);
      // H3 AWQ can fold scales into preceding norms. There is no reliable
      // inverse coordinate transform for an arbitrary adapter in that case.
      if (base.find(base_name + ".pre_quant_scale"))
        throw std::runtime_error("LoRA: AWQ activation scaling is unsupported for " + name);
      float alpha = has_metadata_alpha ? metadata_alpha : static_cast<float>(f.rank);
      if (p.alpha) {
        if (p.alpha->numel() != 1) throw std::runtime_error("LoRA: alpha must be scalar for " + name);
        alpha = values(*p.alpha)[0];
      }
      f.a = values(*p.a);
      f.b = values(*p.b);
      if (p.swap_ffn) {
        if (f.out % 2) throw std::runtime_error("LoRA: SwiGLU output must have two equal halves");
        // Diffusers SwiGLU stores [value; gate]; native H3 stores [gate; value].
        std::rotate(f.b.begin(), f.b.begin() + f.b.size() / 2, f.b.end());
      }
      const float scale = spec.strength * (alpha / static_cast<float>(f.rank));
      if (!std::isfinite(scale)) throw std::runtime_error("LoRA: non-finite scale for " + name);
      for (float& v : f.b) {
        v *= scale;
        if (!std::isfinite(v) || !std::isfinite(bf16_to_f32(f32_to_bf16(v))))
          throw std::runtime_error("LoRA: scaled B overflows BF16 for " + name);
      }
      for (float v : f.a) if (!std::isfinite(bf16_to_f32(f32_to_bf16(v))))
        throw std::runtime_error("LoRA: A overflows BF16 for " + name);
      if (scale != 0.0f) next[name].push_back(std::move(f));
    }
  }
  // Concatenation represents sum_i B_i A_i as one low-rank product. This
  // keeps dispatch count constant even with several adapters on Vulkan.
  for (auto& entry : next) {
    auto& list = entry.second;
    if (list.size() < 2) continue;
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
    for (int row = 0; row < combined.out; ++row) for (const auto& f : list)
      combined.b.insert(combined.b.end(), f.b.begin() + static_cast<size_t>(row) * f.rank,
                         f.b.begin() + static_cast<size_t>(row + 1) * f.rank);
    list.clear();
    list.push_back(std::move(combined));
  }
  factors_ = std::move(next);
}

std::vector<float> LoraAdapters::merged_endpoint_weight(
    const SafeTensors& base, const std::string& name) const {
  if (name != "video_patch_proj" && name != "final_layer.video_out")
    throw std::runtime_error("LoRA: unsupported endpoint merge " + name);
  const auto& weight = base.at(name + ".weight");
  auto result = values(weight);
  if (const auto* updates = find(name)) for (const auto& f : *updates) {
    if (weight.shape != std::vector<int64_t>{f.out, f.in})
      throw std::runtime_error("LoRA: endpoint merge dimensions do not match " + name);
    // Endpoints are small, unquantized F32 matrices. Merge once on the host;
    // both backends then use their existing F32 input/output GEMMs.
    for (int row = 0; row < f.out; ++row) for (int col = 0; col < f.in; ++col) {
      double delta = 0;
      for (int r = 0; r < f.rank; ++r)
        delta += double(f.b[size_t(row) * f.rank + r]) * f.a[size_t(r) * f.in + col];
      float& value = result[size_t(row) * f.in + col];
      value = static_cast<float>(double(value) + delta);
      if (!std::isfinite(value)) throw std::runtime_error("LoRA: merged endpoint overflows " + name);
    }
  }
  return result;
}
}  // namespace slopfab
