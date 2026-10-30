#include "slopfab/dit/hyperflow.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include "slopfab/dit/checkpoint.h"
#include "slopfab/json.h"
#include "slopfab/sha256.h"
#include "slopfab/tensor_convert.h"

namespace slopfab::dit {
namespace {
float number(const std::map<std::string, std::string>& m, const char* key) {
  const double n = json::parse(m.at(key)).as_number();
  const float f = static_cast<float>(n);
  if (!std::isfinite(f))
    throw std::invalid_argument(std::string("Hyperflow: invalid ") + key);
  return f;
}
}

std::optional<HyperflowMetadata>
hyperflow_metadata(const std::map<std::string, std::string>& m) {
  if (!m.count("hyperflow"))
    return {};
  if (m.at("hyperflow") != "true" || m.at("hyperflow_version") != "1.0" ||
      m.at("base_model") != "MiniMaxAI/MiniMax-H3")
    throw std::invalid_argument("Hyperflow: unsupported adapter format or base model");
  HyperflowMetadata result;
  result.gate = number(m, "hyperflow_gate");
  result.alpha = number(m, "lora_alpha");
  const float rank = number(m, "lora_rank");
  if (rank < 1 || rank > 4096 || rank != std::floor(rank))
    throw std::invalid_argument("Hyperflow: invalid rank");
  result.rank = static_cast<int>(rank);
  result.revision = m.at("base_model_revision");
  result.sampling = parse_sampling_settings(
      "{\"version\":1,\"sampler\":\"euler\",\"base_sigmas\":" + m.at("hyperflow_sigmas") +
      ",\"video_sigma_shift\":" + m.at("hyperflow_video_shift") +
      ",\"audio_sigma_shift\":" + m.at("hyperflow_audio_shift") + "}");
  if (result.sampling.base_sigmas->size() != 9 || result.sampling.base_sigmas->front() != 1)
    throw std::invalid_argument("Hyperflow: expected the trained eight-step grid");
  return result;
}

std::shared_ptr<Hyperflow> Hyperflow::load(const SafeTensors& adapter,
                                         const SafeTensors& base, float strength) {
  const auto metadata = hyperflow_metadata(adapter.metadata());
  if (!metadata)
    throw std::invalid_argument("Hyperflow: missing adapter metadata");
  const auto model = resolve_model_descriptor(base);
  if (model.compatibility_architecture != TransformerArchitecture::kPrunedTable &&
      model.compatibility_architecture != TransformerArchitecture::kRef2VAPrunedTable)
    throw std::invalid_argument("Hyperflow requires an ordinary pruned FL2VA or Ref2VA base");
  const std::string variant = model.supports_references ? "ref2va" : "fl2va";
  auto path = std::filesystem::path(adapter.path());
  path.replace_extension(".hyperflow-" + variant + ".safetensors");
  if (!std::filesystem::exists(path))
    throw std::runtime_error("Hyperflow needs its two-time modulation companion: " + path.string() +
        "; run tools/prepare_hyperflow.py with --adapter, --variant " + variant +
        " and --strength " + std::to_string(strength));
  auto result = std::make_shared<Hyperflow>();
  result->archive_.open(path.string());
  const auto& m = result->archive_.metadata();
  if (m.at("slopfab.hyperflow") != "1" || m.at("variant") != variant ||
      m.at("base_model_revision") != metadata->revision || number(m, "strength") != strength)
    throw std::runtime_error("Hyperflow: companion variant, revision or strength mismatch; prepare it again");
  std::string digest;
  for (uint8_t b : sha256_file(adapter.path())) {
    digest += "0123456789abcdef"[b >> 4];
    digest += "0123456789abcdef"[b & 15];
  }
  if (digest != m.at("adapter_sha256"))
    throw std::runtime_error("Hyperflow: companion belongs to different adapter bytes");
  const auto& pairs = result->archive_.at("pairs");
  if (pairs.dtype != DType::kF32 || pairs.shape != std::vector<int64_t>{18, 2})
    throw std::runtime_error("Hyperflow: invalid companion time pairs");
  result->pairs_ = to_f32(pairs);
  // Require exactly both modality intervals plus the two pinned conditions.
  std::vector<float> expected;
  for (float shift : {*metadata->sampling.video_sigma_shift, *metadata->sampling.audio_sigma_shift}) {
    sampler::FlowScheduler scheduler(shift);
    scheduler.set_base_sigmas(*metadata->sampling.base_sigmas);
    for (int i = 0; i < 8; ++i) {
      expected.push_back(scheduler.timesteps()[i]);
      expected.push_back(1.f - scheduler.sigmas()[i + 1]);
    }
  }
  expected.insert(expected.end(), {.999f, .999f, 1.f, 1.f});
  if (result->pairs_ != expected)
    throw std::runtime_error("Hyperflow: companion time pairs do not match trained schedule");
  size_t targets = 0;
  for (const auto& [key, tensor] : base.tensors()) {
    const std::string suffix = ".adaln_proj.linear.weight";
    if (key.size() < suffix.size() || key.compare(key.size() - suffix.size(), suffix.size(), suffix))
      continue;
    const auto name = key.substr(0, key.size() - 7);
    const auto& w = result->archive_.at(name);
    if (tensor.shape.size() != 2 || w.dtype != DType::kF32 ||
        w.shape != std::vector<int64_t>{tensor.shape[0], result->rank()})
      throw std::runtime_error("Hyperflow: invalid modulation shape for " + name);
    for (float v : to_f32(w))
      if (!std::isfinite(v))
        throw std::runtime_error("Hyperflow: non-finite modulation");
    ++targets;
  }
  if (!targets || result->archive_.tensors().size() != targets + 1)
    throw std::runtime_error("Hyperflow: companion modulation targets do not match base");
  return result;
}

std::vector<float> Hyperflow::weight(const std::string& projection) const {
  return to_f32(archive_.at(projection));
}
size_t Hyperflow::weight_bytes() const {
  size_t bytes = 0;
  for (const auto& [key, t] : archive_.tensors())
    if (key != "pairs") bytes += t.nbytes;
  return bytes;
}
std::vector<float> Hyperflow::code(const RowTimesteps& times, size_t rows) const {
  if (times.unique.empty() || times.endpoints.size() != times.unique.size())
    throw std::invalid_argument("Hyperflow requires a (t,r) pair for every distinct timestep");
  if (!rows) rows = times.unique.size();
  if (rows < times.unique.size())
    throw std::invalid_argument("Hyperflow: too few code rows");
  std::vector<float> result(rows * rank(), 0);
  for (size_t i = 0; i < rows; ++i) {
    const auto j = std::min(i, times.unique.size() - 1);
    int match = -1;
    for (int k = 0; k < rank(); ++k)
      if (times.unique[j] == pairs_[2*k] && times.endpoints[j] == pairs_[2*k+1]) match = k;
    if (match < 0)
      throw std::invalid_argument("Hyperflow: time pair outside the prepared schedule");
    result[i * rank() + match] = 1;
  }
  return result;
}

RowTimesteps build_hyperflow_timesteps(const SequenceLayout& layout, const PackedIndices& idx,
                                       float vt, float vr, float at, float ar) {
  using Pair = std::pair<float, float>;
  if (idx.tags.size() != size_t(layout.total_rows()) || layout.num_condition_video < 0 ||
      layout.num_condition_audio < 0 || size_t(layout.num_condition_video) > idx.video.size() ||
      size_t(layout.num_condition_audio) > idx.audio.size())
    throw std::invalid_argument("Hyperflow: invalid packed row layout");
  for (float v : {vt, vr, at, ar})
    if (!std::isfinite(v) || v < 0 || v > 1)
      throw std::invalid_argument("Hyperflow: invalid row time");
  std::vector<Pair> pairs(idx.tags.size(), {vt, vr});
  for (int i = 0; i < layout.num_condition_video; ++i)
    pairs.at(idx.video[i]) = {.999f, .999f};
  for (size_t i = 0; i < idx.audio.size(); ++i)
    pairs.at(idx.audio[i]) = i < size_t(layout.num_condition_audio) ? Pair{1, 1} : Pair{at, ar};
  auto unique = pairs;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  RowTimesteps out;
  for (const auto& p : unique) {
    out.unique.push_back(p.first);
    out.endpoints.push_back(p.second);
  }
  for (size_t i = 0; i < pairs.size(); ++i) {
    const int index = int(std::lower_bound(unique.begin(), unique.end(), pairs[i]) - unique.begin());
    out.indices.push_back(index);
    out.adaln.push_back(index * 3 + std::max(idx.tags[i], 0));
  }
  return out;
}
} // namespace slopfab::dit
