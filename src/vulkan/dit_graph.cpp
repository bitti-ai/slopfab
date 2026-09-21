#include "slopfab/vulkan/dit_graph.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace slopfab::vulkan {
namespace {

void validate_graph_config(const H3MainGraphConfig& config) {
  if (config.layers == 0 || config.layers > 50)
    throw std::invalid_argument("Vulkan H3 graph: layer count must be in [1,50]");
  // Reuse the single-stage's complete shape/capability-independent checks.
  if (config.block.sequence == 0 || config.block.hidden == 0 || config.block.heads == 0 ||
      config.block.head_dim != 128 ||
      uint64_t(config.block.heads) * config.block.head_dim > UINT32_MAX || config.block.ffn == 0 ||
      config.block.timesteps == 0 || config.block.modalities == 0 || config.block.adaln_rank == 0)
    throw std::invalid_argument("Vulkan H3 graph: invalid block configuration");
}

} // namespace

struct ExactH3MainGraph::Impl {
  TensorContext* context = nullptr;
  H3MainGraphConfig config;
  ExactH3BlockScratch scratch;
  std::vector<ExactH3BlockStage> stages;

  Impl(TensorContext& owner, const H3MainGraphConfig& value) : context(&owner), config(value) {
  }
};

ExactH3MainGraph::ExactH3MainGraph() = default;
ExactH3MainGraph::~ExactH3MainGraph() = default;

ExactH3MainGraph::ExactH3MainGraph(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {
}

ExactH3MainGraph::ExactH3MainGraph(ExactH3MainGraph&&) noexcept = default;
ExactH3MainGraph& ExactH3MainGraph::operator=(ExactH3MainGraph&&) noexcept = default;

ExactH3MainGraph ExactH3MainGraph::create(TensorContext& context, const H3MainGraphConfig& config) {
  validate_graph_config(config);
  return ExactH3MainGraph(std::make_shared<Impl>(context, config));
}

void ExactH3MainGraph::load(const SafeTensors& checkpoint) {
  if (!impl_)
    throw std::logic_error("Vulkan H3 graph: empty graph");
  // A second 50-layer allocation would temporarily double the graph's 10+ GiB
  // persistent footprint. Make the bounded lifetime policy explicit instead:
  // callers must unload before loading another checkpoint.
  if (loaded())
    throw std::logic_error("Vulkan H3 graph: unload before load");
  // A corrupt archive, including layer 49's final projection metadata, fails
  // before allocating scratch or any weight. Validation is streamed one
  // layer at a time and retains no converted host tensor between calls.
  for (uint32_t layer = 0; layer < impl_->config.layers; ++layer)
    ExactH3BlockStage::validate_checkpoint(checkpoint, layer, impl_->config.block);

  ExactH3BlockScratch next_scratch =
      ExactH3BlockScratch::create(*impl_->context, impl_->config.block);
  std::vector<ExactH3BlockStage> next;
  next.reserve(impl_->config.layers);
  for (uint32_t layer = 0; layer < impl_->config.layers; ++layer) {
    ExactH3BlockStage stage = ExactH3BlockStage::create(*impl_->context, impl_->config.block);
    stage.load(checkpoint, layer);
    stage.prepare(next_scratch);
    next.push_back(std::move(stage));
  }
  impl_->scratch = std::move(next_scratch);
  impl_->stages = std::move(next);
}

void ExactH3MainGraph::unload() noexcept {
  if (impl_) {
    // Release weights before the shared arena/cache so no prepared stage can
    // retain a scratch allocation through its last reference.
    impl_->stages.clear();
    impl_->scratch = ExactH3BlockScratch();
  }
}

bool ExactH3MainGraph::loaded() const noexcept {
  return impl_ && impl_->stages.size() == impl_->config.layers;
}

uint32_t ExactH3MainGraph::layers() const noexcept {
  return impl_ ? impl_->config.layers : 0;
}

const H3MainGraphConfig& ExactH3MainGraph::config() const noexcept {
  return impl_->config;
}

uint32_t ExactH3MainGraph::required_operators(const H3MainGraphReplayTaps* taps) const {
  return required_operators(0, layers(), taps);
}

uint32_t ExactH3MainGraph::required_operators(uint32_t first_layer, uint32_t layer_count,
                                              const H3MainGraphReplayTaps* taps) const {
  if (!loaded())
    throw std::logic_error("Vulkan H3 graph: not loaded");
  if (layer_count == 0 || first_layer > impl_->config.layers ||
      layer_count > impl_->config.layers - first_layer)
    throw std::invalid_argument("Vulkan H3 graph: invalid layer span");
  if (taps && (taps->count != impl_->config.layers || !taps->boundaries))
    throw std::invalid_argument("Vulkan H3 graph: invalid boundary taps");
  uint64_t total = 0;
  for (uint32_t layer = first_layer; layer < first_layer + layer_count; ++layer) {
    H3BlockReplayTaps block_tap;
    if (taps)
      block_tap.final_residual = &taps->boundaries[layer];
    total += impl_->stages[layer].required_operators(taps ? &block_tap : nullptr);
  }
  if (total > UINT32_MAX)
    throw std::overflow_error("Vulkan H3 graph: operator count overflow");
  return static_cast<uint32_t>(total);
}

void ExactH3MainGraph::record(TensorBatch& batch, DeviceTensor& tokens, DeviceTensor& selectors,
                              DeviceTensor& code, DeviceTensor& cosine, DeviceTensor& sine,
                              const H3AttentionRanges* ranges,
                              const H3MainGraphReplayTaps* taps) const {
  record_layers(batch, tokens, selectors, code, cosine, sine, 0, layers(), ranges, taps);
}

uint32_t ExactH3MainGraph::preflight(DeviceTensor& tokens, DeviceTensor& selectors,
                                     DeviceTensor& code, DeviceTensor& cosine, DeviceTensor& sine,
                                     const H3AttentionRanges* ranges,
                                     const H3MainGraphReplayTaps* taps) const {
  return preflight_layers(tokens, selectors, code, cosine, sine, 0, layers(), ranges, taps);
}

uint32_t ExactH3MainGraph::preflight_layers(DeviceTensor& tokens, DeviceTensor& selectors,
                                            DeviceTensor& code, DeviceTensor& cosine,
                                            DeviceTensor& sine, uint32_t first_layer,
                                            uint32_t layer_count, const H3AttentionRanges* ranges,
                                            const H3MainGraphReplayTaps* taps) const {
  if (!loaded())
    throw std::logic_error("Vulkan H3 graph: not loaded");
  if (layer_count == 0 || first_layer > impl_->config.layers ||
      layer_count > impl_->config.layers - first_layer)
    throw std::invalid_argument("Vulkan H3 graph: invalid layer span");
  const H3BlockConfig& c = impl_->config.block;
  if (c.vsa_tiles && ranges)
    throw std::invalid_argument("Vulkan VSA: frame-banded attention is incompatible");
  const auto tv = tokens.view(), sv = selectors.view(), cv = code.view();
  const auto cosv = cosine.view(), sinv = sine.view();
  if (!impl_->context->owns(tokens) || !impl_->context->owns(selectors) ||
      !impl_->context->owns(code) || !impl_->context->owns(cosine) || !impl_->context->owns(sine) ||
      tv.type != ScalarType::kBFloat16 || tv.layout.rank != 2 ||
      tv.layout.extent[0] != c.sequence || tv.layout.extent[1] != c.hidden ||
      !tv.layout.is_contiguous() || sv.type != ScalarType::kInt32 || sv.layout.rank != 1 ||
      sv.layout.extent[0] != c.sequence || !sv.layout.is_contiguous() ||
      cv.type != ScalarType::kFloat32 || cv.layout.rank != 2 ||
      cv.layout.extent[0] != c.timesteps || cv.layout.extent[1] != c.adaln_rank ||
      !cv.layout.is_contiguous() || cosv.type != ScalarType::kFloat32 ||
      sinv.type != ScalarType::kFloat32 || cosv.layout.rank != 2 || sinv.layout.rank != 2 ||
      cosv.layout.extent[0] != c.sequence || sinv.layout.extent[0] != c.sequence ||
      cosv.layout.extent[1] != 96 || sinv.layout.extent[1] != 96 || !cosv.layout.is_contiguous() ||
      !sinv.layout.is_contiguous() ||
      (ranges && (ranges->sequence() != c.sequence || !ranges->belongs_to(*impl_->context))))
    throw std::invalid_argument("Vulkan H3 graph: invalid activation inputs");
  const std::array<uintptr_t, 5> input_resources{tv.resource, sv.resource, cv.resource,
                                                 cosv.resource, sinv.resource};
  for (size_t i = 0; i < input_resources.size(); ++i)
    for (size_t j = 0; j < i; ++j)
      if (input_resources[i] == input_resources[j])
        throw std::invalid_argument("Vulkan H3 graph: aliased activation inputs");

  if (taps) {
    if (taps->count != impl_->config.layers || !taps->boundaries)
      throw std::invalid_argument("Vulkan H3 graph: invalid boundary taps");
    std::vector<uintptr_t> resources;
    resources.reserve(taps->count + 5);
    resources.insert(resources.end(), input_resources.begin(), input_resources.end());
    for (uint32_t layer = 0; layer < taps->count; ++layer) {
      DeviceTensor& tensor = taps->boundaries[layer];
      const DeviceTensorView view = tensor.view();
      if (!impl_->context->owns(tensor) || view.type != ScalarType::kBFloat16 ||
          view.layout.rank != 2 || view.layout.extent[0] != c.sequence ||
          view.layout.extent[1] != c.hidden || !view.layout.is_contiguous())
        throw std::invalid_argument("Vulkan H3 graph: invalid boundary tap");
      for (uintptr_t resource : resources)
        if (resource == view.resource)
          throw std::invalid_argument("Vulkan H3 graph: aliased boundary tap");
      resources.push_back(view.resource);
    }
  }
  return required_operators(first_layer, layer_count, taps);
}

void ExactH3MainGraph::record_layers(TensorBatch& batch, DeviceTensor& tokens,
                                     DeviceTensor& selectors, DeviceTensor& code,
                                     DeviceTensor& cosine, DeviceTensor& sine, uint32_t first_layer,
                                     uint32_t layer_count, const H3AttentionRanges* ranges,
                                     const H3MainGraphReplayTaps* taps) const {
  const uint32_t operators = preflight_layers(tokens, selectors, code, cosine, sine, first_layer,
                                              layer_count, ranges, taps);
  if (batch.remaining_operator_capacity() < operators)
    throw std::logic_error("Vulkan H3 graph: insufficient batch capacity");

  for (uint32_t layer = first_layer; layer < first_layer + layer_count; ++layer) {
    H3BlockReplayTaps block_tap;
    if (taps)
      block_tap.final_residual = &taps->boundaries[layer];
    impl_->stages[layer].record(batch, tokens, selectors, code, cosine, sine, impl_->scratch,
                                ranges, taps ? &block_tap : nullptr);
  }
}

uint64_t ExactH3MainGraph::persistent_bytes() const noexcept {
  if (!impl_)
    return 0;
  uint64_t total = 0;
  for (const ExactH3BlockStage& stage : impl_->stages) {
    const uint64_t value = stage.persistent_bytes();
    if (value > std::numeric_limits<uint64_t>::max() - total)
      return std::numeric_limits<uint64_t>::max();
    total += value;
  }
  return total;
}

uint64_t ExactH3MainGraph::scratch_bytes() const noexcept {
  return impl_ ? impl_->scratch.reserved_bytes() : 0;
}

uint64_t ExactH3MainGraph::peak_device_bytes() const noexcept {
  const uint64_t persistent = persistent_bytes();
  const uint64_t scratch = scratch_bytes();
  return scratch > std::numeric_limits<uint64_t>::max() - persistent
             ? std::numeric_limits<uint64_t>::max()
             : persistent + scratch;
}

} // namespace slopfab::vulkan
