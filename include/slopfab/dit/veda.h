#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>
#include "slopfab/dit/packing.h"
#include "slopfab/safetensors.h"

namespace slopfab::dit {

struct VedaBudget {
  double value = 0.1;
  bool tiles = false;
  int keep(int real_tokens, int columns, int query_tile) const;
};

struct VedaOptions {
  std::string predictor_path;
  // "trained", a percentage of tiles skipped ("90%"), or a kept tile count.
  std::string generated_sparsity = "trained";
  std::vector<int> full_attention_layers, full_attention_steps;
  bool active(int step, int layer) const;
};

VedaBudget parse_veda_budget(const std::string& text);

using VedaShape = std::array<int, 3>;

struct VedaPlan {
  std::string name;
  VedaShape grid{};
  std::vector<VedaShape> shapes;
  std::vector<std::vector<int>> head_shape;
};

// Video tiles come first, followed by global text/audio/reference tiles.
// Every tile has 128 slots; real packed rows form a prefix, followed by -1.
struct VedaTiles {
  std::vector<int32_t> rows, sizes;
  int video_tiles = 0;
  int video_tokens = 0;
};

VedaTiles build_veda_tiles(const SequenceLayout& layout, const VedaShape& shape);
VedaPlan select_veda_plan(const std::vector<VedaPlan>& plans, const VedaShape& grid);

// The original bundle stays memory mapped. Projection conversion is bounded
// to one layer and rounds to BF16, as in the upstream bundle loader.
class VedaBundle {
public:
  void load(const std::string& path);
  std::vector<uint16_t> projection(int layer, bool query) const;
  int layers = 0, heads = 0, head_dim = 0;
  VedaBudget generated;
  std::vector<VedaPlan> plans;

private:
  SafeTensors file_;
};

// Per-generation state shared by the block stages. References remain global.
struct VedaRequest {
  VedaOptions options;
  VedaBundle bundle;
  SequenceLayout layout;
  VedaPlan plan;
  std::vector<VedaTiles> tilings;
  std::vector<std::vector<int32_t>> keeps;
  int step = -1;
};

std::shared_ptr<VedaRequest> prepare_veda(const VedaOptions& options, const SequenceLayout& layout);

} // namespace slopfab::dit
