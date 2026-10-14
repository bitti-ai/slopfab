// Native implementation of the VEDA predictor bundle and tiling contract:
// https://github.com/veda-sparse/Veda-on-ComfyUI/tree/main/veda_comfy/core
#include "slopfab/dit/veda.h"
#include "slopfab/json.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace slopfab::dit {
namespace {
void require(bool ok, const std::string& message) {
  if (!ok)
    throw std::invalid_argument("VEDA: " + message);
}

double number(const std::string& text) {
  size_t end = 0;
  const double value = std::stod(text, &end);
  require(end == text.size() && std::isfinite(value), "invalid number '" + text + "'");
  return value;
}

int integer(double value) {
  require(std::isfinite(value) && value > 0 && value <= 1048576 && std::floor(value) == value,
          "invalid positive dimension");
  return static_cast<int>(value);
}

void check_shape(const VedaShape& s) {
  require(s[0] > 0 && s[0] <= 128 && s[1] > 0 && s[1] <= 128 && s[2] > 0 && s[2] <= 128 &&
              s[0] * s[1] * s[2] == 128,
          "tile shape must contain 128 tokens");
}

int64_t tile_count(const VedaShape& grid, const VedaShape& shape) {
  int64_t count = 1;
  for (int i = 0; i < 3; ++i)
    count *= (int64_t(grid[i]) + shape[i] - 1) / shape[i];
  return count;
}

float element(const TensorView& t, size_t i) {
  switch (t.dtype) {
  case DType::kF32:
    return static_cast<const float*>(t.data)[i];
  case DType::kBF16:
    return bf16_to_f32(static_cast<const uint16_t*>(t.data)[i]);
  case DType::kF8E4M3:
    return f8_e4m3_to_f32(static_cast<const uint8_t*>(t.data)[i]);
  default:
    throw std::invalid_argument("VEDA: unsupported tensor dtype for " + t.name);
  }
}
} // namespace

VedaBudget parse_veda_budget(const std::string& text) {
  require(!text.empty(), "empty sparsity");
  if (text.back() == '%') {
    const double percent = number(text.substr(0, text.size() - 1));
    require(percent >= 0 && percent < 100, "sparsity must be in [0, 100)%");
    return {1 - percent / 100, false};
  }
  const double count = number(text);
  require(count > 0, "kept tile count must be positive");
  return {count, true};
}

int VedaBudget::keep(int real_tokens, int columns, int query_tile) const {
  require(std::isfinite(value) && value > 0 && real_tokens > 0 && columns > 0 && query_tile >= 0,
          "invalid tile budget");
  if (!tiles && value >= 1)
    return columns;
  const double ideal = (int64_t(real_tokens) + 127) / 128;
  const double budget = tiles ? value : value * ideal * ideal / columns;
  if (budget >= columns)
    return columns;
  const int low = std::max(1, static_cast<int>(std::floor(budget)));
  const double fraction = std::round(std::clamp(budget - low, 0.0, 1.0) * 1e12) / 1e12;
  return std::min(columns, low + int(std::floor((query_tile + 1.0) * fraction) >
                                     std::floor(query_tile * fraction)));
}

bool VedaOptions::active(int step, int layer) const {
  return layer >= 0 &&
         std::find(full_attention_layers.begin(), full_attention_layers.end(), layer) ==
             full_attention_layers.end() &&
         std::find(full_attention_steps.begin(), full_attention_steps.end(), step) ==
             full_attention_steps.end();
}

VedaTiles build_veda_tiles(const SequenceLayout& l, const VedaShape& s) {
  check_shape(s);
  require(l.num_text >= 0 && l.num_condition_video >= 0 && l.num_condition_audio >= 0 &&
              l.num_audio_rows >= 0 && l.num_latent_frames > 0 && l.latent_height > 0 &&
              l.latent_width > 0 && l.latent_height % 2 == 0 && l.latent_width % 2 == 0,
          "invalid H3 sequence geometry");
  const VedaShape grid{l.num_latent_frames, l.latent_height / 2, l.latent_width / 2};
  const int64_t video = int64_t(grid[0]) * grid[1] * grid[2];
  const int64_t prefix =
      int64_t(l.num_text) + l.num_condition_video + l.num_condition_audio + l.num_audio_rows;
  require(video == l.num_video_rows && prefix + video <= std::numeric_limits<int>::max(),
          "video grid does not match packed sequence");
  const int64_t count = tile_count(grid, s) + (prefix + 127) / 128;
  require(count <= 4096, "sequence exceeds the 4096-tile CUDA limit for this plan");
  VedaTiles out;
  out.rows.reserve(size_t(count) * 128);
  out.sizes.reserve(size_t(count));
  for (int hb = 0; hb < grid[1]; hb += s[1])
    for (int wb = 0; wb < grid[2]; wb += s[2])
      for (int tb = 0; tb < grid[0]; tb += s[0]) {
        const size_t begin = out.rows.size();
        for (int t = tb; t < std::min(tb + s[0], grid[0]); ++t)
          for (int h = hb; h < std::min(hb + s[1], grid[1]); ++h)
            for (int w = wb; w < std::min(wb + s[2], grid[2]); ++w)
              out.rows.push_back(int(prefix) + (t * grid[1] + h) * grid[2] + w);
        out.sizes.push_back(int(out.rows.size() - begin));
        out.rows.resize(begin + 128, -1);
      }
  out.video_tiles = int(out.sizes.size());
  out.video_tokens = l.num_video_rows;
  for (int begin = 0; begin < prefix; begin += 128) {
    const int n = int(std::min<int64_t>(128, prefix - begin));
    for (int i = 0; i < n; ++i)
      out.rows.push_back(begin + i);
    out.rows.resize(out.rows.size() + 128 - n, -1);
    out.sizes.push_back(n);
  }
  return out;
}

VedaPlan select_veda_plan(const std::vector<VedaPlan>& plans, const VedaShape& grid) {
  require(!plans.empty() && grid[0] > 0 && grid[1] > 0 && grid[2] > 0,
          "missing plan or invalid grid");
  auto cost = [&](const VedaPlan& p) {
    int64_t padding = 0;
    for (const auto& shape : p.shapes)
      padding += tile_count(grid, shape);
    return std::make_tuple(std::round(std::abs(std::log(double(p.grid[2]) / p.grid[1]) -
                                               std::log(double(grid[2]) / grid[1])) *
                                      1000),
                           std::abs(p.grid[0] - grid[0]), padding);
  };
  VedaPlan best = plans.front();
  for (int transpose = 0; transpose < 2; ++transpose)
    for (auto p : plans) {
      if (transpose) {
        std::swap(p.grid[1], p.grid[2]);
        for (auto& s : p.shapes)
          std::swap(s[1], s[2]);
        p.name += "_T";
      }
      if (cost(p) < cost(best))
        best = std::move(p);
    }
  return best;
}

void VedaBundle::load(const std::string& path) {
  file_.open(path);
  const auto& m = file_.metadata();
  require(m.count("format") && m.at("format") == "miowtion-veda-predictor-v1",
          "not a predictor bundle");
  layers = integer(number(m.at("num_layers")));
  heads = integer(number(m.at("num_heads")));
  head_dim = integer(number(m.at("head_dim")));
  require(layers <= 128 && heads <= 128 && (head_dim == 64 || head_dim == 128),
          "unsupported predictor dimensions");
  if (m.count("target_budget_kind")) {
    const auto& kind = m.at("target_budget_kind");
    require(kind == "ratio" || kind == "tiles", "unknown target budget kind");
    generated = {number(m.at("target_budget_value")), kind == "tiles"};
  } else
    generated = {number(m.at("keep_ratio")), false};
  (void)generated.keep(128, 1, 0);
  const std::string dtype = m.count("dtype") ? m.at("dtype") : "float32";
  const DType storage = dtype == "float32"         ? DType::kF32
                        : dtype == "bfloat16"      ? DType::kBF16
                        : dtype == "float8_e4m3fn" ? DType::kF8E4M3
                                                   : DType::kUnknown;
  require(storage != DType::kUnknown, "unknown predictor storage dtype");
  size_t expected = 0;
  for (int layer = 0; layer < layers; ++layer)
    for (const char* suffix : {".proj_q", ".proj_k"}) {
      const std::string key = "layers." + std::to_string(layer) + suffix;
      const auto& t = file_.at(key);
      require(t.shape == std::vector<int64_t>{heads, 3 * head_dim, head_dim} && t.dtype == storage,
              "invalid projection " + key);
      ++expected;
      if (storage == DType::kF8E4M3) {
        const auto& scale = file_.at(key + ".__scale");
        require(scale.shape == std::vector<int64_t>{heads} && scale.dtype == DType::kF32,
                "invalid per-head FP8 scale for " + key);
        for (int h = 0; h < heads; ++h)
          require(std::isfinite(element(scale, h)) && element(scale, h) > 0, "invalid FP8 scale");
        ++expected;
      }
    }
  require(file_.tensor_count() == expected, "unexpected tensors in predictor bundle");
  plans.clear();
  const auto doc = json::parse(m.at("plans"));
  for (const auto& entry : doc.as_object()) {
    const auto& obj = entry.second.as_object();
    VedaPlan p;
    p.name = obj.at("geometry").as_string();
    const auto& grid = obj.at("grid").as_array();
    require(grid.size() == 3, "plan grid must have three dimensions");
    for (int i = 0; i < 3; ++i)
      p.grid[i] = integer(grid[i].as_number());
    for (const auto& v : obj.at("shapes").as_array()) {
      const auto& text = v.as_string();
      const auto a = text.find('x'), b = text.find('x', a == std::string::npos ? 0 : a + 1);
      require(a != std::string::npos && b != std::string::npos, "invalid tile shape");
      VedaShape s{integer(number(text.substr(0, a))),
                  integer(number(text.substr(a + 1, b - a - 1))),
                  integer(number(text.substr(b + 1)))};
      check_shape(s);
      p.shapes.push_back(s);
    }
    require(!p.shapes.empty(), "empty tile shapes");
    for (const auto& row : obj.at("head_shape").as_array()) {
      std::vector<int> indices;
      for (const auto& v : row.as_array()) {
        const double index = v.as_number();
        require(std::isfinite(index) && index >= 0 && index < p.shapes.size() &&
                    std::floor(index) == index,
                "invalid head shape index");
        indices.push_back(int(index));
      }
      require(indices.size() == size_t(heads), "plan head count mismatch");
      p.head_shape.push_back(std::move(indices));
    }
    require(p.head_shape.size() == size_t(layers), "plan layer count mismatch");
    plans.push_back(std::move(p));
  }
  require(!plans.empty(), "bundle has no tile plans");
}

std::vector<uint16_t> VedaBundle::projection(int layer, bool query) const {
  require(layer >= 0 && layer < layers, "predictor layer out of range");
  const auto key = "layers." + std::to_string(layer) + (query ? ".proj_q" : ".proj_k");
  const auto& t = file_.at(key);
  const auto* scales = file_.find(key + ".__scale");
  std::vector<uint16_t> result(size_t(t.numel()));
  const size_t per_head = size_t(3) * head_dim * head_dim;
  for (size_t i = 0; i < result.size(); ++i) {
    const float value = element(t, i) * (scales ? element(*scales, i / per_head) : 1.0f);
    if (!std::isfinite(value))
      throw std::invalid_argument("VEDA: nonfinite predictor weight");
    result[i] = f32_to_bf16(value);
    if (!std::isfinite(bf16_to_f32(result[i])))
      throw std::invalid_argument("VEDA: predictor weight overflows BF16");
  }
  return result;
}

std::shared_ptr<VedaRequest> prepare_veda(const VedaOptions& options,
                                          const SequenceLayout& layout) {
  auto result = std::make_shared<VedaRequest>();
  result->options = options;
  result->layout = layout;
  result->bundle.load(options.predictor_path);
  result->plan =
      select_veda_plan(result->bundle.plans, {layout.num_latent_frames, layout.latent_height / 2,
                                              layout.latent_width / 2});
  const auto budget = options.generated_sparsity == "trained"
                          ? result->bundle.generated
                          : parse_veda_budget(options.generated_sparsity);
  for (const auto& shape : result->plan.shapes) {
    auto tiles = build_veda_tiles(layout, shape);
    std::vector<int32_t> keeps(tiles.video_tiles);
    for (int row = 0; row < tiles.video_tiles; ++row)
      keeps[row] = budget.keep(tiles.video_tokens, tiles.video_tiles, row);
    result->keeps.push_back(std::move(keeps));
    result->tilings.push_back(std::move(tiles));
  }
  for (int layer : options.full_attention_layers)
    require(layer >= 0 && layer < result->bundle.layers, "full-attention layer out of range");
  for (int step : options.full_attention_steps)
    require(step >= 0, "full-attention step must be nonnegative");
  return result;
}
} // namespace slopfab::dit
