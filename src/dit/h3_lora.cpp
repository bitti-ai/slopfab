#include "h3_lora.h"
#include "slopfab/dit/adaln.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace slopfab::detail::h3 {
namespace {
bool ends(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size()-suffix.size(), suffix.size(), suffix) == 0;
}
std::vector<float> values(const TensorView& tensor) {
  if (tensor.dtype != DType::kF32 && tensor.dtype != DType::kF16 && tensor.dtype != DType::kBF16)
    throw std::runtime_error("LoRA: expected F32/F16/BF16 for " + tensor.name);
  auto result = to_f32(tensor);
  for (float v : result) if (!std::isfinite(v))
    throw std::runtime_error("LoRA: non-finite value in " + tensor.name);
  return result;
}

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
bool adaln_target(const std::string& name) {
  if (name == "final_layer.adaln_proj.linear") return true;
  if (name.compare(0, 7, "blocks.") != 0) return false;
  const size_t dot = name.find('.', 7);
  if (dot == std::string::npos || dot == 7) return false;
  for (size_t i = 7; i < dot; ++i) if (name[i] < '0' || name[i] > '9') return false;
  return name.substr(dot + 1) == "adaln_proj.linear";
}
bool supported(const std::string& name) {
  if (adaln_target(name)) return true;
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
void AdaLNBasis::load(const SafeTensors& base, const SafeTensors& adapter, int width) {
    source = std::make_unique<detail::LoraGrid>();
    source->load(adapter, width, false);
    const auto model = dit::resolve_model_descriptor(base);
    if (!model.adaln_grid_id.empty() && model.adaln_grid_id != source->identity)
      throw std::runtime_error("LoRA: AdaLN grid identity does not match the base model descriptor");
    const auto& e = source->tensor;
    const auto& t = base.at("adaln_t_table");
    if (t.shape != std::vector<int64_t>{dit::AdaLNTable::kRows, dit::AdaLNTable::kRank} || e.shape != std::vector<int64_t>{dit::AdaLNTable::kRows, width})
      throw std::runtime_error("LoRA: AdaLN timestep grid must match the base's 1025 rows and adapter width");
    rows = dit::AdaLNTable::kRows; columns = dit::AdaLNTable::kRank + 1; full = width;
    grid = values(e); table = values(t);
    std::vector<double> q(size_t(rows) * columns), r(size_t(columns) * columns);
    for (int col = 0; col < columns; ++col) {
      for (int row = 0; row < rows; ++row)
        q[size_t(col) * rows + row] = col == columns - 1 ? 1 : table[size_t(row) * (columns - 1) + col];
      for (int pass = 0; pass < 2; ++pass) for (int j = 0; j < col; ++j) {
        double dot = 0;
        for (int row = 0; row < rows; ++row) dot += q[size_t(j) * rows + row] * q[size_t(col) * rows + row];
        r[size_t(j) * columns + col] += dot;
        for (int row = 0; row < rows; ++row) q[size_t(col) * rows + row] -= dot * q[size_t(j) * rows + row];
      }
      double norm = 0;
      for (int row = 0; row < rows; ++row) norm += std::pow(q[size_t(col) * rows + row], 2);
      norm = std::sqrt(norm);
      if (!(norm > 1e-12) || !std::isfinite(norm))
        throw std::runtime_error("LoRA: singular AdaLN timestep table");
      r[size_t(col) * columns + col] = norm;
      for (int row = 0; row < rows; ++row) q[size_t(col) * rows + row] /= norm;
    }
    map.assign(size_t(full) * columns, 0);
    for (int feature = 0; feature < full; ++feature) {
      for (int col = 0; col < columns; ++col)
        for (int row = 0; row < rows; ++row)
          map[size_t(feature) * columns + col] += q[size_t(col) * rows + row] * grid[size_t(row) * full + feature];
      for (int col = columns - 1; col >= 0; --col) {
        auto& v = map[size_t(feature) * columns + col];
        for (int j = col + 1; j < columns; ++j) v -= r[size_t(col) * columns + j] * map[size_t(feature) * columns + j];
        v /= r[size_t(col) * columns + col];
      }
    }
  }
double AdaLNBasis::project(LoraFactors& f, std::vector<float>& bias) const {
    if (f.in != full) throw std::runtime_error("LoRA: inconsistent full AdaLN input widths");
    std::vector<float> a(size_t(f.rank) * (columns - 1));
    std::vector<double> constant(f.rank);
    double error = 0, energy = 0;
    for (int rank = 0; rank < f.rank; ++rank) {
      std::vector<double> fit(columns);
      for (int k = 0; k < full; ++k)
        for (int col = 0; col < columns; ++col)
          fit[col] += f.a[size_t(rank) * full + k] * map[size_t(k) * columns + col];
      for (int col = 0; col < columns - 1; ++col)
        a[size_t(rank) * (columns - 1) + col] = static_cast<float>(fit[col]);
      constant[rank] = fit.back();
      // Check the actual low-rank activation curve, after narrowing the fitted
      // weight to F32. A small whole-grid error need not imply a good A projection.
      for (int row = 0; row < rows; ++row) {
        double actual = 0, predicted = constant[rank];
        for (int k = 0; k < full; ++k) actual += double(f.a[size_t(rank) * full + k]) * grid[size_t(row) * full + k];
        for (int col = 0; col < columns - 1; ++col)
          predicted += a[size_t(rank) * (columns - 1) + col] * double(table[size_t(row) * (columns - 1) + col]);
        error += (actual - predicted) * (actual - predicted);
        energy += actual * actual;
      }
    }
    const double relative = std::sqrt(error / std::max(energy, 1e-30));
    if (!std::isfinite(relative) || relative > 0.01)
      throw std::runtime_error("LoRA: AdaLN conversion exceeds 1% relative fit error; use a matching timestep grid or a pruned adapter");
    bias.resize(f.out);
    for (int row = 0; row < f.out; ++row) {
      double sum = 0;
      for (int rank = 0; rank < f.rank; ++rank) sum += f.b[size_t(row) * f.rank + rank] * constant[rank];
      bias[row] = static_cast<float>(sum);
      if (!std::isfinite(bias[row])) throw std::runtime_error("LoRA: AdaLN bias conversion overflow");
    }
    f.in = columns - 1; f.a = std::move(a);
    for (float v : f.a) if (!std::isfinite(v)) throw std::runtime_error("LoRA: AdaLN conversion overflow");
    return relative;
  }
}
