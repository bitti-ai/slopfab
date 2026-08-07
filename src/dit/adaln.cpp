#include "vidfab/dit/adaln.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "vidfab/tensor_convert.h"

namespace vidfab::dit {
namespace {

constexpr int kIntervals = AdaLNTable::kRows - 1;  // 1024

// Rounds half away from zero, matching what a `round(u)` in the quantiser
// would have done. Only reachable from kNearest, which is not the default.
int nearest_index(double u) {
  const int j = static_cast<int>(std::floor(u + 0.5));
  if (j < 0) return 0;
  if (j > kIntervals) return kIntervals;
  return j;
}

void load_matrix(const SafeTensors& checkpoint, const char* name, int rows, int cols,
                 std::vector<float>& out) {
  const TensorView* view = checkpoint.find(name);
  if (view == nullptr) throw std::runtime_error(std::string(name) + " is missing");
  if (view->shape != std::vector<int64_t>{rows, cols}) {
    throw std::runtime_error(std::string(name) + " has the wrong shape; expected [" +
                             std::to_string(rows) + ", " + std::to_string(cols) + "]");
  }
  to_f32(*view, out);
}

void load_vector(const SafeTensors& checkpoint, const char* name, int size,
                 std::vector<float>& out) {
  const TensorView* view = checkpoint.find(name);
  if (view == nullptr) throw std::runtime_error(std::string(name) + " is missing");
  if (view->shape != std::vector<int64_t>{size}) {
    throw std::runtime_error(std::string(name) + " has the wrong shape; expected [" +
                             std::to_string(size) + "]");
  }
  to_f32(*view, out);
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

}  // namespace

const char* adaln_lookup_name(AdaLNLookup mode) {
  switch (mode) {
    case AdaLNLookup::kLinear:
      return "linear";
    case AdaLNLookup::kNearest:
      return "nearest";
    case AdaLNLookup::kLinearReversed:
      return "linear-reversed";
  }
  return "unknown";
}

std::vector<float> minimax_h3_timestep_sinusoid(float timestep, int freq_dim) {
  if (freq_dim <= 0 || (freq_dim & 1) != 0) {
    throw std::runtime_error("MiniMax-H3 timestep frequency dimension must be positive and even");
  }
  const int half = freq_dim / 2;
  std::vector<float> out(static_cast<size_t>(freq_dim));
  for (int i = 0; i < half; ++i) {
    // diffusers get_timestep_embedding with downscale_freq_shift=0. Compute
    // the frequency in double like torch's exponent construction, then round
    // the stored embedding to float32.
    const double exponent = -std::log(10000.0) * static_cast<double>(i) /
                            static_cast<double>(half);
    const double phase = static_cast<double>(timestep) * std::exp(exponent);
    out[static_cast<size_t>(i)] = static_cast<float>(std::cos(phase));
    out[static_cast<size_t>(half + i)] = static_cast<float>(std::sin(phase));
  }
  return out;
}

void FullAdaLNTimestepEmbedding::load(const SafeTensors& checkpoint, int freq_dim,
                                      int hidden_dim, int output_dim) {
  if (freq_dim <= 0 || hidden_dim <= 0 || output_dim <= 0 || (freq_dim & 1) != 0) {
    throw std::runtime_error("invalid MiniMax-H3 timestep embedding dimensions");
  }
  std::vector<float> in_w, in_b, out_w, out_b;
  load_matrix(checkpoint, "time_embedder.proj_in.weight", hidden_dim, freq_dim, in_w);
  load_vector(checkpoint, "time_embedder.proj_in.bias", hidden_dim, in_b);
  load_matrix(checkpoint, "time_embedder.proj_out.weight", output_dim, hidden_dim, out_w);
  load_vector(checkpoint, "time_embedder.proj_out.bias", output_dim, out_b);
  for (const auto* values : {&in_w, &in_b, &out_w, &out_b}) {
    for (float value : *values) {
      if (!std::isfinite(value)) throw std::runtime_error("time_embedder contains a non-finite entry");
    }
  }
  freq_dim_ = freq_dim;
  hidden_dim_ = hidden_dim;
  output_dim_ = output_dim;
  proj_in_weight_ = std::move(in_w);
  proj_in_bias_ = std::move(in_b);
  proj_out_weight_ = std::move(out_w);
  proj_out_bias_ = std::move(out_b);
}

std::vector<float> FullAdaLNTimestepEmbedding::forward(float timestep) const {
  if (!loaded()) throw std::runtime_error("FullAdaLNTimestepEmbedding::forward: not loaded");
  const std::vector<float> sinusoid = minimax_h3_timestep_sinusoid(timestep, freq_dim_);
  std::vector<float> hidden(static_cast<size_t>(hidden_dim_));
  for (int r = 0; r < hidden_dim_; ++r) {
    double sum = proj_in_bias_[static_cast<size_t>(r)];
    const float* weight = proj_in_weight_.data() + static_cast<size_t>(r) * freq_dim_;
    for (int c = 0; c < freq_dim_; ++c) sum += weight[c] * sinusoid[static_cast<size_t>(c)];
    hidden[static_cast<size_t>(r)] = silu(static_cast<float>(sum));
  }
  std::vector<float> out(static_cast<size_t>(output_dim_));
  for (int r = 0; r < output_dim_; ++r) {
    double sum = proj_out_bias_[static_cast<size_t>(r)];
    const float* weight = proj_out_weight_.data() + static_cast<size_t>(r) * hidden_dim_;
    for (int c = 0; c < hidden_dim_; ++c) sum += weight[c] * hidden[static_cast<size_t>(c)];
    out[static_cast<size_t>(r)] = static_cast<float>(sum);
  }
  return out;
}

std::vector<float> FullAdaLNTimestepEmbedding::forward(
    const std::vector<float>& timesteps) const {
  std::vector<float> out;
  out.reserve(timesteps.size() * static_cast<size_t>(output_dim_));
  for (float timestep : timesteps) {
    std::vector<float> row = forward(timestep);
    out.insert(out.end(), row.begin(), row.end());
  }
  return out;
}

void AdaLNTable::load(const SafeTensors& checkpoint) {
  const TensorView* view = checkpoint.find("adaln_t_table");
  if (view == nullptr) {
    throw std::runtime_error(
        "adaln_t_table is missing. This checkpoint is presumably the unpruned model, which "
        "carries time_embedder.* and a [96768, 2688] adaln_proj instead; that path is not "
        "implemented");
  }
  if (view->shape.size() != 2 || view->shape[0] != kRows || view->shape[1] != kRank) {
    throw std::runtime_error("adaln_t_table has shape " + std::to_string(view->shape.size()) +
                             "-d, expected [1025, 8]");
  }

  to_f32(*view, data_);
  if (data_.size() != static_cast<size_t>(kRows) * kRank) {
    throw std::runtime_error("adaln_t_table did not widen to 1025*8 elements");
  }
  for (float v : data_) {
    if (!std::isfinite(v)) {
      throw std::runtime_error("adaln_t_table contains a non-finite entry");
    }
  }
}

const float* AdaLNTable::row(int index) const {
  if (!loaded()) throw std::runtime_error("AdaLNTable::row: table not loaded");
  if (index < 0 || index >= kRows) {
    throw std::runtime_error("AdaLNTable::row: index " + std::to_string(index) + " out of range");
  }
  return data_.data() + static_cast<size_t>(index) * kRank;
}

std::array<float, AdaLNTable::kRank> AdaLNTable::lookup(float t, AdaLNLookup mode) const {
  if (!loaded()) throw std::runtime_error("AdaLNTable::lookup: table not loaded");

  // Clamp rather than extrapolate. The schedule produces t in [0, 1] by
  // construction, so anything outside is an upstream bug worth surfacing as a
  // pinned endpoint rather than as a plausible extrapolated coefficient.
  double x = static_cast<double>(t);
  if (mode == AdaLNLookup::kLinearReversed) x = 1.0 - x;
  if (x < 0.0) x = 0.0;
  if (x > 1.0) x = 1.0;

  const double u = x * static_cast<double>(kIntervals);
  std::array<float, kRank> out{};

  if (mode == AdaLNLookup::kNearest) {
    const float* r = row(nearest_index(u));
    for (int i = 0; i < kRank; ++i) out[i] = r[i];
    return out;
  }

  int j = static_cast<int>(std::floor(u));
  if (j < 0) j = 0;
  if (j > kIntervals - 1) j = kIntervals - 1;
  const double f = u - static_cast<double>(j);

  const float* lo = row(j);
  const float* hi = row(j + 1);
  for (int i = 0; i < kRank; ++i) {
    // Evaluated in double and rounded once. The coefficients are small and the
    // same c(t) feeds all 51 consumers at every step, so a rounding applied
    // here biases every block's modulation identically and accumulates
    // coherently over the trajectory — the same argument transformer.py:122-126
    // makes about the SiLU in the unpruned model.
    out[i] = static_cast<float>((1.0 - f) * static_cast<double>(lo[i]) +
                                f * static_cast<double>(hi[i]));
  }
  return out;
}

}  // namespace vidfab::dit
