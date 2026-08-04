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
