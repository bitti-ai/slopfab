#include "vidfab/tensor_convert.h"

#include <cmath>
#include <stdexcept>

namespace vidfab {
namespace {

template <typename Stored, typename Fn>
void widen(const TensorView& view, std::vector<float>& out, Fn convert) {
  const auto* src = static_cast<const Stored*>(view.data);
  const size_t n = out.size();
  for (size_t i = 0; i < n; ++i) out[i] = convert(src[i]);
}

}  // namespace

void to_f32(const TensorView& view, std::vector<float>& out) {
  const auto n = static_cast<size_t>(view.numel());
  out.assign(n, 0.0f);
  if (n == 0) return;

  switch (view.dtype) {
    case DType::kF32:
      widen<float>(view, out, [](float v) { return v; });
      break;
    case DType::kF64:
      widen<double>(view, out, [](double v) { return static_cast<float>(v); });
      break;
    case DType::kF16:
      widen<uint16_t>(view, out, [](uint16_t v) { return f16_to_f32(v); });
      break;
    case DType::kBF16:
      widen<uint16_t>(view, out, [](uint16_t v) { return bf16_to_f32(v); });
      break;
    case DType::kF8E4M3:
      widen<uint8_t>(view, out, [](uint8_t v) { return f8_e4m3_to_f32(v); });
      break;
    case DType::kI8:
      widen<int8_t>(view, out, [](int8_t v) { return static_cast<float>(v); });
      break;
    case DType::kU8:
    case DType::kBool:
      widen<uint8_t>(view, out, [](uint8_t v) { return static_cast<float>(v); });
      break;
    case DType::kI16:
      widen<int16_t>(view, out, [](int16_t v) { return static_cast<float>(v); });
      break;
    case DType::kI32:
      widen<int32_t>(view, out, [](int32_t v) { return static_cast<float>(v); });
      break;
    case DType::kI64:
      widen<int64_t>(view, out, [](int64_t v) { return static_cast<float>(v); });
      break;
    case DType::kF8E5M2:
      // E5M2 shares f16's exponent field width; widening is a shift into the
      // f16 layout followed by the existing f16 conversion.
      widen<uint8_t>(view, out, [](uint8_t v) {
        return f16_to_f32(static_cast<uint16_t>(static_cast<uint16_t>(v) << 8));
      });
      break;
    case DType::kUnknown:
      throw std::runtime_error("to_f32: tensor '" + view.name + "' has an unsupported dtype");
  }
}

std::vector<float> to_f32(const TensorView& view) {
  std::vector<float> out;
  to_f32(view, out);
  return out;
}

CompareStats compare(const std::vector<float>& reference, const std::vector<float>& actual) {
  CompareStats stats;
  stats.shape_match = reference.size() == actual.size();
  if (!stats.shape_match) return stats;

  stats.count = static_cast<int64_t>(reference.size());
  if (reference.empty()) {
    stats.shape_match = true;
    return stats;
  }

  // Accumulate in double: summing millions of fp32 errors in fp32 loses the
  // very signal we are trying to measure.
  double sum_abs = 0.0;
  double sum_sq = 0.0;

  for (size_t i = 0; i < reference.size(); ++i) {
    const double r = reference[i];
    const double a = actual[i];

    const bool r_finite = std::isfinite(r);
    const bool a_finite = std::isfinite(a);
    if (!r_finite || !a_finite) {
      // Matching non-finite values agree; a mismatch is counted and skipped so
      // it cannot poison the error statistics.
      const bool both_nan = std::isnan(r) && std::isnan(a);
      const bool both_same_inf = std::isinf(r) && std::isinf(a) && ((r > 0) == (a > 0));
      if (!both_nan && !both_same_inf) ++stats.nan_mismatches;
      continue;
    }

    const double err = std::fabs(r - a);
    sum_abs += err;
    sum_sq += err * err;

    if (err > stats.max_abs_err) {
      stats.max_abs_err = err;
      stats.argmax_abs = static_cast<int64_t>(i);
      stats.lhs_at_argmax = r;
      stats.rhs_at_argmax = a;
    }

    // Guard the denominator so near-zero references do not manufacture huge
    // relative errors from negligible absolute ones.
    const double denom = std::fabs(r);
    if (denom > 1e-6) {
      const double rel = err / denom;
      if (rel > stats.max_rel_err) stats.max_rel_err = rel;
    }
  }

  const auto n = static_cast<double>(reference.size());
  stats.mean_abs_err = sum_abs / n;
  stats.rms_err = std::sqrt(sum_sq / n);
  return stats;
}

}  // namespace vidfab
