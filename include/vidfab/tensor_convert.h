// Host-side conversion of stored tensors to fp32, and tolerance comparison.
//
// This is the backbone of the verification story: reference activations dumped
// from the Python implementation are compared against ours tensor by tensor.
// Bit-exactness across different GEMM implementations is not achievable, so
// correctness is defined as agreement within a stated tolerance.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vidfab/safetensors.h"

namespace vidfab {

// Widens any supported stored dtype to fp32. Packed 4-bit tensors are not
// handled here: they are meaningless without their block scales, so they are
// dequantised by the quantisation layer instead.
void to_f32(const TensorView& view, std::vector<float>& out);

// Convenience overload.
std::vector<float> to_f32(const TensorView& view);

struct CompareStats {
  bool shape_match = false;
  int64_t count = 0;

  double max_abs_err = 0.0;
  double mean_abs_err = 0.0;
  double max_rel_err = 0.0;   // relative to |reference|, guarded near zero
  double rms_err = 0.0;

  // Where the worst absolute error landed, for pointing at a specific element.
  int64_t argmax_abs = -1;
  double lhs_at_argmax = 0.0;
  double rhs_at_argmax = 0.0;

  int64_t nan_mismatches = 0;  // NaN/Inf present in one side but not the other

  // --- whole-tensor quality metrics ----------------------------------------
  //
  // Elementwise tolerances answer "is any single element wrong". These two
  // answer "is this the same tensor", which is the question every quality
  // comparison in this project actually asks, and three separate tracks each
  // grew their own copy before they were put here. **The definitions below are
  // the project's, and they are pinned by test rather than by convention** —
  // two tracks quoting `rel_L2` computed differently would be worse than
  // neither reporting it.
  //
  // Both are computed over the **flattened tensor**, across every element pair
  // where both sides are finite. `finite_count` is that population; it differs
  // from `count` only when `nan_mismatches` is non-zero.

  // rel_L2 = ||reference - actual||_2 / ||reference||_2
  //
  //        = sqrt( sum_i (r_i - a_i)^2 ) / sqrt( sum_i r_i^2 )
  //
  // Normalised by the **reference**, not by `actual` and not by the mean of the
  // two norms. That choice follows the convention `max_rel_err` already uses —
  // `reference` is the denominator — and it makes the metric asymmetric on
  // purpose: `compare(a, b)` and `compare(b, a)` are different questions.
  //
  // Zero reference: 0 when the difference is also zero, +infinity otherwise.
  // Reporting 0 there would call a wrong answer perfect.
  double rel_l2 = 0.0;

  // Pearson correlation over the flattened tensor, **means subtracted**:
  //
  //   corr = S_ra / sqrt(S_rr * S_aa),  S_xy = sum_i (x_i - mean_x)(y_i - mean_y)
  //
  // Not cosine similarity. Skipping the mean subtraction is the plausible
  // wrong form here: latents sit near zero mean, so it looks almost right and
  // is wrong exactly when a run has drifted in level — the case worth
  // detecting. Computed as `S_ra / sqrt(S_rr * S_aa)` rather than
  // `cov / (sd_r * sd_a)` so that identical inputs give exactly +1 and a
  // tensor against its own negation exactly -1.
  //
  // Undefined when either side is constant (zero variance); reported as 0.
  double correlation = 0.0;

  int64_t finite_count = 0;

  bool passes(double abs_tol, double rel_tol) const {
    if (!shape_match) return false;
    if (nan_mismatches != 0) return false;
    return max_abs_err <= abs_tol || max_rel_err <= rel_tol;
  }
};

// Compares two fp32 buffers elementwise. `reference` is the denominator for
// relative error and for `rel_l2`.
CompareStats compare(const std::vector<float>& reference, const std::vector<float>& actual);

}  // namespace vidfab
