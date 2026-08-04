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

  bool passes(double abs_tol, double rel_tol) const {
    if (!shape_match) return false;
    if (nan_mismatches != 0) return false;
    return max_abs_err <= abs_tol || max_rel_err <= rel_tol;
  }
};

// Compares two fp32 buffers elementwise. `reference` is the denominator for
// relative error.
CompareStats compare(const std::vector<float>& reference, const std::vector<float>& actual);

}  // namespace vidfab
