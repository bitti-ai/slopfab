// The campaign's two standard quality metrics: rel_L2 and Pearson correlation.
//
// These live in `compare` because three tracks each grew their own copy — the
// AB2 sampler sweep, the temporal-chunking probe and the banding work — and two
// tracks quoting "rel_L2" computed differently would be worse than neither
// reporting it. So the point of this file is not that the numbers are close to
// right; it is that **the formula is pinned to one definition**.
//
// Every wrong form below is plausible rather than broken. Each produces a
// finite, correctly signed, roughly-right-looking number, and each would pass a
// test that only asserted "correlation is near 1 for similar tensors". So the
// project's habit applies: compute the wrong form too, and assert we do not
// match it.

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "harness.h"
#include "vidfab/tensor_convert.h"

namespace {

// A deliberately asymmetric pair: ||reference|| != ||actual||, and the two
// differ in exactly one element, so every quantity below is hand-computable.
const std::vector<float> kRef = {1.0f, 2.0f, 3.0f, 4.0f};
const std::vector<float> kAct = {1.0f, 2.0f, 3.0f, 5.0f};

// ||r|| = sqrt(30), ||a|| = sqrt(39), ||r - a|| = 1.
const double kNormRef = std::sqrt(30.0);
const double kNormAct = std::sqrt(39.0);

// rel_L2 = 1 / sqrt(30)
const double kRelL2 = 1.0 / kNormRef;
// Pearson: S_ra = 6.5, S_rr = 5, S_aa = 8.75
const double kCorr = 6.5 / std::sqrt(5.0 * 8.75);

double cosine_similarity(const std::vector<float>& x, const std::vector<float>& y) {
  double dot = 0.0;
  double nx = 0.0;
  double ny = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    dot += static_cast<double>(x[i]) * y[i];
    nx += static_cast<double>(x[i]) * x[i];
    ny += static_cast<double>(y[i]) * y[i];
  }
  return dot / std::sqrt(nx * ny);
}

}  // namespace

VIDFAB_TEST(compare_metrics_match_the_stated_formulas) {
  const vidfab::CompareStats s = vidfab::compare(kRef, kAct);

  CHECK(s.shape_match);
  CHECK(s.finite_count == 4);
  CHECK_NEAR(s.rel_l2, kRelL2, 1e-12);
  CHECK_NEAR(s.correlation, kCorr, 1e-12);

  // Cross-check against the metric that was already there: rel_L2 is the rms
  // error rescaled by the reference norm. If these two ever disagree, one of
  // them has changed denominator.
  CHECK_NEAR(s.rel_l2, s.rms_err * std::sqrt(4.0) / kNormRef, 1e-12);
}

VIDFAB_TEST(compare_metrics_known_anchor_values) {
  // The two cases that anchor the definitions against something outside this
  // implementation, and both must be *exact*, not near. `S / sqrt(S*S)` is
  // exactly 1 in IEEE754; `cov / (sd * sd)` is not reliably, which is why the
  // implementation uses the former.
  const vidfab::CompareStats same = vidfab::compare(kRef, kRef);
  CHECK(same.rel_l2 == 0.0);
  CHECK(same.correlation == 1.0);

  std::vector<float> negated;
  for (float v : kRef) negated.push_back(-v);
  const vidfab::CompareStats flipped = vidfab::compare(kRef, negated);
  CHECK(flipped.correlation == -1.0);
  // A negation is a completely different tensor, and rel_L2 must say so: the
  // difference is 2r, so rel_L2 is exactly 2.
  CHECK_NEAR(flipped.rel_l2, 2.0, 1e-12);

  // Scaling `actual` leaves correlation at +1 — it is scale-invariant — while
  // rel_L2 emphatically does not. A metric where both moved together would be
  // reporting the same thing twice.
  std::vector<float> scaled;
  for (float v : kRef) scaled.push_back(v * 3.0f);
  const vidfab::CompareStats s = vidfab::compare(kRef, scaled);
  CHECK_NEAR(s.correlation, 1.0, 1e-12);
  CHECK_NEAR(s.rel_l2, 2.0, 1e-12);
}

VIDFAB_TEST(compare_correlation_rejects_the_uncentred_form) {
  const vidfab::CompareStats s = vidfab::compare(kRef, kAct);

  // Cosine similarity — Pearson without subtracting the means. On this data it
  // is 0.9940 against Pearson's 0.9827: the same shape, the same sign, close
  // enough to look right in a report and wrong enough to mislead.
  const double cosine = cosine_similarity(kRef, kAct);
  CHECK_MSG(std::fabs(s.correlation - cosine) > 1e-3,
            "correlation %.6f must not be the uncentred (cosine) form %.6f", s.correlation, cosine);

  // The decisive test, and the one that matters for this project's data.
  // Pearson is shift-invariant; cosine is not. Latents sit near zero mean,
  // which is exactly where the two agree closely enough to hide the bug — so
  // add a large offset to both sides, where they cannot.
  //
  // Correlation must be unchanged. Cosine would run to ~1.0000.
  std::vector<float> ref_shifted;
  std::vector<float> act_shifted;
  for (size_t i = 0; i < kRef.size(); ++i) {
    ref_shifted.push_back(kRef[i] + 1000.0f);
    act_shifted.push_back(kAct[i] + 1000.0f);
  }
  const vidfab::CompareStats shifted = vidfab::compare(ref_shifted, act_shifted);
  CHECK_NEAR(shifted.correlation, kCorr, 1e-9);

  const double shifted_cosine = cosine_similarity(ref_shifted, act_shifted);
  CHECK_MSG(shifted_cosine > 0.999,
            "the shifted case should make cosine degenerate to ~1 (it is %.6f); if it does not, "
            "this test no longer separates the two forms",
            shifted_cosine);
  CHECK_MSG(std::fabs(shifted.correlation - shifted_cosine) > 0.01,
            "under a large common offset, correlation %.6f must stay put while cosine runs to "
            "%.6f",
            shifted.correlation, shifted_cosine);
}

VIDFAB_TEST(compare_rel_l2_rejects_the_other_normalisations) {
  const vidfab::CompareStats s = vidfab::compare(kRef, kAct);

  // Normalised by `actual` instead of `reference`. 0.1601 against 0.1826.
  const double by_actual = 1.0 / kNormAct;
  CHECK_MSG(std::fabs(s.rel_l2 - by_actual) > 1e-3,
            "rel_L2 %.6f must normalise by the reference, not by actual (%.6f)", s.rel_l2,
            by_actual);

  // Normalised by the mean of the two norms — the other definition in common
  // use, and the one the help text explicitly disclaims. 0.1706 against 0.1826.
  const double by_mean = 1.0 / ((kNormRef + kNormAct) / 2.0);
  CHECK_MSG(std::fabs(s.rel_l2 - by_mean) > 1e-3,
            "rel_L2 %.6f must not normalise by the mean of the two norms (%.6f)", s.rel_l2,
            by_mean);

  // The asymmetry is the observable consequence of that choice, so assert it
  // directly: swapping the arguments must change the answer.
  const vidfab::CompareStats swapped = vidfab::compare(kAct, kRef);
  CHECK_MSG(std::fabs(s.rel_l2 - swapped.rel_l2) > 1e-3,
            "rel_L2 is normalised by the reference and so must be asymmetric, but compare(a,b) "
            "gave %.6f and compare(b,a) gave %.6f",
            s.rel_l2, swapped.rel_l2);
  CHECK_NEAR(swapped.rel_l2, 1.0 / kNormAct, 1e-12);
  // Correlation, by contrast, *is* symmetric, and a swap must not move it.
  CHECK_NEAR(swapped.correlation, s.correlation, 1e-12);
}

VIDFAB_TEST(compare_metrics_are_over_the_flattened_tensor) {
  // Two "rows" of two elements. Row 0 agrees, row 1 is reversed. Any
  // implementation that reduced per row and averaged would report 0 — the mean
  // of +1 and -1 — where the flattened population gives +0.5704.
  const std::vector<float> r = {1.0f, 2.0f, 10.0f, 20.0f};
  const std::vector<float> a = {1.0f, 2.0f, 20.0f, 10.0f};
  const vidfab::CompareStats s = vidfab::compare(r, a);

  // mean 8.25 both sides; S_ra = 132.75, S_rr = S_aa = 232.75.
  CHECK_NEAR(s.correlation, 132.75 / 232.75, 1e-12);
  CHECK_MSG(std::fabs(s.correlation) > 0.5,
            "correlation %.6f looks like a mean of per-row correlations (which would be 0) rather "
            "than the flattened population",
            s.correlation);

  // ||r - a|| = sqrt(100 + 100), ||r|| = sqrt(505).
  CHECK_NEAR(s.rel_l2, std::sqrt(200.0) / std::sqrt(505.0), 1e-12);
}

VIDFAB_TEST(compare_metrics_degenerate_cases) {
  // An all-zero reference has no norm to divide by. Zero difference is a
  // perfect match; anything else is reported as infinite rather than as 0,
  // because calling a wrong answer perfect is the one unacceptable outcome.
  const std::vector<float> zeros = {0.0f, 0.0f, 0.0f};
  CHECK(vidfab::compare(zeros, zeros).rel_l2 == 0.0);
  const vidfab::CompareStats from_zero =
      vidfab::compare(zeros, std::vector<float>{0.0f, 1.0f, 0.0f});
  CHECK(std::isinf(from_zero.rel_l2));

  // A constant tensor has zero variance, so correlation is undefined, not 1.
  // Reported as 0, and the header says so.
  const std::vector<float> flat = {2.0f, 2.0f, 2.0f};
  CHECK(vidfab::compare(flat, flat).correlation == 0.0);
  // ...but rel_L2 still works there, and must still be 0.
  CHECK(vidfab::compare(flat, flat).rel_l2 == 0.0);

  // Empty and mismatched inputs must not produce a metric at all.
  const vidfab::CompareStats mismatch =
      vidfab::compare(kRef, std::vector<float>{1.0f});
  CHECK(!mismatch.shape_match);
  CHECK(mismatch.rel_l2 == 0.0);
  CHECK(mismatch.correlation == 0.0);
}

// The one-pass "computational formula" for Pearson, which is what
// `tests/test_nn_kernels.cu` carries for the nvfp4 correlation figures. It is
// the *same definition* — the means are subtracted, algebraically — but it
// subtracts them by expanding the products, which cancels.
double one_pass_correlation(const std::vector<float>& a, const std::vector<float>& b) {
  double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  const double n = static_cast<double>(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    sa += a[i];
    sb += b[i];
    saa += static_cast<double>(a[i]) * a[i];
    sbb += static_cast<double>(b[i]) * b[i];
    sab += static_cast<double>(a[i]) * b[i];
  }
  const double cov = sab / n - (sa / n) * (sb / n);
  const double va = saa / n - (sa / n) * (sa / n);
  const double vb = sbb / n - (sb / n) * (sb / n);
  return (va > 0 && vb > 0) ? cov / std::sqrt(va * vb) : 0.0;
}

VIDFAB_TEST(compare_correlation_agrees_with_the_kernel_tests_but_is_better_conditioned) {
  // On well-conditioned data the two forms are the same number, which is what
  // says the *definition* here matches the one behind the nvfp4 correlation
  // figures in the README. If this ever fails, the two are measuring different
  // things and every cross-track comparison is void.
  CHECK_NEAR(vidfab::compare(kRef, kAct).correlation, one_pass_correlation(kRef, kAct), 1e-12);

  // The property this implementation promises is that correlation is invariant
  // to a common shift and to a positive scale, *exactly*, because the means are
  // subtracted before anything is squared. `kAct` is not a scaled copy of
  // `kRef`, so this is a real transformation of a real pair, not a tautology.
  std::vector<float> ref_hi;
  std::vector<float> act_hi;
  for (size_t i = 0; i < kRef.size(); ++i) {
    ref_hi.push_back(1.0e8f + (kRef[i] - 1.0f) * 8.0f);
    act_hi.push_back(1.0e8f + (kAct[i] - 1.0f) * 8.0f);
  }
  const double ours = vidfab::compare(ref_hi, act_hi).correlation;
  CHECK_NEAR(ours, kCorr, 1e-12);

  // And the two forms still agree there. That is worth asserting rather than
  // assuming, because it is what makes figures from different tracks
  // comparable at all: the nvfp4 correlations in the README came from the
  // one-pass helper and the chunking probe's came from this one.
  //
  // Two attempts were made to build a case at this project's magnitudes where
  // the one-pass form visibly loses — offsets of 2^23 and 1e8 — and both stayed
  // exact to twelve decimals, because the intermediate sums land exactly
  // representable. The conclusion is the useful one and is why no "must
  // disagree" assertion survives here: at the scales this project actually
  // compares, **the two forms are the same number**, so cross-track figures can
  // be read against each other directly. The two-pass form is kept because it
  // is the one that guarantees exact +/-1 on the anchors, not because the other
  // was measurably wrong.
  CHECK_NEAR(one_pass_correlation(ref_hi, act_hi), kCorr, 1e-12);

  // Positive scale on one side alone must not move it either.
  std::vector<float> act_scaled;
  for (float v : kAct) act_scaled.push_back(v * 1000.0f);
  CHECK_NEAR(vidfab::compare(kRef, act_scaled).correlation, kCorr, 1e-12);
}

VIDFAB_TEST(compare_metrics_use_only_the_finite_population) {
  const float nan_v = std::nanf("");

  // A matching NaN pair is agreement, and is excluded from both metrics rather
  // than poisoning them. The surviving pairs are {1,2} against {1,2}, which
  // correlate at exactly +1 and differ by nothing.
  const std::vector<float> r = {1.0f, 2.0f, nan_v};
  const std::vector<float> a = {1.0f, 2.0f, nan_v};
  const vidfab::CompareStats s = vidfab::compare(r, a);
  CHECK(s.nan_mismatches == 0);
  CHECK(s.count == 3);
  CHECK(s.finite_count == 2);
  CHECK(s.correlation == 1.0);
  CHECK(s.rel_l2 == 0.0);
  CHECK(std::isfinite(s.rel_l2));

  // A NaN against a number is a mismatch, counted, and still must not make
  // either metric non-finite.
  const vidfab::CompareStats bad = vidfab::compare({nan_v, 1.0f, 2.0f}, {0.0f, 1.0f, 2.0f});
  CHECK(bad.nan_mismatches == 1);
  CHECK(bad.finite_count == 2);
  CHECK(std::isfinite(bad.rel_l2));
  CHECK(std::isfinite(bad.correlation));
}
