#pragma once
#include "nn_kernels_fixture.h"

namespace {
// --- attention --------------------------------------------------------------

// The fused kernel's tail handling is invisible at a sequence that divides the
// block sizes evenly. kBr is 64 and kBc is 32, so a prime sequence exercises a
// ragged query tile and a ragged key step at once -- the case where a masked
// column would otherwise contribute exp(0) = 1 to the denominator, which is
// wrong by a factor that grows with how much of the tile is padding.

// sage2's three preparation kernels take head dim as a template parameter, so
// the failure this guards is an index split that is right at one D and wrong at
// the other — invisible unless both widths run in the same process, hence the
// alternating loop.
//
// **The digests are frozen, not printed.** They were captured by running these
// six shapes against a build of this file linked to the pre-refactor
// `sage2_attention_forward` and again against the current one; both produced the
// values below. Comparing round 1 against round 0 of the same build would only
// have proved determinism — a deterministic bit change passes that and moves
// these constants, which is exactly the regression worth catching.
//
// A failure here is not necessarily a bug: it means the numerics moved, and the
// question is whether that was intended. If it was, re-derive the table the same
// way rather than pasting whatever the new build prints.


// head_dim 64 is the other instantiation `supported()` accepts, and until this
// existed nothing exercised it -- every fused test above is 128-wide. The two
// differ in more than a constant: the staging tiles the key block into 2 passes
// rather than 4 and each warp covers a 32-column group of a 64-wide head rather
// than of a 128-wide one, so a mapping that covers D=128 exactly can still
// double-write or skip columns at D=64. A ragged sequence checks that against
// the tail at the same time.


// Dense attention restricted to an explicit key set, for frame banding.
//
// It takes the *ranges* rather than a band width, deliberately. The ranges are
// the specification of what the kernel must do; whether they describe the right
// band is a separate question, answered on the host by
// `packing_banded_key_ranges`. Re-deriving an idealised band here would test the
// two implementations against each other's opinion of the rounding, and the
// first disagreement would get resolved by tuning the reference until it
// matched -- which is backwards, and would silently accept a kernel that
// attends to nearly the right keys.
std::vector<float> cpu_attention_banded(const std::vector<float>& q, const std::vector<float>& k,
                                        const std::vector<float>& v, int seq, int heads,
                                        int kv_heads, int head_dim, float scale,
                                        const std::vector<int32_t>& ranges, int query_tile) {
  const int qld = heads * head_dim;
  const int kvld = kv_heads * head_dim;
  const int group = heads / kv_heads;
  std::vector<float> out(size_t(seq) * qld, 0.0f);

  for (int h = 0; h < heads; ++h) {
    const int kv = h / group;
    for (int i = 0; i < seq; ++i) {
      const size_t t = size_t(i / query_tile) * 4;
      const int lo0 = ranges[t + 0], hi0 = std::min(ranges[t + 1], seq);
      const int lo1 = ranges[t + 2], hi1 = std::min(ranges[t + 3], seq);

      double m = -1e300;
      std::vector<std::pair<int, double>> p;
      p.reserve(size_t(hi0 - lo0) + size_t(std::max(0, hi1 - lo1)));
      const auto score = [&](int j) {
        double dot = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          dot += double(q[size_t(i) * qld + h * head_dim + d]) *
                 k[size_t(j) * kvld + kv * head_dim + d];
        }
        const double s = dot * scale;
        m = std::max(m, s);
        p.emplace_back(j, s);
      };
      for (int j = lo0; j < hi0; ++j) score(j);
      for (int j = lo1; j < hi1; ++j) score(j);

      double sum = 0.0;
      for (auto& e : p) {
        e.second = std::exp(e.second - m);
        sum += e.second;
      }
      for (auto& e : p) {
        const double w = e.second / sum;
        for (int d = 0; d < head_dim; ++d) {
          out[size_t(i) * qld + h * head_dim + d] +=
              float(w * v[size_t(e.first) * kvld + kv * head_dim + d]);
        }
      }
    }
  }
  return out;
}

// Does the kernel attend to *exactly* the keys the host asked for?
//
// Nothing else can answer this. A band one key block too narrow produces
// finite, plausibly-scaled output, passes every norm and shape check, and is
// **faster** -- so it would show up as a speedup that beats its own forecast,
// which is the last thing anyone questions. The band edges here are chosen so
// the rounding is live: R = 42 rows per frame against a 64-row key block, so no
// frame boundary lands on a block boundary and every range is rounded outwards.


// The fused kernel's probability precision is pinned by nothing else in this
// file, and that is a hole rather than an oversight of one change.
//
// `attention_fp16_score_tile` below measures exactly the right quantity, and
// passes `AttentionBackend::kBlocked` at both its workspace sizing and its
// forward call -- it never runs `kFused` at any configuration. Every other
// fused test bounds 1e-3 absolute / 1e-2 relative against the CPU reference,
// and a host model of this kernel's own schedule says fp32, fp16 **and** bf16
// probabilities all pass that bound at every depth from 512 to 32768. So those
// tests cannot see P at all: they would go green whatever it was carried in.
//
// What that leaves unguarded: P feeds a second `mma` whose two operands must
// share a type, so any change that stages V as bf16 -- which is what a
// `cp.async` byte copy of V would require -- silently drops P from fp16's 11
// mantissa bits to bf16's 8. The file header records that fp16 was chosen over
// bf16 deliberately, for exactly those bits. Nothing was checking.
//
// **The bar is 1.10x and it is chosen, not inherited.** A host model of this
// schedule puts fp16 P at 1.00-1.01x the bf16 output floor and bf16 P at
// 1.21-1.27x, flat in K depth from 512 to 32768 -- signal and error both grow
// as sqrt(K), so it does not compound. 1.10x sits between them. Two things make
// that model trustworthy enough to set a bar from: it reproduces the 1.66e-3
// bf16 output floor and the 1.01x fp16 ratio that `attention_fp16_score_tile`
// measured independently, neither of which it was fitted to.
//
// At *this* test's amplitudes the same model puts fp16 P at 1.006x and bf16 P
// at 1.307x, so the two figures to hold in mind are 1.01x and ~1.3x.
//
// Reusing the blocked path's 1.3x would have been the wrong move, and the
// reason is sharper than "it is too loose": bf16 P lands at 1.307x here and at
// 1.21-1.27x at other amplitudes, so it sits *astride* that bar and which side
// it falls depends on the data. A bar a regression clears or misses by seven
// thousandths is not evidence either way -- it is a coin flip wearing whichever
// label it lands on, and a green one ends the conversation. 1.10x is chosen to
// sit clear of both figures: fp16 has ~9 points of headroom, bf16 ~20 points of
// exceedance, and neither depends on the draw.
//
// That test's own comment says its 1.3x "has no power to separate fp16
// probabilities from bf16 ones, because neither is visible through a bf16
// output". The first half is right about its assertion. The second half is too
// strong: the *measurement* separates them cleanly, 1.31x against 1.01x. Both
// comments are left standing because they are about different things.


// Grouped-query: 6 query heads share 2 kv heads. Getting the kv head index wrong
// still produces finite, plausibly-scaled output, so it is pinned against the
// CPU reference rather than against a shape check.


// The score tile is fp16 and doubles as the probability buffer. Two things have
// to hold and neither is visible from the shapes above, where the tile budget
// swallows the whole sequence in a single key block:
//
//   - the online softmax must give the same answer however the keys are split,
//     which is what exercises the running max, the correction factor and the
//     accumulator rescale at all; and
//   - the fp16 tile has to stay inside tolerance against an fp64-ordered
//     reference. The error is *recorded* here, not just bounded, so that a
//     future format change has a number to beat rather than an assertion to
//     satisfy.


}  // namespace
