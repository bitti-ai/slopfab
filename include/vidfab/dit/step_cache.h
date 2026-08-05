// Reusing the previous velocity when the conditioning has barely moved.
//
// This is TeaCache / FBCache / Delta-DiT, and this checkpoint makes it
// unusually clean. In TeaCache the skip indicator has to be a per-model
// polynomial fitted offline against the *input* of the timestep MLP, because
// the thing that actually varies between steps — the modulation the blocks
// see — is only reachable through the MLP. **Here it is reachable directly**:
// the pruned checkpoint deleted `time_embedder.*` and replaced the whole
// timestep front end with `adaln_t_table[row(t)]`, an 8-vector shared by all 51
// consumers (spec 3.4). So
//
//     c(t) = adaln_t_table[row(t)]  in R^8
//
// *is* the entire timestep conditioning, and the distance between two steps'
// conditioning is an exact, calibration-free, microsecond-cost quantity rather
// than a fitted proxy for one.
//
// The signature of a step is the **pair** (c(t_v), c(t_a)), 16 floats. For t2va
// there are up to two distinct timesteps per step — video on the shift-12 grid,
// audio on the shift-3 grid, text sharing the video one (spec 3.4 note 3,
// 7.5) — and the two grids move at very different rates, so a distance built
// over the video timestep alone would misread how much the audio branch's
// modulation has changed.
//
// What is *not* modelled: the latent itself. The conditioning is the only thing
// this can see, so a step where the conditioning barely moves but the sample is
// changing fast is a step this will happily skip. That is the same bet every
// member of this family makes, and it is why the flags default off.
//
// The decision logic lives here, in the core library, rather than in the
// denoise loop, because it is the part that is worth pinning by test and the
// denoise loop is only reachable through a 19.6 GiB checkpoint and a GPU.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "vidfab/dit/adaln.h"

namespace vidfab::dit {

struct StepCacheConfig {
  // Accumulated relative-L1 movement of the conditioning signature that must
  // build up before a step is recomputed. **Zero disables the whole feature**
  // and is the default. Dimensionless: 1.0 means the conditioning has moved,
  // in total since the last computed step, by as much as its own magnitude.
  float threshold = 0.0f;

  // Steps at the start of the schedule that are always computed. The
  // trajectory is most sensitive early — an error there is amplified by every
  // remaining step — and step 0 has no velocity to reuse at all, the buffers
  // being zero-filled. Values below `kMinWarmup` are raised to it.
  int warmup = 3;

  // Calibration-free fixed-interval fallback: compute every n-th step, skip the
  // rest. Zero disables. This exists so the threshold path has something
  // honest to be judged against — a schedule-blind policy that costs nothing to
  // reason about. When both are set this one wins, because a fixed interval and
  // an adaptive one ORed together is neither.
  int skip_every = 0;

  // False when this is a plain run and nothing may be skipped.
  bool enabled() const { return threshold > 0.0f || skip_every > 0; }
};

// Two steps are always computed no matter what the flags say, and both are
// enforced here rather than documented:
//
//   * The first `kMinWarmup`. Step 0 in particular has nothing to reuse — the
//     velocity buffers are still zeros — so skipping it does not degrade the
//     sample, it destroys it.
//   * The last. At the terminal step `sigma_next = 0`, so `ratio = 0` and
//     `x_next = denoised` exactly (spec 7.3): a stale velocity there is written
//     straight into the output with no damping at all.
constexpr int kMinWarmup = 2;

// Relative L1 between a signature and its reference, summed over all `n`
// components of the pair.
//
//     d(a, b) = sum|a - b| / sum|b|
//
// Relative rather than absolute because the table's columns span three orders
// of magnitude — their ranges decay geometrically from 0.870 to 0.000653, being
// the singular values of a truncated SVD (adaln.h) — so an absolute threshold
// would be a threshold on column 0 and nothing else, and its useful value would
// depend on the checkpoint's scale. L1 rather than L2 for the same reason: L2
// is dominated by the leading component even harder than L1 is. This is
// TeaCache's own `mean|dx| / mean|x|` metric, with the mean's divisor cancelling
// between numerator and denominator.
//
// Returns 0 when the reference is all zeros, which cannot happen for a real
// table row but would otherwise be an infinity that skips the rest of the run.
float conditioning_distance(const float* a, const float* reference, int n);

// The compute-or-skip decision for one denoising run, stepped in order.
//
// Accumulating *increments* — `acc += d(c_i, c_{i-1})` every step, reset to zero
// only on a step that is actually computed — is what makes one scalar threshold
// behave the same way on the shift-12 video grid and the shift-3 audio one. The
// obvious alternative, `d(c_i, c_last_computed)`, is a chord where this is an
// arc length, and it is wrong in a way that is silent: a schedule that moves out
// and back within a step or two has a small chord and a large arc, so the chord
// form under-counts exactly the movement that the reused velocity is stalest
// against. The arc length is also an upper bound on the chord, so this errs
// towards recomputing.
class StepCache {
 public:
  StepCache(const StepCacheConfig& config, int num_steps);

  bool enabled() const { return config_.enabled(); }

  // Call exactly once per step, in order. `code` is that step's conditioning
  // signature — for t2va, `c(t_v)` followed by `c(t_a)`, 16 floats. Pass
  // `nullptr` when the cache is disabled and the signature was not built;
  // the answer is then unconditionally true.
  //
  // Returns true when the transformer must run, false when the velocity from
  // the previous computed step is to be reused.
  bool should_compute(int step, const float* code, int len);

  int computed() const { return computed_; }
  int skipped() const { return skipped_; }

  // Accumulated movement since the last computed step, for reporting.
  float accumulator() const { return accumulator_; }

  // The warmup actually in force, after the `kMinWarmup` floor.
  int warmup() const { return warmup_; }

 private:
  StepCacheConfig config_;
  int num_steps_ = 0;
  int warmup_ = kMinWarmup;
  float accumulator_ = 0.0f;
  std::vector<float> previous_;
  bool have_previous_ = false;
  int computed_ = 0;
  int skipped_ = 0;
};

// Which steps compute, for a whole schedule at once. 1 = compute, 0 = reuse.
// `codes[i]` is step i's signature; every entry must be the same length.
//
// This is the same object the loop drives, run to completion — not a
// reimplementation of it — so a test of this is a test of the run.
std::vector<uint8_t> plan_step_cache(const StepCacheConfig& config,
                                     const std::vector<std::vector<float>>& codes);

// The pair form: `schedule[i]` is `(t_v, t_a)` for step i, and `code` turns one
// timestep into its 8-vector. In a real run `code` is the transformer's AdaLN
// lookup, which honours the configured lookup mode (linear by default, and
// deliberately a knob — spec 3.5 leaves the grid semantics unresolved).
using CodeFn = std::function<std::array<float, AdaLNTable::kRank>(float t)>;

std::vector<uint8_t> plan_step_cache(const StepCacheConfig& config,
                                     const std::vector<std::pair<float, float>>& schedule,
                                     const CodeFn& code);

}  // namespace vidfab::dit
