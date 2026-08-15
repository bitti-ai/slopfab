// The block cache's span placement and its compute-or-reuse decision.
//
// Both halves are worth testing without a GPU for the same reason the step
// cache's decision is: every failure mode here is silent. A span placed one
// block off still produces video, and a schedule that reuses the terminal step
// produces video built on a residual captured at a timestep the sample never
// passed through. Neither shows up as an error, only as a sample that drifted,
// and telling that apart from the drift the feature is *expected* to cause
// needs the decision pinned separately from the run.
//
// The wrong forms below are the ones that are plausible rather than merely
// broken, each differing from the real policy in exactly one decision.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "harness.h"
#include "vidfab/dit/block_cache.h"

namespace {

using vidfab::dit::BlockCache;
using vidfab::dit::BlockCacheConfig;
using vidfab::dit::BlockSpan;
using vidfab::dit::kMinBlockWarmup;
using vidfab::dit::plan_block_cache;
using vidfab::dit::resolve_block_span;

BlockCacheConfig cfg(int span, int interval, int warmup = 3, int start = -1) {
  BlockCacheConfig c;
  c.span = span;
  c.interval = interval;
  c.warmup = warmup;
  c.start = start;
  return c;
}

int count(const std::vector<uint8_t>& plan) {
  int n = 0;
  for (uint8_t v : plan) n += v;
  return n;
}

// 1 at each listed index, 0 elsewhere.
std::vector<uint8_t> at(size_t n, const std::vector<int>& indices) {
  std::vector<uint8_t> out(n, 0);
  for (int i : indices) out[static_cast<size_t>(i)] = 1;
  return out;
}

enum class Wrong {
  kNoTerminal,    // last step not protected
  kAnchorAtZero,  // interval phased from step 0 instead of from the warmup
};

std::vector<uint8_t> wrong_plan(const BlockCacheConfig& c, int steps, Wrong which) {
  std::vector<uint8_t> out(static_cast<size_t>(std::max(0, steps)), 0);
  const int warmup = std::max(kMinBlockWarmup, c.warmup);
  bool have = false;
  for (int i = 0; i < steps; ++i) {
    bool forced = !c.enabled() || i < warmup || !have;
    if (which != Wrong::kNoTerminal) forced = forced || i >= steps - 1;

    const int phase = which == Wrong::kAnchorAtZero ? i : i - warmup;
    const bool compute = forced || (phase % c.interval) == 0;
    if (compute) have = true;
    out[static_cast<size_t>(i)] = compute ? 1 : 0;
  }
  return out;
}

// Whether the terminal step would have been a reuse but for the protection.
// Comparing against `Wrong::kNoTerminal` only says anything when it would: with
// 10 steps, warmup 3 and interval 3 the last step satisfies `(9-3) % 3 == 0`
// and computes on its own, so the unprotected form agrees and the comparison
// asserts nothing. This is exactly the coincidence that would let the terminal
// protection be deleted without a test noticing.
bool terminal_is_off_interval(const BlockCacheConfig& c, int steps) {
  const int warmup = std::max(kMinBlockWarmup, c.warmup);
  return ((steps - 1 - warmup) % c.interval) != 0;
}

}  // namespace

VIDFAB_TEST(block_span_centres_by_default) {
  // 50 blocks, span 10 -> [20, 30). The point of centring is that neither end
  // block is in the span; assert that directly rather than only the indices,
  // because that is the property the default exists to guarantee.
  const BlockSpan s = resolve_block_span(cfg(10, 2), 50);
  CHECK(s.begin == 20);
  CHECK(s.end == 30);
  CHECK(s.size() == 10);
  CHECK(s.valid());
  CHECK(s.begin > 0);
  CHECK(s.end < 50);

  // Odd remainder biases low, which is what keeps the final block — the one
  // feeding the final norm — out of the span for every width but the full one.
  const BlockSpan odd = resolve_block_span(cfg(9, 2), 50);
  CHECK(odd.begin == 20);
  CHECK(odd.end == 29);
  CHECK(odd.end < 50);
}

VIDFAB_TEST(block_span_honours_explicit_start) {
  const BlockSpan s = resolve_block_span(cfg(4, 2, 3, /*start=*/7), 50);
  CHECK(s.begin == 7);
  CHECK(s.end == 11);
}

VIDFAB_TEST(block_span_clamps_rather_than_failing) {
  // A span wider than the stack means "everything", not an error after a
  // 19.6 GiB load.
  const BlockSpan all = resolve_block_span(cfg(999, 2), 50);
  CHECK(all.begin == 0);
  CHECK(all.end == 50);

  // A start past the end leaves nothing to cache, and must report itself as
  // invalid rather than as a zero-width span the forward pass would treat as
  // live.
  const BlockSpan past = resolve_block_span(cfg(4, 2, 3, /*start=*/60), 50);
  CHECK(!past.valid());

  // Explicit start plus span running off the end is truncated, not wrapped.
  const BlockSpan tail = resolve_block_span(cfg(10, 2, 3, /*start=*/45), 50);
  CHECK(tail.begin == 45);
  CHECK(tail.end == 50);
}

VIDFAB_TEST(block_span_disabled_is_invalid) {
  CHECK(!resolve_block_span(cfg(0, 2), 50).valid());   // span 0
  CHECK(!resolve_block_span(cfg(10, 1), 50).valid());  // interval 1
  CHECK(!resolve_block_span(cfg(10, 2), 0).valid());   // no blocks
}

VIDFAB_TEST(block_cache_plan_protects_warmup_and_terminal) {
  // 10 steps, warmup 3, interval 2. Steps 0,1,2 warm up; the interval is phased
  // from the end of the warmup so step 3 computes, 4 reuses, 5 computes, 6
  // reuses, 7 computes, 8 reuses; step 9 is terminal and forced.
  const std::vector<uint8_t> plan = plan_block_cache(cfg(10, 2, 3), 10);
  CHECK(plan == at(10, {0, 1, 2, 3, 5, 7, 9}));
  CHECK(count(plan) == 7);

  // The terminal step is the one whose residual is written into the output
  // undamped, so it must be computed no matter where the interval lands. Swept
  // rather than spot-checked, because whether the protection is load-bearing
  // depends on where the interval happens to fall.
  int asserted = 0;
  for (int steps = 6; steps <= 24; ++steps) {
    for (int interval = 2; interval <= 5; ++interval) {
      const BlockCacheConfig c = cfg(10, interval, 3);
      const std::vector<uint8_t> p = plan_block_cache(c, steps);
      CHECK(p.back() == 1);
      if (terminal_is_off_interval(c, steps)) {
        // Here the protection is the only reason the last step computes, so
        // dropping it is visible.
        CHECK(p != wrong_plan(c, steps, Wrong::kNoTerminal));
        ++asserted;
      }
    }
  }
  CHECK(asserted > 0);
}

VIDFAB_TEST(block_cache_interval_is_phased_from_the_warmup) {
  // Anchoring the interval at step 0 makes the first post-warmup step compute
  // or reuse depending on whether the warmup happens to divide the interval —
  // a silent coupling between two flags that look independent. With warmup 3
  // and interval 2 the anchored form reuses step 3 and computes step 4, which
  // is the opposite phase.
  const BlockCacheConfig c = cfg(10, 2, 3);
  const std::vector<uint8_t> real = plan_block_cache(c, 12);
  CHECK(real != wrong_plan(c, 12, Wrong::kAnchorAtZero));
  CHECK(real[3] == 1);

  // The property that matters is phase-independence of the warmup: the first
  // step after any warmup is always a computed one.
  for (int w = 1; w <= 6; ++w) {
    const std::vector<uint8_t> p = plan_block_cache(cfg(10, 3, w), 20);
    CHECK(p[static_cast<size_t>(w)] == 1);
    for (int i = 0; i < w; ++i) CHECK(p[static_cast<size_t>(i)] == 1);
  }
}

VIDFAB_TEST(block_cache_never_reuses_an_absent_delta) {
  // The floor on the warmup. `--block-cache-warmup 0` must not become "reuse
  // from step 0", which would add uninitialised device memory to the residual
  // stream — the one way this feature reaches a NaN rather than a drift.
  const std::vector<uint8_t> p = plan_block_cache(cfg(10, 2, /*warmup=*/0), 8);
  CHECK(p[0] == 1);

  // The `have_delta` guard is deliberately redundant with that floor and
  // cannot be distinguished through the planner, which always captures on its
  // first computed step. It is reachable only by a caller that resets its
  // buffers, so it is driven directly: one that never captures never gets a
  // reuse, however long the run.
  BlockCache cache(cfg(10, 2, 0), 8);
  for (int i = 0; i < 8; ++i) CHECK(cache.should_compute(i, /*have_delta=*/false));
  CHECK(cache.reused() == 0);
  CHECK(cache.computed() == 8);
}

VIDFAB_TEST(block_cache_disabled_computes_everything) {
  // The shipped default must leave every step computing, so that a build with
  // this feature present is identical to one without it until asked.
  const std::vector<uint8_t> off = plan_block_cache(cfg(0, 2), 20);
  CHECK(count(off) == 20);
  CHECK(!BlockCache(cfg(0, 2), 20).enabled());

  // Interval 1 disables too, rather than caching for nothing.
  CHECK(count(plan_block_cache(cfg(10, 1), 20)) == 20);
}

VIDFAB_TEST(block_cache_counts_add_up) {
  const BlockCacheConfig c = cfg(10, 2, 3);
  BlockCache cache(c, 20);
  bool have = false;
  for (int i = 0; i < 20; ++i) {
    if (cache.should_compute(i, have)) have = true;
  }
  CHECK(cache.computed() + cache.reused() == 20);
  // The planner and the stepped object are the same policy, so their tallies
  // must agree — that is what makes a test of the planner a test of the run.
  CHECK(cache.computed() == count(plan_block_cache(c, 20)));
  CHECK(cache.warmup() == 3);

  // Longer intervals reuse more, which is the only monotonicity the feature
  // promises.
  const int r2 = 20 - count(plan_block_cache(cfg(10, 2, 3), 20));
  const int r4 = 20 - count(plan_block_cache(cfg(10, 4, 3), 20));
  CHECK(r4 > r2);
}
