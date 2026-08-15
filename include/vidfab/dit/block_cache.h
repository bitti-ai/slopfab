// Reusing one contiguous span of blocks' combined residual across steps.
//
// The step cache next door (step_cache.h) skips a whole model evaluation: 100%
// of a step, some of the time. This skips a *fraction of every step* instead —
// a contiguous run of transformer blocks whose combined contribution to the
// residual stream is reused from the last step that computed it. Same family
// (FORA, Delta-DiT, TaylorSeer), finer granularity.
//
// **The two cannot currently be enabled together, and main.cpp refuses the
// combination.** Not because they conflict in principle but because of where
// each one lives: the step cache skips a step by not calling `forward` at all,
// and this cache's decision is made *inside* `forward`. So a step the step
// cache skips is a step this one never sees — the interval stops counting real
// steps, a delta believed to be one step old silently becomes three or more,
// and the reuse percentage reported at the end divides by only the steps that
// ran, understating the staleness instead of revealing it. Lifting the
// restriction means giving this class a `skip_step(i)` that the denoise loop
// calls on skipped steps, so the phase and the counters stay defined; it does
// not mean deleting the check.
//
// **Why a span and not per-block.** The published block caches store one
// residual per block. That is not affordable here and the arithmetic is not
// close: the residual stream is `[seq, 5376]` bf16, 844 MB at the production
// geometry, and 50 of those is 42 GB against a 32 GB card already holding 19.6
// GiB of weights. One delta for a whole span is 844 MB and fits in the
// headroom. So the cached quantity is
//
//     delta = x_after_span - x_before_span
//
// for one contiguous `[begin, end)` run, captured on a computed step and added
// to the live `x` on a reused one. The delta lives in **one** buffer, not two:
// it holds the "before" state until the subtract overwrites it in place.
//
// The span is skipped entirely when reused, which is where the time comes from.
// The saving is
//
//     (span / num_layers) * (reused steps / total steps)
//
// and it is worth recording that this crude prediction is what actually
// happens. Measured 11 Aug 2026 at 4,116 packed rows over 29 steps, against the
// profiler's own step clock: span 10 interval 2 predicted 8.28% and measured
// 8.28%; span 20 interval 2 predicted 16.55% and measured 16.55%; span 20
// interval 3 predicted 22.07% and measured 22.00%. Nothing is lost to overhead
// because the 50 blocks are ~99% of a step and the cache's own bookkeeping —
// snapshot, capture and reuse together — is 0.072 ms against a 920 ms step.
// Run-to-run noise on that rig was 1.9%, which is larger than every gap above.
//
// **That measurement transfers to the production geometry even though it was
// taken at 4,116 rows.** The obvious objection is that attention is O(rows²)
// and is ~50% of a step at 78,482 rows against a much smaller share at 4,116,
// so the mix inside a block is completely different. It does not matter: the
// prediction is `span/num_layers x reuse_fraction`, and every block has the
// same mix as every other block, whatever that mix happens to be. Skipping 20
// of 50 blocks removes 40% of the block work regardless of how that work
// divides between attention and FFN.
//
// The one figure that does *not* transfer is the 0.072 ms. At 4,116 rows the
// stream and the delta total 88.6 MB and largely sit in L2; at 78,482 rows they
// are 1.69 GB and cannot. Expect single-digit milliseconds there — still
// ~0.008% of a 37.5 s step, so the ratio holds and the conclusion is unchanged.
//
// **What this bets on.** That the span's contribution to the residual changes
// slowly between adjacent steps. That is the same bet as the step cache, made
// over blocks instead of over the whole model, and it is why the flags default
// off. The failure mode is not a crash — it is a sample that drifts, so the
// honest way to judge a setting is to generate the same seed with and without.
//
// **What it does not model.** The delta is reused unscaled and unrotated. The
// incoming `x` has moved since the delta was captured, so the reused delta is
// the right correction for a state that no longer exists. Blocks near the ends
// of the stack are the worst offenders in the literature, which is why the
// default span is centred rather than anchored at 0.
//
// **If drift shows up, do not reach for precision.** The entire error budget
// here is staleness. Storing the delta in bf16 costs nothing worth having: the
// subtract is done in fp32 and is exact for any exponent separation the stream
// can produce, the store's rounding is the same one `add_gated` would have
// applied on the path this replaces, and bf16 shares fp32's exponent range so
// there is no overflow to find. An fp32 delta would double the buffer and buy
// nothing. The axes that actually matter are a scale factor on the reused
// delta, a shorter interval, and a narrower span.
#pragma once

#include <cstdint>
#include <vector>

namespace vidfab::dit {

struct BlockCacheConfig {
  // Number of consecutive blocks whose combined residual is cached. **Zero
  // disables the whole feature** and is the default.
  int span = 0;

  // First block of the span. Negative means centre it in the stack, which is
  // the default and the only value with a defensible justification: the first
  // and last blocks carry the fastest-changing residuals, so a span that
  // swallows them degrades the sample far out of proportion to the time saved.
  int start = -1;

  // Recompute the span exactly every n-th step, reuse it on the others. 2 means
  // alternate. Values below 2 disable the feature rather than reusing nothing,
  // because an interval of 1 computes every step and would pay the snapshot and
  // subtract for nothing.
  int interval = 2;

  // Steps at the start of the schedule where the span is always computed. Same
  // reasoning as the step cache's: the trajectory is most sensitive early, and
  // step 0 has no delta to reuse at all.
  int warmup = 3;

  bool enabled() const { return span > 0 && interval >= 2; }
};

// The resolved span, half-open `[begin, end)` over block indices.
struct BlockSpan {
  int begin = 0;
  int end = 0;

  int size() const { return end - begin; }
  bool valid() const { return end > begin; }
};

// Places `config.span` blocks inside a stack of `num_layers`.
//
// Centring is `(num_layers - span) / 2`, and an explicit `start` is used as
// given. Both are then clamped into `[0, num_layers]`; a span wider than the
// stack becomes the whole stack rather than an error, so that `--block-cache-
// span 999` means "everything" instead of failing after a 19.6 GiB load.
//
// Returns an invalid span when the config is disabled, which is what the
// forward pass tests rather than re-deriving the enable condition.
BlockSpan resolve_block_span(const BlockCacheConfig& config, int num_layers);

// The compute-or-reuse decision for one denoising run, stepped in order.
//
// Deliberately schedule-blind, unlike `StepCache`: this one keys on the step
// index alone. The conditioning-distance indicator that makes the step cache
// calibration-free is a statement about how much the *whole* model's output
// moved, and there is no reason to believe it transfers to one span of blocks
// without measuring it. A fixed interval is the honest thing to ship first, and
// it is the baseline any adaptive rule would have to beat.
class BlockCache {
 public:
  BlockCache() = default;
  BlockCache(const BlockCacheConfig& config, int num_steps);

  bool enabled() const { return config_.enabled(); }
  const BlockCacheConfig& config() const { return config_; }

  // Call exactly once per step, in order. Returns true when the span must be
  // evaluated and its delta captured, false when the cached delta is to be
  // added instead.
  //
  // `have_delta` is the caller's own answer to "is there anything to reuse" —
  // the first computed step has to happen before any reuse can, and the warmup
  // already guarantees that. Passing false forces a compute regardless, so a
  // caller that resets its buffers cannot silently add uninitialised memory to
  // the residual stream.
  bool should_compute(int step, bool have_delta);

  int computed() const { return computed_; }
  int reused() const { return reused_; }

  // The warmup actually in force, after the one-step floor.
  int warmup() const { return warmup_; }

 private:
  BlockCacheConfig config_;
  int num_steps_ = 0;
  int warmup_ = 1;
  int computed_ = 0;
  int reused_ = 0;
};

// At least this many steps are computed at the start no matter what the flags
// say. One, not the step cache's two: there is no velocity buffer here that
// starts out zeroed, only a delta that starts out absent, and `should_compute`
// forces a compute while it is absent anyway. The floor exists so that a
// `--block-cache-warmup 0` cannot express "reuse from step 0", which would mean
// reusing a delta captured at a timestep the sample never passed through.
constexpr int kMinBlockWarmup = 1;

// Which steps compute the span, for a whole schedule at once. 1 = compute,
// 0 = reuse. Same object the forward pass drives, run to completion, so a test
// of this is a test of the run.
std::vector<uint8_t> plan_block_cache(const BlockCacheConfig& config, int num_steps);

}  // namespace vidfab::dit
