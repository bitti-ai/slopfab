#include "vidfab/dit/block_cache.h"

#include <algorithm>

namespace vidfab::dit {

BlockSpan resolve_block_span(const BlockCacheConfig& config, int num_layers) {
  BlockSpan span;
  if (!config.enabled() || num_layers <= 0) return span;

  const int width = std::min(config.span, num_layers);
  // Centring biases low on an odd remainder, which keeps the last block — the
  // one whose residual feeds the final norm directly — outside the span for
  // every width the stack can hold except the full one.
  const int begin = config.start < 0 ? (num_layers - width) / 2 : config.start;

  span.begin = std::clamp(begin, 0, num_layers);
  span.end = std::min(span.begin + width, num_layers);
  // An explicit start past the end leaves nothing to cache. Report that as an
  // invalid span rather than a zero-width one at the end of the stack, so the
  // caller's `valid()` test catches it.
  if (span.end <= span.begin) {
    span.begin = 0;
    span.end = 0;
  }
  return span;
}

BlockCache::BlockCache(const BlockCacheConfig& config, int num_steps)
    : config_(config), num_steps_(std::max(0, num_steps)) {
  warmup_ = std::max(kMinBlockWarmup, config_.warmup);
}

bool BlockCache::should_compute(int step, bool have_delta) {
  // The terminal step, for the same reason the step cache protects it: at the
  // last evaluation `sigma_next = 0`, so `x_next = denoised` exactly (spec 7.3)
  // and whatever the span contributed is written into the output with no
  // damping at all. Costing one span evaluation to keep the final frame off a
  // stale delta is the cheapest insurance in the loop.
  const bool forced = !config_.enabled() || !have_delta || step < warmup_ ||
                      step >= num_steps_ - 1;

  // Phase the interval from the end of the warmup rather than from step 0, so
  // that `warmup` and `interval` are independent: anchoring at 0 makes the
  // first post-warmup step a compute or a reuse depending on whether the warmup
  // happens to divide the interval, which is a silent coupling between two
  // flags that look unrelated.
  //
  // The divisor is floored rather than trusted. `enabled()` already requires it
  // to be at least 2, and `forced` short-circuits when it is not — but that
  // makes the modulo safe *contingently*, on the order of a `||`, and the next
  // person to hoist this expression for readability would introduce a division
  // by zero that no test covers because no valid config reaches it.
  const int interval = std::max(2, config_.interval);
  const bool compute = forced || ((step - warmup_) % interval) == 0;

  if (compute) {
    ++computed_;
  } else {
    ++reused_;
  }
  return compute;
}

std::vector<uint8_t> plan_block_cache(const BlockCacheConfig& config, int num_steps) {
  const int steps = std::max(0, num_steps);
  BlockCache cache(config, steps);
  std::vector<uint8_t> out(static_cast<size_t>(steps), 0);
  // Mirrors the forward pass: the delta exists from the first computed step
  // onwards and is never dropped mid-run.
  bool have_delta = false;
  for (int i = 0; i < steps; ++i) {
    const bool compute = cache.should_compute(i, have_delta);
    if (compute) have_delta = true;
    out[static_cast<size_t>(i)] = compute ? 1 : 0;
  }
  return out;
}

}  // namespace vidfab::dit
