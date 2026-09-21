// The step cache's compute-or-skip decision.
//
// This is the part of step caching worth testing without a GPU: given a
// schedule, a threshold and a warmup, which steps run the transformer. Every
// failure mode here is silent — the run still produces a video, just one built
// on a velocity that was staler than intended, or one that quietly evaluated
// every step and bought nothing.
//
// Four wrong implementations are plausible rather than merely broken, so each
// is computed alongside the real one and asserted *not* to match:
//
//   * no terminal protection, or protecting one step too many. At the last step
//     `sigma_next = 0` so `ratio = 0` and `x_next = denoised` exactly (spec
//     7.3) — a reused velocity there is the output.
//   * resetting the accumulator on a skipped step instead of a computed one.
//     That silently degrades the whole feature to "warmup and last only",
//     because a single increment rarely crosses a useful threshold.
//   * measuring the chord to the last computed step rather than the arc length
//     through every step since. The chord under-counts exactly the case the
//     cache is worst at — conditioning that moves out and back.
//
// The distances below are all exact ratios of the synthetic codes, so the
// expected schedules are enumerated rather than approximated.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "harness.h"
#include "slopfab/dit/step_cache.h"

namespace {

using slopfab::dit::AdaLNTable;
using slopfab::dit::conditioning_distance;
using slopfab::dit::kMinWarmup;
using slopfab::dit::plan_step_cache;
using slopfab::dit::StepCache;
using slopfab::dit::StepCacheConfig;

constexpr int kSignature = 2 * AdaLNTable::kRank; // c(t_v) then c(t_a)

// A signature whose components are all `v`. The relative-L1 distance between
// two such is then |v - w| / |w| exactly, independent of the width, which is
// what lets the expected schedules below be written down.
std::vector<float> flat(double v) {
  return std::vector<float>(static_cast<size_t>(kSignature), static_cast<float>(v));
}

std::vector<uint8_t> ones(size_t n) {
  return std::vector<uint8_t>(n, 1);
}

// Wrong forms, each differing from `StepCache` in exactly one decision.
enum class Wrong {
  kNoTerminal,     // last step not protected
  kProtectLastTwo, // one step too many protected
  kResetOnSkip,    // accumulator cleared whether or not the step was computed
  kChord,          // distance to the last computed step, not accumulated
};

std::vector<uint8_t> wrong_plan(const StepCacheConfig& cfg,
                                const std::vector<std::vector<float>>& codes, Wrong which) {
  const int n = static_cast<int>(codes.size());
  const int warmup = std::max(kMinWarmup, cfg.warmup);
  std::vector<uint8_t> out(codes.size(), 0);
  float acc = 0.0f;
  std::vector<float> previous;
  std::vector<float> last_computed;
  for (int i = 0; i < n; ++i) {
    const std::vector<float>& c = codes[static_cast<size_t>(i)];
    bool forced = !cfg.enabled() || i < warmup;
    if (which == Wrong::kNoTerminal) {
      // no terminal clause at all
    } else if (which == Wrong::kProtectLastTwo) {
      forced = forced || i >= n - 2;
    } else {
      forced = forced || i >= n - 1;
    }

    bool compute = true;
    if (!forced) {
      if (cfg.skip_every > 0) {
        compute = (i % cfg.skip_every) == 0;
      } else if (which == Wrong::kChord) {
        const float d =
            last_computed.empty()
                ? 0.0f
                : conditioning_distance(c.data(), last_computed.data(), static_cast<int>(c.size()));
        compute = d >= cfg.threshold;
      } else {
        if (!previous.empty()) {
          acc += conditioning_distance(c.data(), previous.data(), static_cast<int>(c.size()));
        }
        compute = acc >= cfg.threshold;
      }
    }
    if (which == Wrong::kResetOnSkip) {
      acc = 0.0f; // unconditionally, which is the bug
    } else if (compute) {
      acc = 0.0f;
    }
    previous = c;
    if (compute)
      last_computed = c;
    out[static_cast<size_t>(i)] = compute ? 1 : 0;
  }
  return out;
}

int count(const std::vector<uint8_t>& plan) {
  int n = 0;
  for (uint8_t v : plan)
    n += v;
  return n;
}

// 1 at each listed index, 0 elsewhere.
std::vector<uint8_t> at(size_t n, const std::vector<int>& indices) {
  std::vector<uint8_t> out(n, 0);
  for (int i : indices)
    out[static_cast<size_t>(i)] = 1;
  return out;
}

} // namespace

SLOPFAB_TEST(step_cache_distance_is_relative) {
  const std::vector<float> a = flat(1.0);
  const std::vector<float> b = flat(1.1);

  CHECK_NEAR(conditioning_distance(a.data(), a.data(), kSignature), 0.0, 0.0);
  CHECK_NEAR(conditioning_distance(b.data(), a.data(), kSignature), 0.1, 1e-6);

  // Relative, so scaling both sides leaves it alone. This is the property that
  // makes one threshold portable across a table whose eight columns span three
  // orders of magnitude.
  const std::vector<float> a10 = flat(10.0);
  const std::vector<float> b10 = flat(11.0);
  CHECK_NEAR(conditioning_distance(b10.data(), a10.data(), kSignature),
             conditioning_distance(b.data(), a.data(), kSignature), 1e-6);

  // A zero reference is a division by zero waiting to skip the rest of a run.
  const std::vector<float> zero = flat(0.0);
  CHECK_NEAR(conditioning_distance(a.data(), zero.data(), kSignature), 0.0, 0.0);
  CHECK_NEAR(conditioning_distance(nullptr, a.data(), kSignature), 0.0, 0.0);
}

SLOPFAB_TEST(step_cache_off_computes_everything) {
  // Codes that move violently, so nothing but the disable can be keeping the
  // count at N.
  std::vector<std::vector<float>> codes;
  for (int i = 0; i < 30; ++i)
    codes.push_back(flat(1.0 + 5.0 * i));

  StepCacheConfig cfg; // defaults: threshold 0, skip_every 0
  CHECK(!cfg.enabled());
  CHECK(plan_step_cache(cfg, codes) == ones(codes.size()));

  // The warmup flag alone must not switch anything on.
  cfg.warmup = 7;
  CHECK(!cfg.enabled());
  CHECK(plan_step_cache(cfg, codes) == ones(codes.size()));

  StepCache cache(cfg, static_cast<int>(codes.size()));
  for (size_t i = 0; i < codes.size(); ++i) {
    CHECK(cache.should_compute(static_cast<int>(i), nullptr, 0));
  }
  CHECK(cache.skipped() == 0);
  CHECK(cache.computed() == static_cast<int>(codes.size()));
}

SLOPFAB_TEST(step_cache_warmup_and_last_are_unconditional) {
  // A monotone schedule with a threshold so large it can never be reached:
  // what survives is exactly the two guarantees that are enforced in code.
  std::vector<std::vector<float>> codes;
  for (int i = 0; i < 20; ++i)
    codes.push_back(flat(1.0 + 0.001 * i));

  StepCacheConfig cfg;
  cfg.threshold = 1e6f;
  cfg.warmup = 5;

  const std::vector<uint8_t> plan = plan_step_cache(cfg, codes);
  CHECK(plan == at(codes.size(), {0, 1, 2, 3, 4, 19}));
  CHECK(count(plan) == 6);

  // The last step, spelled out: `>= n - 1`, not `> n - 1` and not `>= n - 2`.
  CHECK(plan[19] == 1);
  CHECK(plan[18] == 0);
  CHECK(plan != wrong_plan(cfg, codes, Wrong::kNoTerminal));
  CHECK(plan != wrong_plan(cfg, codes, Wrong::kProtectLastTwo));
  CHECK(wrong_plan(cfg, codes, Wrong::kNoTerminal)[19] == 0);
  CHECK(wrong_plan(cfg, codes, Wrong::kProtectLastTwo)[18] == 1);
}

SLOPFAB_TEST(step_cache_warmup_has_a_floor) {
  std::vector<std::vector<float>> codes;
  for (int i = 0; i < 12; ++i)
    codes.push_back(flat(1.0 + 0.001 * i));

  // Step 0 has no velocity to reuse — the buffers are zeros — and step 1 would
  // reuse one predicted from pure noise. A caller asking for warmup 0 gets 2.
  StepCacheConfig cfg;
  cfg.threshold = 1e6f;
  cfg.warmup = 0;
  const std::vector<uint8_t> plan = plan_step_cache(cfg, codes);
  CHECK(plan == at(codes.size(), {0, 1, 11}));

  StepCache cache(cfg, 12);
  CHECK(cache.warmup() == kMinWarmup);
}

SLOPFAB_TEST(step_cache_accumulates_across_skipped_steps) {
  // Geometric codes, so every consecutive distance is exactly 0.1 and the
  // crossings can be written down: the accumulator reaches 0.4 on the fourth
  // skipped step in a row and no sooner.
  std::vector<std::vector<float>> codes;
  double v = 1.0;
  for (int i = 0; i < 12; ++i) {
    codes.push_back(flat(v));
    v *= 1.1;
  }

  StepCacheConfig cfg;
  cfg.threshold = 0.35f;
  cfg.warmup = 1; // raised to 2 by the floor

  const std::vector<uint8_t> plan = plan_step_cache(cfg, codes);
  CHECK(plan == at(codes.size(), {0, 1, 5, 9, 11}));

  // Resetting on a skip never lets three increments of 0.1 add up to 0.35, so
  // the whole feature silently collapses to warmup-plus-last. The two forms
  // agree on those steps, which is why this is invisible without the diff.
  const std::vector<uint8_t> reset_on_skip = wrong_plan(cfg, codes, Wrong::kResetOnSkip);
  CHECK(plan != reset_on_skip);
  CHECK(reset_on_skip == at(codes.size(), {0, 1, 11}));

  // Counters agree with the plan and account for every step.
  StepCache cache(cfg, static_cast<int>(codes.size()));
  for (size_t i = 0; i < codes.size(); ++i) {
    cache.should_compute(static_cast<int>(i), codes[i].data(), kSignature);
  }
  CHECK(cache.computed() == 5);
  CHECK(cache.skipped() == 7);
  CHECK(cache.computed() + cache.skipped() == static_cast<int>(codes.size()));
}

SLOPFAB_TEST(step_cache_measures_arc_not_chord) {
  // Conditioning that moves out and back. The chord to the last computed step
  // is near zero every other step, so a chord-based rule never recomputes —
  // and the reused velocity is at its stalest exactly there.
  std::vector<std::vector<float>> codes;
  for (int i = 0; i < 10; ++i)
    codes.push_back(flat((i % 2 == 0) ? 1.0 : 1.15));

  StepCacheConfig cfg;
  cfg.threshold = 0.5f;
  cfg.warmup = 2;

  const std::vector<uint8_t> plan = plan_step_cache(cfg, codes);
  CHECK(plan == at(codes.size(), {0, 1, 5, 9}));

  const std::vector<uint8_t> chord = wrong_plan(cfg, codes, Wrong::kChord);
  CHECK(plan != chord);
  CHECK(chord == at(codes.size(), {0, 1, 9}));
}

SLOPFAB_TEST(step_cache_skip_every_is_warmup_plus_interval_plus_last) {
  std::vector<std::vector<float>> codes;
  for (int i = 0; i < 20; ++i)
    codes.push_back(flat(1.0 + 0.7 * i));

  StepCacheConfig cfg;
  cfg.skip_every = 4;
  cfg.warmup = 3;
  CHECK(cfg.enabled());

  const std::vector<uint8_t> plan = plan_step_cache(cfg, codes);
  CHECK(plan == at(codes.size(), {0, 1, 2, 4, 8, 12, 16, 19}));
  CHECK(count(plan) == 8);

  // Blind to the schedule by construction: the same interval on wildly
  // different codes produces the same plan. That is what makes it the honest
  // baseline for the threshold.
  std::vector<std::vector<float>> flat_codes(20, flat(1.0));
  CHECK(plan_step_cache(cfg, flat_codes) == plan);

  // n = 1 degenerates to computing everything rather than to a division bug.
  cfg.skip_every = 1;
  CHECK(plan_step_cache(cfg, codes) == ones(codes.size()));
}

SLOPFAB_TEST(step_cache_skip_every_wins_over_threshold) {
  // The CLI rejects the combination; the library resolves it, and which way it
  // resolves is worth pinning rather than discovering.
  std::vector<std::vector<float>> codes;
  for (int i = 0; i < 20; ++i)
    codes.push_back(flat(1.0 + 0.7 * i));

  StepCacheConfig both;
  both.skip_every = 4;
  both.warmup = 3;
  both.threshold = 1e-9f; // would compute every step if it were consulted

  StepCacheConfig interval = both;
  interval.threshold = 0.0f;
  CHECK(plan_step_cache(both, codes) == plan_step_cache(interval, codes));
}

SLOPFAB_TEST(step_cache_uses_both_timesteps) {
  // t2va has two distinct timesteps per step on grids of different shift (spec
  // 7.5). A distance built over the video one alone cannot see the audio
  // branch's conditioning moving, so a schedule whose video timestep is frozen
  // would never recompute.
  const slopfab::dit::CodeFn code = [](float t) {
    std::array<float, AdaLNTable::kRank> c{};
    c[0] = t;
    return c;
  };

  StepCacheConfig cfg;
  cfg.threshold = 0.2f;
  cfg.warmup = 2;

  std::vector<std::pair<float, float>> moving;
  std::vector<std::pair<float, float>> frozen;
  for (int i = 0; i < 10; ++i) {
    moving.emplace_back(0.5f, 0.5f + 0.05f * static_cast<float>(i));
    frozen.emplace_back(0.5f, 0.5f);
  }

  const std::vector<uint8_t> plan = plan_step_cache(cfg, moving, code);
  CHECK(plan == at(moving.size(), {0, 1, 6, 9}));

  // Nothing moves at all: only the two guarantees fire.
  CHECK(plan_step_cache(cfg, frozen, code) == at(frozen.size(), {0, 1, 9}));
  CHECK(count(plan) > count(plan_step_cache(cfg, frozen, code)));
}

SLOPFAB_TEST(step_cache_enabled_is_the_one_definition) {
  // `StepCacheConfig::enabled()` is what the CLI uses to decide whether
  // `--sampler ab2` is refused, so this is not an accessor test — it pins the
  // exact combination matrix that guard implements.
  //
  // The row that matters most is the last one. `--sampler ab2` with caching off
  // is the floor control for the whole threshold sweep: two legitimate
  // integrations of the same ODE at identical evaluation counts, which is the
  // only way to know whether a threshold's diff means anything or sits inside
  // the distance two valid trajectories land apart anyway. If `enabled()` ever
  // returns true for a default-constructed config, that run becomes illegal and
  // the sweep loses its interpretation while every individual number in it
  // still looks fine.
  StepCacheConfig off;
  CHECK(!off.enabled());

  StepCacheConfig warmup_only;
  warmup_only.warmup = 12;
  CHECK(!warmup_only.enabled()); // warmup alone is inert, not an opt-in

  StepCacheConfig explicit_zero;
  explicit_zero.threshold = 0.0f;
  explicit_zero.skip_every = 0;
  CHECK(!explicit_zero.enabled());

  StepCacheConfig thresholded;
  thresholded.threshold = 1e-6f;
  CHECK(thresholded.enabled());

  StepCacheConfig interval;
  interval.skip_every = 1;
  CHECK(interval.enabled()); // degenerate but opted in: it took the flag path
}

SLOPFAB_TEST(step_cache_short_schedules) {
  StepCacheConfig cfg;
  cfg.threshold = 1e-9f;
  cfg.warmup = 8; // longer than the schedule

  for (int n = 1; n <= 4; ++n) {
    std::vector<std::vector<float>> codes;
    for (int i = 0; i < n; ++i)
      codes.push_back(flat(1.0 + i));
    const std::vector<uint8_t> plan = plan_step_cache(cfg, codes);
    CHECK_MSG(plan == ones(codes.size()), "a %d-step schedule must evaluate every step", n);
  }
}
