#include "vidfab/dit/step_cache.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vidfab::dit {

float conditioning_distance(const float* a, const float* reference, int n) {
  if (a == nullptr || reference == nullptr || n <= 0) return 0.0f;
  double delta = 0.0;
  double scale = 0.0;
  for (int i = 0; i < n; ++i) {
    delta += std::fabs(static_cast<double>(a[i]) - static_cast<double>(reference[i]));
    scale += std::fabs(static_cast<double>(reference[i]));
  }
  if (!(scale > 0.0)) return 0.0f;
  return static_cast<float>(delta / scale);
}

StepCache::StepCache(const StepCacheConfig& config, int num_steps)
    : config_(config), num_steps_(std::max(0, num_steps)) {
  if (config_.skip_every < 0) config_.skip_every = 0;
  // The floor, not the default. A caller passing 0 gets 2, because step 0 has
  // no velocity to reuse and step 1 reuses one drawn from pure noise.
  warmup_ = std::max(kMinWarmup, config_.warmup);
}

bool StepCache::should_compute(int step, const float* code, int len) {
  if (code == nullptr) len = 0;

  // The two unconditional computes. `step >= num_steps_ - 1` rather than
  // `step > num_steps_ - 1`: the schedule has `num_steps_` model evaluations
  // indexed 0..num_steps_-1, so the last one is `num_steps_ - 1`, and the
  // terminal step is where a stale velocity is written into the output
  // undamped (spec 7.3).
  const bool forced = !config_.enabled() || step < warmup_ || step >= num_steps_ - 1;

  bool compute = true;
  if (!forced) {
    if (config_.skip_every > 0) {
      // Fixed interval. Deliberately blind to the schedule — that is the whole
      // point of having it as the comparison.
      compute = (step % config_.skip_every) == 0;
    } else {
      // Arc length since the last computed step, not the chord to it.
      if (have_previous_ && len == static_cast<int>(previous_.size())) {
        accumulator_ += conditioning_distance(code, previous_.data(), len);
      }
      compute = accumulator_ >= config_.threshold;
    }
  }

  // Reset only on a computed step. Resetting on a skipped one turns the
  // accumulation into a per-step comparison, which never crosses a threshold
  // larger than a single increment and silently degrades to "compute the
  // warmup and the last step and nothing else".
  if (compute) accumulator_ = 0.0f;

  if (len > 0) {
    previous_.assign(code, code + len);
    have_previous_ = true;
  }

  if (compute) {
    ++computed_;
  } else {
    ++skipped_;
  }
  return compute;
}

std::vector<uint8_t> plan_step_cache(const StepCacheConfig& config,
                                     const std::vector<std::vector<float>>& codes) {
  const int steps = static_cast<int>(codes.size());
  StepCache cache(config, steps);
  std::vector<uint8_t> out(codes.size(), 0);
  for (int i = 0; i < steps; ++i) {
    const std::vector<float>& c = codes[static_cast<size_t>(i)];
    out[static_cast<size_t>(i)] =
        cache.should_compute(i, c.empty() ? nullptr : c.data(), static_cast<int>(c.size())) ? 1 : 0;
  }
  return out;
}

void build_signature(const CodeFn& code, float t_video, float t_audio, std::vector<float>& out) {
  if (!code) throw std::runtime_error("build_signature: no code function");
  const std::array<float, AdaLNTable::kRank> cv = code(t_video);
  const std::array<float, AdaLNTable::kRank> ca = code(t_audio);
  out.resize(2 * static_cast<size_t>(AdaLNTable::kRank));
  for (int k = 0; k < AdaLNTable::kRank; ++k) {
    out[static_cast<size_t>(k)] = cv[static_cast<size_t>(k)];
    out[static_cast<size_t>(AdaLNTable::kRank + k)] = ca[static_cast<size_t>(k)];
  }
}

std::vector<uint8_t> plan_step_cache(const StepCacheConfig& config,
                                     const std::vector<std::pair<float, float>>& schedule,
                                     const CodeFn& code) {
  if (!code) throw std::runtime_error("plan_step_cache: no code function");
  std::vector<std::vector<float>> codes;
  codes.reserve(schedule.size());
  std::vector<float> row;
  for (const std::pair<float, float>& t : schedule) {
    build_signature(code, t.first, t.second, row);
    codes.push_back(row);
  }
  return plan_step_cache(config, codes);
}

}  // namespace vidfab::dit
