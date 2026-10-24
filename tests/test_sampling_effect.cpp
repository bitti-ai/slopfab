#include "harness.h"
#include "slopfab/sampling_effect.h"
#include <limits>
using namespace slopfab;

SLOPFAB_TEST(sampling_effect_noop_edit_cancel_and_invalid) {
  const float x[] = {.12345f, -2.f}, v[] = {.23456f, 3.f};
  SamplingStep step;
  step.count = 2;
  step.latent = x;
  step.sigma = .37f;
  std::vector<float> modified, prediction;
  CHECK(apply_sampling_effect(
      [](const SamplingStep&) {
        return true;
      },
      step, v, modified, prediction));
  CHECK(modified[0] == v[0] && modified[1] == v[1]);
  CHECK(apply_sampling_effect(
      [](const SamplingStep& s) {
        s.denoised[1] += 2 * s.sigma;
        return true;
      },
      step, v, modified, prediction));
  CHECK(modified[0] == v[0]);
  CHECK_NEAR(modified[1], 5.f, 1e-6);
  CHECK(!apply_sampling_effect(
      [](const SamplingStep&) {
        return false;
      },
      step, v, modified, prediction));
  bool rejected = false;
  try {
    apply_sampling_effect(
        [](const SamplingStep& s) {
          s.denoised[0] = std::numeric_limits<float>::quiet_NaN();
          return true;
        },
        step, v, modified, prediction);
  } catch (const std::exception&) {
    rejected = true;
  }
  CHECK(rejected);
}
