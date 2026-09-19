#include "harness.h"
#include "slopfab/dit/motion_cache.h"
#include "slopfab/pipeline.h"

#include <limits>
#include <stdexcept>

using namespace slopfab::dit;
namespace {
SequenceLayout layout(int frames = 2) {
  SequenceLayout l;
  l.num_latent_frames = frames;
  l.latent_height = l.latent_width = 2;
  l.num_video_rows = frames;
  l.num_audio_latents = 2;
  l.num_audio_rows = 4;
  return l;
}
MotionCacheConfig enabled() {
  MotionCacheConfig c;
  c.enabled = true;
  c.warmup_steps = 2;
  c.start_percent = 0;
  c.end_percent = 1;
  c.reuse_threshold = .2f;
  return c;
}
void prime(MotionCache& cache) {
  std::vector<float> v(8, 1), a(4, 1), vv(8, 2), av(4, 2);
  CHECK(cache.should_compute(0, .9f, v.data(), a.data()));
  cache.update(.9f, v.data(), a.data(), vv.data(), av.data());
  v.assign(8, 2); a.assign(4, 2); vv.assign(8, 4); av.assign(4, 4);
  CHECK(cache.should_compute(1, .8f, v.data(), a.data()));
  cache.update(.8f, v.data(), a.data(), vv.data(), av.data());
}
}

SLOPFAB_TEST(motion_cache_residual_sign_accumulation_and_limits) {
  MotionCache cache(enabled(), layout(), 4, 1, 10, 12);
  prime(cache);
  std::vector<float> v(8, 2.1f), a(4, 2.1f), vv(8), av(4);
  CHECK(!cache.should_compute(2, .7f, v.data(), a.data()));
  CHECK_NEAR(cache.score(), .05, 1e-6);
  cache.reuse(v.data(), a.data(), vv.data(), av.data());
  for (float value : vv) CHECK_NEAR(value, 3.9, 1e-6);
  for (float value : av) CHECK_NEAR(value, 3.9, 1e-6);
  v.assign(8, 2.2f); a.assign(4, 2.2f);
  CHECK(!cache.should_compute(3, .6f, v.data(), a.data()));
  CHECK(cache.should_compute(4, .5f, v.data(), a.data())); // skip limit
  CHECK(cache.computed() == 3 && cache.skipped() == 2);

  auto config = enabled(); config.reuse_threshold = .12f;
  MotionCache accumulated(config, layout(), 4, 1, 10, 12);
  prime(accumulated);
  v.assign(8, 2.1f); a.assign(4, 2.1f);
  CHECK(!accumulated.should_compute(2, .7f, v.data(), a.data()));
  v.assign(8, 2.2f); a.assign(4, 2.2f);
  CHECK(accumulated.should_compute(3, .6f, v.data(), a.data())); // .05 + .10
}

SLOPFAB_TEST(motion_cache_audio_controls_joint_reuse_and_nonfinite_recomputes) {
  MotionCache cache(enabled(), layout(), 4, 1, 10, 12);
  prime(cache);
  std::vector<float> v(8, 2), a(4, 3);
  CHECK(cache.should_compute(2, .7f, v.data(), a.data()));
  CHECK_NEAR(cache.score(), .5, 1e-6);
  a.assign(4, 2);
  a[2] = std::numeric_limits<float>::quiet_NaN(); // other stereo channel
  CHECK(cache.should_compute(3, .6f, v.data(), a.data()));
  a.assign(4, 2); v[0] = std::numeric_limits<float>::infinity();
  CHECK(cache.should_compute(4, .5f, v.data(), a.data()));
}

SLOPFAB_TEST(motion_cache_motion_weights_prioritize_moving_frames) {
  for (int changed_frame : {0, 2}) {
    auto l = layout(3);
    l.num_audio_latents = l.num_audio_rows = 0;
    MotionCache cache(enabled(), l, 4, 1, 10, 12);
    // Frame deltas [1,1,4] give normalized weights [.75,.75,1.5].
    std::vector<float> v{ -1,-1,-1,-1, 0,0,0,0, 4,4,4,4 }, vv(12, 1);
    CHECK(cache.should_compute(0, .9f, v.data(), nullptr));
    cache.update(.9f, v.data(), nullptr, vv.data(), nullptr);
    for (float& x : v) ++x;
    vv.assign(12, 2);
    CHECK(cache.should_compute(1, .8f, v.data(), nullptr));
    cache.update(.8f, v.data(), nullptr, vv.data(), nullptr);
    v[size_t(changed_frame) * 4] += 1;
    CHECK(cache.should_compute(2, .7f, v.data(), nullptr) == (changed_frame == 2));
    CHECK_NEAR(cache.score(), changed_frame == 2 ? .25 : .125, 1e-6);
  }
}

SLOPFAB_TEST(motion_cache_subsamples_latent_coordinates_not_packed_features) {
  auto c = enabled(); c.subsample_factor = 3;
  auto l = layout();
  l.latent_height = l.latent_width = 4;
  l.num_video_rows = 8;
  l.num_audio_latents = 5; l.num_audio_rows = 10;
  MotionCache cache(c, l, 8, 2, 10, 12);
  std::vector<float> v(64, 1), a(20, 1), vv(64, 2), av(20, 2);
  cache.update(.9f, v.data(), a.data(), vv.data(), av.data());
  v.assign(64, 2); a.assign(20, 2); vv.assign(64, 4); av.assign(20, 4);
  cache.update(.8f, v.data(), a.data(), vv.data(), av.data());
  v[1] = 100; // (h=0,w=1) not sampled
  a[2] = 100; // t=1 not sampled
  CHECK(!cache.should_compute(2, .7f, v.data(), a.data()));
  CHECK_NEAR(cache.score(), 0, 0);
  v[31] = 100; // channel 1, h=3,w=3: inside a patch, sampled
  CHECK(cache.should_compute(3, .6f, v.data(), a.data()));
}

SLOPFAB_TEST(motion_cache_warmup_sigma_range_terminal_and_reset) {
  auto c = enabled(); c.start_percent = .15f; c.end_percent = .95f;
  MotionCache cache(c, layout(), 4, 1, 10, 12);
  prime(cache);
  std::vector<float> v(8, 2), a(4, 2);
  CHECK(cache.should_compute(2, 1, v.data(), a.data()));
  CHECK(!cache.should_compute(3, .9f, v.data(), a.data()));
  CHECK(cache.should_compute(4, .3f, v.data(), a.data())); // shift-12 end sigma ~.387
  CHECK(cache.should_compute(9, .8f, v.data(), a.data())); // terminal always fresh
  MotionCache fresh(c, layout(), 4, 1, 10, 12);
  CHECK(fresh.should_compute(3, .8f, v.data(), a.data()));
  c.enabled = false;
  MotionCache off(c, layout(), 4, 1, 10, 12);
  CHECK(off.should_compute(3, .8f, nullptr, nullptr));
  c.enabled = true; c.reuse_threshold = 0;
  MotionCache zero(c, layout(), 4, 1, 10, 12);
  CHECK(zero.should_compute(3, .8f, nullptr, nullptr));
}

SLOPFAB_TEST(motion_cache_validation_and_plan) {
  for (int invalid = 0; invalid < 8; ++invalid) {
    auto c = enabled();
    if (invalid == 0) c.reuse_threshold = std::numeric_limits<float>::quiet_NaN();
    if (invalid == 1) c.motion_strength = -1;
    if (invalid == 2) c.warmup_steps = 1;
    if (invalid == 3) c.max_consecutive_skips = 0;
    if (invalid == 4) c.start_percent = c.end_percent;
    if (invalid == 5) c.end_percent = 2;
    if (invalid == 6) c.subsample_factor = 0;
    if (invalid == 7) c.start_percent = std::numeric_limits<float>::infinity();
    bool rejected = false;
    try { c.validate(); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
  }
  slopfab::GenerateRequest request;
  request.motion_cache = enabled();
  auto plan = slopfab::resolve_plan(request);
  CHECK(slopfab::describe_plan(request, plan).find("MotionCache") != std::string::npos);
  request.skip_every = 2;
  bool rejected = false;
  try { slopfab::resolve_plan(request); } catch (const std::invalid_argument&) { rejected = true; }
  CHECK(rejected);
}
