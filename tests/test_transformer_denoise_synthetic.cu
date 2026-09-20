#include "detail/transformer_fixture.h"

SLOPFAB_TEST_CATEGORY(denoise_motion_cache_residual_trajectory_and_reset, "synthetic") {
  // v = constant - x has a constant residual. Reusing that residual must
  // exactly reproduce fresh evaluations, including different modality grids.
  auto layout = tiny_layout();
  layout.num_condition_video = 1;
  layout.num_condition_audio = 1;
  layout.condition_audio_is_explicit = true;
  auto indices = slopfab::dit::build_indices(layout);
  // build_indices is the FL2VA packer; place the Ref2VA audio anchor in
  // its modality list, preserving the condition-first ordering.
  indices.video.erase(indices.video.begin() + 1);
  indices.audio.insert(indices.audio.begin(), layout.num_text + 1);
  indices.tags[layout.num_text + 1] = slopfab::dit::kTagAudio;
  std::vector<float> anchor_v(96, 7), anchor_a(32, 9);
  auto run = [&](bool enabled, bool cancel) {
    slopfab::sampler::FlowScheduler video(12), audio(3);
    video.set_timesteps(14); audio.set_timesteps(14);
    Transformer model;
    auto in = make_denoise_inputs(layout, indices, video, audio);
    in.condition_video_rows = &anchor_v;
    in.condition_audio_rows = &anchor_a;
    in.motion_cache.enabled = enabled;
    in.motion_cache.reuse_threshold = 1;
    in.motion_cache.start_percent = 0;
    in.motion_cache.end_percent = 1;
    in.motion_cache.warmup_steps = 2;
    std::vector<uint8_t> invoked(video.num_steps(), 0);
    in.velocity = [&](int step, const RowTimesteps&, const float* v, const float* a,
                      float* vv, float* av) {
      invoked[step] = 1;
      for (size_t i = 0; i < anchor_v.size(); ++i) CHECK(v[i] == 7);
      for (size_t i = 0; i < anchor_a.size(); ++i) CHECK(a[i] == 9);
      for (size_t i = 0; i < indices.video.size() * 96; ++i) vv[i] = 4 - v[i];
      for (size_t i = 0; i < indices.audio.size() * 32; ++i) av[i] = 8 - a[i];
    };
    const auto out = slopfab::dit::denoise(model, in,
        [&](int step, int) { return !cancel || step < 3; });
    invoked.resize(out.decisions.size());
    CHECK(out.decisions == invoked);
    CHECK(out.steps_computed + out.steps_skipped == static_cast<int>(out.decisions.size()));
    if (enabled) CHECK(out.steps_skipped > 0);
    return out;
  };
  const auto fresh = run(false, false);
  const auto cached = run(true, false);
  CHECK_CLOSE(fresh.video_rows, cached.video_rows, 1e-6, "MotionCache video residual sign");
  CHECK_CLOSE(fresh.audio_rows, cached.audio_rows, 1e-6, "MotionCache audio residual sign");
  const auto cancelled = run(true, true);
  CHECK(cancelled.decisions.size() == 4);
  const auto repeated = run(true, false);
  CHECK(cached.decisions == repeated.decisions);
  CHECK(cached.video_rows == repeated.video_rows);
  CHECK(cached.audio_rows == repeated.audio_rows);
  CHECK(cached.decisions.front() == 1 && cached.decisions.back() == 1);
}

SLOPFAB_TEST_CATEGORY(denoise_skips_exactly_the_planned_steps, "synthetic") {
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = slopfab::dit::build_indices(layout);

  // A code function standing in for the AdaLN table: c(t) = (t, 0...). The
  // relative-L1 distance between consecutive signatures is then a known
  // function of the two schedules, and the planner and the loop must derive it
  // from the same timesteps.
  const slopfab::dit::CodeFn code = [](float t) {
    std::array<float, slopfab::dit::AdaLNTable::kRank> c{};
    c[0] = t;
    c[1] = 0.5f * t;
    return c;
  };

  struct Case {
    const char* name;
    float threshold;
    int warmup;
    int skip_every;
  };
  const Case cases[] = {
      {"off", 0.0f, 3, 0},
      {"threshold", 0.30f, 3, 0},
      {"threshold, warmup 6", 0.30f, 6, 0},
      {"skip-every 3", 0.0f, 3, 3},
      {"threshold below any increment", 1e-9f, 2, 0},
      {"threshold above every increment", 1e6f, 2, 0},
  };

  for (const Case& c : cases) {
    slopfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
    video.set_timesteps(16);
    audio.set_timesteps(16);

    Transformer model;  // never used: `velocity` and `code` short-circuit it
    slopfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);
    in.code = code;
    in.cache.threshold = c.threshold;
    in.cache.warmup = c.warmup;
    in.cache.skip_every = c.skip_every;

    std::vector<uint8_t> invoked(video.timesteps().size(), 0);
    in.velocity = [&](int step, const RowTimesteps&, const float*, const float*, float* vv,
                      float* av) {
      invoked[static_cast<size_t>(step)] = 1;
      std::fill(vv, vv + layout.num_video_rows * 96, 0.25f);
      std::fill(av, av + layout.num_audio_rows * 32, -0.5f);
    };

    // The planner, driven from the same two schedules the loop reads.
    std::vector<std::pair<float, float>> schedule;
    for (size_t i = 0; i < video.timesteps().size(); ++i) {
      schedule.emplace_back(video.timesteps()[i], audio.timesteps()[i]);
    }
    const std::vector<uint8_t> planned = slopfab::dit::plan_step_cache(in.cache, schedule, code);

    const slopfab::dit::DenoiseOutputs out = slopfab::dit::denoise(model, in);

    CHECK_MSG(out.decisions == planned, "%s: loop decisions differ from plan_step_cache", c.name);
    CHECK_MSG(invoked == planned, "%s: forward ran on steps the plan did not choose", c.name);
    CHECK_MSG(out.decisions.size() == video.timesteps().size(),
              "%s: %zu decisions for %zu evaluations", c.name, out.decisions.size(),
              video.timesteps().size());

    int computed = 0;
    for (uint8_t v : planned) computed += v;
    CHECK_MSG(out.steps_computed == computed, "%s: steps_computed %d, plan says %d", c.name,
              out.steps_computed, computed);
    CHECK_MSG(out.steps_skipped == static_cast<int>(planned.size()) - computed,
              "%s: steps_skipped disagrees with the plan", c.name);

    // The two guarantees, asserted against what actually ran rather than
    // against the planner that was already checked for them on the host.
    CHECK_MSG(invoked.front() == 1, "%s: step 0 was skipped and has no velocity to reuse", c.name);
    CHECK_MSG(invoked.back() == 1, "%s: the terminal step was skipped", c.name);
  }
}

SLOPFAB_TEST_CATEGORY(denoise_cache_disabled_changes_nothing, "synthetic") {
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = slopfab::dit::build_indices(layout);

  auto run = [&](bool set_inert_flags) {
    slopfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
    video.set_timesteps(14);
    audio.set_timesteps(14);
    Transformer model;
    slopfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);
    if (set_inert_flags) {
      // Explicitly zero, plus a warmup that must not switch anything on by
      // itself. `--cache-warmup` alone is inert and this is where that is
      // enforced end to end rather than at the predicate.
      in.cache.threshold = 0.0f;
      in.cache.skip_every = 0;
      in.cache.warmup = 9;
    }
    in.velocity = [&](int step, const RowTimesteps&, const float* v, const float*, float* vv,
                      float* av) {
      // Velocity that depends on the step and on the current latents, so a
      // reused one would diverge immediately rather than coincidentally match.
      const float s = 0.1f * static_cast<float>(step + 1);
      for (int r = 0; r < layout.num_video_rows * 96; ++r) vv[r] = s * (v[r] + 0.3f);
      std::fill(av, av + layout.num_audio_rows * 32, s);
    };
    return slopfab::dit::denoise(model, in);
  };

  const slopfab::dit::DenoiseOutputs a = run(false);
  const slopfab::dit::DenoiseOutputs b = run(true);

  CHECK(a.steps_skipped == 0);
  CHECK(b.steps_skipped == 0);
  CHECK(a.decisions == b.decisions);
  for (uint8_t d : a.decisions) CHECK(d == 1);
  // Exact equality, not a tolerance: the default path must be the same
  // arithmetic in the same order, not merely close to it.
  CHECK(a.video_rows == b.video_rows);
  CHECK(a.audio_rows == b.audio_rows);
}

SLOPFAB_TEST_CATEGORY(denoise_zero_velocity_is_a_fixed_point, "synthetic") {
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = slopfab::dit::build_indices(layout);
  slopfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
  video.set_timesteps(8);
  audio.set_timesteps(8);

  Transformer model;  // never used: `velocity` short-circuits the forward pass
  slopfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);

  // Capture the starting latents by recording the first call's inputs.
  std::vector<float> first_video, first_audio;
  int calls = 0;
  in.velocity = [&](int step, const RowTimesteps&, const float* v, const float* a, float* vv,
                    float* av) {
    if (step == 0) {
      first_video.assign(v, v + layout.num_video_rows * 96);
      first_audio.assign(a, a + layout.num_audio_rows * 32);
    }
    ++calls;
    std::fill(vv, vv + layout.num_video_rows * 96, 0.0f);
    std::fill(av, av + layout.num_audio_rows * 32, 0.0f);
  };

  const slopfab::dit::DenoiseOutputs out = slopfab::dit::denoise(model, in);
  CHECK(calls == static_cast<int>(video.timesteps().size()));

  // v = 0 makes `denoised` equal x_t, so x_next = ratio*x + (1-ratio)*x = x for
  // every ratio. Any drift means the update is not the one in spec 7.3.
  CHECK_CLOSE(first_video, out.video_rows, 1e-6, "video latents under zero velocity");
  CHECK_CLOSE(first_audio, out.audio_rows, 1e-6, "audio latents under zero velocity");
}

SLOPFAB_TEST_CATEGORY(denoise_pinned_target_audio_is_clean_and_never_stepped, "synthetic") {
  SequenceLayout layout = tiny_layout();
  layout.condition_audio_is_explicit = true;
  const auto idx = slopfab::dit::build_indices(layout);
  slopfab::sampler::FlowScheduler video(3), audio(3);
  video.set_timesteps(4); audio.set_timesteps(4);
  Transformer model;
  auto in = make_denoise_inputs(layout, idx, video, audio);
  const std::vector<float> initial_audio = slopfab::test::make_data(size_t(layout.num_audio_rows) * 32, 78);
  const std::vector<float> initial_video(size_t(layout.num_video_rows) * 96, .5f);
  in.init_audio_rows = &initial_audio;
  in.init_video_rows = &initial_video;
  in.pin_target_audio = true;
  int calls = 0;
  in.velocity = [&](int, const RowTimesteps& rt, const float*, const float* a, float* vv, float* av) {
    ++calls;
    CHECK(std::vector<float>(a, a + initial_audio.size()) == initial_audio);
    for (int row : idx.audio) CHECK(rt.unique[rt.indices[row]] == 1.0f);
    std::fill(vv, vv + initial_video.size(), .25f);
    std::fill(av, av + initial_audio.size(), 1000.0f);
  };
  in.boundary = [&](int, const std::vector<float>&, const std::vector<float>& a) { CHECK(a == initial_audio); };
  const auto result = slopfab::dit::denoise(model, in);
  CHECK(calls == 3);
  CHECK(result.audio_rows == initial_audio);
  CHECK(result.video_rows != initial_video);
  in.init_audio_rows = nullptr;
  bool rejected = false;
  try { slopfab::dit::denoise(model, in); } catch (const std::exception&) { rejected = true; }
  CHECK(rejected);
}

SLOPFAB_TEST_CATEGORY(denoise_accepts_video_only_still_layout, "synthetic") {
  SequenceLayout layout = tiny_layout();
  layout.num_audio_latents = 0;
  layout.num_audio_rows = 0;
  layout.num_latent_frames = 1;
  layout.num_video_rows = layout.rows_per_frame();
  const PackedIndices idx = slopfab::dit::build_indices(layout);

  slopfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
  video.set_timesteps(4);
  audio.set_timesteps(4);

  Transformer model;
  slopfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);
  int calls = 0;
  in.velocity = [&](int, const RowTimesteps& rt, const float*, const float* audio_rows,
                    float* vv, float* audio_velocity) {
    ++calls;
    CHECK(audio_rows != nullptr);
    CHECK(audio_velocity != nullptr);
    CHECK(rt.indices.size() == idx.text.size() + idx.video.size());
    std::fill(vv, vv + layout.num_video_rows * 96, 0.0f);
  };

  const slopfab::dit::DenoiseOutputs out = slopfab::dit::denoise(model, in);
  CHECK(calls == static_cast<int>(video.timesteps().size()));
  CHECK(out.video_rows.size() == static_cast<size_t>(layout.num_video_rows) * 96);
  CHECK(out.audio_rows.empty());
}

SLOPFAB_TEST_CATEGORY(denoise_constant_velocity_matches_cpu_euler, "synthetic") {
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = slopfab::dit::build_indices(layout);
  slopfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
  video.set_timesteps(12);
  audio.set_timesteps(12);

  Transformer model;
  slopfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);

  const float kVideoV = 0.75f;
  const float kAudioV = -0.4f;
  std::vector<float> start_video, start_audio;
  std::vector<float> seen_video_t, seen_audio_t;
  in.velocity = [&](int step, const RowTimesteps& rt, const float* v, const float* a, float* vv,
                    float* av) {
    if (step == 0) {
      start_video.assign(v, v + layout.num_video_rows * 96);
      start_audio.assign(a, a + layout.num_audio_rows * 32);
    }
    // The video timestep is what text rows inherit, so it is the maximum of the
    // unique set exactly when t_v > t_a. Record both to check the schedules are
    // advanced independently.
    seen_video_t.push_back(rt.unique.empty() ? 0.0f : rt.unique.front());
    seen_audio_t.push_back(rt.unique.empty() ? 0.0f : rt.unique.back());
    std::fill(vv, vv + layout.num_video_rows * 96, kVideoV);
    std::fill(av, av + layout.num_audio_rows * 32, kAudioV);
  };

  const slopfab::dit::DenoiseOutputs out = slopfab::dit::denoise(model, in);

  // Independent host integration of the same schedule.
  auto integrate = [](const slopfab::sampler::FlowScheduler& sched, std::vector<float> x, float v) {
    for (size_t i = 0; i + 1 < sched.sigmas().size(); ++i) {
      const float sigma_from_t = 1.0f - sched.timesteps()[i];
      const float ratio = sched.sigmas()[i + 1] / sched.sigmas()[i];
      for (float& e : x) {
        const float denoised = e + sigma_from_t * v;  // a PLUS (spec 7.3)
        e = ratio * e + (1.0f - ratio) * denoised;
      }
    }
    return x;
  };
  CHECK_CLOSE(integrate(video, start_video, kVideoV), out.video_rows, 1e-5, "video Euler trajectory");
  CHECK_CLOSE(integrate(audio, start_audio, kAudioV), out.audio_rows, 1e-5, "audio Euler trajectory");

  // Both shifts map sigma = 1 to itself, so step 0 conditions every row on
  // t = 0 and the unique set collapses to one entry — the one step where t2va
  // does *not* have two distinct timesteps. Every later step must, or the two
  // schedules are not being advanced independently.
  CHECK(seen_video_t.front() == seen_audio_t.front());
  int distinct = 0;
  for (size_t i = 1; i < seen_video_t.size(); ++i) {
    if (seen_video_t[i] != seen_audio_t[i]) ++distinct;
  }
  CHECK_MSG(distinct == static_cast<int>(seen_video_t.size()) - 1,
            "only %d of %zu steps after the first had two distinct timesteps; the video and "
            "audio schedules are not being advanced independently",
            distinct, seen_video_t.size() - 1);
}

SLOPFAB_TEST_CATEGORY(denoise_is_deterministic, "synthetic") {
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = slopfab::dit::build_indices(layout);
  slopfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
  video.set_timesteps(6);
  audio.set_timesteps(6);

  Transformer model;
  auto run = [&](uint64_t seed) {
    slopfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);
    in.seed = seed;
    in.velocity = [&](int, const RowTimesteps&, const float* v, const float* a, float* vv,
                      float* av) {
      // A velocity that depends on the state, so a divergence anywhere in the
      // trajectory propagates rather than cancelling.
      for (int i = 0; i < layout.num_video_rows * 96; ++i) vv[i] = 0.1f * v[i];
      for (int i = 0; i < layout.num_audio_rows * 32; ++i) av[i] = -0.2f * a[i];
    };
    return slopfab::dit::denoise(model, in);
  };

  const slopfab::dit::DenoiseOutputs a = run(11);
  const slopfab::dit::DenoiseOutputs b = run(11);
  const slopfab::dit::DenoiseOutputs c = run(12);
  CHECK_CLOSE(a.video_rows, b.video_rows, 0.0, "same seed, same video latents");
  CHECK_CLOSE(a.audio_rows, b.audio_rows, 0.0, "same seed, same audio latents");
  CHECK(a.video_rows != c.video_rows);
}
