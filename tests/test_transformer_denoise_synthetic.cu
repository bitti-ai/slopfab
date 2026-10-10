#include "detail/transformer_fixture.h"
#include "slopfab/sampler/noise.h"
#include "slopfab/generate.h"

SLOPFAB_TEST_CATEGORY(denoise_locked_continuation_overlap, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::sampler;
  const std::vector<int32_t> tags{dit::kTagText, dit::kTagVideo, dit::kTagText};
  const std::vector<dit::ReferenceGeometry> refs{{dit::ReferenceKind::kVideo, 1, 4, 4, 1, true}};
  const auto packed = dit::build_ref2va_packed_sequence(tags, refs, 2, 4, 4, 3);
  const auto& layout = packed.layout;
  const auto& indices = packed.indices;
  std::vector<float> anchors_v(size_t(layout.num_condition_video) * 96, 13);
  std::vector<float> anchors_a(size_t(layout.num_condition_audio) * 32, 17);
  std::vector<float> initial_v(size_t(layout.num_video_rows) * 96, .25f);
  std::vector<float> initial_a(size_t(layout.num_audio_rows) * 32, -.5f);
  ContinuationConstraint constraint;
  constraint.video.target_values_per_channel = initial_v.size();
  constraint.video.original.assign(96, 2.f);
  constraint.audio.channels = 2;
  constraint.audio.target_values_per_channel = initial_a.size() / 2;
  constraint.audio.original.assign(64, 3.f);
  std::fill(constraint.audio.original.begin() + 32, constraint.audio.original.end(), 5.f);
  // Exercise both ordinary prefix locking and the bridge's two-sided context.
  for (bool suffix : {false, true}) {
    if (suffix) {
      constraint.video.suffix_values_per_channel = 96;
      constraint.video.original.insert(constraint.video.original.end(), 96, 7.f);
      constraint.audio.suffix_values_per_channel = 32;
      constraint.audio.original.assign(128, 0.f);
      for (int c = 0; c < 2; ++c) {
        std::fill_n(constraint.audio.original.begin() + c * 64, 32, c ? 5.f : 3.f);
        std::fill_n(constraint.audio.original.begin() + c * 64 + 32, 32, c ? 11.f : 9.f);
      }
    }
    for (auto kind : {SamplerKind::kEuler, SamplerKind::kAb2, SamplerKind::kRenoise}) {
      for (bool cancel : {false, true}) {
        FlowScheduler video(12), audio(3);
        // A non-unit first sigma also exercises projection before the first call.
        video.set_sigmas({.8f, .4f, 0});
        audio.set_sigmas({.9f, .6f, .2f, 0});
        video.set_sampler(kind);
        audio.set_sampler(kind);
        Transformer model;
        auto in = make_denoise_inputs(layout, indices, video, audio);
        in.continuation = &constraint;
        in.init_video_rows = &initial_v;
        in.init_audio_rows = &initial_a;
        in.condition_video_rows = &anchors_v;
        in.condition_audio_rows = &anchors_a;
        const size_t vi[] = {0, 0, 1, 1}, ai[] = {0, 1, 1, 2};
        const size_t vn[] = {0, 1, 1, 2}, an[] = {1, 1, 2, 3};
        const auto check = [&](const float* v, const float* a, float vs, float as) {
          for (size_t i = 0; i < 96; ++i)
            CHECK(v[i] == (1 - vs) * 2.f + vs * .25f);
          for (size_t c = 0; c < 2; ++c)
            for (size_t i = 0; i < 32; ++i)
              CHECK(a[c * constraint.audio.target_values_per_channel + i] ==
                    (1 - as) * (c ? 5.f : 3.f) + as * -.5f);
          if (suffix) {
            for (size_t i = initial_v.size() - 96; i < initial_v.size(); ++i)
              CHECK(v[i] == (1 - vs) * 7.f + vs * .25f);
            for (size_t c = 0; c < 2; ++c)
              for (size_t i = 0; i < 32; ++i)
                CHECK(a[(c + 1) * constraint.audio.target_values_per_channel - 32 + i] ==
                      (1 - as) * (c ? 11.f : 9.f) + as * -.5f);
          }
        };
        int calls = 0, boundaries = 0;
        in.velocity = [&](int step, const RowTimesteps&, const float* v, const float* a, float* vv,
                          float* av) {
          ++calls;
          CHECK(std::equal(anchors_v.begin(), anchors_v.end(), v));
          CHECK(std::equal(anchors_a.begin(), anchors_a.end(), a));
          check(v + anchors_v.size(), a + anchors_a.size(), video.sigmas()[vi[step]],
                audio.sigmas()[ai[step]]);
          std::fill_n(vv, anchors_v.size() + initial_v.size(), 1.f);
          std::fill_n(av, anchors_a.size() + initial_a.size(), 1.f);
        };
        in.boundary = [&](int step, const std::vector<float>& v, const std::vector<float>& a) {
          ++boundaries;
          check(v.data(), a.data(), video.sigmas()[vn[step]], audio.sigmas()[an[step]]);
          if (step == 0)
            CHECK(v[96] == initial_v[96]); // video did not advance on the audio-only step
        };
        const auto output = dit::denoise(model, in, [&](int, int total) {
          CHECK(total == 4);
          return !cancel;
        });
        CHECK(calls == (cancel ? 1 : 4) && boundaries == calls);
        CHECK(output.steps_computed == calls && output.steps_skipped == 0);
        CHECK(constraint.video.noise.empty() && constraint.audio.noise.empty()); // reusable config
        if (!cancel) {
          check(output.video_rows.data(), output.audio_rows.data(), 0, 0);
          CHECK(output.video_rows[96] != initial_v[96]); // suffix remains generated
        }
      }
    }
  }
}

SLOPFAB_TEST_CATEGORY(denoise_independent_audio_steps, "synthetic") {
  using namespace slopfab::sampler;
  const auto layout = tiny_layout();
  const auto indices = slopfab::dit::build_indices(layout);
  const std::vector<float> initial_video(size_t(layout.num_video_rows) * 96, .25f);
  const std::vector<float> initial_audio(size_t(layout.num_audio_rows) * 32, -.5f);
  // Explicit oracle for two video and three audio updates. Also swap them to
  // exercise fewer audio updates, and test every CUDA integrator.
  for (bool swap : {false, true}) {
    for (auto kind : {SamplerKind::kEuler, SamplerKind::kAb2, SamplerKind::kRenoise}) {
      FlowScheduler video(12), audio(3);
      video.set_timesteps(swap ? 4 : 3);
      audio.set_timesteps(swap ? 3 : 4);
      video.set_sampler(kind);
      audio.set_sampler(kind);
      auto expected_video_scheduler = video, expected_audio_scheduler = audio;
      auto expected_video = initial_video, expected_audio = initial_audio;
      Transformer model;
      auto in = make_denoise_inputs(layout, indices, video, audio);
      in.init_video_rows = &initial_video;
      in.init_audio_rows = &initial_audio;
      const int short_indices[] = {0, 0, 1, 1};
      const int long_indices[] = {0, 1, 1, 2};
      const bool advance_short[] = {false, true, false, true};
      const bool advance_long[] = {true, false, true, true};
      int calls = 0, boundaries = 0, progress_calls = 0;
      in.velocity = [&](int step, const RowTimesteps& row, const float* v, const float* a,
                        float* vv, float* av) {
        ++calls;
        const int vi = swap ? long_indices[step] : short_indices[step];
        const int ai = swap ? short_indices[step] : long_indices[step];
        const auto expected_times = slopfab::dit::build_row_timesteps(
            layout, indices, video.timesteps()[vi], audio.timesteps()[ai]);
        CHECK(row.unique == expected_times.unique && row.indices == expected_times.indices);
        CHECK(std::equal(expected_video.begin(), expected_video.end(), v));
        CHECK(std::equal(expected_audio.begin(), expected_audio.end(), a));
        // Coupled predictions depend on both current latents and the call.
        const float v_velocity = .1f * a[0] + .01f * float(step + 1);
        const float a_velocity = .2f * v[0] - .02f * float(step + 1);
        std::fill(vv, vv + initial_video.size(), v_velocity);
        std::fill(av, av + initial_audio.size(), a_velocity);
        const auto advance = [&](FlowScheduler& scheduler, int index, std::vector<float>& state,
                                 float velocity, NoiseStream stream) {
          std::vector<float> velocities(state.size(), velocity), noise(state.size());
          if (kind == SamplerKind::kRenoise && size_t(index + 1) < scheduler.num_steps())
            fill_renoise_normal(in.seed, index, stream, noise.data(), noise.size());
          scheduler.step(index, state.data(), velocities.data(), state.size(), state.data(),
                         noise.data());
        };
        if (swap ? advance_long[step] : advance_short[step])
          advance(expected_video_scheduler, vi, expected_video, v_velocity,
                  NoiseStream::kVideoLatents);
        if (swap ? advance_short[step] : advance_long[step])
          advance(expected_audio_scheduler, ai, expected_audio, a_velocity,
                  NoiseStream::kAudioLatents);
      };
      in.boundary = [&](int, const std::vector<float>& v, const std::vector<float>& a) {
        ++boundaries;
        CHECK(v == expected_video && a == expected_audio);
      };
      const auto result = slopfab::dit::denoise(model, in, [&](int step, int total) {
        CHECK(total == 4 && step == progress_calls++);
        return true;
      });
      CHECK(calls == 4 && boundaries == 4 && progress_calls == 4);
      CHECK(result.steps_computed == 4 && result.steps_skipped == 0);
      CHECK(result.video_rows == expected_video && result.audio_rows == expected_audio);
      // Restart and stop at the first audio/video-only boundary.
      expected_video = initial_video;
      expected_audio = initial_audio;
      calls = boundaries = 0;
      const auto cancelled = slopfab::dit::denoise(model, in, [](int, int total) {
        CHECK(total == 4);
        return false;
      });
      CHECK(calls == 1 && boundaries == 1 && cancelled.steps_computed == 1);
      CHECK(cancelled.video_rows == expected_video && cancelled.audio_rows == expected_audio);
      CHECK(swap ? cancelled.audio_rows == initial_audio : cancelled.video_rows == initial_video);
    }
  }
}

SLOPFAB_TEST_CATEGORY(denoise_inpaint_constrains_each_boundary_and_keeps_anchors, "synthetic") {
  auto layout = tiny_layout();
  layout.num_latent_frames = 1;
  layout.num_video_rows = layout.rows_per_frame();
  layout.num_audio_rows = layout.num_audio_latents = 0;
  layout.num_condition_video = 1;
  const auto indices = slopfab::dit::build_indices(layout);
  slopfab::sampler::FlowScheduler video, audio;
  video.set_sigmas({.75f, .5f, .25f, 0});
  audio.set_sigmas({.75f, .5f, .25f, 0});
  const size_t count = size_t(layout.num_video_rows) * 96;
  slopfab::InpaintConstraint constraint;
  constraint.original.assign(count, 2);
  constraint.noise.assign(count, 10);
  constraint.mask.assign(count, 0);
  constraint.mask[1] = 1;
  std::vector<float> anchors(96, 123);
  Transformer model;
  auto in = make_denoise_inputs(layout, indices, video, audio);
  in.inpaint = &constraint;
  in.condition_video_rows = &anchors;
  int calls = 0, boundaries = 0;
  in.velocity = [&](int step, const RowTimesteps&, const float* v, const float*, float* vv,
                    float*) {
    ++calls;
    CHECK(v[0] == 123);
    CHECK(v[96] == 2 + 8 * video.sigmas()[step]);
    CHECK(v[97] == 8); // zero velocity leaves the editable cell at its initial value
    std::fill(vv, vv + count + 96, 0.0f);
  };
  in.boundary = [&](int step, const std::vector<float>& v, const std::vector<float>& a) {
    ++boundaries;
    CHECK(v[0] == 2 + 8 * video.sigmas()[step + 1]);
    CHECK(v[1] == 8);
    CHECK(a.empty());
  };
  auto result = slopfab::dit::denoise(model, in);
  CHECK(calls == 3 && boundaries == 3);
  CHECK(result.video_rows[0] == 2 && result.video_rows[1] == 8);
  video.reset();
  audio.reset();
  calls = boundaries = 0;
  result = slopfab::dit::denoise(model, in, [](int, int) {
    return false;
  });
  CHECK(calls == 1 && boundaries == 1);
  CHECK(result.video_rows[0] == 6);
}

SLOPFAB_TEST_CATEGORY(denoise_renoise_inpaint_and_outpaint_trajectories, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::sampler;
  auto image = std::make_shared<RGBImage>();
  image->width = image->height = 64;
  image->pixels.resize(64 * 64 * 3, 123);
  for (bool invert : {false, true}) {
    for (float strength : {1.f, .5f, .01f}) {
      GenerateRequest request;
      request.still_image = true;
      request.schedule = ScheduleKind::kDmad4Step;
      request.image_edit = {image, 17, 17, 32, 32, strength, 0, invert};
      const auto plan = resolve_plan(request);
      auto layout = plan.layout;
      layout.num_condition_video = 1;
      const auto indices = dit::build_indices(layout);
      FlowScheduler video(12), audio(2);
      video.set_sigmas(plan.video_sigmas);
      audio.set_sigmas(plan.audio_sigmas);
      video.set_sampler(SamplerKind::kRenoise);
      audio.set_sampler(SamplerKind::kRenoise);
      InpaintConstraint constraint;
      constraint.mask = edit_mask_rows(request.image_edit, 64, 64);
      const size_t count = constraint.mask.size();
      constraint.original.assign(count, 2);
      constraint.noise.assign(count, 10);
      std::vector<float> anchors(96, 123);
      Transformer model;
      auto in = make_denoise_inputs(layout, indices, video, audio);
      in.seed = 42;
      in.inpaint = &constraint;
      in.condition_video_rows = &anchors;
      auto expected = constraint.initial(video.sigmas().front());
      int boundaries = 0;
      in.velocity = [&](int, const RowTimesteps&, const float* v, const float*, float* vv, float*) {
        CHECK(std::equal(anchors.begin(), anchors.end(), v));
        CHECK_CLOSE(expected, std::vector<float>(v + 96, v + 96 + count), 1e-6,
                    "preserved rows reach the next model evaluation");
        std::fill(vv, vv + 96 + count, .25f);
      };
      in.boundary = [&](int step, const std::vector<float>& v, const std::vector<float>& a) {
        ++boundaries;
        std::vector<float> noise(count);
        fill_renoise_normal(in.seed, step, NoiseStream::kVideoLatents, noise.data(), count);
        const float sigma = video.sigmas()[step], next = video.sigmas()[step + 1];
        for (size_t i = 0; i < count; ++i) {
          const float clean = constraint.mask[i] ? expected[i] + sigma * .25f : 2.f;
          expected[i] = (1 - next) * clean + next * noise[i];
        }
        CHECK_CLOSE(expected, v, 1e-6, "DMAD edit boundary");
        CHECK(a.empty());
      };
      const auto result = dit::denoise(model, in);
      CHECK(boundaries == int(video.num_steps()));
      expected = constraint.initial(video.sigmas().front());
      const auto repeated = dit::denoise(model, in);
      CHECK(result.video_rows == repeated.video_rows);
      expected = constraint.initial(video.sigmas().front());
      boundaries = 0;
      const auto cancelled = dit::denoise(model, in, [](int, int) { return false; });
      CHECK(boundaries == 1);
      CHECK_CLOSE(expected, cancelled.video_rows, 1e-6, "cancelled DMAD edit boundary");
    }
  }
}

SLOPFAB_TEST_CATEGORY(denoise_outpaint_langevin_preserves_anchors_and_counts_calls, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::sampler;
  dit::SequenceLayout layout;
  layout.num_text = 1;
  layout.num_latent_frames = 1;
  layout.latent_width = layout.latent_height = 4;
  layout.num_video_rows = 4;
  layout.num_condition_video = 1;
  const auto indices = dit::build_indices(layout);
  std::vector<float> anchors(96, 123);
  InpaintConstraint constraint;
  constraint.original.assign(384, .2f);
  constraint.noise.assign(384, .8f);
  constraint.mask.assign(384, 1);
  std::fill_n(constraint.mask.begin(), 96, 0);
  for (auto kind : {SamplerKind::kEuler, SamplerKind::kRenoise}) {
    for (int inner : {0, 1, 3}) {
      constraint.langevin_steps = inner;
      FlowScheduler video(12), audio(3);
      video.set_sigmas({.75f, .4f, 0});
      audio.set_sigmas({.75f, .4f, 0});
      video.set_sampler(kind);
      audio.set_sampler(kind);
      Transformer model;
      auto in = make_denoise_inputs(layout, indices, video, audio);
      in.seed = 42;
      in.inpaint = &constraint;
      in.condition_video_rows = &anchors;
      int calls = 0, boundaries = 0;
      in.velocity = [&](int, const RowTimesteps&, const float* v, const float*, float* vv, float*) {
        ++calls;
        CHECK(std::equal(anchors.begin(), anchors.end(), v));
        std::fill_n(vv, anchors.size() + constraint.mask.size(), .25f);
      };
      in.boundary = [&](int step, const std::vector<float>& v, const std::vector<float>& a) {
        ++boundaries;
        CHECK(a.empty());
        std::vector<float> noise(384);
        fill_renoise_normal(in.seed, step, NoiseStream::kVideoLatents, noise.data(), noise.size());
        const float sigma = video.sigmas()[step + 1];
        for (size_t j = 0; j < v.size(); ++j) {
          CHECK(std::isfinite(v[j]));
          if (constraint.mask[j] == 0)
            CHECK_NEAR(v[j], (1 - sigma) * .2f + sigma * (kind == SamplerKind::kRenoise ? noise[j] : .8f), 1e-6);
        }
      };
      const auto result = dit::denoise(model, in);
      CHECK(boundaries == 2 && calls == 2 * (inner + 1));
      CHECK(result.steps_computed == calls && result.decisions.size() == 2);
      calls = 0;
      const auto repeated = dit::denoise(model, in);
      CHECK(result.video_rows == repeated.video_rows);
      calls = boundaries = 0;
      const auto cancelled = dit::denoise(model, in, [](int, int) { return false; });
      CHECK(calls == 1 && cancelled.steps_computed == 1);
      CHECK(boundaries == (inner ? 0 : 1));
    }
  }
}

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
    video.set_timesteps(14);
    audio.set_timesteps(14);
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
    in.velocity = [&](int step, const RowTimesteps&, const float* v, const float* a, float* vv,
                      float* av) {
      invoked[step] = 1;
      for (size_t i = 0; i < anchor_v.size(); ++i)
        CHECK(v[i] == 7);
      for (size_t i = 0; i < anchor_a.size(); ++i)
        CHECK(a[i] == 9);
      for (size_t i = 0; i < indices.video.size() * 96; ++i)
        vv[i] = 4 - v[i];
      for (size_t i = 0; i < indices.audio.size() * 32; ++i)
        av[i] = 8 - a[i];
    };
    const auto out = slopfab::dit::denoise(model, in, [&](int step, int) {
      return !cancel || step < 3;
    });
    invoked.resize(out.decisions.size());
    CHECK(out.decisions == invoked);
    CHECK(out.steps_computed + out.steps_skipped == static_cast<int>(out.decisions.size()));
    if (enabled)
      CHECK(out.steps_skipped > 0);
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

    Transformer model; // never used: `velocity` and `code` short-circuit it
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
    for (uint8_t v : planned)
      computed += v;
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
      for (int r = 0; r < layout.num_video_rows * 96; ++r)
        vv[r] = s * (v[r] + 0.3f);
      std::fill(av, av + layout.num_audio_rows * 32, s);
    };
    return slopfab::dit::denoise(model, in);
  };

  const slopfab::dit::DenoiseOutputs a = run(false);
  const slopfab::dit::DenoiseOutputs b = run(true);

  CHECK(a.steps_skipped == 0);
  CHECK(b.steps_skipped == 0);
  CHECK(a.decisions == b.decisions);
  for (uint8_t d : a.decisions)
    CHECK(d == 1);
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

  Transformer model; // never used: `velocity` short-circuits the forward pass
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
  video.set_timesteps(4);
  audio.set_timesteps(4);
  Transformer model;
  auto in = make_denoise_inputs(layout, idx, video, audio);
  const std::vector<float> initial_audio =
      slopfab::test::make_data(size_t(layout.num_audio_rows) * 32, 78);
  const std::vector<float> initial_video(size_t(layout.num_video_rows) * 96, .5f);
  in.init_audio_rows = &initial_audio;
  in.init_video_rows = &initial_video;
  in.pin_target_audio = true;
  int calls = 0;
  in.velocity = [&](int, const RowTimesteps& rt, const float*, const float* a, float* vv,
                    float* av) {
    ++calls;
    CHECK(std::vector<float>(a, a + initial_audio.size()) == initial_audio);
    for (int row : idx.audio)
      CHECK(rt.unique[rt.indices[row]] == 1.0f);
    std::fill(vv, vv + initial_video.size(), .25f);
    std::fill(av, av + initial_audio.size(), 1000.0f);
  };
  in.boundary = [&](int, const std::vector<float>&, const std::vector<float>& a) {
    CHECK(a == initial_audio);
  };
  const auto result = slopfab::dit::denoise(model, in);
  CHECK(calls == 3);
  CHECK(result.audio_rows == initial_audio);
  CHECK(result.video_rows != initial_video);
  in.init_audio_rows = nullptr;
  bool rejected = false;
  try {
    slopfab::dit::denoise(model, in);
  } catch (const std::exception&) {
    rejected = true;
  }
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
  in.velocity = [&](int, const RowTimesteps& rt, const float*, const float* audio_rows, float* vv,
                    float* audio_velocity) {
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
        const float denoised = e + sigma_from_t * v; // a PLUS (spec 7.3)
        e = ratio * e + (1.0f - ratio) * denoised;
      }
    }
    return x;
  };
  CHECK_CLOSE(integrate(video, start_video, kVideoV), out.video_rows, 1e-5,
              "video Euler trajectory");
  CHECK_CLOSE(integrate(audio, start_audio, kAudioV), out.audio_rows, 1e-5,
              "audio Euler trajectory");

  // Both shifts map sigma = 1 to itself, so step 0 conditions every row on
  // t = 0 and the unique set collapses to one entry — the one step where t2va
  // does *not* have two distinct timesteps. Every later step must, or the two
  // schedules are not being advanced independently.
  CHECK(seen_video_t.front() == seen_audio_t.front());
  int distinct = 0;
  for (size_t i = 1; i < seen_video_t.size(); ++i) {
    if (seen_video_t[i] != seen_audio_t[i])
      ++distinct;
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
      for (int i = 0; i < layout.num_video_rows * 96; ++i)
        vv[i] = 0.1f * v[i];
      for (int i = 0; i < layout.num_audio_rows * 32; ++i)
        av[i] = -0.2f * a[i];
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

SLOPFAB_TEST_CATEGORY(denoise_dmad_video_audio_trajectory_and_repeatability, "synthetic") {
  using namespace slopfab;
  auto layout = tiny_layout();
  const auto indices = dit::build_indices(layout);
  sampler::FlowScheduler video(12), audio(2);
  video.set_timesteps(5);
  audio.set_timesteps(5);
  video.set_sampler(sampler::SamplerKind::kRenoise);
  audio.set_sampler(sampler::SamplerKind::kRenoise);
  Transformer model;
  auto inputs = make_denoise_inputs(layout, indices, video, audio);
  inputs.seed = 42;
  const size_t nv = indices.video.size() * 96, na = indices.audio.size() * 32;
  std::vector<float> initial_v(nv, .25f), initial_a(na, -.5f);
  inputs.init_video_rows = &initial_v;
  inputs.init_audio_rows = &initial_a;
  inputs.velocity = [&](int, const RowTimesteps&, const float*, const float*, float* v, float* a) {
    std::fill(v, v + nv, 2.0f);
    std::fill(a, a + na, -1.0f);
  };
  auto expected_v = initial_v, expected_a = initial_a;
  inputs.boundary = [&](int step, const std::vector<float>& v, const std::vector<float>& a) {
    const auto reference = [&](std::vector<float>& x, const sampler::FlowScheduler& scheduler,
                               sampler::NoiseStream modality, float velocity) {
      std::vector<float> noise(x.size());
      sampler::fill_renoise_normal(42, step, modality, noise.data(), noise.size());
      const float sigma = scheduler.sigmas()[step], next = scheduler.sigmas()[step + 1];
      for (size_t i = 0; i < x.size(); ++i)
        x[i] = (1 - next) * (x[i] + sigma * velocity) + next * noise[i];
    };
    reference(expected_v, video, sampler::NoiseStream::kVideoLatents, 2);
    reference(expected_a, audio, sampler::NoiseStream::kAudioLatents, -1);
    CHECK_CLOSE(expected_v, v, 1e-6, "DMAD video boundary");
    CHECK_CLOSE(expected_a, a, 1e-6, "DMAD audio boundary");
  };
  const auto first = dit::denoise(model, inputs);
  CHECK(first.steps_computed == 4 && first.steps_skipped == 0);
  inputs.boundary = {};
  const auto repeated = dit::denoise(model, inputs);
  CHECK(first.video_rows == repeated.video_rows && first.audio_rows == repeated.audio_rows);
  inputs.seed = 43;
  const auto other = dit::denoise(model, inputs);
  CHECK(first.video_rows != other.video_rows && first.audio_rows != other.audio_rows);
}
