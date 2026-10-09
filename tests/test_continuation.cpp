#include "harness.h"
#include "latent_fixture.h"
#include "slopfab/continuation.h"
#include "slopfab/pipeline.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {
template <class F> bool rejects(F fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
}

SLOPFAB_TEST(latent_bridge_preservation_and_clocks) {
  LatentFixture fixture;
  fixture.write(107);
  const auto left = slopfab::LatentClip::load(fixture.path.string());
  auto right = std::make_shared<slopfab::LatentClip>(*left);
  for (auto& v : right->video_rows)
    v += 10000;
  for (auto& v : right->audio_rows)
    v += 20000;
  for (int lm : {0, 17, 34})
    for (int rm : {0, 17, 51})
      for (int context : {5, 22, 39})
        for (int gap : {0, 12, 13, 29, 30, 63}) {
          slopfab::LatentBridge b{left, right, lm, rm, context};
          const auto p = slopfab::plan_bridge(b, gap);
          CHECK(p.gap_frames >= gap && p.gap_frames % 17 == 12);
          CHECK(p.output_frames == 214 + p.gap_frames);
          CHECK(p.window_frames == lm + rm + p.gap_frames + 2 * context);
          CHECK(p.left_video_start % 5 == 0 && p.right_video_drop % 5 == 0);
          const size_t frame = 192;
          std::vector<float> video(
              size_t(slopfab::dit::video_latent_num_frames(p.window_frames)) * frame, -7);
          std::vector<float> audio(size_t(p.window_audio_latents) * 64, -8);
          std::fill(audio.begin() + size_t(p.window_audio_latents) * 32, audio.end(), -9.f);
          const auto joined = slopfab::join_bridge(b, p, video, audio);
          CHECK(joined.frames == p.output_frames);
          CHECK(std::equal(left->video_rows.begin(),
                           left->video_rows.begin() + size_t(p.left_video_keep) * frame,
                           joined.video_rows.begin()));
          CHECK(std::equal(right->video_rows.begin() + size_t(p.right_video_drop) * frame,
                           right->video_rows.end(),
                           joined.video_rows.begin() +
                               size_t(p.left_video_keep + p.generated_video_latents) * frame));
          CHECK(joined.video_rows[size_t(p.left_video_keep) * frame] == -7);
          const size_t output_a = size_t((int64_t(joined.frames) * 5 + 1) / 3);
          const size_t source_a = 178; // round(107 * 40 / 24)
          CHECK(joined.audio_rows.size() == output_a * 64);
          for (size_t c = 0; c < 2; ++c) {
            CHECK(std::equal(left->audio_rows.begin() + c * source_a * 32,
                             left->audio_rows.begin() + (c * source_a + p.left_audio_keep) * 32,
                             joined.audio_rows.begin() + c * output_a * 32));
            CHECK(std::equal(right->audio_rows.begin() + (c * source_a + p.right_audio_drop) * 32,
                             right->audio_rows.begin() + (c + 1) * source_a * 32,
                             joined.audio_rows.begin() +
                                 (c * output_a + p.left_audio_keep + p.generated_audio_latents) *
                                     32));
            CHECK(joined.audio_rows[(c * output_a + p.left_audio_keep) * 32] == (c ? -9 : -8));
          }
          auto constraints = slopfab::make_bridge_constraint(b, p);
          for (auto* constraint : {&constraints.video, &constraints.audio}) {
            const size_t width = constraint->target_values_per_channel;
            const size_t kept = constraint->original.size() / constraint->channels;
            const size_t tail = constraint->suffix_values_per_channel;
            const size_t head = kept - tail;
            std::vector<float> initial(width * constraint->channels);
            for (size_t i = 0; i < initial.size(); ++i)
              initial[i] = float(i % 19) / 16;
            constraint->capture_noise(initial.data(), initial.size());
            for (float sigma : {1.f, .5f, 0.f}) {
              std::vector<float> rows(initial.size(), -33);
              constraint->apply(rows.data(), rows.size(), sigma);
              for (size_t c = 0; c < constraint->channels; ++c)
                for (size_t i = 0; i < width; ++i) {
                  if (i >= head && i < width - tail)
                    CHECK(rows[c * width + i] == -33);
                  else {
                    const size_t src = c * kept + (i < head ? i : kept - (width - i));
                    CHECK(rows[c * width + i] ==
                          (1 - sigma) * constraint->original[src] + sigma * initial[c * width + i]);
                  }
                }
            }
          }
        }
}

SLOPFAB_TEST(latent_bridge_guides_and_validation) {
  LatentFixture fixture;
  fixture.write(73);
  const auto clip = slopfab::LatentClip::load(fixture.path.string());
  slopfab::GenerateRequest req;
  req.bridge = slopfab::LatentBridge{clip, clip, 17, 34, 22};
  req.num_frames = 30;
  const auto plan = slopfab::resolve_plan(req);
  CHECK(plan.aligned_frames == 192 && plan.sampling_frames == 141);
  std::vector<slopfab::dit::ReferenceGeometry> refs{
      {slopfab::dit::ReferenceKind::kImage, 1, 2, 4, 0}};
  std::vector<float> video(192, 42), audio;
  slopfab::append_bridge_guides(*req.bridge, plan.bridge, 7, refs, video, audio);
  CHECK(refs.size() == 3 && refs[1].target_aligned && refs[2].target_aligned);
  const auto packed = slopfab::dit::build_ref2va_packed_sequence(
      {1, 1, 1}, refs, plan.layout.num_latent_frames, 2, 4, plan.layout.num_audio_latents);
  CHECK(video.size() == size_t(packed.layout.num_condition_video) * 96);
  CHECK(audio.size() == size_t(packed.layout.num_condition_audio) * 32);
  const auto& indices = packed.indices.video;
  const int cv = packed.layout.num_condition_video;
  for (int side = 0; side < 2; ++side)
    for (int i = 0; i < 14; ++i) {
      const int guide = indices[2 + side * 14 + i];
      const int target_offset = side ? packed.layout.num_video_rows - 14 : 0;
      const int target = indices[cv + target_offset + i];
      for (int axis = 0; axis < 3; ++axis)
        CHECK_NEAR(packed.position_ids[size_t(guide) * 3 + axis],
                   packed.position_ids[size_t(target) * 3 + axis], 1e-10);
    }
  auto b = *req.bridge;
  // Both stereo guides must use the exact target audio clock, including when
  // the right video context begins at a fractional 40 Hz coordinate.
  int guide_offset = 0;
  for (int side = 0; side < 2; ++side) {
    const int count = side ? plan.bridge.right_context_audio : plan.bridge.left_context_audio;
    const int target_offset = side ? plan.layout.num_audio_latents - count : 0;
    for (int c = 0; c < 2; ++c)
      for (int i = 0; i < count; ++i) {
        const int guide = packed.indices.audio[guide_offset + c * count + i];
        const int target =
            packed.indices.audio[packed.layout.num_condition_audio +
                                 c * plan.layout.num_audio_latents + target_offset + i];
        for (int axis = 0; axis < 3; ++axis)
          CHECK_NEAR(packed.position_ids[size_t(guide) * 3 + axis],
                     packed.position_ids[size_t(target) * 3 + axis], 1e-10);
      }
    guide_offset += 2 * count;
  }
  refs.back().target_audio_time_offset = std::numeric_limits<double>::infinity();
  CHECK(rejects([&] {
    slopfab::dit::build_ref2va_packed_sequence({1}, refs, plan.layout.num_latent_frames, 2, 4,
                                               plan.layout.num_audio_latents);
  }));
  for (int invalid : {-1, 1, 22}) {
    b.left_margin_frames = invalid;
    CHECK(rejects([&] {
      slopfab::plan_bridge(b, 12);
    }));
  }
  b = *req.bridge;
  b.right_margin_frames = 68;
  CHECK(rejects([&] {
    slopfab::plan_bridge(b, 12);
  }));
  b = *req.bridge;
  CHECK(rejects([&] {
    slopfab::plan_bridge(b, -1);
  }));
  CHECK(rejects([&] {
    slopfab::plan_bridge(b, std::numeric_limits<int>::max());
  }));
  auto invalid_plan = plan.bridge;
  ++invalid_plan.right_audio_drop;
  CHECK(rejects([&] {
    slopfab::make_bridge_constraint(b, invalid_plan);
  }));
  b.right.reset();
  CHECK(rejects([&] {
    slopfab::plan_bridge(b, 12);
  }));
  req.continuation = clip;
  CHECK(rejects([&] {
    slopfab::resolve_plan(req);
  }));
  req.continuation.reset();
  req.motion_cache.enabled = true;
  CHECK(rejects([&] {
    slopfab::resolve_plan(req);
  }));
}

SLOPFAB_TEST(continuation_locked_overlap_constraints) {
  LatentFixture fixture;
  fixture.write();
  const auto source = slopfab::LatentClip::load(fixture.path.string());
  for (int overlap : {5, 22, 39}) {
    const auto plan = slopfab::plan_continuation(*source, overlap, 34);
    auto constraint = slopfab::make_continuation_constraint(*source, plan);
    const size_t nv = size_t(plan.overlap_video_latents) * 2 * 96;
    CHECK(constraint.video.original ==
          std::vector<float>(source->video_rows.end() - nv, source->video_rows.end()));
    const size_t na = size_t(plan.overlap_audio_latents) * 32;
    const size_t source_channel = size_t(source->layout().num_audio_latents) * 32;
    for (size_t c = 0; c < 2; ++c)
      CHECK(std::equal(constraint.audio.original.begin() + c * na,
                       constraint.audio.original.begin() + (c + 1) * na,
                       source->audio_rows.begin() + (c + 1) * source_channel - na));
    for (auto* prefix : {&constraint.video, &constraint.audio}) {
      const size_t count = prefix->channels * prefix->target_values_per_channel;
      const size_t kept = prefix->original.size() / prefix->channels;
      std::vector<float> initial(count);
      for (size_t i = 0; i < count; ++i)
        initial[i] = float(i % 31) / 16;
      CHECK(rejects([&] {
        prefix->apply(initial.data(), count, .5f);
      })); // noise must be captured
      prefix->capture_noise(initial.data(), count);
      for (float sigma : {1.f, .8f, .3f, 0.f}) {
        std::vector<float> rows(count, -123.f);
        prefix->apply(rows.data(), count, sigma);
        for (size_t c = 0; c < prefix->channels; ++c) {
          for (size_t i = 0; i < prefix->target_values_per_channel; ++i) {
            const size_t index = c * prefix->target_values_per_channel + i;
            const float expected =
                i < kept ? (1 - sigma) * prefix->original[c * kept + i] + sigma * initial[index]
                         : -123.f;
            CHECK(rows[index] == expected);
          }
        }
      }
      for (float sigma : {-1.f, 2.f, std::numeric_limits<float>::quiet_NaN()})
        CHECK(rejects([&] {
          prefix->apply(initial.data(), count, sigma);
        }));
      CHECK(rejects([&] {
        prefix->capture_noise(initial.data(), count - 1);
      }));
    }
  }
  slopfab::GenerateRequest request;
  CHECK(!request.continuation_lock_overlap);
  request.continuation_lock_overlap = true;
  CHECK(rejects([&] {
    slopfab::resolve_plan(request);
  }));
  request.continuation = source;
  request.sampling.audio_steps = 7;
  const auto plan = slopfab::resolve_plan(request);
  CHECK(slopfab::describe_plan(request, plan).find("locked video and audio") != std::string::npos);
  request.skip_every = 2;
  CHECK(rejects([&] {
    slopfab::resolve_plan(request);
  }));
  request.skip_every = 0;
  request.block_cache_span = 1;
  CHECK(rejects([&] {
    slopfab::resolve_plan(request);
  }));
  request.block_cache_span = 0;
  request.motion_cache.enabled = true;
  CHECK(rejects([&] {
    slopfab::resolve_plan(request);
  }));
}

SLOPFAB_TEST(continuation_archive_roundtrip_and_validation) {
  LatentFixture fixture, saved;
  fixture.write();
  auto clip = slopfab::LatentClip::load(fixture.path.string());
  CHECK(clip->frames == 39 && clip->width == 64 && clip->height == 32);
  CHECK(clip->layout().num_latent_frames == 12);
  CHECK(clip->layout().num_audio_latents == 65);
  auto copy = *clip;
  copy.transformer = "D:\\models\\a\"b\n.st";
  copy.save(saved.path.string());
  auto loaded = slopfab::LatentClip::load(saved.path.string());
  CHECK(loaded->transformer == copy.transformer);
  CHECK(loaded->video_rows == clip->video_rows);
  CHECK(loaded->audio_rows == clip->audio_rows);
  // Replace an existing archive only after a complete write.
  copy.save(saved.path.string());
  CHECK(slopfab::LatentClip::load(saved.path.string())->audio_rows == clip->audio_rows);
  auto nonfinite = copy;
  nonfinite.video_rows[0] = std::numeric_limits<float>::quiet_NaN();
  nonfinite.save(saved.path.string());
  CHECK(rejects([&] {
    slopfab::LatentClip::load(saved.path.string());
  }));
  copy.save(saved.path.string());
  std::filesystem::resize_file(saved.path, std::filesystem::file_size(saved.path) - 4);
  CHECK(rejects([&] {
    slopfab::LatentClip::load(saved.path.string());
  }));
  fixture.write(39, true, "future-version");
  CHECK(rejects([&] {
    slopfab::LatentClip::load(fixture.path.string());
  }));
  fixture.write(39, false);
  auto synthetic = slopfab::LatentClip::load(fixture.path.string());
  CHECK(rejects([&] {
    slopfab::plan_continuation(*synthetic, 22, 17);
  }));
  copy.video_rows.pop_back();
  CHECK(rejects([&] {
    copy.save(saved.path.string());
  }));
  CHECK(rejects([&] {
    slopfab::plan_continuation(*clip, 6, 17);
  }));
  CHECK(rejects([&] {
    slopfab::plan_continuation(*clip, 56, 17);
  }));
  CHECK(rejects([&] {
    slopfab::plan_continuation(*clip, 22, 0);
  }));
  CHECK(rejects([&] {
    slopfab::plan_continuation(*clip, 22, std::numeric_limits<int>::max());
  }));
  auto invalid = slopfab::plan_continuation(*clip, 22, 17);
  invalid.overlap_audio_latents = -1;
  CHECK(rejects([&] {
    slopfab::join_continuation(*clip, invalid, {}, {});
  }));
  const auto minimal = slopfab::plan_continuation(*clip, 5, 1);
  CHECK(minimal.extension_frames == 17 && minimal.window_frames == 22 &&
        minimal.overlap_video_latents == 2);
}

SLOPFAB_TEST(continuation_cumulative_audio_and_prefix_preservation) {
  LatentFixture fixture;
  fixture.write();
  auto clip = *slopfab::LatentClip::load(fixture.path.string());
  for (int iteration = 0; iteration < 30; ++iteration) {
    auto plan = slopfab::plan_continuation(clip, 22, 17);
    // Rational 40 Hz / 24 fps boundaries: differences vary with global phase.
    const int old_a = (clip.frames * 5 + 1) / 3;
    const int end_a = ((clip.frames + 17) * 5 + 1) / 3;
    const int start_a = ((clip.frames - 22) * 5 + 1) / 3;
    CHECK(plan.overlap_audio_latents == old_a - start_a);
    CHECK(plan.window_audio_latents == end_a - start_a);
    std::vector<float> video(12 * 2 * 96, -10.0f), audio(size_t(plan.window_audio_latents) * 64);
    std::fill(audio.begin(), audio.begin() + size_t(plan.window_audio_latents) * 32, 111.0f);
    std::fill(audio.begin() + size_t(plan.window_audio_latents) * 32, audio.end(), 222.0f);
    auto next = slopfab::join_continuation(clip, plan, video, audio);
    CHECK(next.frames == clip.frames + 17);
    CHECK(std::equal(clip.video_rows.begin(), clip.video_rows.end(), next.video_rows.begin()));
    CHECK(next.video_rows[clip.video_rows.size()] == -10.0f);
    CHECK(next.layout().num_audio_latents == end_a);
    for (int c = 0; c < 2; ++c) {
      CHECK(std::equal(clip.audio_rows.begin() + size_t(c) * old_a * 32,
                       clip.audio_rows.begin() + size_t(c + 1) * old_a * 32,
                       next.audio_rows.begin() + size_t(c) * end_a * 32));
      CHECK(next.audio_rows[(size_t(c) * end_a + old_a) * 32] == (c ? 222.0f : 111.0f));
    }
    clip = std::move(next);
  }
}

SLOPFAB_TEST(continuation_guide_positions_and_planning) {
  LatentFixture fixture;
  fixture.write();
  auto clip = slopfab::LatentClip::load(fixture.path.string());
  slopfab::GenerateRequest req;
  req.continuation = clip;
  req.num_frames = 18; // rounds to 34 new frames
  auto plan = slopfab::resolve_plan(req);
  CHECK(plan.aligned_frames == 73 && plan.sampling_frames == 56);
  CHECK(plan.canvas_width == 64 && plan.canvas_height == 32);
  CHECK(plan.layout.num_latent_frames == 17);
  std::vector<slopfab::dit::ReferenceGeometry> geometry;
  std::vector<float> video, audio;
  slopfab::append_continuation_guide(*clip, plan.continuation, 7, geometry, video, audio);
  CHECK(video.size() == 7 * 2 * 96);
  CHECK(audio.size() == size_t(plan.continuation.overlap_audio_latents) * 64);
  const auto l = clip->layout();
  CHECK(audio.front() ==
        clip->audio_rows[(l.num_audio_latents - plan.continuation.overlap_audio_latents) * 32]);
  CHECK(audio[size_t(plan.continuation.overlap_audio_latents) * 32] ==
        clip->audio_rows[(2 * l.num_audio_latents - plan.continuation.overlap_audio_latents) * 32]);
  auto packed = slopfab::dit::build_ref2va_packed_sequence({1, 1, 1}, geometry, 17, 2, 4,
                                                           plan.layout.num_audio_latents);
  const auto& idx = packed.indices;
  for (int i = 0; i < geometry[0].video_rows(); ++i)
    for (int d = 0; d < 3; ++d)
      CHECK_NEAR(packed.position_ids[size_t(idx.video[i]) * 3 + d],
                 packed.position_ids[size_t(idx.video[geometry[0].video_rows() + i]) * 3 + d], 0);
  CHECK_NEAR(packed.position_ids[size_t(idx.video[0]) * 3], 3, 0);
  CHECK_NEAR(packed.position_ids[size_t(idx.audio[0]) * 3], 3, 0);
  const auto times = slopfab::dit::build_row_timesteps(packed.layout, idx, .5f, .3f, .999f, 1.0f);
  CHECK_NEAR(times.unique[times.indices[idx.video[0]]], .999f, 0);
  CHECK_NEAR(times.unique[times.indices[idx.audio[0]]], 1, 0);
  // Ordinary references advance the clock; the temporal guide must not.
  geometry.insert(geometry.begin(), {slopfab::dit::ReferenceKind::kImage, 1, 2, 4, 0});
  packed = slopfab::dit::build_ref2va_packed_sequence({1, 1, 1}, geometry, 17, 2, 4,
                                                      plan.layout.num_audio_latents);
  CHECK_NEAR(packed.position_ids[size_t(packed.indices.video[2]) * 3], 4, 0);
  CHECK_NEAR(packed.position_ids[size_t(packed.layout.video_start()) * 3], 4, 0);
  req.canvas_width = 32;
  req.canvas_height = 32;
  CHECK(rejects([&] {
    slopfab::resolve_plan(req);
  }));
  req.canvas_width = 0;
  req.canvas_height = 0;
  req.still_image = true;
  CHECK(rejects([&] {
    slopfab::resolve_plan(req);
  }));
}
