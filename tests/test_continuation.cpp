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
