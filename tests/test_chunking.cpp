// Temporal chunking for the frame-banding quality probe.
//
// The arithmetic here is the silently-wrong kind: a chunk placed half a latent
// frame off, an audio slice taken from the wrong stereo channel, or a
// cross-fade whose weights do not sum to one all produce correctly shaped
// output that a shape check waves through and the eye reads as "the model is
// bad at this". So the tests pin the placement, the tiling, the slice identity
// and the partition of unity directly.

#include <cmath>
#include <string>
#include <vector>

#include "harness.h"
#include "slopfab/dit/chunking.h"
#include "slopfab/pipeline.h"
#include "slopfab/sampler/noise.h"

namespace {

slopfab::GenerateRequest square_request(int frames) {
  slopfab::GenerateRequest r;
  r.prompt = "a test";
  r.aspect_w = 1;
  r.aspect_h = 1;
  r.num_frames = frames;
  r.num_inference_steps = 30;
  return r;
}

// The probe's own geometry, resolved rather than asserted from memory:
// `--frames 15` snaps up to 22 pixel frames and `--frames 45` to 56, and three
// 22-frame chunks at a stride of 5 latent frames compose to exactly the 56.
slopfab::dit::SequenceLayout full_layout() {
  return slopfab::resolve_plan(square_request(45)).layout;
}

slopfab::dit::SequenceLayout chunk_layout() {
  return slopfab::resolve_plan(square_request(15)).layout;
}

bool rejects_stride_not_multiple_of_five() {
  // 12 latent frames of chunk inside 22, two chunks: stride 10 is legal, so
  // build an illegal one instead — 7 inside 15 over 2 chunks is a stride of 8.
  slopfab::dit::SequenceLayout full;
  slopfab::dit::SequenceLayout chunk;
  full.latent_height = chunk.latent_height = 48;
  full.latent_width = chunk.latent_width = 48;
  full.num_latent_frames = 15;
  chunk.num_latent_frames = 7;
  full.num_audio_latents = 60;
  chunk.num_audio_latents = 30;
  try {
    slopfab::dit::resolve_chunk_plan(full, chunk, 2);
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

} // namespace

SLOPFAB_TEST(chunking_probe_geometry_resolves) {
  const slopfab::GeneratePlan full = slopfab::resolve_plan(square_request(45));
  const slopfab::GeneratePlan chunk = slopfab::resolve_plan(square_request(15));

  // `align_num_frames` snaps up to the next 17k+5. The probe's two request
  // sizes were chosen because they land exactly on a tiling, and that is worth
  // pinning: if either snapped elsewhere the chunks would stop composing.
  CHECK(full.aligned_frames == 56);
  CHECK(chunk.aligned_frames == 22);
  CHECK(full.canvas_height == 768 && full.canvas_width == 768);
  CHECK(chunk.canvas_height == 768 && chunk.canvas_width == 768);
  CHECK(full.layout.num_latent_frames == 17);
  CHECK(chunk.layout.num_latent_frames == 7);
  CHECK(full.layout.rows_per_frame() == 576);
  CHECK(full.layout.num_video_rows == 17 * 576);
  CHECK(chunk.layout.num_video_rows == 7 * 576);
  CHECK(full.layout.num_audio_latents == 93);
  CHECK(chunk.layout.num_audio_latents == 37);
}

SLOPFAB_TEST(chunking_plan_tiles_exactly) {
  const slopfab::dit::ChunkPlan p =
      slopfab::dit::resolve_chunk_plan(full_layout(), chunk_layout(), 3);

  CHECK(p.latent_stride == 5);
  CHECK(p.frame_overlap == 2);
  CHECK(p.frame_offset.size() == 3);
  CHECK(p.frame_offset[0] == 0 && p.frame_offset[1] == 5 && p.frame_offset[2] == 10);
  CHECK(p.frame_offset[2] + p.chunk_latent_frames == p.full_latent_frames);

  // 93 - 37 = 56 over two gaps, so 28 apart, and 56 + 37 lands exactly on 93.
  CHECK(p.audio_stride == 28);
  CHECK(p.audio_overlap == 9);
  CHECK(p.audio_offset[0] == 0 && p.audio_offset[1] == 28 && p.audio_offset[2] == 56);

  // The two clocks are not commensurate here: latent frame 10 sits at pixel
  // frame 34, which is 56.67 audio latents, against the offset's 56. Two
  // thirds of an audio latent is 17 ms — under one video frame at 24 fps, and
  // reported rather than rounded away.
  CHECK_NEAR(p.worst_audio_drift_latents, 2.0 / 3.0, 1e-9);

  CHECK(rejects_stride_not_multiple_of_five());
}

SLOPFAB_TEST(chunking_noise_slice_matches_the_full_draw) {
  const slopfab::dit::SequenceLayout full = full_layout();
  const slopfab::dit::SequenceLayout chunk = chunk_layout();
  const slopfab::dit::ChunkPlan plan = slopfab::dit::resolve_chunk_plan(full, chunk, 3);
  const uint64_t seed = 11;

  // The reference: the full field, patchified at full geometry.
  const std::vector<float> field = slopfab::sampler::video_noise(
      seed, full.num_latent_frames, full.latent_height, full.latent_width, 24);
  std::vector<float> full_rows(static_cast<size_t>(full.num_video_rows) * 96);
  slopfab::dit::patchify_video(field.data(), full, full_rows.data());

  const std::vector<float> audio_field =
      slopfab::sampler::audio_noise(seed, full.num_audio_latents, 32);

  const int R = full.rows_per_frame();
  for (int k = 0; k < 3; ++k) {
    std::vector<float> v;
    std::vector<float> a;
    slopfab::dit::slice_chunk_noise(seed, full, chunk, plan, k, &v, &a);
    CHECK(v.size() == static_cast<size_t>(chunk.num_video_rows) * 96);
    CHECK(a.size() == static_cast<size_t>(chunk.num_audio_rows) * 32);

    // Video rows are frame-major, so chunk k's rows must be bit-identical to
    // the full run's rows starting at frame offset * R. Anything else means
    // the slice was taken in the wrong space.
    const size_t begin = static_cast<size_t>(plan.frame_offset[static_cast<size_t>(k)]) * R * 96;
    std::vector<float> expect(full_rows.begin() + static_cast<ptrdiff_t>(begin),
                              full_rows.begin() + static_cast<ptrdiff_t>(begin + v.size()));
    CHECK_CLOSE(expect, v, 0.0, ("video noise slice, chunk " + std::to_string(k)).c_str());

    // Audio is channel-major over the 2A rows: getting this wrong takes both
    // halves from channel 0 and is invisible in every shape.
    const int A = full.num_audio_latents;
    const int Ac = chunk.num_audio_latents;
    const int a0 = plan.audio_offset[static_cast<size_t>(k)];
    std::vector<float> audio_expect;
    for (int c = 0; c < 2; ++c) {
      const size_t off = (static_cast<size_t>(c) * A + a0) * 32;
      audio_expect.insert(audio_expect.end(), audio_field.begin() + static_cast<ptrdiff_t>(off),
                          audio_field.begin() + static_cast<ptrdiff_t>(off + Ac * 32));
    }
    CHECK_CLOSE(audio_expect, a, 0.0, ("audio noise slice, chunk " + std::to_string(k)).c_str());
  }

  // And the slice is genuinely not what a chunk-sized draw would give — if it
  // were, the whole seed policy would be a no-op and the probe would be
  // measuring three independent samples without saying so.
  std::vector<float> v0;
  std::vector<float> a0;
  slopfab::dit::slice_chunk_noise(seed, full, chunk, plan, 0, &v0, &a0);
  const std::vector<float> own = slopfab::sampler::video_noise(
      seed, chunk.num_latent_frames, chunk.latent_height, chunk.latent_width, 24);
  std::vector<float> own_rows(static_cast<size_t>(chunk.num_video_rows) * 96);
  slopfab::dit::patchify_video(own.data(), chunk, own_rows.data());
  size_t differing = 0;
  for (size_t i = 0; i < v0.size(); ++i) {
    if (v0[i] != own_rows[i])
      ++differing;
  }
  CHECK_MSG(differing > v0.size() / 2,
            "a sliced chunk should differ from an independent chunk-sized draw almost everywhere, "
            "but only %zu of %zu elements differ",
            differing, v0.size());
}

SLOPFAB_TEST(chunking_blend_is_a_partition_of_unity) {
  const slopfab::dit::SequenceLayout full = full_layout();
  const slopfab::dit::SequenceLayout chunk = chunk_layout();
  const slopfab::dit::ChunkPlan plan = slopfab::dit::resolve_chunk_plan(full, chunk, 3);

  // Every chunk holds the constant 1. A correct cross-fade must return 1
  // everywhere: any weight that does not sum to one shows up as a dark or
  // bright band exactly at a seam, which is the artefact this probe is trying
  // to *measure* rather than to create.
  const size_t vn = static_cast<size_t>(chunk.num_video_rows) * 96;
  const size_t an = static_cast<size_t>(chunk.num_audio_rows) * 32;
  const std::vector<std::vector<float>> video(3, std::vector<float>(vn, 1.0f));
  const std::vector<std::vector<float>> audio(3, std::vector<float>(an, 1.0f));

  std::vector<float> vout;
  std::vector<float> aout;
  slopfab::dit::blend_chunks(full, chunk, plan, video, audio, &vout, &aout);
  CHECK(vout.size() == static_cast<size_t>(full.num_video_rows) * 96);
  CHECK(aout.size() == static_cast<size_t>(full.num_audio_rows) * 32);
  CHECK_CLOSE(std::vector<float>(vout.size(), 1.0f), vout, 1e-6f, "video partition of unity");
  CHECK_CLOSE(std::vector<float>(aout.size(), 1.0f), aout, 1e-6f, "audio partition of unity");
}

SLOPFAB_TEST(chunking_slice_then_blend_is_the_identity) {
  const slopfab::dit::SequenceLayout full = full_layout();
  const slopfab::dit::SequenceLayout chunk = chunk_layout();
  const slopfab::dit::ChunkPlan plan = slopfab::dit::resolve_chunk_plan(full, chunk, 3);
  const uint64_t seed = 11;

  // Slicing a field into chunks and cross-fading it straight back must return
  // the field, because every cross-fade is then between two copies of the same
  // numbers. This is the end-to-end check on the placement: a chunk written to
  // the wrong offset, an off-by-one in a ramp, or a video/audio offset that
  // disagree all break it, and none of them changes a shape.
  //
  // It is also the only check that couples `slice_chunk_noise` and
  // `blend_chunks` — each is self-consistent on its own, and a consistent pair
  // of mistakes between them is exactly what a probe would misread as the
  // model's fault.
  std::vector<std::vector<float>> video;
  std::vector<std::vector<float>> audio;
  for (int k = 0; k < 3; ++k) {
    std::vector<float> v;
    std::vector<float> a;
    slopfab::dit::slice_chunk_noise(seed, full, chunk, plan, k, &v, &a);
    video.push_back(std::move(v));
    audio.push_back(std::move(a));
  }

  std::vector<float> vout;
  std::vector<float> aout;
  slopfab::dit::blend_chunks(full, chunk, plan, video, audio, &vout, &aout);

  const std::vector<float> field = slopfab::sampler::video_noise(
      seed, full.num_latent_frames, full.latent_height, full.latent_width, 24);
  std::vector<float> expect_video(static_cast<size_t>(full.num_video_rows) * 96);
  slopfab::dit::patchify_video(field.data(), full, expect_video.data());
  const std::vector<float> expect_audio =
      slopfab::sampler::audio_noise(seed, full.num_audio_latents, 32);

  // fp32 only: the blend multiplies by weights that sum to one, so the overlap
  // frames carry a rounding of order 1e-7 against a unit-variance field.
  CHECK_CLOSE(expect_video, vout, 1e-6f, "slice -> blend round trip, video");
  CHECK_CLOSE(expect_audio, aout, 1e-6f, "slice -> blend round trip, audio");
}

SLOPFAB_TEST(chunking_blend_places_each_chunk_where_the_plan_says) {
  const slopfab::dit::SequenceLayout full = full_layout();
  const slopfab::dit::SequenceLayout chunk = chunk_layout();
  const slopfab::dit::ChunkPlan plan = slopfab::dit::resolve_chunk_plan(full, chunk, 3);
  const int R = full.rows_per_frame();

  // Chunk k holds the constant (k + 1). The interior of each chunk must come
  // back as that constant exactly, and the two overlap frames must be the
  // symmetric 3:1 / 1:3 mix the window promises.
  std::vector<std::vector<float>> video;
  std::vector<std::vector<float>> audio;
  for (int k = 0; k < 3; ++k) {
    video.emplace_back(static_cast<size_t>(chunk.num_video_rows) * 96, static_cast<float>(k + 1));
    audio.emplace_back(static_cast<size_t>(chunk.num_audio_rows) * 32, static_cast<float>(k + 1));
  }

  std::vector<float> vout;
  std::vector<float> aout;
  slopfab::dit::blend_chunks(full, chunk, plan, video, audio, &vout, &aout);

  auto frame_value = [&](int f) {
    return vout[static_cast<size_t>(f) * R * 96];
  };
  for (int f = 0; f <= 4; ++f)
    CHECK_NEAR(frame_value(f), 1.0, 1e-6);
  CHECK_NEAR(frame_value(5), 0.75 * 1.0 + 0.25 * 2.0, 1e-6);
  CHECK_NEAR(frame_value(6), 0.25 * 1.0 + 0.75 * 2.0, 1e-6);
  for (int f = 7; f <= 9; ++f)
    CHECK_NEAR(frame_value(f), 2.0, 1e-6);
  CHECK_NEAR(frame_value(10), 0.75 * 2.0 + 0.25 * 3.0, 1e-6);
  CHECK_NEAR(frame_value(11), 0.25 * 2.0 + 0.75 * 3.0, 1e-6);
  for (int f = 12; f <= 16; ++f)
    CHECK_NEAR(frame_value(f), 3.0, 1e-6);

  // Audio: 9 latents of overlap, so the ramp is 0.5/9 .. 8.5/9, and both
  // stereo channels must be blended — the second channel starts at A*32 into
  // the row buffer and a blend that forgot it would leave it as chunk 0's.
  const int A = full.num_audio_latents;
  for (int c = 0; c < 2; ++c) {
    auto latent_value = [&](int a) {
      return aout[(static_cast<size_t>(c) * A + a) * 32];
    };
    CHECK_NEAR(latent_value(0), 1.0, 1e-6);
    CHECK_NEAR(latent_value(27), 1.0, 1e-6);
    CHECK_NEAR(latent_value(28), (8.5 / 9.0) * 1.0 + (0.5 / 9.0) * 2.0, 1e-6);
    CHECK_NEAR(latent_value(36), (0.5 / 9.0) * 1.0 + (8.5 / 9.0) * 2.0, 1e-6);
    CHECK_NEAR(latent_value(45), 2.0, 1e-6);
    CHECK_NEAR(latent_value(56), (8.5 / 9.0) * 2.0 + (0.5 / 9.0) * 3.0, 1e-6);
    CHECK_NEAR(latent_value(92), 3.0, 1e-6);
  }
}
