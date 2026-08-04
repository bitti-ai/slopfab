// Token-packing tests.
//
// Nothing here touches the GPU and nothing here is expensive, which is exactly
// why it is worth doing thoroughly: every bug this file can catch is otherwise
// invisible. A patchifier that walks the 2x2 patch in the wrong order, a
// spatial grid built with torch's linspace instead of numpy's, or a row-index
// convention that disagrees between the patchifier and the rotary grid all
// produce correctly-shaped tensors full of plausible numbers, and the only
// symptom is that the video is wrong.
//
// The reference values come from docs/transformer_spec.md sections 1.1, 1.4,
// 2.3 and 7.5, which cite ref/diffusers/modular/packing.py by line.

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "harness.h"
#include "vidfab/dit/packing.h"

namespace {

using namespace vidfab::dit;

// The reference t2va request: 16:9, short edge 768, 10 s at 24 fps.
SequenceLayout reference_layout(int num_text) {
  const int frames = align_num_frames(240);
  SequenceLayout l;
  l.num_text = num_text;
  l.num_condition_video = 0;
  l.num_latent_frames = video_latent_num_frames(frames);
  l.latent_height = 48;
  l.latent_width = 84;
  l.num_audio_latents = audio_latents_for_frames(frames);
  l.num_audio_rows = 2 * l.num_audio_latents;
  l.num_video_rows = l.num_latent_frames * l.rows_per_frame();
  return l;
}

VIDFAB_TEST(packing_geometry) {
  // Canvas: 16:9 resolves to 768x1344, which is exactly the area cap.
  int h = 0;
  int w = 0;
  resolve_canvas_size(16, 9, &h, &w);
  CHECK(h == 768);
  CHECK(w == 1344);

  // Square resolves to the short edge on both axes.
  resolve_canvas_size(1, 1, &h, &w);
  CHECK(h == 768);
  CHECK(w == 768);

  // Portrait is the transpose of landscape.
  int h2 = 0;
  int w2 = 0;
  resolve_canvas_size(9, 16, &h2, &w2);
  CHECK(h2 == 1344);
  CHECK(w2 == 768);

  // Every axis is a multiple of 32 across the supported ratio range.
  for (int num = 1; num <= 4; ++num) {
    resolve_canvas_size(num, 1, &h, &w);
    CHECK(h % 32 == 0 && w % 32 == 0);
    resolve_canvas_size(1, num, &h, &w);
    CHECK(h % 32 == 0 && w % 32 == 0);
  }

  // Outside 1:4 .. 4:1 is rejected rather than clamped.
  CHECK(::vidfab::test::throws([] {
    int a = 0;
    int b = 0;
    resolve_canvas_size(5, 1, &a, &b);
  }));

  // Frame alignment snaps UP to 17k + 5, and an already-aligned count is a
  // fixed point.
  CHECK(align_num_frames(240) == 243);
  CHECK(align_num_frames(124) == 124);
  CHECK(align_num_frames(1) == 5);
  CHECK(align_num_frames(6) == 22);
  for (int n = 1; n < 300; ++n) {
    const int aligned = align_num_frames(n);
    CHECK(aligned >= n);
    CHECK(aligned % 17 == 5);
    CHECK(align_num_frames(aligned) == aligned);
  }

  // 17k + 5 pixel frames -> 5k + 2 latent frames.
  CHECK(video_latent_num_frames(243) == 72);
  CHECK(video_latent_num_frames(124) == 37);
  CHECK(video_latent_num_frames(5) == 2);
  CHECK(::vidfab::test::throws([] { video_latent_num_frames(100); }));

  // 40 audio latents per second at 24 fps.
  CHECK(audio_latents_for_frames(243) == 405);
  CHECK(audio_latents_for_frames(124) == 207);
}

VIDFAB_TEST(packing_worked_example) {
  // docs/transformer_spec.md section 1.1, the reproducible 768p request.
  const int L = 128;
  const SequenceLayout l = reference_layout(L);

  CHECK(l.num_latent_frames == 72);
  CHECK(l.rows_per_frame() == 24 * 42);
  CHECK(l.rows_per_frame() == 1008);
  CHECK(l.num_video_rows == 72576);
  CHECK(l.num_audio_latents == 405);
  CHECK(l.num_audio_rows == 810);
  CHECK(l.total_rows() == 73386 + L);

  CHECK(l.condition_start() == L);
  CHECK(l.audio_start() == L);
  CHECK(l.video_start() == L + 810);

  // The default 124-frame request.
  SequenceLayout d;
  d.num_text = L;
  d.num_latent_frames = video_latent_num_frames(124);
  d.latent_height = 48;
  d.latent_width = 84;
  d.num_audio_latents = audio_latents_for_frames(124);
  d.num_audio_rows = 2 * d.num_audio_latents;
  d.num_video_rows = d.num_latent_frames * d.rows_per_frame();
  CHECK(d.num_latent_frames == 37);
  CHECK(d.num_video_rows == 37296);
  CHECK(d.num_audio_rows == 414);
  CHECK(d.total_rows() == 37710 + L);
}

VIDFAB_TEST(packing_indices_are_a_permutation) {
  const SequenceLayout l = reference_layout(37);
  const PackedIndices idx = build_indices(l);
  const int S = l.total_rows();

  CHECK(static_cast<int>(idx.text.size()) == l.num_text);
  CHECK(static_cast<int>(idx.audio.size()) == l.num_audio_rows);
  CHECK(static_cast<int>(idx.video.size()) == l.num_video_rows + l.num_condition_video);

  // The three index sets are disjoint and their union is [0, S) — the spec
  // relies on the scatter being a permutation, and a port that gets an offset
  // wrong silently drops or doubles rows.
  std::vector<int> seen(static_cast<size_t>(S), 0);
  for (int i : idx.text) ++seen[static_cast<size_t>(i)];
  for (int i : idx.audio) ++seen[static_cast<size_t>(i)];
  for (int i : idx.video) ++seen[static_cast<size_t>(i)];
  const int total_seen = std::accumulate(seen.begin(), seen.end(), 0);
  CHECK(total_seen == S);
  bool exactly_once = true;
  for (int c : seen) exactly_once = exactly_once && (c == 1);
  CHECK(exactly_once);

  // Tags: text rows 1, audio rows 2, video rows 0.
  CHECK(static_cast<int>(idx.tags.size()) == S);
  bool tags_ok = true;
  for (int i = 0; i < l.num_text; ++i) tags_ok = tags_ok && idx.tags[static_cast<size_t>(i)] == kTagText;
  for (int i : idx.audio) tags_ok = tags_ok && idx.tags[static_cast<size_t>(i)] == kTagAudio;
  for (int i : idx.video) tags_ok = tags_ok && idx.tags[static_cast<size_t>(i)] == kTagVideo;
  CHECK(tags_ok);
}

// The patchifier is checked against the index arithmetic written out longhand
// in the spec, not against its own implementation. Values are made unique per
// (c, f, y, x) so that any transposition shows up as a mismatch rather than as
// a coincidence.
VIDFAB_TEST(packing_patchify_index_arithmetic) {
  SequenceLayout l;
  l.num_text = 0;
  l.num_latent_frames = 3;
  l.latent_height = 6;
  l.latent_width = 8;
  l.num_audio_latents = 0;
  l.num_audio_rows = 0;
  l.num_video_rows = l.num_latent_frames * l.rows_per_frame();

  const int C = 24;
  const int F = l.num_latent_frames;
  const int Hl = l.latent_height;
  const int Wl = l.latent_width;
  const size_t n = static_cast<size_t>(C) * F * Hl * Wl;

  std::vector<float> latents(n);
  for (int c = 0; c < C; ++c) {
    for (int f = 0; f < F; ++f) {
      for (int y = 0; y < Hl; ++y) {
        for (int x = 0; x < Wl; ++x) {
          const size_t i = ((static_cast<size_t>(c) * F + f) * Hl + y) * Wl + x;
          latents[i] = static_cast<float>(((c * 16 + f) * 32 + y) * 64 + x);
        }
      }
    }
  }

  std::vector<float> rows(static_cast<size_t>(l.num_video_rows) * 96);
  patchify_video(latents.data(), l, rows.data());

  // row     = (f * (Hl/2) + hh) * (Wl/2) + ww
  // feature = c * 4 + dh * 2 + dw
  // value   = latents[c, f, 2*hh + dh, 2*ww + dw]
  const int half_h = Hl / 2;
  const int half_w = Wl / 2;
  bool ok = true;
  for (int f = 0; f < F && ok; ++f) {
    for (int hh = 0; hh < half_h && ok; ++hh) {
      for (int ww = 0; ww < half_w && ok; ++ww) {
        const size_t row = (static_cast<size_t>(f) * half_h + hh) * half_w + ww;
        for (int c = 0; c < C && ok; ++c) {
          for (int dh = 0; dh < 2 && ok; ++dh) {
            for (int dw = 0; dw < 2 && ok; ++dw) {
              const float want =
                  static_cast<float>(((c * 16 + f) * 32 + (2 * hh + dh)) * 64 + (2 * ww + dw));
              ok = rows[row * 96 + c * 4 + dh * 2 + dw] == want;
            }
          }
        }
      }
    }
  }
  CHECK(ok);

  // Within one row the layout is channel-major over the 2x2 patch, so the four
  // entries of channel 0 are contiguous and channel 1 starts at offset 4. A
  // patch-major layout would interleave them; assert the difference directly.
  CHECK(rows[0 * 96 + 0] == latents[0]);                                 // c=0, dh=0, dw=0
  CHECK(rows[0 * 96 + 1] == latents[1]);                                 // c=0, dh=0, dw=1
  CHECK(rows[0 * 96 + 2] == latents[static_cast<size_t>(Wl)]);           // c=0, dh=1, dw=0
  CHECK(rows[0 * 96 + 4] ==
        latents[static_cast<size_t>(F) * Hl * Wl]);                      // c=1, dh=0, dw=0

  // Round trip.
  std::vector<float> back(n, -1.0f);
  unpatchify_video(rows.data(), l, back.data());
  CHECK(back == latents);
}

VIDFAB_TEST(packing_unpack_audio) {
  const int A = 5;
  const int dim = 32;
  std::vector<float> rows(static_cast<size_t>(2 * A) * dim);
  for (int c = 0; c < 2; ++c) {
    for (int a = 0; a < A; ++a) {
      for (int d = 0; d < dim; ++d) {
        rows[(static_cast<size_t>(c) * A + a) * dim + d] =
            static_cast<float>((c * 100 + a) * 100 + d);
      }
    }
  }

  std::vector<float> out(rows.size());
  unpack_audio(rows.data(), A, out.data());

  // (2A, 32) -> (2, 32, A): channel-major outer, then channel dim, then time.
  bool ok = true;
  for (int c = 0; c < 2 && ok; ++c) {
    for (int d = 0; d < dim && ok; ++d) {
      for (int a = 0; a < A && ok; ++a) {
        const float want = static_cast<float>((c * 100 + a) * 100 + d);
        ok = out[(static_cast<size_t>(c) * dim + d) * A + a] == want;
      }
    }
  }
  CHECK(ok);
}

VIDFAB_TEST(packing_rotary_coordinates) {
  const int L = 11;
  const SequenceLayout l = reference_layout(L);
  const std::vector<double> pos = build_position_ids(l);
  CHECK(pos.size() == static_cast<size_t>(l.total_rows()) * 3);

  // Text rows: (i, 0, 0).
  for (int i = 0; i < L; ++i) {
    CHECK_NEAR(pos[static_cast<size_t>(i) * 3 + 0], static_cast<double>(i), 0.0);
    CHECK_NEAR(pos[static_cast<size_t>(i) * 3 + 1], 0.0, 0.0);
    CHECK_NEAR(pos[static_cast<size_t>(i) * 3 + 2], 0.0, 0.0);
  }

  // Video temporal grid: T(0) = L, then 5/3 * (1, 4, 4, 4, 4) cumulative. Each
  // group of five latent frames advances (5/3) * 17 = 28.333..., i.e. exactly
  // 17 pixel frames.
  const int R = l.rows_per_frame();
  const int vs = l.video_start();
  const double t0 = pos[static_cast<size_t>(vs) * 3];
  CHECK_NEAR(t0, static_cast<double>(L), 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(vs + R) * 3], L + 5.0 / 3.0, 1e-12);
  CHECK_NEAR(pos[static_cast<size_t>(vs + 2 * R) * 3], L + 5.0 / 3.0 + 20.0 / 3.0, 1e-12);
  CHECK_NEAR(pos[static_cast<size_t>(vs + 5 * R) * 3], L + (5.0 / 3.0) * 17.0, 1e-12);
  CHECK_NEAR(pos[static_cast<size_t>(vs + 10 * R) * 3], L + (5.0 / 3.0) * 34.0, 1e-12);

  // Spatial grids for Hl, Wl = 48, 84: aspect-normalised and centred, so the
  // spans are [3.905, 28.095) on h and [-5.166, 37.166) on w. Zero therefore
  // falls inside the w range — there is no reserved band separating text from
  // media on the spatial axes.
  const double sqrt_area = std::sqrt(48.0 * 84.0);
  const double ratio_h = 48.0 / sqrt_area;
  const double left_h = (1.0 - ratio_h) / 2.0;
  const double ratio_w = 84.0 / sqrt_area;
  const double left_w = (1.0 - ratio_w) / 2.0;

  CHECK_NEAR(pos[static_cast<size_t>(vs) * 3 + 1], left_h * 32.0, 1e-12);
  CHECK_NEAR(pos[static_cast<size_t>(vs) * 3 + 2], left_w * 32.0, 1e-12);
  CHECK_NEAR((left_h + ratio_h) * 32.0, 28.0947, 1e-3);
  CHECK_NEAR((left_w + ratio_w) * 32.0, 37.1660, 1e-3);

  // Golden values taken from numpy itself, to 17 significant figures, compared
  // with **zero tolerance**. These pin the exact float64 expression, which is
  // the point: numpy's linspace computes `arange(n) * ((stop - start) / n) +
  // start`, and `(left + ratio) - left` is not `ratio` in float64. Reusing
  // `ratio` puts the width grid one ulp off. A loose tolerance here would let
  // that back in, and the rotary angle error it causes is real — the T
  // coordinate alone reaches a few thousand, where fp32 ulp is 2.4e-4.
  CHECK_NEAR(pos[static_cast<size_t>(vs) * 3 + 1], 3.9051368637047297, 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(vs + 42) * 3 + 1], 4.9130421250626686, 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(vs + 23 * 42) * 3 + 1], 27.08695787493733, 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(vs) * 3 + 2], -5.1660104885167222, 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(vs + 1) * 3 + 2], -4.1581052271587833, 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(vs + 41) * 3 + 2], 36.158105227158771, 0.0);

  // Row r within a frame is hh*(Wl/2) + ww — the same order the patchifier
  // uses. Checking element 1 pins that the fastest-varying axis is width.
  const double step_w = ratio_w / 42.0;
  CHECK_NEAR(pos[static_cast<size_t>(vs + 1) * 3 + 2], (left_w + step_w) * 32.0, 1e-12);
  CHECK_NEAR(pos[static_cast<size_t>(vs + 1) * 3 + 1], left_h * 32.0, 1e-12);
  const double step_h = ratio_h / 24.0;
  CHECK_NEAR(pos[static_cast<size_t>(vs + 42) * 3 + 1], (left_h + step_h) * 32.0, 1e-12);
  CHECK_NEAR(pos[static_cast<size_t>(vs + 42) * 3 + 2], left_w * 32.0, 1e-12);

  // A square canvas spans exactly [0, 32).
  SequenceLayout sq = l;
  sq.latent_height = 48;
  sq.latent_width = 48;
  sq.num_video_rows = sq.num_latent_frames * sq.rows_per_frame();
  const std::vector<double> sq_pos = build_position_ids(sq);
  const int sq_vs = sq.video_start();
  CHECK_NEAR(sq_pos[static_cast<size_t>(sq_vs) * 3 + 1], 0.0, 1e-15);
  CHECK_NEAR(sq_pos[static_cast<size_t>(sq_vs) * 3 + 2], 0.0, 1e-15);
  CHECK_NEAR(sq_pos[static_cast<size_t>(sq_vs + 23) * 3 + 2], 32.0 * 23.0 / 24.0, 1e-12);

  // Audio: t = L + a with the origin at L rather than audio_start, h = 0, and
  // the two channels pinned to opposite extremes of the width grid.
  const int as = l.audio_start();
  const int A = l.num_audio_latents;
  CHECK_NEAR(pos[static_cast<size_t>(as) * 3 + 0], static_cast<double>(L), 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(as + 3) * 3 + 0], static_cast<double>(L + 3), 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(as + A) * 3 + 0], static_cast<double>(L), 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(as) * 3 + 1], 0.0, 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(as + A) * 3 + 1], 0.0, 0.0);
  CHECK_NEAR(pos[static_cast<size_t>(as) * 3 + 2], left_w * 32.0, 1e-12);
  CHECK_NEAR(pos[static_cast<size_t>(as + A) * 3 + 2], (left_w + 41.0 * step_w) * 32.0, 1e-12);
}

VIDFAB_TEST(packing_row_timesteps) {
  const int L = 4;
  const SequenceLayout l = reference_layout(L);
  const PackedIndices idx = build_indices(l);

  // Audio ahead of video: the audio timestep sorts second.
  {
    const RowTimesteps rt = build_row_timesteps(l, idx, 0.25f, 0.75f);
    CHECK(rt.unique.size() == 2);
    CHECK_NEAR(rt.unique[0], 0.25, 0.0);
    CHECK_NEAR(rt.unique[1], 0.75, 0.0);
    CHECK(rt.indices[0] == 0);                                           // text -> video t
    CHECK(rt.indices[static_cast<size_t>(l.audio_start())] == 1);        // audio
    CHECK(rt.indices[static_cast<size_t>(l.video_start())] == 0);        // video

    // adaln = timestep_index * 3 + tag.
    CHECK(rt.adaln[0] == 0 * 3 + kTagText);
    CHECK(rt.adaln[static_cast<size_t>(l.audio_start())] == 1 * 3 + kTagAudio);
    CHECK(rt.adaln[static_cast<size_t>(l.video_start())] == 0 * 3 + kTagVideo);
  }

  // Reversed: the same two schedules cross, and the identity of index 0 flips.
  // This is the behaviour that must be reproduced rather than fixed, because
  // it feeds the AdaLN table row.
  {
    const RowTimesteps rt = build_row_timesteps(l, idx, 0.75f, 0.25f);
    CHECK(rt.unique.size() == 2);
    CHECK_NEAR(rt.unique[0], 0.25, 0.0);
    CHECK(rt.indices[0] == 1);                                           // text -> video t, now index 1
    CHECK(rt.indices[static_cast<size_t>(l.audio_start())] == 0);
    CHECK(rt.adaln[0] == 1 * 3 + kTagText);
    CHECK(rt.adaln[static_cast<size_t>(l.audio_start())] == 0 * 3 + kTagAudio);
  }

  // Equal timesteps collapse to one entry, and every adaln index then depends
  // on the tag alone.
  {
    const RowTimesteps rt = build_row_timesteps(l, idx, 0.5f, 0.5f);
    CHECK(rt.unique.size() == 1);
    CHECK(rt.adaln[0] == kTagText);
    CHECK(rt.adaln[static_cast<size_t>(l.audio_start())] == kTagAudio);
    CHECK(rt.adaln[static_cast<size_t>(l.video_start())] == kTagVideo);
  }

  // Every adaln index is in range for an 18-vector table of 3 modalities.
  const RowTimesteps rt = build_row_timesteps(l, idx, 0.9f, 0.1f);
  bool in_range = true;
  for (int32_t a : rt.adaln) in_range = in_range && a >= 0 && a < 2 * 3;
  CHECK(in_range);
}

}  // namespace
