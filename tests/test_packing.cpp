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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "harness.h"
#include "slopfab/dit/packing.h"
#include "slopfab/dit/rope.h"
#include "slopfab/sampler/scheduler.h"

namespace {

using namespace slopfab::dit;

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

SLOPFAB_TEST(packing_geometry) {
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
  CHECK(::slopfab::test::throws([] {
    int a = 0;
    int b = 0;
    resolve_canvas_size(5, 1, &a, &b);
  }));

  // An explicit canvas is checked, not derived. The load-bearing property is
  // that naming the default resolution and asking for 16:9 land on the same
  // canvas — otherwise adding the flag would have moved the default.
  int ah = 0;
  int aw = 0;
  resolve_canvas_size(16, 9, &ah, &aw);
  validate_canvas_size(ah, aw);
  CHECK(aw == 1344 && ah == 768);

  // Multiple of 32 on both axes. 16 is the VAE's compression factor and would
  // look plausible; it is rejected because the 2x2 patch grid then loses its
  // last row or column silently.
  validate_canvas_size(768, 1344);
  validate_canvas_size(512, 512);
  CHECK(::slopfab::test::throws([] { validate_canvas_size(768, 1360); }));  // 1360 = 16*85
  CHECK(::slopfab::test::throws([] { validate_canvas_size(784, 1344); }));  // 784  = 16*49
  CHECK(::slopfab::test::throws([] { validate_canvas_size(0, 1344); }));
  CHECK(::slopfab::test::throws([] { validate_canvas_size(768, -32); }));

  // The same 1:4..4:1 range the aspect path enforces, applied to the canvas
  // the caller named rather than to a ratio they asked for.
  validate_canvas_size(768, 3072);                                          // exactly 4:1
  validate_canvas_size(3072, 768);                                          // exactly 1:4
  CHECK(::slopfab::test::throws([] { validate_canvas_size(768, 3104); }));   // just over 4:1
  CHECK(::slopfab::test::throws([] { validate_canvas_size(3104, 768); }));   // just over 1:4

  // The area cap is deliberately NOT enforced here — that is the difference
  // between the two entry points, and a test that accepted an over-budget
  // canvas by accident would look identical to one that meant to.
  validate_canvas_size(1088, 1920);
  CHECK(canvas_exceeds_trained_area(1088, 1920));
  CHECK(!canvas_exceeds_trained_area(768, 1344));   // exactly the budget is not over it
  CHECK(!canvas_exceeds_trained_area(768, 768));

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
  CHECK(::slopfab::test::throws([] { video_latent_num_frames(100); }));

  // 40 audio latents per second at 24 fps.
  CHECK(audio_latents_for_frames(243) == 405);
  CHECK(audio_latents_for_frames(124) == 207);
}

SLOPFAB_TEST(packing_worked_example) {
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

SLOPFAB_TEST(packing_indices_are_a_permutation) {
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
SLOPFAB_TEST(packing_patchify_index_arithmetic) {
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

SLOPFAB_TEST(packing_unpack_audio) {
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

SLOPFAB_TEST(packing_rotary_coordinates) {
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

SLOPFAB_TEST(packing_row_timesteps) {
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

// ---------------------------------------------------------------------------
// build_row_timesteps: the sort-free rewrite against the sort it replaced.
// ---------------------------------------------------------------------------
//
// `build_row_timesteps` used to materialise an S-element `row_t`, sort it, and
// `std::unique` it, which at S = 37 710 is an O(S log S) way to compute a
// two-valued function. The rewrite is a min/max of two scalars and a single
// O(S) pass. That is only safe if it is *byte for byte* the same answer, so
// the old body is kept here verbatim as the reference and the two are compared
// exactly — not within a tolerance — over every shape the schedule can reach.

// The implementation as it stood at 2480c82, unchanged. Do not "improve" it:
// its value is that it is the thing being replaced.
RowTimesteps row_timesteps_by_sort(const SequenceLayout& layout, const PackedIndices& idx,
                                   float video_t, float audio_t) {
  const int total = layout.total_rows();

  std::vector<float> row_t(static_cast<size_t>(total), video_t);
  for (size_t i = static_cast<size_t>(layout.num_condition_video); i < idx.audio.size(); ++i) {
    row_t[static_cast<size_t>(idx.audio[i])] = audio_t;
  }

  RowTimesteps out;
  out.unique = row_t;
  std::sort(out.unique.begin(), out.unique.end());
  out.unique.erase(std::unique(out.unique.begin(), out.unique.end()), out.unique.end());

  out.indices.resize(static_cast<size_t>(total));
  out.adaln.resize(static_cast<size_t>(total));
  for (int s = 0; s < total; ++s) {
    const auto it =
        std::lower_bound(out.unique.begin(), out.unique.end(), row_t[static_cast<size_t>(s)]);
    const int32_t ti = static_cast<int32_t>(it - out.unique.begin());
    out.indices[static_cast<size_t>(s)] = ti;
    out.adaln[static_cast<size_t>(s)] = ti * 3 + std::max(idx.tags[static_cast<size_t>(s)], 0);
  }
  return out;
}

// Bitwise float equality, not `==`: `==` would call +0.0 and -0.0 identical,
// and the point of this comparison is that nothing at all moved.
bool same_bits(float a, float b) {
  uint32_t ua = 0;
  uint32_t ub = 0;
  std::memcpy(&ua, &a, sizeof(ua));
  std::memcpy(&ub, &b, sizeof(ub));
  return ua == ub;
}

bool identical(const RowTimesteps& a, const RowTimesteps& b) {
  if (a.unique.size() != b.unique.size()) return false;
  if (a.indices.size() != b.indices.size()) return false;
  if (a.adaln.size() != b.adaln.size()) return false;
  for (size_t i = 0; i < a.unique.size(); ++i) {
    if (!same_bits(a.unique[i], b.unique[i])) return false;
  }
  for (size_t i = 0; i < a.indices.size(); ++i) {
    if (a.indices[i] != b.indices[i]) return false;
  }
  for (size_t i = 0; i < a.adaln.size(); ++i) {
    if (a.adaln[i] != b.adaln[i]) return false;
  }
  return true;
}

// A small t2va-shaped layout. Kept small so the cross-product of layouts and
// timestep pairs stays instant; the shape of the answer does not depend on S,
// and the real 73 390-row layout is exercised separately below.
SequenceLayout small_layout(int num_text, int num_audio_latents, int num_latent_frames) {
  SequenceLayout l;
  l.num_text = num_text;
  l.num_condition_video = 0;
  l.num_latent_frames = num_latent_frames;
  l.latent_height = 4;
  l.latent_width = 6;
  l.num_audio_latents = num_audio_latents;
  l.num_audio_rows = 2 * num_audio_latents;
  l.num_video_rows = num_latent_frames * l.rows_per_frame();
  return l;
}

SLOPFAB_TEST(packing_row_timesteps_matches_sort) {
  std::vector<SequenceLayout> layouts;
  std::vector<std::string> names;

  layouts.push_back(reference_layout(4));
  names.push_back("reference 10 s t2va");

  layouts.push_back(small_layout(3, 5, 2));
  names.push_back("small t2va");

  // No audio rows at all: `audio_t` never occurs, so `unique` is one entry and
  // every index is 0 whatever the two timesteps are.
  layouts.push_back(small_layout(3, 0, 2));
  names.push_back("no audio rows");

  // Non-zero C. The audio loop starts C entries into `idx.audio`, so the first
  // C audio rows keep the *video* timestep even though they are audio rows —
  // and the C conditioning video rows sit between the text and the audio.
  {
    SequenceLayout l = small_layout(3, 5, 2);
    l.num_condition_video = 3;
    layouts.push_back(l);
    names.push_back("C = 3");
  }

  // C larger than the audio row count: nothing is overridden at all.
  {
    SequenceLayout l = small_layout(3, 2, 2);
    l.num_condition_video = 9;
    layouts.push_back(l);
    names.push_back("C past the end of idx.audio");
  }

  // Every row is an overridden audio row, so `video_t` occurs nowhere and
  // `unique` holds the *audio* timestep alone. A rewrite that assumes the video
  // timestep is always present puts a value in `unique` that no row carries.
  {
    SequenceLayout l;
    l.num_audio_latents = 3;
    l.num_audio_rows = 6;
    layouts.push_back(l);
    names.push_back("audio only");
  }

  // Degenerate: no rows. `unique` must come back empty, not one entry.
  layouts.push_back(SequenceLayout{});
  names.push_back("empty");

  std::vector<std::pair<float, float>> pairs = {
      {0.25f, 0.75f},  // video first
      {0.75f, 0.25f},  // audio first — the flip
      {0.5f, 0.5f},    // exactly equal, collapses to one entry
      {0.0f, 0.0f},   {1.0f, 1.0f},  {0.0f, 1.0f},
      {1.0f, 0.0f},   {1.0f, 0.999f}, {0.999f, 1.0f},
  };
  // One ulp apart in both directions: the closest two distinct floats the
  // schedule could ever hand over, where a `fabs(a - b) < eps` style compare
  // would wrongly collapse them to one entry.
  const float t = 0.87345f;
  pairs.push_back({t, std::nextafter(t, 1.0f)});
  pairs.push_back({std::nextafter(t, 1.0f), t});
  pairs.push_back({t, std::nextafter(t, 0.0f)});

  int compared = 0;
  for (size_t li = 0; li < layouts.size(); ++li) {
    const PackedIndices idx = build_indices(layouts[li]);
    for (const auto& p : pairs) {
      const RowTimesteps want = row_timesteps_by_sort(layouts[li], idx, p.first, p.second);
      const RowTimesteps got = build_row_timesteps(layouts[li], idx, p.first, p.second);
      CHECK_MSG(identical(want, got), "row timesteps differ from the sort reference: %s, v=%.9g a=%.9g",
                names[li].c_str(), static_cast<double>(p.first), static_cast<double>(p.second));
      ++compared;
    }
  }
  CHECK(compared == static_cast<int>(layouts.size() * pairs.size()));
}

SLOPFAB_TEST(packing_row_timesteps_over_the_real_schedule) {
  // The two schedulers a real request runs: shift 12 for video, shift 3 for
  // audio, stepped together. This is where the identity of index 0 is actually
  // decided, so walk every step of a full 50-step schedule and require the
  // rewrite to agree with the sort at every one.
  slopfab::sampler::FlowScheduler video(12.0f);
  slopfab::sampler::FlowScheduler audio(3.0f);
  video.set_timesteps(50);
  audio.set_timesteps(50);
  const std::vector<float>& vt = video.timesteps();
  const std::vector<float>& at = audio.timesteps();
  CHECK(vt.size() == at.size());

  const SequenceLayout l = reference_layout(7);
  const PackedIndices idx = build_indices(l);

  int one_entry = 0;
  int video_is_zero = 0;
  int audio_is_zero = 0;
  for (size_t i = 0; i < vt.size(); ++i) {
    const RowTimesteps want = row_timesteps_by_sort(l, idx, vt[i], at[i]);
    const RowTimesteps got = build_row_timesteps(l, idx, vt[i], at[i]);
    CHECK_MSG(identical(want, got), "schedule step %zu: v=%.9g a=%.9g", i,
              static_cast<double>(vt[i]), static_cast<double>(at[i]));
    if (got.unique.size() == 1) {
      ++one_entry;
    } else if (got.indices[static_cast<size_t>(l.video_start())] == 0) {
      ++video_is_zero;
    } else {
      ++audio_is_zero;
    }
  }

  // The test would still pass if the schedule only ever produced one regime, so
  // say which regimes it actually covered. Both grids start at sigma = 1, where
  // the shift is the identity and the two timesteps are equal, and separate
  // afterwards — so a real 50-step schedule visits the collapsed case and the
  // two-entry case both.
  CHECK_MSG(one_entry > 0, "no step collapsed to a single unique timestep (of %zu)", vt.size());
  CHECK_MSG(video_is_zero + audio_is_zero > 0, "no step produced two unique timesteps (of %zu)",
            vt.size());
  CHECK_MSG(one_entry + video_is_zero + audio_is_zero == static_cast<int>(vt.size()),
            "regimes: %d collapsed, %d video-first, %d audio-first, of %zu steps", one_entry,
            video_is_zero, audio_is_zero, vt.size());
}

SLOPFAB_TEST(packing_row_timesteps_not_the_plausible_wrong_forms) {
  // Three rewrites of this function are plausible rather than merely broken,
  // and all three produce correctly shaped output. Compute each wrong form and
  // require the implementation not to match it.
  const SequenceLayout l = reference_layout(5);
  const PackedIndices idx = build_indices(l);
  const size_t text_row = 0;
  const size_t audio_row = static_cast<size_t>(l.audio_start());
  const size_t video_row = static_cast<size_t>(l.video_start());

  // (1) Descending `unique`. torch.unique(sorted=True) is ascending; taking
  //     max first is the natural mistake when you already have both scalars.
  {
    const RowTimesteps rt = build_row_timesteps(l, idx, 0.75f, 0.25f);
    CHECK(rt.unique.size() == 2);
    CHECK(rt.unique[0] < rt.unique[1]);
    const std::vector<float> descending = {0.75f, 0.25f};
    CHECK(rt.unique != descending);
    CHECK_NEAR(rt.unique[0], 0.25, 0.0);
    CHECK_NEAR(rt.unique[1], 0.75, 0.0);
  }

  // (2) `unique` always of size 2. Equal timesteps must *collapse* — the first
  //     step of every real schedule is exactly this case, because at sigma = 1
  //     both shifts are the identity.
  {
    const RowTimesteps rt = build_row_timesteps(l, idx, 0.5f, 0.5f);
    CHECK(rt.unique.size() == 1);
    CHECK(rt.unique.size() != 2);
    bool all_zero = true;
    for (int32_t v : rt.indices) all_zero = all_zero && v == 0;
    CHECK(all_zero);
    // ...and the adaln index then depends on the tag alone, which is what a
    // stale `ti = 1` for the audio rows would break.
    CHECK(rt.adaln[audio_row] == kTagAudio);
    CHECK(rt.adaln[audio_row] != 1 * 3 + kTagAudio);
  }

  // (3) The index not flipping when the two grids cross: video pinned to 0 and
  //     audio to 1 regardless of order. Correct whenever video_t < audio_t and
  //     silently wrong — a wrong AdaLN table row for every row in the sequence
  //     — whenever it is not.
  {
    const RowTimesteps lo = build_row_timesteps(l, idx, 0.25f, 0.75f);
    const RowTimesteps hi = build_row_timesteps(l, idx, 0.75f, 0.25f);

    CHECK(lo.indices[video_row] == 0 && lo.indices[audio_row] == 1);
    CHECK(hi.indices[video_row] == 1 && hi.indices[audio_row] == 0);
    CHECK(lo.indices[video_row] != hi.indices[video_row]);
    CHECK(lo.indices[audio_row] != hi.indices[audio_row]);

    // Text rows follow the video timestep, so they flip with it.
    CHECK(lo.indices[text_row] == 0 && hi.indices[text_row] == 1);
    CHECK(hi.adaln[text_row] == 1 * 3 + kTagText);
    CHECK(hi.adaln[text_row] != 0 * 3 + kTagText);
  }

  // (4) The audio loop starting at 0 instead of `num_condition_audio`. With
  //     C = 3 the first three audio rows keep the video timestep; starting at
  //     zero would give them the audio one, which changes only 3 rows of
  //     73 000 and nothing else.
  {
    SequenceLayout c = reference_layout(5);
    c.num_condition_audio = 3;
    c.condition_audio_is_explicit = true;
    const PackedIndices cidx = build_indices(c);
    const RowTimesteps rt = build_row_timesteps(c, cidx, 0.75f, 0.25f);
    const size_t first_audio = static_cast<size_t>(c.audio_start());
    // Audio rows [0, 3) of the audio block are not stepped: video timestep,
    // which here sorts second.
    CHECK(rt.indices[first_audio + 0] == 1);
    CHECK(rt.indices[first_audio + 1] == 1);
    CHECK(rt.indices[first_audio + 2] == 1);
    CHECK(rt.indices[first_audio + 3] == 0);
    // Their tag is still audio, so the adaln row is the video *timestep* with
    // the audio modality.
    CHECK(rt.adaln[first_audio + 0] == 1 * 3 + kTagAudio);
    CHECK(rt.adaln[first_audio + 3] == 0 * 3 + kTagAudio);
  }
}

// Frame-banded attention is lossy, so what it drops has to be exactly what was
// intended. Every failure below is silent: the model still runs, the output is
// still finite and plausibly scaled, and only the samples change.
SLOPFAB_TEST(packing_banded_key_ranges) {
  // The default request: 124 frames at 16:9 -> 37 latent frames of 48x84.
  SequenceLayout layout;
  layout.num_text = 17;
  layout.num_audio_rows = 414;
  layout.num_latent_frames = 37;
  layout.latent_height = 48;
  layout.latent_width = 84;
  layout.num_video_rows = layout.num_latent_frames * layout.rows_per_frame();

  const int S = layout.total_rows();
  const int R = layout.rows_per_frame();
  const int vstart = layout.video_start();
  CHECK(R == 1008);
  CHECK(S == 37727);

  const int kQueryTile = 128;
  const int kKeyAlign = 64;

  // Band off must be exactly full attention, every tile, or the flag's default
  // is not the behaviour it replaces.
  {
    const BandedKeyRanges off = build_banded_key_ranges(layout, 0, kQueryTile, kKeyAlign);
    bool all_global = true;
    for (int t = 0; t < off.num_query_tiles; ++t) {
      if (off.ranges[size_t(t) * 4 + 0] != 0) all_global = false;
      if (off.ranges[size_t(t) * 4 + 1] < S) all_global = false;
      if (off.ranges[size_t(t) * 4 + 3] != 0) all_global = false;
    }
    CHECK(all_global);
  }

  const BandedKeyRanges b = build_banded_key_ranges(layout, 9, kQueryTile, kKeyAlign);
  CHECK(b.num_query_tiles == (S + kQueryTile - 1) / kQueryTile);

  bool covers_own_rows = true;   // a query must always see its own frame
  bool keeps_conditioning = true;  // ...and the text/audio prefix
  bool ordered_disjoint = true;
  bool aligned = true;
  bool in_bounds = true;
  int banded_tiles = 0;
  int max_keys = 0;

  for (int t = 0; t < b.num_query_tiles; ++t) {
    const int lo0 = b.ranges[size_t(t) * 4 + 0], hi0 = b.ranges[size_t(t) * 4 + 1];
    const int lo1 = b.ranges[size_t(t) * 4 + 2], hi1 = b.ranges[size_t(t) * 4 + 3];
    const int q0 = t * kQueryTile;
    const int q_last = std::min(q0 + kQueryTile, S) - 1;

    if (lo0 % kKeyAlign || hi0 % kKeyAlign || lo1 % kKeyAlign || hi1 % kKeyAlign) aligned = false;
    if (lo0 > hi0 || lo1 > hi1) ordered_disjoint = false;
    if (hi1 > 0 && lo1 <= hi0) ordered_disjoint = false;  // merged, or double-counted
    if (hi0 > ((S + kKeyAlign - 1) / kKeyAlign) * kKeyAlign) in_bounds = false;
    if (hi1 > ((S + kKeyAlign - 1) / kKeyAlign) * kKeyAlign) in_bounds = false;

    const auto covered = [&](int row) {
      return (row >= lo0 && row < hi0) || (hi1 > lo1 && row >= lo1 && row < hi1);
    };
    // Self-attention: every row of this tile must be able to see itself.
    for (int r = q0; r <= q_last; ++r) {
      if (!covered(r)) covers_own_rows = false;
    }
    // The prompt. Losing this is the failure that would look like the model
    // ignoring its conditioning while every shape and norm stayed plausible.
    for (int r = 0; r < vstart; ++r) {
      if (!covered(r)) keeps_conditioning = false;
    }
    const int keys = b.keys_for_tile(t);
    max_keys = std::max(max_keys, keys);
    if (keys < S) ++banded_tiles;
  }

  CHECK(aligned);
  CHECK(ordered_disjoint);
  CHECK(in_bounds);
  CHECK_MSG(covers_own_rows, "a banded query tile cannot see its own rows");
  CHECK_MSG(keeps_conditioning, "a banded query tile cannot see the text/audio prefix");
  // Most tiles must actually be banded, or the flag is doing nothing.
  CHECK(banded_tiles > b.num_query_tiles * 3 / 4);
  // The global tiles hold `seq_end` keys, which exceeds S because the ranges are
  // rounded out to whole key blocks. That is the intended conservative
  // direction, so the bound is against the aligned end, not against S.
  const int seq_end = ((S + kKeyAlign - 1) / kKeyAlign) * kKeyAlign;
  CHECK(max_keys <= seq_end);

  // Every interior tile -- one whose band reaches neither end of the video --
  // must see the prefix plus its band and nothing more. A 128-row query tile is
  // narrower than a 1008-row frame, so it spans one frame or straddles two,
  // giving 2*9+1 or 2*9+2 frames; the bound has to allow both, plus at most one
  // key block of outward rounding at each of the two edges.
  {
    const int band = 9;
    int checked = 0;
    bool within_model = true;
    for (int t = 0; t < b.num_query_tiles; ++t) {
      const int q0 = t * kQueryTile;
      const int q_last = std::min(q0 + kQueryTile, S) - 1;
      if (q0 < vstart) continue;
      const int f_first = (q0 - vstart) / R;
      const int f_last = (q_last - vstart) / R;
      if (f_first - band < 0 || f_last + band + 1 > layout.num_latent_frames) continue;
      const int frames = (f_last + band + 1) - (f_first - band);
      const int lo_bound = vstart + frames * R;
      const int hi_bound = lo_bound + 4 * kKeyAlign;
      const int keys = b.keys_for_tile(t);
      if (keys < lo_bound || keys > hi_bound) within_model = false;
      ++checked;
    }
    CHECK(checked > 0);
    CHECK_MSG(within_model, "a banded tile's key count does not match the cost model");
  }

  // A band wide enough to reach both ends must collapse to full attention --
  // the two ranges merge rather than leaving a gap in the middle.
  {
    const BandedKeyRanges wide = build_banded_key_ranges(layout, 64, kQueryTile, kKeyAlign);
    bool all_full = true;
    for (int t = 0; t < wide.num_query_tiles; ++t) {
      if (wide.keys_for_tile(t) < S) all_full = false;
      if (wide.ranges[size_t(t) * 4 + 3] != 0) all_full = false;
    }
    CHECK(all_full);
  }

  // Tiles holding any text or audio row attend globally, including the one that
  // straddles the video boundary -- that tile has rows on both sides and the
  // frame arithmetic would go negative for the audio ones.
  {
    const int straddle = vstart / kQueryTile;
    CHECK(b.ranges[size_t(straddle) * 4 + 0] == 0);
    CHECK(b.ranges[size_t(straddle) * 4 + 1] >= S);
    CHECK(b.ranges[size_t(0) * 4 + 1] >= S);
  }
}

SLOPFAB_TEST(h3_rope_tables_are_canonical_serialized_bytes) {
  const std::vector<double> positions = {
      0.0, 0.0, 0.0, 1.25, -2.5, 4096.125, 16777217.0, 3.5, -9.75};
  const H3RopeTables tables = build_h3_rope_tables(positions, 10000.0f, 16);
  CHECK(tables.rows == 3);
  CHECK(tables.frequency_dim == 16);
  CHECK(tables.cosine.size() == 3 * 96);
  CHECK(tables.sine.size() == tables.cosine.size());
  bool duplicated = true;
  for (size_t i = 0; i < tables.cosine.size(); ++i) {
    const size_t column = i % 96;
    if (column < 48) {
      duplicated &= std::memcmp(&tables.cosine[i], &tables.cosine[i + 48], 4) == 0;
      duplicated &= std::memcmp(&tables.sine[i], &tables.sine[i + 48], 4) == 0;
    }
  }
  CHECK(duplicated);
  uint64_t hash = 1469598103934665603ull;
  auto hash_bytes = [&](const std::vector<float>& values) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(float); ++i) {
      hash ^= bytes[i];
      hash *= 1099511628211ull;
    }
  };
  hash_bytes(tables.cosine);
  hash_bytes(tables.sine);
  CHECK_MSG(hash == 0xdfd06ee912173e0full,
            "H3 RoPE canonical table hash is %016llx",
            static_cast<unsigned long long>(hash));
  auto rejects = [](auto&& call) {
    try { call(); } catch (const std::exception&) { return true; }
    return false;
  };
  CHECK(rejects([&] { build_h3_rope_tables({}, 10000.0f, 16); }));
  CHECK(rejects([&] { build_h3_rope_tables({0.0, 0.0, 0.0}, 0.0f, 16); }));
  CHECK(rejects([&] { build_h3_rope_tables({0.0, 0.0, 0.0}, 0.5f, 16); }));
  CHECK(rejects([&] { build_h3_rope_tables(
      {0.0, std::numeric_limits<double>::infinity(), 0.0}, 10000.0f, 16); }));
  CHECK(rejects([&] { build_h3_rope_tables(
      {0.0, std::numeric_limits<double>::max(), 0.0}, 10000.0f, 16); }));
}

}  // namespace
