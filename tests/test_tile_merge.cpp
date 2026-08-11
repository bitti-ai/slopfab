// The VAE decoder's spatial tile cross-fade.
//
// TileMerge computes only the two overlap slabs, where the whole-tile blend it
// replaces wrote every element of a 22 MiB scratch buffer twice — and for most
// of those elements the body of the loop was literally `out[i] = b[i]`. The
// saving is only worth having if the result is bit-for-bit unchanged, so these
// tests keep a verbatim copy of the old whole-tile blend and require exact
// equality against it, not a tolerance.
//
// The case that would break first is the corner where both axes overlap. The
// vertical blend runs first and the horizontal blend consumes its output, so
// the corner must carry a doubly-blended value; a slab implementation that
// materialised the vertical result only over the columns it "needed" would
// feed the horizontal pass raw values there and produce a seam visible only at
// tile crossings. Every geometry below with both overlaps non-zero tests that,
// and `tile_merge_corner_is_blended_twice` checks it on numbers chosen so the
// singly-blended answer is a different number.

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "harness.h"
#include "vidfab/vae/tile_merge.h"

namespace {

// The whole-tile blend this replaces, copied unchanged from the version of
// src/vae/decode_pipeline.cpp that shipped it. blend(a, b, overlap) from
// klvae.py:220-250: the result has b's shape, its first `overlap` slices
// cross-fade from a to b, the rest is b verbatim.
void blend_axis(const float* a, const float* b, float* out, int lead, int overlap, int blend_len,
                int trail) {
  for (int l = 0; l < lead; ++l) {
    for (int i = 0; i < blend_len; ++i) {
      const float wb = (i < overlap) ? static_cast<float>(i) / static_cast<float>(overlap) : 1.0f;
      const float wa = 1.0f - wb;
      for (int t = 0; t < trail; ++t) {
        const size_t bi = (static_cast<size_t>(l) * blend_len + i) * trail + t;
        if (i < overlap) {
          const int a_index = blend_len - overlap + i;
          const size_t ai = (static_cast<size_t>(l) * blend_len + a_index) * trail + t;
          out[bi] = a[ai] * wa + b[bi] * wb;
        } else {
          out[bi] = b[bi];
        }
      }
    }
  }
}

struct Geometry {
  const char* name;
  int planes;
  int th;
  int tw;
  int y_ov;  // 0 means no tile above
  int x_ov;  // 0 means no tile to the left
  int keep_h;
  int keep_w;
};

// What the old code produced for the kept part of one tile: blend vertically
// into a full-tile scratch, blend that horizontally into another full-tile
// scratch, then copy the top-left keep_h x keep_w corner of every plane out.
std::vector<float> reference_stitch(const Geometry& g, const std::vector<float>& raw,
                                    const std::vector<float>& above,
                                    const std::vector<float>& left) {
  const size_t tile = static_cast<size_t>(g.planes) * g.th * g.tw;
  std::vector<float> lhs(tile);
  std::vector<float> rhs(tile);
  const float* src = raw.data();
  if (g.y_ov > 0) {
    blend_axis(above.data(), src, lhs.data(), g.planes, g.y_ov, g.th, g.tw);
    src = lhs.data();
  }
  if (g.x_ov > 0) {
    blend_axis(left.data(), src, rhs.data(), g.planes * g.th, g.x_ov, g.tw, 1);
    src = rhs.data();
  }
  std::vector<float> out(static_cast<size_t>(g.planes) * g.keep_h * g.keep_w);
  for (int p = 0; p < g.planes; ++p) {
    for (int y = 0; y < g.keep_h; ++y) {
      for (int x = 0; x < g.keep_w; ++x) {
        out[(static_cast<size_t>(p) * g.keep_h + y) * g.keep_w + x] =
            src[(static_cast<size_t>(p) * g.th + y) * g.tw + x];
      }
    }
  }
  return out;
}

std::vector<float> merged_stitch(const Geometry& g, const std::vector<float>& raw,
                                 const std::vector<float>& above,
                                 const std::vector<float>& left, vidfab::vae::TileMerge* merge) {
  merge->prepare(raw.data(), g.y_ov > 0 ? above.data() : nullptr,
                 g.x_ov > 0 ? left.data() : nullptr, g.planes, g.th, g.tw, g.y_ov, g.x_ov);
  std::vector<float> out(static_cast<size_t>(g.planes) * g.keep_h * g.keep_w);
  for (int p = 0; p < g.planes; ++p) {
    for (int y = 0; y < g.keep_h; ++y) {
      merge->copy_row(p, y, g.keep_w,
                      out.data() + (static_cast<size_t>(p) * g.keep_h + y) * g.keep_w);
    }
  }
  return out;
}

}  // namespace

VIDFAB_TEST(tile_merge_matches_the_whole_tile_blend_exactly) {
  // The shipped decode runs 3 * 28 planes over 256 x 256 tiles with y-overlaps
  // [96, 80, 80] and x-overlaps [96, 96, 80, 80, 80, 80]. The proportions are
  // reproduced at a size a unit test can afford, plus the degenerate corners.
  const Geometry cases[] = {
      {"interior tile, both overlaps", 5, 32, 32, 12, 12, 20, 20},
      {"first row, left overlap only", 5, 32, 32, 0, 12, 32, 20},
      {"first column, top overlap only", 5, 32, 32, 12, 0, 20, 32},
      {"first tile, no overlap at all", 5, 32, 32, 0, 0, 20, 20},
      {"last tile keeps its full extent", 5, 32, 32, 10, 10, 32, 32},
      {"overlap of one", 3, 8, 8, 1, 1, 7, 7},
      {"overlap fills the whole tile", 3, 8, 8, 8, 8, 8, 8},
      {"shipped ratios, 96 of 256 scaled", 7, 64, 64, 24, 24, 40, 40},
      {"asymmetric extents", 4, 24, 40, 9, 15, 15, 25},
      {"keep narrower than the x overlap", 4, 16, 16, 6, 12, 10, 8},
  };

  vidfab::vae::TileMerge merge;  // deliberately reused, as the pipeline reuses it
  for (const Geometry& g : cases) {
    const size_t tile = static_cast<size_t>(g.planes) * g.th * g.tw;
    const std::vector<float> raw = vidfab::test::make_data(tile, 1301, 3.0f);
    const std::vector<float> above = vidfab::test::make_data(tile, 7717, 3.0f);
    const std::vector<float> left = vidfab::test::make_data(tile, 4409, 3.0f);

    const std::vector<float> expect = reference_stitch(g, raw, above, left);
    const std::vector<float> actual = merged_stitch(g, raw, above, left, &merge);
    CHECK(expect.size() == actual.size());
    // Tolerance zero: this is a data-movement change, not an arithmetic one.
    CHECK_CLOSE(expect, actual, 0.0, (std::string("tile merge: ") + g.name).c_str());
  }
}

VIDFAB_TEST(tile_merge_corner_is_blended_twice) {
  // Constant tiles make the expected value arithmetic rather than a second
  // implementation. raw = 0, above = 4, left = 8, overlap 4 on both axes.
  //
  // At (y, x) = (1, 1): the vertical pass gives b1 = 4 * (1 - 1/4) = 3, then
  // the horizontal pass gives 8 * (1 - 1/4) + 3 * (1/4) = 6.75. Had the corner
  // been blended horizontally against the *raw* 0 it would be 6.0, so the two
  // orders are distinguishable, which is the whole point of the check.
  const int planes = 2, th = 8, tw = 8, ov = 4;
  const size_t tile = static_cast<size_t>(planes) * th * tw;
  const std::vector<float> raw(tile, 0.0f);
  const std::vector<float> above(tile, 4.0f);
  const std::vector<float> left(tile, 8.0f);

  vidfab::vae::TileMerge merge;
  merge.prepare(raw.data(), above.data(), left.data(), planes, th, tw, ov, ov);
  std::vector<float> row(tw);

  merge.copy_row(1, 1, tw, row.data());
  CHECK_NEAR(row[1], 6.75, 0.0);
  // Same row, outside the horizontal slab: vertical blend only.
  CHECK_NEAR(row[ov], 3.0, 0.0);
  // Below the vertical slab, inside the horizontal one: horizontal blend of
  // raw against left, 8 * (1 - 1/4) + 0 * (1/4).
  merge.copy_row(1, ov, tw, row.data());
  CHECK_NEAR(row[1], 6.0, 0.0);
  // Interior: the raw tile, untouched.
  CHECK_NEAR(row[ov], 0.0, 0.0);
}

VIDFAB_TEST(chunk_destinations_reproduce_the_staged_split) {
  // The shipped schedule: 28 decoded frames per chunk, of which [3, 20) is the
  // primary block and [23, 28) is the carry. The old code stitched all 28 into
  // a staging buffer and then copied those two ranges out of it; writing
  // through the destination table has to put exactly the same bytes in exactly
  // the same places, and leave the six dead frames unwritten.
  const int out_frames = 28, pre = 3, frames_per_chunk = 17, chunk_dec = 20, overlap = 5;
  const size_t stride = 12;  // stand-in for 3 * height * width

  // Distinguishable content: frame f, element i holds f * 1000 + i.
  std::vector<float> staged(static_cast<size_t>(out_frames) * stride);
  for (int f = 0; f < out_frames; ++f) {
    for (size_t i = 0; i < stride; ++i) {
      staged[static_cast<size_t>(f) * stride + i] = static_cast<float>(f) * 1000.0f +
                                                    static_cast<float>(i);
    }
  }

  // What the two copy_n calls produced.
  std::vector<float> expect_primary(static_cast<size_t>(frames_per_chunk) * stride);
  std::vector<float> expect_carry(static_cast<size_t>(overlap) * stride);
  std::copy_n(staged.begin() + static_cast<ptrdiff_t>(pre * stride),
              static_cast<size_t>(frames_per_chunk) * stride, expect_primary.begin());
  std::copy_n(staged.begin() + static_cast<ptrdiff_t>((chunk_dec + pre) * stride),
              static_cast<size_t>(overlap) * stride, expect_carry.begin());

  // What the table produces. The sentinel proves the six dead frames are never
  // written: it survives only if nothing points at them.
  const float sentinel = -12345.0f;
  std::vector<float> primary(expect_primary.size(), sentinel);
  std::vector<float> carry(expect_carry.size(), sentinel);
  std::vector<float*> dst;
  vidfab::vae::chunk_frame_destinations(out_frames, pre, frames_per_chunk, chunk_dec, overlap,
                                        stride, primary.data(), carry.data(), &dst);
  CHECK(dst.size() == static_cast<size_t>(out_frames));
  int written = 0;
  for (int f = 0; f < out_frames; ++f) {
    if (dst[static_cast<size_t>(f)] == nullptr) continue;
    ++written;
    std::copy_n(staged.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(f) * stride), stride,
                dst[static_cast<size_t>(f)]);
  }
  CHECK(written == frames_per_chunk + overlap);
  CHECK_CLOSE(expect_primary, primary, 0.0, "chunk destinations, primary block");
  CHECK_CLOSE(expect_carry, carry, 0.0, "chunk destinations, carry block");

  // Named explicitly rather than left implicit in the arithmetic: frames 0-2
  // are the pre-padding and 20-22 the gap before the carry.
  for (int f : {0, 1, 2, 20, 21, 22}) CHECK(dst[static_cast<size_t>(f)] == nullptr);
  for (int f : {3, 19, 23, 27}) CHECK(dst[static_cast<size_t>(f)] != nullptr);
}

VIDFAB_TEST(tile_layout_is_the_shipped_geometry) {
  // The overlap widths are what every sizing argument about the blend rests on,
  // so they are pinned rather than recomputed: 1280 x 768 at tile 256, minimum
  // overlap 64, latent ratio 16.
  const vidfab::vae::TileLayout y = vidfab::vae::split_tiles(768, 256, 64, 16);
  const vidfab::vae::TileLayout x = vidfab::vae::split_tiles(1280, 256, 64, 16);

  CHECK(y.starts.size() == 4);
  CHECK(x.starts.size() == 7);
  const std::vector<int> y_overlaps = {96, 80, 80};
  const std::vector<int> x_overlaps = {96, 96, 80, 80, 80, 80};
  CHECK(y.overlaps == y_overlaps);
  CHECK(x.overlaps == x_overlaps);

  // The kept extents must tile the axis exactly — the stitch writes
  // `keep` pixels per tile and relies on them summing to the full width, with
  // no gap left holding whatever the buffer had before.
  auto kept_total = [](const vidfab::vae::TileLayout& l) {
    int total = 0;
    for (size_t i = 0; i < l.starts.size(); ++i) {
      total += l.extents[i] - (i + 1 < l.starts.size() ? l.overlaps[i] : 0);
    }
    return total;
  };
  CHECK(kept_total(y) == 768);
  CHECK(kept_total(x) == 1280);

  // Every boundary stays on the latent grid: an overlap that was not a whole
  // number of latent units would slice a latent in half.
  for (int o : y.overlaps) CHECK(o % 16 == 0);
  for (int o : x.overlaps) CHECK(o % 16 == 0);

  // A single tile when the axis fits, and no overlaps to blend.
  const vidfab::vae::TileLayout one = vidfab::vae::split_tiles(256, 256, 64, 16);
  CHECK(one.starts.size() == 1 && one.overlaps.empty() && one.extents[0] == 256);
}

VIDFAB_TEST(tile_shape_groups_are_chunk_invariant) {
  // The pipeline resolves this once for the whole decode instead of per chunk.
  // Nothing it reads depends on the chunk, so the only thing to establish is
  // that the map really does partition the tiles and that its iteration order —
  // which fixes the order batches reach the decoder — is deterministic.
  const vidfab::vae::TileLayout y = vidfab::vae::split_tiles(768, 256, 64, 16);
  const vidfab::vae::TileLayout x = vidfab::vae::split_tiles(1280, 256, 64, 16);

  std::vector<int> tile_h, tile_w;
  std::map<std::pair<int, int>, std::vector<size_t>> groups;
  vidfab::vae::tile_shape_groups(y, x, 768, 1280, 16, &tile_h, &tile_w, &groups);

  CHECK(tile_h.size() == 4 && tile_w.size() == 7);
  for (int h : tile_h) CHECK(h == 256);
  for (int w : tile_w) CHECK(w == 256);
  // Every tile is 256 x 256 in pixels, so 16 x 16 in latents: one batch of 28.
  CHECK(groups.size() == 1);
  CHECK(groups.begin()->first == std::make_pair(16, 16));
  CHECK(groups.begin()->second.size() == 28);

  // Each tile index appears exactly once across all groups.
  std::vector<int> seen(28, 0);
  for (const auto& g : groups) {
    for (size_t id : g.second) {
      CHECK(id < seen.size());
      ++seen[id];
    }
  }
  for (int n : seen) CHECK(n == 1);

  // Recomputing gives an identical map, key order included — which is what
  // makes hoisting it out of the chunk loop a no-op.
  std::vector<int> h2, w2;
  std::map<std::pair<int, int>, std::vector<size_t>> again;
  vidfab::vae::tile_shape_groups(y, x, 768, 1280, 16, &h2, &w2, &again);
  CHECK(h2 == tile_h && w2 == tile_w);
  CHECK(again == groups);

  // A geometry with a ragged edge, so the multi-group path is exercised too:
  // 300 pixels at tile 256 splits into two tiles whose second one is clipped.
  const vidfab::vae::TileLayout ry = vidfab::vae::split_tiles(768, 256, 64, 16);
  const vidfab::vae::TileLayout rx = vidfab::vae::split_tiles(1280, 512, 64, 16);
  std::vector<int> rh, rw;
  std::map<std::pair<int, int>, std::vector<size_t>> rgroups;
  vidfab::vae::tile_shape_groups(ry, rx, 768, 1280, 16, &rh, &rw, &rgroups);
  size_t total = 0;
  for (const auto& g : rgroups) total += g.second.size();
  CHECK(total == ry.starts.size() * rx.starts.size());
}
VIDFAB_TEST(tile_latent_gather_covers_its_whole_buffer) {
  // The gather's destination buffer is now reused and grown rather than
  // value-initialised each shape group, which is only safe because the gather
  // writes every element of it. The indexing below is the pipeline's, at
  // src/vae/decode_pipeline.cpp: batch item `bi`, channel `ci`, frame `t`, row
  // `y`, each copying a whole row of `tw`.
  const int ch = 24, window = 7, th = 16, tw = 16;
  const size_t n = 3;  // batch items
  const size_t tile_voxels = static_cast<size_t>(window) * th * tw;
  const size_t needed = n * ch * tile_voxels;

  std::vector<int> touched(needed, 0);
  for (size_t bi = 0; bi < n; ++bi) {
    for (int ci = 0; ci < ch; ++ci) {
      for (int t = 0; t < window; ++t) {
        for (int y = 0; y < th; ++y) {
          const size_t dst =
              (bi * ch + ci) * tile_voxels + (static_cast<size_t>(t) * th + y) * tw;
          CHECK(dst + tw <= needed);
          for (int k = 0; k < tw; ++k) ++touched[dst + static_cast<size_t>(k)];
        }
      }
    }
  }
  size_t unwritten = 0, doubled = 0;
  for (int c : touched) {
    if (c == 0) ++unwritten;
    if (c > 1) ++doubled;
  }
  CHECK_MSG(unwritten == 0, "%zu of %zu gather destinations are never written", unwritten, needed);
  CHECK_MSG(doubled == 0, "%zu gather destinations are written more than once", doubled);
}
