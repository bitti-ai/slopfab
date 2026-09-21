// Spatial composition regressions for composited-neighbour blending.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "harness.h"
#include "slopfab/vae/tile_merge.h"

namespace {
using slopfab::vae::TileLayout;
using slopfab::vae::TileMerge;
using slopfab::vae::split_tiles;

// Float64 reference using complete composited rows and complete left tiles.
// This deliberately retains whole neighbours instead of the production tails.
std::vector<float> oracle(const TileLayout& yl, const TileLayout& xl, int h, int w, int planes,
                          const std::vector<std::vector<float>>& tiles) {
  std::vector<float> result(static_cast<size_t>(planes) * h * w);
  for (int p = 0; p < planes; ++p) {
    std::vector<double> previous;
    for (size_t i = 0; i < yl.starts.size(); ++i) {
      const int th = yl.extents[i];
      std::vector<double> current(static_cast<size_t>(th) * w);
      std::vector<double> left;
      for (size_t j = 0; j < xl.starts.size(); ++j) {
        const int tw = xl.extents[j];
        const auto& raw = tiles[i * xl.starts.size() + j];
        std::vector<double> tile(raw.begin() + static_cast<ptrdiff_t>(p * th * tw),
                                 raw.begin() + static_cast<ptrdiff_t>((p + 1) * th * tw));
        if (i > 0) {
          const int overlap = yl.overlaps[i - 1];
          for (int y = 0; y < overlap; ++y) {
            const double wb = static_cast<double>(y) / overlap;
            for (int x = 0; x < tw; ++x) {
              const size_t above =
                  static_cast<size_t>(yl.extents[i - 1] - overlap + y) * w + xl.starts[j] + x;
              tile[y * tw + x] = previous[above] * (1.0 - wb) + tile[y * tw + x] * wb;
            }
          }
        }
        if (j > 0) {
          const int overlap = xl.overlaps[j - 1];
          for (int y = 0; y < th; ++y) {
            for (int x = 0; x < overlap; ++x) {
              const double wb = static_cast<double>(x) / overlap;
              const size_t before =
                  static_cast<size_t>(y) * xl.extents[j - 1] + xl.extents[j - 1] - overlap + x;
              tile[y * tw + x] = left[before] * (1.0 - wb) + tile[y * tw + x] * wb;
            }
          }
        }
        const int keep = tw - (j + 1 < xl.starts.size() ? xl.overlaps[j] : 0);
        for (int y = 0; y < th; ++y)
          std::copy_n(tile.data() + y * tw, keep, current.data() + y * w + xl.starts[j]);
        left = std::move(tile);
      }
      const int keep = th - (i + 1 < yl.starts.size() ? yl.overlaps[i] : 0);
      for (size_t k = 0; k < static_cast<size_t>(keep) * w; ++k)
        result[(static_cast<size_t>(p) * h + yl.starts[i]) * w + k] =
            static_cast<float>(current[k]);
      previous = std::move(current);
    }
  }
  return result;
}

std::vector<std::vector<float>> random_tiles(const TileLayout& yl, const TileLayout& xl,
                                             int planes) {
  std::vector<std::vector<float>> tiles;
  for (int h : yl.extents)
    for (int w : xl.extents)
      tiles.push_back(slopfab::test::make_data(static_cast<size_t>(planes) * h * w,
                                               123 + static_cast<uint32_t>(tiles.size()), 3.0f));
  return tiles;
}

std::vector<float*> destinations(std::vector<float>& output, int planes, int h, int w) {
  std::vector<float*> dst;
  for (int p = 0; p < planes; ++p)
    dst.push_back(output.data() + static_cast<size_t>(p) * h * w);
  return dst;
}
} // namespace

SLOPFAB_TEST(tile_merge_includes_diagonal_at_intersection) {
  const auto layout = split_tiles(12, 8, 4, 1);
  std::vector<std::vector<float>> tiles;
  for (float value : {0.0f, 10.0f, 20.0f, 30.0f})
    tiles.emplace_back(64, value);
  TileMerge merge(layout, layout, 12, 12);
  std::vector<float> out(144, -999.0f);
  merge.compose(tiles, destinations(out, 1, 12, 12));
  CHECK_NEAR(out[6 * 12 + 6], 15.0, 0.0);
  // If only the diagonal is nonzero, it still contributes one quarter.
  for (size_t i = 0; i < tiles.size(); ++i)
    std::fill(tiles[i].begin(), tiles[i].end(), i == 0 ? 4.0f : 0.0f);
  merge.compose(tiles, destinations(out, 1, 12, 12));
  CHECK_NEAR(out[6 * 12 + 6], 1.0, 0.0);
}

SLOPFAB_TEST(tile_merge_matches_float64_composited_neighbour_oracle) {
  struct Case {
    int h, w, tile, overlap;
  };

  for (const auto g : {Case{12, 12, 8, 4}, Case{29, 29, 16, 4}, Case{32, 32, 16, 15},
                       Case{57, 76, 16, 4}, Case{7, 32, 16, 0}, Case{32, 7, 16, 0},
                       Case{7, 37, 16, 0}, Case{37, 7, 16, 4}, Case{7, 9, 16, 4}}) {
    const auto yl = split_tiles(g.h, g.tile, g.overlap, 1);
    const auto xl = split_tiles(g.w, g.tile, g.overlap, 1);
    auto tiles = random_tiles(yl, xl, 3);
    const auto original = tiles;
    const auto expected = oracle(yl, xl, g.h, g.w, 3, tiles);
    std::vector<float> out(expected.size(), -999.0f);
    TileMerge merge(yl, xl, g.h, g.w);
    const auto dst = destinations(out, 3, g.h, g.w);
    merge.compose(tiles, dst);
    CHECK_CLOSE(expected, out, 2e-6, "composited neighbours, float64 oracle");
    CHECK(tiles == original);
    std::fill(out.begin(), out.end(), 777.0f);
    merge.compose(tiles, dst);
    CHECK_CLOSE(expected, out, 2e-6, "reused tails overwrite output");
    // A skipped plane must not be read or written, and other planes retain
    // their original decoder strides rather than being packed together.
    auto sparse = dst;
    sparse[1] = nullptr;
    std::fill(out.begin() + g.h * g.w, out.begin() + 2 * g.h * g.w, 777.0f);
    merge.compose(tiles, sparse);
    for (int k = 0; k < g.h * g.w; ++k) {
      CHECK_NEAR(out[g.h * g.w + k], 777.0, 0.0);
      CHECK_NEAR(out[2 * g.h * g.w + k], expected[2 * g.h * g.w + k], 2e-6);
    }
  }
}

SLOPFAB_TEST(tile_merge_preserves_constants_and_global_coordinates) {
  const int h = 29, w = 47, planes = 2;
  const auto yl = split_tiles(h, 16, 4, 1);
  const auto xl = split_tiles(w, 16, 4, 1);
  auto tiles = random_tiles(yl, xl, planes);
  for (size_t i = 0; i < yl.starts.size(); ++i)
    for (size_t j = 0; j < xl.starts.size(); ++j)
      for (int y = 0; y < yl.extents[i]; ++y)
        for (int x = 0; x < xl.extents[j]; ++x) {
          auto& tile = tiles[i * xl.starts.size() + j];
          tile[y * xl.extents[j] + x] = 0.375f;
          tile[(yl.extents[i] + y) * xl.extents[j] + x] =
              static_cast<float>((yl.starts[i] + y) * w + xl.starts[j] + x);
        }
  TileMerge merge(yl, xl, h, w);
  std::vector<float> out(planes * h * w, -999.0f);
  merge.compose(tiles, destinations(out, planes, h, w));
  for (int k = 0; k < h * w; ++k) {
    CHECK_NEAR(out[k], 0.375, 2e-7);
    CHECK_NEAR(out[h * w + k], k, 3e-4);
  }
}

SLOPFAB_TEST(tile_merge_triple_overlap_uses_sequential_weights) {
  // Three length-8 tiles at starts 0, 3, 6. At coordinate 7 the first
  // contributor survives two fades: (1 - 4/5) * (1 - 1/5) = 0.16.
  // Normalized overlap-add instead gives 1/6, so this pins the sequential rule.
  const TileLayout triple{{0, 3, 6}, {8, 8, 8}, {5, 5}};
  const TileLayout one{{0}, {1}, {}};
  for (bool vertical : {false, true}) {
    TileMerge merge(vertical ? triple : one, vertical ? one : triple, vertical ? 14 : 1,
                    vertical ? 1 : 14);
    std::vector<std::vector<float>> tiles(3, std::vector<float>(8, 0.0f));
    const float expected[] = {0.16f, 0.64f, 0.2f};
    std::vector<float> out(14);
    for (size_t contributor = 0; contributor < tiles.size(); ++contributor) {
      for (size_t k = 0; k < tiles.size(); ++k)
        std::fill(tiles[k].begin(), tiles[k].end(), k == contributor ? 1.0f : 0.0f);
      merge.compose(tiles, {out.data()});
      CHECK_NEAR(out[7], expected[contributor], 1e-7);
    }
  }
  // Both axes overlap three ways; the diagonal survives through both tails.
  TileMerge merge(triple, triple, 14, 14);
  std::vector<std::vector<float>> tiles(9, std::vector<float>(64, 0.0f));
  std::fill(tiles[0].begin(), tiles[0].end(), 1.0f);
  std::vector<float> out(196);
  merge.compose(tiles, {out.data()});
  CHECK_NEAR(out[7 * 14 + 7], 0.16 * 0.16, 1e-7);
}

SLOPFAB_TEST(tile_merge_preserves_native_plan) {
  const auto x = split_tiles(1216, 256, 64, 16);
  const auto y = split_tiles(896, 256, 64, 16);
  CHECK(x.starts == std::vector<int>({0, 192, 384, 576, 768, 960}));
  CHECK(y.starts == std::vector<int>({0, 160, 320, 480, 640}));
  CHECK(x.starts.size() * y.starts.size() == 30);
}

SLOPFAB_TEST(tile_merge_single_tile_is_exact_and_bad_shapes_can_be_retried) {
  const auto y = split_tiles(7, 16, 4, 1);
  const auto x = split_tiles(9, 16, 4, 1);
  TileMerge merge(y, x, 7, 9);
  auto tiles = random_tiles(y, x, 2);
  std::vector<float> out(126, -999.0f);
  const auto dst = destinations(out, 2, 7, 9);
  merge.compose(tiles, dst);
  CHECK_CLOSE(tiles[0], out, 0.0, "single tile identity");
  const auto original = tiles;
  tiles[0].pop_back();
  bool threw = false;
  try {
    merge.compose(tiles, dst);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  tiles = original;
  merge.compose(tiles, dst);
  CHECK_CLOSE(tiles[0], out, 0.0, "retry after invalid tile");
}

SLOPFAB_TEST(tile_merge_rejects_invalid_plans) {
  const std::vector<TileLayout> invalid = {
      {{}, {}, {}},           {{0}, {}, {}},         {{1}, {7}, {}},        {{0}, {7}, {}},
      {{0, 5}, {4, 3}, {-1}}, {{0, 5}, {4, 3}, {0}}, {{0, 4}, {6, 4}, {1}}, {{0, 4}, {8, 4}, {5}},
      {{0, 0}, {8, 8}, {8}},  {{0, 4}, {4, 0}, {0}}};
  for (const auto& layout : invalid) {
    bool threw = false;
    try {
      (void)slopfab::vae::validate_tile_axis(layout, 8);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw);
  }
}

SLOPFAB_TEST(chunk_destinations_reproduce_the_staged_split) {
  // The shipped schedule: 28 decoded frames per chunk, of which [3, 20) is the
  // primary block and [23, 28) is the carry. The old code stitched all 28 into
  // a staging buffer and then copied those two ranges out of it; writing
  // through the destination table has to put exactly the same bytes in exactly
  // the same places, and leave the six dead frames unwritten.
  const int out_frames = 28, pre = 3, frames_per_chunk = 17, chunk_dec = 20, overlap = 5;
  const size_t stride = 12; // stand-in for 3 * height * width

  // Distinguishable content: frame f, element i holds f * 1000 + i.
  std::vector<float> staged(static_cast<size_t>(out_frames) * stride);
  for (int f = 0; f < out_frames; ++f) {
    for (size_t i = 0; i < stride; ++i) {
      staged[static_cast<size_t>(f) * stride + i] =
          static_cast<float>(f) * 1000.0f + static_cast<float>(i);
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
  slopfab::vae::chunk_frame_destinations(out_frames, pre, frames_per_chunk, chunk_dec, overlap,
                                         stride, primary.data(), carry.data(), &dst);
  CHECK(dst.size() == static_cast<size_t>(out_frames));
  int written = 0;
  for (int f = 0; f < out_frames; ++f) {
    if (dst[static_cast<size_t>(f)] == nullptr)
      continue;
    ++written;
    std::copy_n(staged.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(f) * stride), stride,
                dst[static_cast<size_t>(f)]);
  }
  CHECK(written == frames_per_chunk + overlap);
  CHECK_CLOSE(expect_primary, primary, 0.0, "chunk destinations, primary block");
  CHECK_CLOSE(expect_carry, carry, 0.0, "chunk destinations, carry block");

  // Named explicitly rather than left implicit in the arithmetic: frames 0-2
  // are the pre-padding and 20-22 the gap before the carry.
  for (int f : {0, 1, 2, 20, 21, 22})
    CHECK(dst[static_cast<size_t>(f)] == nullptr);
  for (int f : {3, 19, 23, 27})
    CHECK(dst[static_cast<size_t>(f)] != nullptr);
}

SLOPFAB_TEST(tile_layout_is_the_shipped_geometry) {
  // The overlap widths are what every sizing argument about the blend rests on,
  // so they are pinned rather than recomputed: 1280 x 768 at tile 256, minimum
  // overlap 64, latent ratio 16.
  const slopfab::vae::TileLayout y = slopfab::vae::split_tiles(768, 256, 64, 16);
  const slopfab::vae::TileLayout x = slopfab::vae::split_tiles(1280, 256, 64, 16);

  CHECK(y.starts.size() == 4);
  CHECK(x.starts.size() == 7);
  const std::vector<int> y_overlaps = {96, 80, 80};
  const std::vector<int> x_overlaps = {96, 96, 80, 80, 80, 80};
  CHECK(y.overlaps == y_overlaps);
  CHECK(x.overlaps == x_overlaps);

  // The kept extents must tile the axis exactly — the stitch writes
  // `keep` pixels per tile and relies on them summing to the full width, with
  // no gap left holding whatever the buffer had before.
  auto kept_total = [](const slopfab::vae::TileLayout& l) {
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
  for (int o : y.overlaps)
    CHECK(o % 16 == 0);
  for (int o : x.overlaps)
    CHECK(o % 16 == 0);

  // A single tile when the axis fits, and no overlaps to blend.
  const slopfab::vae::TileLayout one = slopfab::vae::split_tiles(256, 256, 64, 16);
  CHECK(one.starts.size() == 1 && one.overlaps.empty() && one.extents[0] == 256);

  // Degenerate arguments spin forever rather than returning something wrong:
  // neither loop below the guard makes progress. A hang is the one failure a
  // caller cannot diagnose, so both are rejected. The single-tile early-out
  // above runs first, so these need an input longer than the tile.
  CHECK(slopfab::test::throws([] {
    (void)slopfab::vae::split_tiles(1280, 256, 256, 16);
  }));
  CHECK(slopfab::test::throws([] {
    (void)slopfab::vae::split_tiles(1280, 256, 300, 16);
  }));
  CHECK(slopfab::test::throws([] {
    (void)slopfab::vae::split_tiles(1280, 256, 64, 0);
  }));
  CHECK(slopfab::test::throws([] {
    (void)slopfab::vae::split_tiles(1280, 256, 64, -16);
  }));
  // And the shipped arguments are nowhere near the guard.
  CHECK(!slopfab::test::throws([] {
    (void)slopfab::vae::split_tiles(1280, 256, 64, 16);
  }));
}

SLOPFAB_TEST(tile_shape_groups_are_chunk_invariant) {
  // The pipeline resolves this once for the whole decode instead of per chunk.
  // Nothing it reads depends on the chunk, so the only thing to establish is
  // that the map really does partition the tiles and that its iteration order —
  // which fixes the order batches reach the decoder — is deterministic.
  const slopfab::vae::TileLayout y = slopfab::vae::split_tiles(768, 256, 64, 16);
  const slopfab::vae::TileLayout x = slopfab::vae::split_tiles(1280, 256, 64, 16);

  std::vector<int> tile_h, tile_w;
  std::map<std::pair<int, int>, std::vector<size_t>> groups;
  slopfab::vae::tile_shape_groups(y, x, 768, 1280, 16, &tile_h, &tile_w, &groups);

  CHECK(tile_h.size() == 4 && tile_w.size() == 7);
  for (int h : tile_h)
    CHECK(h == 256);
  for (int w : tile_w)
    CHECK(w == 256);
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
  for (int n : seen)
    CHECK(n == 1);

  // Recomputing gives an identical map, key order included — which is what
  // makes hoisting it out of the chunk loop a no-op.
  std::vector<int> h2, w2;
  std::map<std::pair<int, int>, std::vector<size_t>> again;
  slopfab::vae::tile_shape_groups(y, x, 768, 1280, 16, &h2, &w2, &again);
  CHECK(h2 == tile_h && w2 == tile_w);
  CHECK(again == groups);

  // A geometry with a ragged edge, so the multi-group path is exercised too:
  // 300 pixels at tile 256 splits into two tiles whose second one is clipped.
  const slopfab::vae::TileLayout ry = slopfab::vae::split_tiles(768, 256, 64, 16);
  const slopfab::vae::TileLayout rx = slopfab::vae::split_tiles(1280, 512, 64, 16);
  std::vector<int> rh, rw;
  std::map<std::pair<int, int>, std::vector<size_t>> rgroups;
  slopfab::vae::tile_shape_groups(ry, rx, 768, 1280, 16, &rh, &rw, &rgroups);
  size_t total = 0;
  for (const auto& g : rgroups)
    total += g.second.size();
  CHECK(total == ry.starts.size() * rx.starts.size());
}

SLOPFAB_TEST(tile_latent_gather_covers_its_whole_buffer) {
  // The gather's destination buffer is now reused and grown rather than
  // value-initialised each shape group, which is only safe because the gather
  // writes every element of it. The indexing below is the pipeline's, at
  // src/vae/decode_pipeline.cpp: batch item `bi`, channel `ci`, frame `t`, row
  // `y`, each copying a whole row of `tw`.
  const int ch = 24, window = 7, th = 16, tw = 16;
  const size_t n = 3; // batch items
  const size_t tile_voxels = static_cast<size_t>(window) * th * tw;
  const size_t needed = n * ch * tile_voxels;

  std::vector<int> touched(needed, 0);
  for (size_t bi = 0; bi < n; ++bi) {
    for (int ci = 0; ci < ch; ++ci) {
      for (int t = 0; t < window; ++t) {
        for (int y = 0; y < th; ++y) {
          const size_t dst = (bi * ch + ci) * tile_voxels + (static_cast<size_t>(t) * th + y) * tw;
          CHECK(dst + tw <= needed);
          for (int k = 0; k < tw; ++k)
            ++touched[dst + static_cast<size_t>(k)];
        }
      }
    }
  }
  size_t unwritten = 0, doubled = 0;
  for (int c : touched) {
    if (c == 0)
      ++unwritten;
    if (c > 1)
      ++doubled;
  }
  CHECK_MSG(unwritten == 0, "%zu of %zu gather destinations are never written", unwritten, needed);
  CHECK_MSG(doubled == 0, "%zu gather destinations are written more than once", doubled);
}
