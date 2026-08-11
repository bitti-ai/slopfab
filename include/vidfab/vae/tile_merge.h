// Cross-fade of one spatial tile against its already-decoded neighbours.
//
// The reference blends whole tiles (klvae.py:220-250): blend(a, b, overlap)
// returns a buffer of b's shape whose first `overlap` slices cross-fade from a
// to b and whose remaining slices are b verbatim. Done literally that is two
// full-tile passes per tile — 22 MiB read and written each — for a result that
// differs from the raw tile only inside the overlap slabs, which at the shipped
// geometry are 96 of 256 rows and 96 of 256 columns. The horizontal pass in
// particular copied 100% of the tile to change 96 columns of it.
//
// So only the two slabs are materialised, and the stitch reads the untouched
// interior straight out of the raw tile.
//
// THE ORDERING THAT MATTERS: the vertical blend runs first and the horizontal
// blend consumes its output (spec section 1.5), so in the corner where both
// apply the horizontal pass must read the *vertically blended* value, not the
// raw one. That is why the vertical slab is materialised across the full tile
// width rather than only the columns the horizontal slab covers: those extra
// columns are exactly the ones the horizontal pass reads. Every blended element
// keeps the identical `a * wa + b * wb` expression with the identical weights,
// so the result is bit-for-bit what the two full-tile passes produced.
#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vidfab::vae {

struct TileLayout {
  std::vector<int> starts;    // tile start position in pixels
  std::vector<int> extents;   // tile length in pixels
  std::vector<int> overlaps;  // overlap between tile i and i+1, size = N-1
};

// Mirrors split_tiles(..., is_decoder=True) (klvae.py:192-218). Positions are
// computed in pixel space; the caller divides by vae_ratio to slice latents.
inline TileLayout split_tiles(int input_len, int tile_size, int overlap_min, int ratio) {
  TileLayout layout;
  if (tile_size >= input_len) {
    layout.starts.push_back(0);
    layout.extents.push_back(input_len);
    return layout;
  }

  // Both of these hang rather than misbehave, which is why they are checked
  // rather than left to the caller. The loop below grows `n` until
  // `n * (tile_size - overlap_min) + overlap_min - input_len` turns
  // non-negative, and that expression does not increase with `n` once the
  // overlap swallows the tile. The round-robin after it decrements `surplus` by
  // `min(surplus, ratio)`, which never reaches zero for a non-positive ratio.
  // Reachable API since this moved into a header, and a spin is a far worse
  // failure than a thrown message.
  if (overlap_min >= tile_size || ratio <= 0) {
    throw std::runtime_error("vae: tile overlap must be smaller than the tile and the latent "
                             "ratio must be positive");
  }

  int n = (input_len + tile_size - 1) / tile_size;
  while (tile_size * n - overlap_min * (n - 1) - input_len < 0) ++n;

  std::vector<int> overlaps(static_cast<size_t>(n - 1), overlap_min);
  int surplus = tile_size * n - overlap_min * (n - 1) - input_len;
  // The surplus is absorbed by widening overlaps in whole latent units,
  // round-robin, so every tile boundary stays aligned to the latent grid.
  for (int i = 0; surplus > 0; i = (i + 1) % (n - 1)) {
    const int bump = std::min(surplus, ratio);
    overlaps[static_cast<size_t>(i)] += bump;
    surplus -= bump;
  }

  int pos = 0;
  for (int i = 0; i < n; ++i) {
    layout.starts.push_back(pos);
    layout.extents.push_back(tile_size);
    if (i < n - 1) pos += tile_size - overlaps[static_cast<size_t>(i)];
  }
  layout.overlaps = std::move(overlaps);
  return layout;
}

// Per-axis tile extents in pixels, and the tiles grouped by latent shape.
//
// Equal-shape tiles are decoded in one batch, so they are collected by
// (latent height, latent width); the ragged edge shapes, if the geometry has
// any, form batches of their own. Nothing here depends on the temporal chunk,
// which is why the pipeline resolves it once for the whole decode instead of
// rebuilding the map — allocations and all — for every chunk. std::map orders
// its keys, so the batches always reach the decoder in the same order.
inline void tile_shape_groups(const TileLayout& ytiles, const TileLayout& xtiles, int H_px,
                              int W_px, int patch, std::vector<int>* tile_h,
                              std::vector<int>* tile_w,
                              std::map<std::pair<int, int>, std::vector<size_t>>* groups) {
  tile_h->assign(ytiles.starts.size(), 0);
  tile_w->assign(xtiles.starts.size(), 0);
  groups->clear();
  for (size_t ti = 0; ti < ytiles.starts.size(); ++ti) {
    for (size_t tj = 0; tj < xtiles.starts.size(); ++tj) {
      const int th = std::min(ytiles.extents[ti], H_px - ytiles.starts[ti]) / patch;
      const int tw = std::min(xtiles.extents[tj], W_px - xtiles.starts[tj]) / patch;
      (*tile_h)[ti] = th * patch;
      (*tile_w)[tj] = tw * patch;
      (*groups)[{th, tw}].push_back(ti * xtiles.starts.size() + tj);
    }
  }
}

// Scratch for one tile's overlap slabs. Reused across tiles and chunks: the
// buffers only ever grow, so after the first tile every prepare() is
// allocation-free.
class TileMerge {
 public:
  // `raw` is the freshly decoded tile, [planes][th][tw]. `above` and `left`
  // are the raw neighbouring tiles, or null when this tile has no neighbour on
  // that axis; `y_ov` / `x_ov` are then ignored.
  //
  // Both neighbours are indexed with *this* tile's th/tw strides, exactly as
  // the whole-tile blend did. split_tiles gives every tile the same extents, so
  // the two agree; keeping the arithmetic identical keeps the results identical.
  void prepare(const float* raw, const float* above, const float* left, int planes, int th, int tw,
               int y_ov, int x_ov) {
    raw_ = raw;
    th_ = th;
    tw_ = tw;
    y_ov_ = (above != nullptr) ? y_ov : 0;
    x_ov_ = (left != nullptr) ? x_ov : 0;

    if (y_ov_ > 0) {
      const size_t need = static_cast<size_t>(planes) * y_ov_ * tw;
      if (top_.size() < need) top_.resize(need);
      for (int p = 0; p < planes; ++p) {
        for (int i = 0; i < y_ov_; ++i) {
          const float wb = static_cast<float>(i) / static_cast<float>(y_ov_);
          const float wa = 1.0f - wb;
          const size_t b_row = (static_cast<size_t>(p) * th + i) * tw;
          const size_t a_row = (static_cast<size_t>(p) * th + (th - y_ov_ + i)) * tw;
          const size_t o_row = (static_cast<size_t>(p) * y_ov_ + i) * tw;
          for (int t = 0; t < tw; ++t) {
            top_[o_row + t] = above[a_row + t] * wa + raw[b_row + t] * wb;
          }
        }
      }
    }

    if (x_ov_ > 0) {
      const size_t need = static_cast<size_t>(planes) * th * x_ov_;
      if (cols_.size() < need) cols_.resize(need);
      for (int p = 0; p < planes; ++p) {
        for (int y = 0; y < th; ++y) {
          // Rows inside the vertical slab take their `b` from the vertically
          // blended value; the rest are still raw.
          const float* brow = (y < y_ov_)
                                  ? top_.data() + (static_cast<size_t>(p) * y_ov_ + y) * tw
                                  : raw + (static_cast<size_t>(p) * th + y) * tw;
          const float* arow = left + (static_cast<size_t>(p) * th + y) * tw;
          float* orow = cols_.data() + (static_cast<size_t>(p) * th + y) * x_ov_;
          for (int j = 0; j < x_ov_; ++j) {
            const float wb = static_cast<float>(j) / static_cast<float>(x_ov_);
            const float wa = 1.0f - wb;
            orow[j] = arow[tw - x_ov_ + j] * wa + brow[j] * wb;
          }
        }
      }
    }
  }

  // Writes the first `count` columns of merged row (p, y) to `dst`. The row is
  // assembled from up to two pieces: the horizontal slab, then whichever of the
  // vertical slab or the raw tile owns the interior.
  void copy_row(int p, int y, int count, float* dst) const {
    const int from_cols = std::min(count, x_ov_);
    if (from_cols > 0) {
      const float* src = cols_.data() + (static_cast<size_t>(p) * th_ + y) * x_ov_;
      std::copy_n(src, from_cols, dst);
    }
    if (from_cols < count) {
      const float* src = (y < y_ov_)
                             ? top_.data() + (static_cast<size_t>(p) * y_ov_ + y) * tw_
                             : raw_ + (static_cast<size_t>(p) * th_ + y) * tw_;
      std::copy_n(src + from_cols, count - from_cols, dst + from_cols);
    }
  }

 private:
  std::vector<float> top_;   // [planes][y_ov][tw], the vertical slab
  std::vector<float> cols_;  // [planes][th][x_ov], the horizontal slab
  const float* raw_ = nullptr;
  int th_ = 0;
  int tw_ = 0;
  int y_ov_ = 0;
  int x_ov_ = 0;
};

// Where each of a chunk's `out_frames` decoded frames belongs.
//
// Only two ranges survive a chunk: [pre, pre + frames_per_chunk) is the primary
// block and belongs at its final place in the assembled video, and
// [chunk_dec + pre, + overlap) is carried into the next chunk. At the shipped
// schedule that is frames [3, 20) and [23, 28) of 28 — the other six are read
// by nothing, so their entry is null and the stitch skips them.
//
// The stitch used to write all 28 frames into a staging buffer and two copies
// afterwards moved the kept ranges out of it. Giving it the destinations
// directly is the same data with one pass fewer, and `frame_stride` is
// 3 * height * width, one whole frame of planar RGB.
inline void chunk_frame_destinations(int out_frames, int pre, int frames_per_chunk, int chunk_dec,
                                     int overlap, size_t frame_stride, float* primary, float* carry,
                                     std::vector<float*>* dst) {
  dst->assign(static_cast<size_t>(out_frames), nullptr);
  for (int f = 0; f < out_frames; ++f) {
    if (f >= pre && f - pre < frames_per_chunk) {
      (*dst)[static_cast<size_t>(f)] = primary + static_cast<size_t>(f - pre) * frame_stride;
    } else if (f >= chunk_dec + pre && f - chunk_dec - pre < overlap) {
      (*dst)[static_cast<size_t>(f)] =
          carry + static_cast<size_t>(f - chunk_dec - pre) * frame_stride;
    }
  }
}

}  // namespace vidfab::vae
