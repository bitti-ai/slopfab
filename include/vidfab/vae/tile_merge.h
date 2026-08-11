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
#include <vector>

namespace vidfab::vae {

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

}  // namespace vidfab::vae
