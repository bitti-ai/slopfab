// Spatial cross-fades against already-composited VAE neighbours.
#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace slopfab::vae {

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

inline void validate_tile_axis(const TileLayout& layout, int length) {
  const size_t n = layout.starts.size();
  if (n == 0 || layout.extents.size() != n || layout.overlaps.size() != n - 1 ||
      length <= 0 || layout.starts.front() != 0) {
    throw std::runtime_error("vae: invalid tile axis plan");
  }
  for (size_t i = 0; i < n; ++i) {
    const int start = layout.starts[i], extent = layout.extents[i];
    if (start < 0 || start >= length || extent <= 0 || extent > length - start) {
      throw std::runtime_error("vae: invalid tile extent");
    }
    if (i > 0) {
      const int overlap = layout.overlaps[i - 1];
      if (start <= layout.starts[i - 1] || overlap < 0 ||
          overlap > std::min(layout.extents[i - 1], extent) ||
          layout.starts[i - 1] + layout.extents[i - 1] - start != overlap) {
        throw std::runtime_error("vae: inconsistent tile overlap geometry");
      }
    }
  }
  if (layout.starts.back() + layout.extents.back() != length) {
    throw std::runtime_error("vae: tile plan does not cover the axis");
  }
}

class TileMerge {
 public:
  TileMerge(const TileLayout& ytiles, const TileLayout& xtiles, int height, int width)
      : ytiles_(ytiles), xtiles_(xtiles), height_(height), width_(width) {
    validate_tile_axis(ytiles, height);
    validate_tile_axis(xtiles, width);
    if (ytiles.starts.size() == 1 && xtiles.starts.size() == 1) return;
    const int th = *std::max_element(ytiles.extents.begin(), ytiles.extents.end());
    const int tw = *std::max_element(xtiles.extents.begin(), xtiles.extents.end());
    const int y_overlap = ytiles.overlaps.empty() ? 0 :
        *std::max_element(ytiles.overlaps.begin(), ytiles.overlaps.end());
    const int x_overlap = xtiles.overlaps.empty() ? 0 :
        *std::max_element(xtiles.overlaps.begin(), xtiles.overlaps.end());
    // Process one retained plane at a time in the backend's host float format.
    // Only composited overlap tails survive a tile; no accumulation band or
    // normalization weights are needed. All scratch is reused across chunks.
    tile_.resize(static_cast<size_t>(th) * tw);
    left_.resize(static_cast<size_t>(th) * x_overlap);
    strip_.resize(static_cast<size_t>(y_overlap) * width);
    next_strip_.resize(strip_.size());
  }

  // Tiles are row-major [plane][tile height][tile width]. Each destination is
  // one full output plane, or null for a discarded temporal frame. No input is
  // modified; destinations are overwritten, including when this object is reused.
  void compose(const std::vector<std::vector<float>>& tiles,
               const std::vector<float*>& destinations) {
    const size_t nx = xtiles_.starts.size();
    if (tiles.size() != ytiles_.starts.size() * nx)
      throw std::runtime_error("vae: decoded tile count does not match the plan");
    for (size_t i = 0; i < tiles.size(); ++i) {
      if (tiles[i].size() != destinations.size() * ytiles_.extents[i / nx] *
                                xtiles_.extents[i % nx])
        throw std::runtime_error("vae: decoded tile shape does not match the plan");
    }
    const size_t pixels = static_cast<size_t>(height_) * width_;
    for (size_t p = 0; p < destinations.size(); ++p) {
      float* dst = destinations[p];
      if (dst == nullptr) continue;
      if (tiles.size() == 1) {
        std::copy_n(tiles[0].data() + p * pixels, pixels, dst);
        continue;
      }
      for (size_t i = 0; i < ytiles_.starts.size(); ++i) {
        const int th = ytiles_.extents[i];
        const int top = i > 0 ? ytiles_.overlaps[i - 1] : 0;
        const int bottom = i + 1 < ytiles_.starts.size() ? ytiles_.overlaps[i] : 0;
        for (size_t j = 0; j < nx; ++j) {
          const int tw = xtiles_.extents[j];
          const int left = j > 0 ? xtiles_.overlaps[j - 1] : 0;
          const int right = j + 1 < nx ? xtiles_.overlaps[j] : 0;
          const int keep_w = tw - right;
          const float* raw = tiles[i * nx + j].data() + p * th * tw;
          std::copy_n(raw, static_cast<size_t>(th) * tw, tile_.data());
          // The previous row's strip contains horizontally composited pixels
          // across the full canvas width, including diagonal contributors.
          for (int y = 0; y < top; ++y) {
            const float wb = static_cast<float>(y) / top;
            for (int x = 0; x < tw; ++x) {
              float& value = tile_[static_cast<size_t>(y) * tw + x];
              value = strip_[static_cast<size_t>(y) * width_ + xtiles_.starts[j] + x] *
                          (1.0f - wb) + value * wb;
            }
          }
          for (int y = 0; y < th; ++y) {
            float* row = tile_.data() + static_cast<size_t>(y) * tw;
            for (int x = 0; x < left; ++x) {
              const float wb = static_cast<float>(x) / left;
              row[x] = left_[static_cast<size_t>(y) * left + x] * (1.0f - wb) + row[x] * wb;
            }
          }
          // Save the right tail AFTER both blends and BEFORE cropping columns.
          // Save bottom rows with those columns cropped to build one full strip.
          for (int y = 0; y < th; ++y) {
            const float* row = tile_.data() + static_cast<size_t>(y) * tw;
            if (right > 0)
              std::copy_n(row + keep_w, right, left_.data() + static_cast<size_t>(y) * right);
            if (y >= th - bottom)
              std::copy_n(row, keep_w, next_strip_.data() +
                  static_cast<size_t>(y - (th - bottom)) * width_ + xtiles_.starts[j]);
            else
              std::copy_n(row, keep_w, dst +
                  static_cast<size_t>(ytiles_.starts[i] + y) * width_ + xtiles_.starts[j]);
          }
        }
        strip_.swap(next_strip_);
      }
    }
  }

 private:
  TileLayout ytiles_, xtiles_;
  int height_, width_;
  std::vector<float> tile_, left_, strip_, next_strip_;
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

}  // namespace slopfab::vae
