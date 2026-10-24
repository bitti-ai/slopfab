#include "slopfab/dit/vsa.h"
#include <algorithm>
#include <stdexcept>

namespace slopfab::dit {
VsaTiles build_vsa_tiles(const SequenceLayout& l, int tile_size, int keep_denominator) {
  const int t = l.num_latent_frames, h = l.latent_height / 2, w = l.latent_width / 2;
  if (t <= 0 || h <= 0 || w <= 0 || l.latent_height % 2 || l.latent_width % 2 || l.num_text < 0 ||
      l.num_audio_rows < 0 || l.num_condition_video < 0 || l.num_condition_audio < 0 ||
      int64_t(t) * h * w != l.num_video_rows || (tile_size != 64 && tile_size != 128) ||
      keep_denominator < 1)
    throw std::invalid_argument("VSA-H3 invalid packed tile geometry");
  VsaTiles tiles;
  tiles.tile_size = tile_size;
  tiles.keep_denominator = keep_denominator;
  tiles.row_tiles.resize(l.total_rows(), -1);
  auto append = [&](const std::vector<int32_t>& rows) {
    const int tile = int(tiles.sizes.size());
    tiles.sizes.push_back(int(rows.size()));
    const size_t start = tiles.rows.size();
    tiles.rows.resize(start + tile_size, -1);
    std::copy(rows.begin(), rows.end(), tiles.rows.begin() + start);
    for (int row : rows)
      tiles.row_tiles.at(row) = tile;
  };
  auto dense = [&](int offset, int count) {
    for (int start = 0; start < count; start += tile_size) {
      std::vector<int32_t> rows;
      for (int i = start; i < std::min(start + tile_size, count); ++i)
        rows.push_back(offset + i);
      append(rows);
    }
  };
  dense(0, l.num_text);
  int cursor = l.num_text;
  for (const auto& s : l.reference_segments) {
    if (s.start != cursor || s.rows <= 0 || s.rows > l.audio_start() - cursor || s.frames < 0 ||
        (s.frames &&
         (s.height <= 0 || s.width <= 0 || int64_t(s.frames) * s.height * s.width != s.rows)))
      throw std::invalid_argument("VSA-H3 invalid reference segment");
    if (!s.frames)
      dense(s.start, s.rows);
    cursor += s.rows;
  }
  if (cursor != l.audio_start())
    throw std::invalid_argument("VSA-H3 requires complete reference segment geometry");
  dense(l.audio_start(), l.num_audio_rows);
  tiles.prefix_tiles = int(tiles.sizes.size());
  auto video = [&](int offset, int frames, int height, int width) {
    const int tw = tile_size == 128 ? 8 : 4;
    for (int z = 0; z < frames; z += 4)
      for (int y = 0; y < height; y += 4)
        for (int x = 0; x < width; x += tw) {
          std::vector<int32_t> rows;
          for (int iz = z; iz < std::min(z + 4, frames); ++iz)
            for (int iy = y; iy < std::min(y + 4, height); ++iy)
              for (int ix = x; ix < std::min(x + tw, width); ++ix)
                rows.push_back(offset + (iz * height + iy) * width + ix);
          append(rows);
        }
    tiles.region_ends.push_back(int(tiles.sizes.size()));
  };
  for (const auto& s : l.reference_segments)
    if (s.frames)
      video(s.start, s.frames, s.height, s.width);
  video(l.video_start(), t, h, w);
  return tiles;
}
}
