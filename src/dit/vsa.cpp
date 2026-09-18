#include "slopfab/dit/vsa.h"

#include <algorithm>
#include <stdexcept>

namespace slopfab::dit {

VsaTiles build_vsa_tiles(const SequenceLayout& layout) {
  const int t = layout.num_latent_frames;
  const int h = layout.latent_height / 2, w = layout.latent_width / 2;
  if (t <= 0 || h <= 0 || w <= 0 || layout.latent_height % 2 ||
      layout.latent_width % 2 || layout.num_text < 0 || layout.num_audio_rows < 0 ||
      layout.num_condition_video != 0 || layout.num_condition_audio != 0 ||
      int64_t(t) * h * w != layout.num_video_rows)
    throw std::runtime_error("VSA-H3 requires text/audio followed by an unconditioned video grid");
  VsaTiles tiles;
  tiles.row_tiles.resize(layout.total_rows(), -1);
  auto append = [&](const std::vector<int32_t>& rows) {
    const int tile = static_cast<int>(tiles.sizes.size());
    tiles.sizes.push_back(static_cast<int32_t>(rows.size()));
    const size_t start = tiles.rows.size();
    tiles.rows.resize(start + 64, -1);
    std::copy(rows.begin(), rows.end(), tiles.rows.begin() + start);
    for (int row : rows) tiles.row_tiles[row] = tile;
  };
  int offset = 0;
  for (int segment : {layout.num_text, layout.num_audio_rows}) {
    for (int start = 0; start < segment; start += 64) {
      std::vector<int32_t> rows;
      for (int i = start; i < std::min(start + 64, segment); ++i)
        rows.push_back(offset + i);
      append(rows);
    }
    offset += segment;
  }
  tiles.prefix_tiles = static_cast<int>(tiles.sizes.size());
  for (int bt = 0; bt < t; bt += 4)
    for (int bh = 0; bh < h; bh += 4)
      for (int bw = 0; bw < w; bw += 4) {
        std::vector<int32_t> rows;
        for (int z = bt; z < std::min(bt + 4, t); ++z)
          for (int y = bh; y < std::min(bh + 4, h); ++y)
            for (int x = bw; x < std::min(bw + 4, w); ++x)
              rows.push_back(offset + (z * h + y) * w + x);
        append(rows);
      }
  return tiles;
}

}  // namespace slopfab::dit
