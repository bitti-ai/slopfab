#include "network.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace slopfab::syncnet {
void Tracker::update(int frame, const std::vector<Box>& boxes, bool scene_cut) {
  if (scene_cut)
    active_.clear();
  active_.erase(std::remove_if(active_.begin(), active_.end(),
                               [&](size_t i) {
                                 return frame - tracks_[i].observations.back().frame > 10;
                               }),
                active_.end());

  struct Match {
    float overlap;
    size_t track, box;
  };

  std::vector<Match> matches;
  for (size_t i : active_)
    for (size_t j = 0; j < boxes.size(); ++j) {
      float overlap = iou(tracks_[i].observations.back().box, boxes[j]);
      if (overlap > .5f)
        matches.push_back({overlap, i, j});
    }
  std::stable_sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
    return a.overlap > b.overlap;
  });
  std::vector<bool> used_track(tracks_.size()), used_box(boxes.size());
  for (const auto& m : matches)
    if (!used_track[m.track] && !used_box[m.box]) {
      tracks_[m.track].observations.push_back({frame, boxes[m.box]});
      used_track[m.track] = used_box[m.box] = true;
    }
  for (size_t j = 0; j < boxes.size(); ++j)
    if (!used_box[j]) {
      active_.push_back(tracks_.size());
      tracks_.push_back({int(tracks_.size()), {{frame, boxes[j]}}});
    }
}

std::vector<Box> track_boxes(const Track& track) {
  if (track.observations.empty())
    return {};
  const auto& o = track.observations;
  int first = o.front().frame, last = o.back().frame;
  if (last < first || last - first > 15000)
    throw std::invalid_argument("SyncNet: invalid track");
  std::vector<std::array<float, 3>> centers(size_t(last - first + 1));
  size_t j = 0;
  for (int f = first; f <= last; ++f) {
    while (j + 1 < o.size() && o[j + 1].frame < f)
      ++j;
    const auto& a = o[j];
    const auto& b = o[std::min(j + 1, o.size() - 1)];
    float t = a.frame == b.frame ? 0 : float(f - a.frame) / (b.frame - a.frame);
    float x1 = a.box.x1 + t * (b.box.x1 - a.box.x1), y1 = a.box.y1 + t * (b.box.y1 - a.box.y1);
    float x2 = a.box.x2 + t * (b.box.x2 - a.box.x2), y2 = a.box.y2 + t * (b.box.y2 - a.box.y2);
    centers[f - first] = {(x1 + x2) / 2, (y1 + y2) / 2, std::max(x2 - x1, y2 - y1) / 2};
  }
  std::vector<Box> result;
  for (int f = 0; f < int(centers.size()); ++f) {
    float smoothed[3];
    for (int c = 0; c < 3; ++c) {
      std::array<float, 13> values;
      for (int k = -6; k <= 6; ++k)
        values[k + 6] = centers[std::clamp(f + k, 0, int(centers.size()) - 1)][c];
      std::nth_element(values.begin(), values.begin() + 6, values.end());
      smoothed[c] = values[6];
    }
    float cx = smoothed[0], cy = smoothed[1], bs = smoothed[2];
    result.push_back({std::floor(cx - 1.4f * bs), std::floor(cy - bs), std::floor(cx + 1.4f * bs),
                      std::floor(cy + 1.8f * bs), 1});
  }
  return result;
}
} // namespace slopfab::syncnet
