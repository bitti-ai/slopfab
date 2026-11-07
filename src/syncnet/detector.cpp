#include "network.h"
#include <algorithm>
#include <cmath>

namespace slopfab::syncnet {
float iou(const Box& a, const Box& b) {
  float intersection = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                       std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  float area = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection;
  return area > 0 ? intersection / area : 0;
}

std::vector<Box> detect(Network& net, const Image& rgb, float threshold) {
  Image x = rgb;
  const float means[] = {123, 117, 104};
  for (size_t i = 0; i < x.data.size(); ++i)
    x.data[i] -= means[i % 3];
  std::vector<Box> proposals;
  int level = 0;
  auto head = [&](Image source, const char* norm = nullptr) {
    if (norm)
      net.l2norm(source, norm);
    auto loc = net.conv(source, "loc." + std::to_string(level), 1, 1);
    auto conf = net.conv(source, "conf." + std::to_string(level), 1, 1);
    float step = float(4 << level), size = float(16 << level);
    for (int p = 0; p < loc.h * loc.w; ++p) {
      float bg = conf.data[size_t(p) * conf.c];
      if (level == 0)
        bg = std::max({bg, conf.data[size_t(p) * 4 + 1], conf.data[size_t(p) * 4 + 2]});
      float probability = 1.f / (1.f + std::exp(bg - conf.data[size_t(p) * conf.c + conf.c - 1]));
      if (probability <= threshold)
        continue;
      const float* d = loc.data.data() + size_t(p) * 4;
      float cx = (p % loc.w + .5f) * step + d[0] * .1f * size;
      float cy = (p / loc.w + .5f) * step + d[1] * .1f * size;
      float w = size * std::exp(d[2] * .2f), h = size * std::exp(d[3] * .2f);
      if (std::isfinite(cx + cy + w + h) && w > 0 && h > 0)
        proposals.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, probability});
    }
    ++level;
  };
  for (int i = 0; i < 35; ++i) {
    if (i == 4 || i == 9 || i == 16 || i == 23 || i == 30)
      x = Network::pool(x, 2, 2, 2, 2, 0, i == 16);
    else if (i == 0 || i == 2 || i == 5 || i == 7 || i == 10 || i == 12 || i == 14 || i == 17 ||
             i == 19 || i == 21 || i == 24 || i == 26 || i == 28 || i == 31 || i == 33)
      x = net.conv(x, "vgg." + std::to_string(i), 1,
                   i == 33   ? 0
                   : i == 31 ? 6
                             : 1,
                   i == 31 ? 6 : 1, true);
    if (i == 15)
      head(x, "L2Norm3_3");
    if (i == 22)
      head(x, "L2Norm4_3");
    if (i == 29)
      head(x, "L2Norm5_3");
  }
  head(x);
  for (int i = 0; i < 4; ++i) {
    x = net.conv(x, "extras." + std::to_string(i), i % 2 ? 2 : 1, i % 2, 1, true);
    if (i % 2)
      head(x);
  }
  std::stable_sort(proposals.begin(), proposals.end(), [](const Box& a, const Box& b) {
    return a.score > b.score;
  });
  if (proposals.size() > 5000)
    proposals.resize(5000);
  // Upstream runs .3 NMS, then .1 NMS across scales. Keep both stages.
  for (float overlap : {.3f, .1f}) {
    std::vector<Box> kept;
    for (const auto& b : proposals) {
      bool suppressed = false;
      for (const auto& a : kept)
        if (iou(a, b) > overlap) {
          suppressed = true;
          break;
        }
      if (!suppressed)
        kept.push_back(b);
      if (kept.size() == 750)
        break;
    }
    proposals = std::move(kept);
  }
  return proposals;
}
} // namespace slopfab::syncnet
