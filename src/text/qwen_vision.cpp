#include "vidfab/text/qwen_vision.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vidfab::text {
namespace {
constexpr int kFactor = 32;
constexpr double kMinPixels = 65536.0;
constexpr double kMaxPixels = 16777216.0;

int round_factor(double value) {
  return std::max(kFactor, static_cast<int>(std::round(value / kFactor)) * kFactor);
}
}  // namespace

size_t QwenImageGrid::patch_count() const {
  return static_cast<size_t>(temporal) * height * width;
}

size_t QwenImageGrid::merged_token_count() const { return patch_count() / 4; }

QwenImageGrid qwen3vl_image_grid(int width, int height) {
  if (width <= 0 || height <= 0) throw std::runtime_error("Qwen image: invalid size");
  const double ratio = std::max(width, height) / static_cast<double>(std::min(width, height));
  if (ratio > 200.0) throw std::runtime_error("Qwen image: aspect ratio exceeds 200:1");

  int resized_h = round_factor(height);
  int resized_w = round_factor(width);
  double pixels = static_cast<double>(resized_h) * resized_w;
  if (pixels > kMaxPixels) {
    const double beta = std::sqrt(static_cast<double>(height) * width / kMaxPixels);
    resized_h = std::max(kFactor, static_cast<int>(std::floor(height / beta / kFactor)) * kFactor);
    resized_w = std::max(kFactor, static_cast<int>(std::floor(width / beta / kFactor)) * kFactor);
  } else if (pixels < kMinPixels) {
    const double beta = std::sqrt(kMinPixels / (static_cast<double>(height) * width));
    resized_h = static_cast<int>(std::ceil(height * beta / kFactor)) * kFactor;
    resized_w = static_cast<int>(std::ceil(width * beta / kFactor)) * kFactor;
  }
  return {1, resized_h / 16, resized_w / 16};
}

QwenPixelValues qwen3vl_patchify_resized_rgb(const std::vector<uint8_t>& rgb,
                                             int width, int height) {
  if (width <= 0 || height <= 0 || width % kFactor || height % kFactor)
    throw std::runtime_error("Qwen image: resized dimensions must be positive multiples of 32");
  if (rgb.size() != static_cast<size_t>(width) * height * 3)
    throw std::runtime_error("Qwen image: RGB byte count does not match dimensions");

  QwenPixelValues out;
  out.grid = {1, height / 16, width / 16};
  constexpr int patch = 16, merge = 2, temporal = 2, channels = 3;
  constexpr int row_width = channels * temporal * patch * patch;
  out.rows.resize(out.grid.patch_count() * row_width);

  // Equivalent to reshape(t,tp,c,h/m,m,p,w/m,m,p), then transpose
  // (t,h/m,w/m,m,m,c,tp,p,p). This keeps each 2x2 merge group contiguous.
  size_t dst = 0;
  for (int tile_y = 0; tile_y < height / (merge * patch); ++tile_y)
    for (int tile_x = 0; tile_x < width / (merge * patch); ++tile_x)
      for (int merge_y = 0; merge_y < merge; ++merge_y)
        for (int merge_x = 0; merge_x < merge; ++merge_x)
          for (int channel = 0; channel < channels; ++channel)
            for (int time = 0; time < temporal; ++time)
              for (int py = 0; py < patch; ++py)
                for (int px = 0; px < patch; ++px) {
                  (void)time;  // a still image is duplicated over the temporal patch
                  const int y = (tile_y * merge + merge_y) * patch + py;
                  const int x = (tile_x * merge + merge_x) * patch + px;
                  out.rows[dst++] = rgb[(static_cast<size_t>(y) * width + x) * 3 + channel] /
                                           127.5f -
                                       1.0f;
                }
  return out;
}

std::vector<int32_t> qwen3vl_image_block(const std::vector<int32_t>& label_ids,
                                         size_t merged_tokens, int32_t vision_start_id,
                                         int32_t image_pad_id, int32_t vision_end_id) {
  if (merged_tokens == 0) throw std::runtime_error("Qwen image: empty vision block");
  std::vector<int32_t> ids;
  ids.reserve(label_ids.size() + merged_tokens + 2);
  ids.insert(ids.end(), label_ids.begin(), label_ids.end());
  ids.push_back(vision_start_id);
  ids.insert(ids.end(), merged_tokens, image_pad_id);
  ids.push_back(vision_end_id);
  return ids;
}

}  // namespace vidfab::text
