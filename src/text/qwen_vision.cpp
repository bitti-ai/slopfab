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
