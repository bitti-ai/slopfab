#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vidfab::text {

// Host-side shape contract of the Qwen3-VL image processor shipped with H3.
// The visual forward pass consumes unmerged 16x16 patches; the language model
// receives one token for each 2x2 spatial group.
struct QwenImageGrid {
  int temporal = 1;
  int height = 0;
  int width = 0;

  size_t patch_count() const;
  size_t merged_token_count() const;
};

// Matches Qwen2VLImageProcessorFast.smart_resize for H3's processor config:
// factor=patch_size*merge_size=32, min_pixels=65536, max_pixels=16777216.
// Throws for invalid sizes and aspect ratios greater than 200:1.
QwenImageGrid qwen3vl_image_grid(int width, int height);

// MiniMax's image presentation, before the verbatim prompt. No chat template,
// BOS, EOS, im_start, or im_end tokens are added.
std::vector<int32_t> qwen3vl_image_block(const std::vector<int32_t>& label_ids,
                                         size_t merged_tokens,
                                         int32_t vision_start_id = 151652,
                                         int32_t image_pad_id = 151655,
                                         int32_t vision_end_id = 151653);

}  // namespace vidfab::text
