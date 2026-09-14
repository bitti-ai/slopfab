#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>

#include "slopfab/safetensors.h"
#include "slopfab/text/limits.h"

namespace slopfab::text {

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

struct QwenPixelValues {
  QwenImageGrid grid;
  // Row-major [grid_t*grid_h*grid_w, 3*2*16*16], matching the processor's
  // pixel_values. Values are RGB normalized by (x-.5)/.5.
  std::vector<float> rows;
};

struct QwenVisionConfig {
  int hidden_size = 1152;
  int intermediate_size = 4304;
  int num_heads = 16;
  int depth = 27;
  int position_side = 48;
  int merge_size = 2;
  int output_size = 5120;
};

// Host metadata consumed by the CUDA visual tower. `prefix` permits both the
// original HF `model.visual.*` naming and the flattened repack's `visual.*`.
struct QwenVisionCheckpoint {
  const SafeTensors* checkpoint = nullptr;
  std::string prefix;
  QwenVisionConfig config;
};

// Validates all 351 BF16 tensors and returns their resolved prefix. This is a
// real loader boundary: partial towers and subtly different Qwen variants are
// rejected before any 1.19 GB device upload begins.
QwenVisionCheckpoint load_qwen3vl_vision_checkpoint(const SafeTensors& checkpoint);

struct QwenVisionPositions {
  // Learned absolute-position row for each unmerged patch.
  std::vector<int32_t> learned;
  // (temporal, height, width), one triplet per unmerged patch, in the same
  // merge-group-major order as QwenPixelValues::rows.
  std::vector<int32_t> rotary_thw;
};

QwenVisionPositions qwen3vl_vision_positions(const QwenImageGrid& grid,
                                             int position_side = 48,
                                             int merge_size = 2);

struct QwenMultimodalPlan {
  // Axis-major [3, token_count] positions for the decoder's interleaved mRoPE.
  std::vector<int32_t> position_ids;
  // Decoder row for each merged visual output. The same indices are used for
  // the main scatter and all three DeepStack additions.
  std::vector<int32_t> image_rows;
};

QwenMultimodalPlan qwen3vl_multimodal_plan(const std::vector<int32_t>& token_ids,
                                           const std::vector<QwenImageGrid>& grids,
                                           int32_t vision_start_id = 151652,
                                           int32_t image_pad_id = 151655,
                                           int32_t vision_end_id = 151653);

// DeepStack features are extracted from vision layers 8/16/24, then injected
// after text-decoder layers 0/1/2 respectively.
int qwen3vl_deepstack_slot(int text_decoder_layer);

// Qwen vision uses 2-D rotary embedding within each 72-wide attention head.
// Eighteen frequencies come from height and eighteen from width, then the
// half-split layout is duplicated to 72 channels.
void qwen3vl_vision_rope_tables(const QwenVisionPositions& positions,
                               std::vector<float>& cos, std::vector<float>& sin,
                               int head_dim = 72, float theta = 10000.0f);
void qwen3vl_decoder_rope_tables(const QwenMultimodalPlan& plan, int tokens,
                                 std::vector<float>& cos, std::vector<float>& sin,
                                 int head_dim = 128, float theta = 5.0e6f);

struct QwenVisionEmbedding {
  int tokens = 0;
  int hidden = 5120;
  std::vector<uint16_t> main;
  std::vector<uint16_t> deepstack[3];
};

// Optional exact-mode authority trace. All visual-block residuals are captured
// into one device arena and copied to this host vector only after the tower has
// completed; enabling a trace therefore does not add host seams between blocks.
struct QwenVisionTrace {
  int tokens = 0;
  int hidden = 1152;
  std::vector<uint16_t> block_residuals;
};

class QwenVisionEncoder {
 public:
  QwenVisionEncoder();
  ~QwenVisionEncoder();
  QwenVisionEncoder(const QwenVisionEncoder&) = delete;
  QwenVisionEncoder& operator=(const QwenVisionEncoder&) = delete;
  void load(const SafeTensors& checkpoint);
  void unload();
  QwenVisionEmbedding encode(const std::vector<QwenPixelValues>& images);
  QwenVisionEmbedding encode_exact(const std::vector<QwenPixelValues>& images,
                                   QwenVisionTrace* trace = nullptr);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Matches Qwen2VLImageProcessorFast.smart_resize for H3's processor config:
// factor=patch_size*merge_size=32, min_pixels=65536, max_pixels=16777216.
// Throws for invalid sizes and aspect ratios greater than 200:1.
QwenImageGrid qwen3vl_image_grid(int width, int height);

// Public H3 reference images keep their 2048-short-edge keyframe geometry,
// which can exceed the exact vision tower's 16,384-patch capacity at a
// non-square aspect. This separate presentation grid preserves the source
// aspect on the processor's factor-32 lattice while bounding unmerged patches.
QwenImageGrid qwen3vl_conditioning_grid(int width, int height);

// Returns decoder token count after adding all merged image pads and their
// start/end sentinels to already-tokenized labels/prompt. Every grid and the
// aggregate are validated with checked arithmetic before model allocation.
size_t qwen3vl_conditioning_token_count(
    const std::vector<QwenImageGrid>& grids, size_t nonvision_tokens,
    size_t max_prompt_tokens = kMaxPromptTokens);

// Patchifies an image already resized to the grid selected above. Keeping
// interpolation outside this primitive makes its byte-to-row mapping exact
// and independently testable. A still is repeated for the temporal size 2.
QwenPixelValues qwen3vl_patchify_resized_rgb(const std::vector<uint8_t>& rgb,
                                             int width, int height);
QwenPixelValues qwen3vl_patchify_resized_rgb_pair(const std::vector<uint8_t>& first,
    const std::vector<uint8_t>& second, int width, int height);

// MiniMax's image presentation, before the verbatim prompt. No chat template,
// BOS, EOS, im_start, or im_end tokens are added.
std::vector<int32_t> qwen3vl_image_block(const std::vector<int32_t>& label_ids,
                                         size_t merged_tokens,
                                         int32_t vision_start_id = 151652,
                                         int32_t image_pad_id = 151655,
                                         int32_t vision_end_id = 151653);

}  // namespace slopfab::text
