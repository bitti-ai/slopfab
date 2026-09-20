#pragma once
#include "session_state.h"
#include "slopfab/text/qwen_vision.h"
#include <functional>
namespace slopfab::generation {
struct PromptInputs {
  text::PromptEmbedding& fixed_prompt;
  const std::vector<std::string>& reference_identities;
  const std::vector<RGBImage>& reference_images;
  std::vector<text::QwenImageGrid>& reference_conditioning_grids;
  std::vector<int32_t>& reference_conditioning_ids;
  std::vector<text::QwenPixelValues>& media_qwen_pairs;
  std::function<text::Tokenizer&()> tokenizer;
};
bool prepare_multimodal_prompt(const GenerateRequest&, const RunOptions&,
    const std::vector<PreparedReference>&, PromptInputs&, RunResult&);
bool encode_h3_prompt(const GenerateRequest&, const RunOptions&, ReusedGenerationModels&,
    PromptInputs&, text::PromptEmbedding&, RunResult&);
}
