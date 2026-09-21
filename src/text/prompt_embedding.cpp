#include "slopfab/text/prompt_embedding.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include "slopfab/dit/packing.h"
#include "slopfab/tensor_convert.h"

namespace slopfab::text {
PromptEmbedding read_prompt_embedding(const std::string& path, bool require_tags) {
  SafeTensors file;
  file.open(path);
  const TensorView& view = file.at("prompt_embedding");
  if (view.dtype != DType::kF32 || view.shape.size() != 2 || view.shape[0] <= 0 ||
      view.shape[0] > kMaxPromptTokens || view.shape[1] != 5120)
    throw std::runtime_error(
        "fixed prompt: prompt_embedding must be F32 [L,5120] within token capacity");
  PromptEmbedding result;
  result.num_tokens = static_cast<int>(view.shape[0]);
  result.hidden_size = 5120;
  result.data = to_f32(view);
  for (float value : result.data)
    if (!std::isfinite(value))
      throw std::runtime_error("fixed prompt: non-finite embedding value");
  result.modality_tags.assign(result.num_tokens, dit::kTagText);
  const auto* tags = file.find("text_token_tags");
  if (!tags && require_tags)
    throw std::runtime_error("fixed prompt: reference conditioning requires text_token_tags [L]");
  if (tags) {
    if ((tags->dtype != DType::kI32 && tags->dtype != DType::kI64) ||
        tags->shape != std::vector<int64_t>{result.num_tokens})
      throw std::runtime_error("fixed prompt: text_token_tags must be I32 or I64 [L]");
    for (int i = 0; i < result.num_tokens; ++i) {
      int64_t tag = 0;
      const auto* data = static_cast<const unsigned char*>(tags->data);
      if (tags->dtype == DType::kI64)
        std::memcpy(&tag, data + size_t(i) * 8, 8);
      else {
        int32_t value;
        std::memcpy(&value, data + size_t(i) * 4, 4);
        tag = value;
      }
      if (tag != dit::kTagText && tag != dit::kTagVideo && tag != dit::kTagAudio)
        throw std::runtime_error("fixed prompt: invalid modality tag");
      result.modality_tags[i] = static_cast<int32_t>(tag);
    }
  }
  return result;
}
} // namespace slopfab::text
