#include "slopfab/text/prompt_embedding.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include "slopfab/dit/packing.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/safetensors_write.h"

namespace slopfab::text {
void write_prompt_embedding(const std::string& path, const PromptEmbedding& embedding,
                            const std::map<std::string, std::string>& metadata) {
  if (embedding.num_tokens <= 0 || embedding.num_tokens > kMaxPromptTokens ||
      embedding.hidden_size != 5120 ||
      embedding.data.size() != size_t(embedding.num_tokens) * 5120)
    throw std::invalid_argument("prompt embedding: expected F32 [L,5120] within token capacity");
  for (float value : embedding.data)
    if (!std::isfinite(value))
      throw std::invalid_argument("prompt embedding: non-finite value");
  auto tags = embedding.modality_tags;
  if (tags.empty())
    tags.assign(embedding.num_tokens, dit::kTagText);
  if (tags.size() != size_t(embedding.num_tokens))
    throw std::invalid_argument("prompt embedding: tag count disagrees with token count");
  for (auto tag : tags)
    if (tag != dit::kTagText && tag != dit::kTagVideo && tag != dit::kTagAudio)
      throw std::invalid_argument("prompt embedding: invalid modality tag");
  auto info = metadata;
  info["slopfab_prompt_embedding"] = "h3-prompt-v1";
  write_safetensors_atomic(path,
                          {{"prompt_embedding", {embedding.num_tokens, 5120}, embedding.data},
                           {"text_token_tags", {embedding.num_tokens}, {}, DType::kI32, tags}},
                          info);
}

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
