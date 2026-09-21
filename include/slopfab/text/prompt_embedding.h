#pragma once

#include "slopfab/text/encoder.h"

namespace slopfab::text {
// Fixed/captured conditioning. Reference runs require explicit modality tags;
// legacy text-only captures may omit them. No tokenizer or encoder is loaded.
PromptEmbedding read_prompt_embedding(const std::string& path, bool require_tags = false);
} // namespace slopfab::text
