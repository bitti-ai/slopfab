#pragma once

#include "slopfab/text/encoder.h"

namespace slopfab::text {
// Fixed/captured conditioning. Reference runs require explicit modality tags;
// legacy text-only captures may omit them. No tokenizer or encoder is loaded.
PromptEmbedding read_prompt_embedding(const std::string& path, bool require_tags = false);

// Validates the complete conditioning payload before atomically saving it.
// Missing tags are filled with text tags for text-only callers.
void write_prompt_embedding(const std::string& path, const PromptEmbedding& embedding,
                            const std::map<std::string, std::string>& metadata = {});
} // namespace slopfab::text
