#pragma once

namespace slopfab::text {

// Shared conditioning budget for text, image/video tokens, and modality labels.
// This is an implementation resource bound, not the model's context length.
inline constexpr int kMaxPromptTokens = 32768;

} // namespace slopfab::text
