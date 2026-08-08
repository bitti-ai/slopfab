#pragma once

namespace vidfab {

// User-selected implementation for MiniMax H3 self-attention.
enum class AttentionMode {
  kNone,    // Unfused, memory-bounded reference path.
  kFlash2,  // Exact BF16 FlashAttention-2-style fused kernel.
  kSage2,   // Quantized SageAttention2 kernel.
  kSol,     // Training-free block routing with zeroth-order correction.
};

inline const char* attention_mode_name(AttentionMode mode) {
  switch (mode) {
    case AttentionMode::kNone: return "none";
    case AttentionMode::kFlash2: return "flash2";
    case AttentionMode::kSage2: return "sage2";
    case AttentionMode::kSol: return "sol";
  }
  return "unknown";
}

}  // namespace vidfab
