#pragma once

namespace vidfab {

// User-selected implementation for MiniMax H3 self-attention.
enum class AttentionMode {
  kNone,    // Unfused, memory-bounded reference path.
  kFlash2,  // Exact BF16 FlashAttention-2-style fused kernel.
  kSage2,   // Quantized SageAttention2 kernel.
  kSol,     // Training-free block routing with zeroth-order correction.
  kSolExperimental,  // Experimental SM120 TMA/WMMA pipeline; explicitly opt-in.
};

inline const char* attention_mode_name(AttentionMode mode) {
  switch (mode) {
    case AttentionMode::kNone: return "none";
    case AttentionMode::kFlash2: return "flash2";
    case AttentionMode::kSage2: return "sage2";
    case AttentionMode::kSol: return "sol";
    case AttentionMode::kSolExperimental: return "sol-experimental";
  }
  return "unknown";
}

inline bool is_sol_attention(AttentionMode mode) {
  return mode == AttentionMode::kSol || mode == AttentionMode::kSolExperimental;
}

}  // namespace vidfab
