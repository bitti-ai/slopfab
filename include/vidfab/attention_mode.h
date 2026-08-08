#pragma once

#include <limits>

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

struct SolSchedule {
  float beta = 1.0f;
  float error_k = 0.0f;
  float error_v = 0.0f;
  int step_begin = 10;
  int step_end = std::numeric_limits<int>::max();
  int step_every = 1;
  int layer_begin = 2;
  int layer_end = std::numeric_limits<int>::max();
  int layer_every = 1;

  bool active(int step, int layer) const {
    return step >= step_begin && step <= step_end && step_every > 0 &&
           (step - step_begin) % step_every == 0 && layer >= layer_begin &&
           layer <= layer_end && layer_every > 0 &&
           (layer - layer_begin) % layer_every == 0;
  }
};

}  // namespace vidfab
