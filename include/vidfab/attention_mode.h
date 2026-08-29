#pragma once

#include <limits>
#include <string_view>

#include "vidfab/device_tensor.h"

namespace vidfab {

// User-selected implementation for MiniMax H3 self-attention.
enum class AttentionMode {
  kNone,    // Unfused, memory-bounded reference path.
  kFlash2,  // Exact BF16 FlashAttention-2-style fused kernel.
  kSage2,   // Quantized SageAttention2 kernel.
  kSol,     // Training-free block routing with zeroth-order correction.
  kSolExperimental,  // Experimental SM120 TMA/WMMA pipeline; explicitly opt-in.
  // Pinned deterministic cooperative attention shared by CUDA and Vulkan.
  // This is distinct from kNone: it has no materialized score tensor and does
  // not silently fall back to the legacy blocked reference.
  kExact,
};

inline const char* attention_mode_name(AttentionMode mode) {
  switch (mode) {
    case AttentionMode::kNone: return "none";
    case AttentionMode::kFlash2: return "flash2";
    case AttentionMode::kSage2: return "sage2";
    case AttentionMode::kSol: return "sol";
    case AttentionMode::kSolExperimental: return "sol-experimental";
    case AttentionMode::kExact: return "exact";
  }
  return "unknown";
}

inline bool parse_attention_mode(std::string_view name,
                                 AttentionMode* mode) noexcept {
  if (mode == nullptr) return false;
  if (name == "none") *mode = AttentionMode::kNone;
  else if (name == "flash2") *mode = AttentionMode::kFlash2;
  else if (name == "sage2") *mode = AttentionMode::kSage2;
  else if (name == "sol") *mode = AttentionMode::kSol;
  else if (name == "sol-experimental") *mode = AttentionMode::kSolExperimental;
  else if (name == "exact") *mode = AttentionMode::kExact;
  else return false;
  return true;
}

// Backend capability contract, intentionally separate from orchestration.
// Vulkan has exact attention primitives but still lacks the complete neural
// model runner; accepting kExact here must not be reported as generation
// support by a caller.
inline bool attention_mode_supported(DeviceBackend backend,
                                     AttentionMode mode) noexcept {
  switch (backend) {
    case DeviceBackend::kCuda:
      return mode == AttentionMode::kNone || mode == AttentionMode::kFlash2 ||
             mode == AttentionMode::kSage2 || mode == AttentionMode::kSol ||
             mode == AttentionMode::kSolExperimental || mode == AttentionMode::kExact;
    case DeviceBackend::kVulkan:
      return mode == AttentionMode::kExact;
  }
  return false;
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
