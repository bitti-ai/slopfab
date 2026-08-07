// The rank-8 AdaLN timestep code.
//
// Our checkpoint does not ship the reference's AdaLN. The released
// `transformer_minimax_h3.py` computes
//
//     m(t) = W_full @ SiLU(time_embedder(Timesteps(t))) + b,   W_full: 96768 x 2688
//
// but the pruned checkpoint deleted `time_embedder.*` entirely and replaced the
// whole front end with a table lookup producing an 8-vector:
//
//     m(t) = W_8 @ c(t) + b,   W_8: 96768 x 8,   c(t) = adaln_t_table[row(t)]
//
// This is the 13 billion parameters `ref/README.md:141` says can be precomputed
// and cached: 50 x 96768 x 2688 = 13.0e9. One table, 51 consumers — the 50
// blocks plus `final_layer`.
//
// **Do not apply SiLU to a table row.** The activation sits *before* the
// projection in the reference, so it is already baked into the table.
//
// This header owns the lookup only. The 51 GEMVs that turn `c(t)` into
// modulation vectors happen on the device, because that is where `W_8` lives.
//
// What the table actually is, measured on the real tensor (see
// docs/transformer_spec.md section 10.1): a **uniform grid of 1024 intervals
// over [0, 1]**, smooth in the row index, whose per-column ranges decay
// geometrically (0.870, 0.315, 0.127, 0.103, 0.0102, 0.00222, 0.00118,
// 0.000653) — the singular-value profile of a truncated SVD, which settles the
// question in section 10.2 of whether the factorisation is exact.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "vidfab/safetensors.h"

namespace vidfab::dit {

// How a continuous timestep selects a row.
enum class AdaLNLookup {
  // Default. j = floor(t*1024), blend with j+1.
  //
  // Settled by measurement, not preference: dropping every other row and
  // interpolating the omitted ones back recovers them to 2.4e-5 absolute,
  // which is at worst 0.2% of a column's range. Nearest-neighbour incurs up
  // to half a grid step, 1.5-3.5% of range on the small components — one to
  // two orders of magnitude worse.
  kLinear,
  // c = table[round(t*1024)]. Kept so the choice above stays falsifiable.
  kNearest,
  // Linear, but indexing by sigma = 1 - t instead of by t.
  //
  // The grid is uniform, so "over t" and "over sigma" differ only by reversing
  // the row order — a single bit, and the one thing about this table that
  // could not be recovered from the local files. Settling it needs
  // SiLU(time_embedder(t)) rebuilt from the ORIGINAL MiniMaxAI/MiniMax-H3
  // shards, and the pruned checkpoint deleted exactly those weights.
  //
  // Evidence favours the forward direction and it is the default: row 0 has
  // the larger norm (0.5387 against 0.4252) and the larger step to its
  // neighbour, consistent with Timesteps(0) being the distinguished point of
  // the sinusoid manifold — with flip_sin_to_cos, s(0) is exactly
  // (1...1, 0...0), all 128 cosines maximal and all 128 sines zero.
  //
  // If the direction is wrong the symptom is specific: the sample degrades
  // along the denoising trajectory instead of improving. Flip to this and
  // re-run.
  kLinearReversed,
};

const char* adaln_lookup_name(AdaLNLookup mode);

class AdaLNTable {
 public:
  static constexpr int kRows = 1025;  // 2^10 + 1: a dyadic grid, endpoints included
  static constexpr int kRank = 8;

  // Reads `adaln_t_table` and validates its shape and finiteness. Throws with
  // a specific message if the tensor is missing or the wrong shape.
  void load(const SafeTensors& checkpoint);

  bool loaded() const { return !data_.empty(); }

  // The 8-vector for a timestep in [0, 1]. Values outside the range clamp to
  // the endpoints rather than extrapolating: the schedule never leaves [0, 1],
  // so an out-of-range t is a bug upstream and silently extrapolating an SVD
  // coefficient would hide it.
  std::array<float, kRank> lookup(float t, AdaLNLookup mode = AdaLNLookup::kLinear) const;

  // Raw row access, for verification.
  const float* row(int index) const;

 private:
  std::vector<float> data_;  // [1025, 8] fp32
};

// Conventional MiniMax-H3 timestep front end used by the unpruned Ref2VA
// transformer. Timesteps are unscaled values in [0, 1]. The sinusoid is the
// diffusers Timesteps(freq_dim, flip_sin_to_cos=true,
// downscale_freq_shift=0) layout: all cosine channels, then all sine channels.
std::vector<float> minimax_h3_timestep_sinusoid(float timestep, int freq_dim = 256);

// Host reference and loader contract for
//   proj_out(SiLU(proj_in(time_proj(t)))).
// The real model dimensions are 256 -> 5376 -> 2688; dimensions are arguments
// so small exact fixtures can exercise the same arithmetic in unit tests.
class FullAdaLNTimestepEmbedding {
 public:
  void load(const SafeTensors& checkpoint, int freq_dim = 256, int hidden_dim = 5376,
            int output_dim = 2688);
  bool loaded() const { return !proj_in_weight_.empty(); }
  int freq_dim() const { return freq_dim_; }
  int hidden_dim() const { return hidden_dim_; }
  int output_dim() const { return output_dim_; }

  std::vector<float> forward(float timestep) const;
  std::vector<float> forward(const std::vector<float>& timesteps) const;

 private:
  int freq_dim_ = 0;
  int hidden_dim_ = 0;
  int output_dim_ = 0;
  std::vector<float> proj_in_weight_, proj_in_bias_;
  std::vector<float> proj_out_weight_, proj_out_bias_;
};

}  // namespace vidfab::dit
