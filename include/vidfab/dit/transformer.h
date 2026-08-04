// The 33B MiniMax H3 omni transformer, t2va path.
//
// 50 pre-norm blocks at width 5376 that widen to 7168 inside attention, a
// 2-block token refiner on the text stream, and a rank-8 factorised AdaLN.
// Fully specified in docs/transformer_spec.md; the four things most likely to
// go silently wrong are:
//
//   1. The AdaLN is *not* the one in the released reference. Our checkpoint
//      replaced the timestep MLP with `adaln_t_table [1025, 8]` and contracted
//      every modulation projection onto that 8-dim basis. Do not apply SiLU to
//      the table rows — the activation is already baked in (spec 3.4).
//   2. `mlp.fc2` ships no `input_scale` and is tagged
//      `"full_precision_matrix_mult": true`. It must not take an fp8 GEMM.
//   3. `qkv_proj` is contiguous `[Wq; Wk; Wv]`, already de-interleaved.
//      Verified on blocks 0, 30 and 35 (spec 10.3).
//   4. QK-norm runs *before* RoPE, and `mlp.fc1`'s first half is the gate.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vidfab/dit/adaln.h"
#include "vidfab/dit/packing.h"
#include "vidfab/safetensors.h"

namespace vidfab::dit {

struct TransformerConfig {
  int hidden_size = 5376;
  int num_layers = 50;
  int num_attention_heads = 56;
  int attention_head_dim = 128;  // inner_dim = 56*128 = 7168, wider than hidden
  int ffn_dim = 14336;
  int in_channels = 24;
  int audio_in_channels = 32;
  int text_dim = 5120;
  int num_refiner_layers = 2;
  int rope_freq_dim = 16;
  float rope_theta = 10000.0f;
  float norm_eps = 1e-5f;
  int adaln_rank = 8;
  int adaln_table_rows = 1025;

  int inner_dim() const { return num_attention_heads * attention_head_dim; }
  int video_patch_dim() const { return in_channels * 4; }  // patch (1,2,2)
};

class Transformer {
 public:
  Transformer();
  ~Transformer();
  Transformer(const Transformer&) = delete;
  Transformer& operator=(const Transformer&) = delete;

  void load(const SafeTensors& checkpoint, const TransformerConfig& config = {});
  const TransformerConfig& config() const;
  size_t weight_bytes() const;
  void unload();

  void set_adaln_lookup(AdaLNLookup mode);
  AdaLNLookup adaln_lookup() const;

  // Runs the token refiner over the conditioning embedding and caches the
  // result. Position-agnostic and timestep-independent, so it runs once per
  // request rather than once per step — the reference re-runs it every step
  // only because its forward is stateless.
  //
  // `prompt_embeds` is `[L, 5120]` fp32 from the text encoder.
  void prepare_text(const float* prompt_embeds, int num_tokens);

  // Uploads the packed-sequence geometry: rotary tables, index tensors, tags.
  // Once per request.
  void prepare_sequence(const SequenceLayout& layout, const PackedIndices& indices,
                        const std::vector<double>& position_ids);

  // One denoising evaluation. `video_latents` is `[V, 96]` and `audio_latents`
  // is `[Sa, 32]`, both fp32 host buffers; the velocity predictions come back
  // in the same shapes.
  //
  // Exactly one forward pass per step: the released checkpoints are
  // CFG-distilled, there is no guider and no negative prompt.
  void forward(const float* video_latents, const float* audio_latents,
               const RowTimesteps& row_timesteps, float* video_velocity, float* audio_velocity);

  // Bytes of device memory the forward pass needs on top of the weights, for a
  // given sequence length. Lets the pipeline fail fast with a useful message
  // rather than mid-run.
  size_t activation_bytes(const SequenceLayout& layout) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vidfab::dit
