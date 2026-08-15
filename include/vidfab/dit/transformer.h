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

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "vidfab/attention_mode.h"

#include "vidfab/dit/adaln.h"
#include "vidfab/dit/block_cache.h"
#include "vidfab/dit/checkpoint.h"
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
  int timestep_freq_dim = 256;
  int timestep_hidden_dim = 5376;
  int timestep_embed_dim = 2688;

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

  // Frame-banded attention: a video row attends to +/- this many latent frames
  // instead of the whole sequence. 0 is off and is the default.
  //
  // **This is lossy and it changes the sample**, unlike every other knob on this
  // class. Text and audio rows keep global attention, and every video row keeps
  // the text/audio prefix, so the conditioning path is untouched; what it drops
  // is distant video-to-video attention. Set before `prepare_sequence`, which is
  // where the per-query-tile key ranges are built.
  void set_attention_band(int frames);
  int attention_band() const;
  void set_attention_mode(AttentionMode mode);
  AttentionMode attention_mode() const;
  void set_sol_schedule(const SolSchedule& schedule);
  void set_denoise_step(int step);

  // Block-span residual caching (block_cache.h). **Lossy, like the attention
  // band and unlike every other knob here**: it changes the sample.
  //
  // `num_steps` is the schedule length, needed because the terminal step is
  // forced to compute and the class cannot see the loop. Call before
  // `prepare_sequence` — that is where the `[seq, hidden]` delta buffer is
  // sized, and enabling it afterwards would find it unallocated. `forward`
  // checks and throws rather than leaving that to this comment.
  //
  // Cannot be combined with the step cache: see the refusal in main.cpp and the
  // reason in block_cache.h.
  //
  // The decision is driven off `set_denoise_step`, which the loop already
  // calls, rather than pushed in per step: the span boundaries live here
  // anyway, so splitting the two halves of one policy across the loop boundary
  // would only create a way for them to disagree.
  void set_block_cache(const BlockCacheConfig& config, int num_steps);
  const BlockCacheConfig& block_cache_config() const;

  // The span actually in force, after centring and clamping. Invalid when the
  // feature is off. For reporting — a run that says which blocks it skipped is
  // the difference between a reproducible measurement and an anecdote.
  BlockSpan block_cache_span() const;

  // The loaded stack's depth. This, not `config().num_layers`, is what the span
  // is resolved against, and the two are not the same for every architecture —
  // reporting a span out of the configured depth would describe a stack the
  // span was never placed in.
  int num_blocks() const;

  // Spans evaluated and spans reused so far, across every step since `load`.
  int block_cache_computed() const;
  int block_cache_reused() const;
  // `c(t)` — the 8-vector that is the *entire* timestep conditioning for this
  // checkpoint, shared by all 51 AdaLN consumers (spec 3.4). Goes through the
  // configured lookup mode, so it is exactly the vector `build_modulation`
  // expands; reading the table directly would silently pin the mode, which
  // spec 3.5 deliberately leaves as a knob.
  //
  // Public because the step cache outside this class needs it and it is
  // microseconds of host work — the alternative, a fitted proxy for how much
  // the conditioning moved, is what TeaCache has to do on models that still
  // carry a timestep MLP.
  std::array<float, AdaLNTable::kRank> adaln_code(float t) const;

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

  // The rank-8 modulation of one block, evaluated at `timesteps` and copied
  // back to the host as `[6 params][T*3 modality rows][hidden]` — the layout
  // the modulate kernels index, parameter-outer.
  //
  // For verification only, and worth its place in the public surface: spec
  // 3.2's parameter order is the single most consequential index in the model
  // and it cannot be checked from any shape. A wrong assignment still produces
  // finite, plausibly scaled video. What identifies a correct one is the
  // *statistics* — `scale_msa` and `scale_mlp` carry large non-zero means
  // because of the `1 + scale` parameterisation, while the shifts and gates sit
  // on zero — and reading those needs the numbers.
  std::vector<float> debug_modulation(int block_index, const std::vector<float>& timesteps);

  // The cached refiner output, `[L, hidden]` widened to fp32. Verification
  // only: the text stream is invisible in the model's output except through
  // attention, so a bug in it shows up as a few percent of error smeared over
  // every video row and nowhere that localises it.
  std::vector<float> debug_text_cache() const;

  // One labelled snapshot of the residual stream from inside `prepare_text`.
  struct DebugStage {
    std::string label;   // "condition_proj", "attn", "ffn", "final_norm"
    int rows = 0;
    int dim = 0;
    std::vector<float> data;
  };

  // Re-runs `prepare_text` capturing the residual stream at every stage
  // boundary, in order: condition_proj, then attn/ffn for each refiner block,
  // then final_norm.
  //
  // This exists to localise a numerical disagreement to a single operation.
  // Comparing only the *end* of the refiner cannot distinguish "one op is
  // wrong" from "bf16 rounding accumulated everywhere", and those have very
  // different consequences while looking identical from outside.
  std::vector<DebugStage> debug_text_stages(const float* prompt_embeds, int num_tokens);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vidfab::dit
