// Qwen3-VL-32B as MiniMax H3's text conditioner.
//
// H3 does not use a text encoder in the usual sense. It runs the *decoder*
// stack of a 32B VLM for exactly 50 layers and takes the raw residual stream —
// no final norm, no LM head, no chat template, no special tokens. The shipped
// checkpoint is already truncated: its metadata reads
// `{"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"}`
// and it contains `model.layers.0` through `.49` and nothing after.
//
// The `visual.*` tower is present in the file but is **not loaded**: it is
// reached only by the fl2va keyframe path, and this port is t2va only.
//
// Consequences that shape the interface:
//
//   - Output is `[L, 5120]` fp32, the residual stream after layer 49 (0-based).
//     No final norm is applied — not even an identity one. docs/text_encoder_spec.md
//     section 1.4 is emphatic: a "helpfully completed" model reintroduces
//     exactly the bug encoders.py:142-149 exists to prevent, and it produces
//     well-scaled output. Sanity check: the rows' RMS is in the hundreds, not 1.
//   - The prompt is tokenised with `add_special_tokens=False`. No BOS, no EOS,
//     no `<|im_start|>`, no system prompt. Spec section 1.2.
//   - Attention is strictly CAUSAL (spec section 3). The reference's
//     `attention_mask = ones_like(input_ids)` is a *padding* mask meaning
//     "nothing is padded", not a request for bidirectional attention.
//   - Weights are int8 with ConvRot and per-output-channel scales, ~24.4 GB
//     resident. That does not co-exist on one 32 GB card with the 19.3 GB
//     transformer, so `encode` is expected to run once and then `unload`
//     before the denoiser loads. The pipeline enforces the ordering. The
//     streaming residency mode below removes even that constraint.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vidfab/dtype.h"
#include "vidfab/safetensors.h"
#include "vidfab/text/tokenizer.h"

namespace vidfab::text {

// Where the 24.4 GB of int8 weights live between `load` and `encode`.
//
// Both modes compute the same thing; only the schedule differs. Streaming
// exists because text encoding runs *once per request* while the transformer
// that follows needs 19.3 GB in the same process, so a conditioner that fits in
// ~2 GB is worth having even though it is slower.
enum class Residency {
  // Free VRAM decides: resident when the card has room for the weights plus
  // working set, streaming otherwise. The default.
  kAuto,
  // All 50 layers uploaded once. Fastest encode, ~24.4 GB held until unload().
  kResident,
  // Each layer staged through pinned host memory and uploaded just before use,
  // double-buffered against compute. ~1 GB of layer buffers.
  kStreaming,
};

struct EncoderConfig {
  int hidden_size = 5120;
  int num_layers = 50;  // the checkpoint is pre-truncated; this is the whole file
  int num_attention_heads = 64;
  int num_key_value_heads = 8;  // grouped-query, 8 kv heads
  int head_dim = 128;
  int intermediate_size = 25600;
  float rms_norm_eps = 1e-6f;  // note: 1e-6, unlike the transformer's 1e-5
  float rope_theta = 5.0e6f;
  int vocab_size = 151936;

  Residency residency = Residency::kAuto;

  // The reference truncates nothing (spec section 10.6), so neither do we: a
  // silently shortened prompt is worse than a hard error. Buffers are sized for
  // this bound; raising it is safe but costs O(L^2) attention.
  int max_prompt_tokens = 8192;
};

// `[num_tokens, hidden_size]` fp32, host side. A few thousand rows at most, so
// there is no reason to keep it on the device between stages.
struct PromptEmbedding {
  int num_tokens = 0;
  int hidden_size = 5120;
  std::vector<float> data;
};

// Measured, not estimated. Reported so a caller can choose a residency mode on
// evidence and so the two modes can be compared without external tooling.
struct EncoderStats {
  double load_seconds = 0.0;
  double last_encode_seconds = 0.0;
  int last_num_tokens = 0;

  size_t weight_bytes = 0;      // device bytes held by the 50 layers
  size_t workspace_bytes = 0;   // device arena high-water mark
  size_t activation_bytes = 0;  // persistent per-encode buffers
  // weights + workspace + activations. Nothing is freed during encode, so this
  // is the peak, not an average.
  size_t peak_device_bytes = 0;
  size_t host_pinned_bytes = 0;
};

class Encoder {
 public:
  Encoder();
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  // Validates the checkpoint and prepares the 50 decoder layers. Skips
  // `visual.*` entirely. The embedding table stays on the host: only `L` of its
  // 151936 rows are ever read, so gathering on the host saves 1.6 GB of device
  // memory.
  //
  // `checkpoint` must outlive the encoder: the embedding table is read from its
  // mapping at every encode, and in kStreaming mode so are the layer weights.
  void load(const SafeTensors& checkpoint, const EncoderConfig& config = {});

  const EncoderConfig& config() const;
  size_t weight_bytes() const;
  Residency residency() const;  // the mode actually chosen, never kAuto
  const EncoderStats& stats() const;

  // Frees all device memory. Call before loading the transformer.
  void unload();

  // Runs the 50 layers over `token_ids` and returns the unnormalised residual
  // stream. `token_ids` must come from `Tokenizer::encode` with no special
  // tokens added.
  PromptEmbedding encode(const std::vector<int32_t>& token_ids);

  // Convenience: tokenise then encode.
  PromptEmbedding encode(const Tokenizer& tokenizer, const std::string& prompt);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// --- host-side pieces, exposed for testing ----------------------------------
//
// Everything below is CUDA-free and lives in src/text/encoder.cpp. They are
// public because each one is a place where a silent error is possible and a
// direct test is cheap.

// The 18 tensors of one decoder layer that this port actually loads. The seven
// `comfy_quant` blobs are validated and discarded; nothing else exists under
// `model.layers.i.` (spec section 8.3).
enum class LayerTensor {
  kQWeight,
  kQScale,
  kKWeight,
  kKScale,
  kVWeight,
  kVScale,
  kOWeight,
  kOScale,
  kGateWeight,
  kGateScale,
  kUpWeight,
  kUpScale,
  kDownWeight,
  kDownScale,
  kInputLayerNorm,
  kPostAttentionLayerNorm,
  kQNorm,
  kKNorm,
  kCount,
};

constexpr int kLayerTensorCount = static_cast<int>(LayerTensor::kCount);

struct TensorSpec {
  const char* suffix;  // appended to "model.layers.<i>."
  DType dtype;
  int64_t dim0;
  int64_t dim1;  // 0 marks a 1-D tensor
};

// Expected name, dtype and shape of each of the 18, resolved from `config`.
TensorSpec layer_tensor_spec(const EncoderConfig& config, LayerTensor which);

// Byte offsets of one layer's tensors inside a single packed blob, in the order
// of `LayerTensor`. Packing a layer into one contiguous block is what makes the
// resident and streaming paths the same code: one host memcpy, one H2D copy.
struct LayerLayout {
  size_t offset[kLayerTensorCount] = {};
  size_t bytes[kLayerTensorCount] = {};
  size_t total_bytes = 0;
};

LayerLayout make_layer_layout(const EncoderConfig& config);

// Throws std::runtime_error naming the offending tensor on the first problem.
// Checks the tensor count, every shape and dtype of every layer, that all 350
// quantised weights are int8 + ConvRot at group 256, and that no `model.norm`
// or `lm_head` is present — the last because their absence is what makes
// "output of the last layer, unnormalised" the right answer (spec section 1.4).
void validate_checkpoint(const SafeTensors& checkpoint, const EncoderConfig& config);

// Copies one layer's 18 tensors out of the mapping into `dst`, which must hold
// `layout.total_bytes`.
void pack_layer(const SafeTensors& checkpoint, const EncoderConfig& config, int layer,
                const LayerLayout& layout, uint8_t* dst);

// inv_freq[j] = theta^(-2j/head_dim), j in [0, head_dim/2). For Qwen3-VL that
// is 5e6^(-j/64) over j in [0, 64) — spec section 2.5.
std::vector<float> rope_inv_freq(int head_dim, float theta);

// `cos` and `sin` come out `[num_tokens, head_dim]` fp32 with the half period
// duplicated, which is the layout `launch_rope_neox` and the reference's
// `cat((freqs, freqs), -1)` both expect.
//
// There are no 3-D positions here. Qwen3-VL's interleaved mrope degenerates
// *bitwise* to 1-D RoPE for a text-only prompt because all three position axes
// carry the token index (spec section 2.3), so `mrope_section` and
// `mrope_interleaved` have no effect and are deliberately not implemented.
// **That degeneracy is a property of the input, not of the model.** Anyone
// adding fl2va keyframes must implement the real 3-D interleaved mrope here;
// spec section 2.2 has the frequency-slot table.
void build_rope_tables(int num_tokens, const std::vector<float>& inv_freq, std::vector<float>& cos,
                       std::vector<float>& sin);

// Gathers `ids.size()` rows of a BF16 `[vocab, hidden]` embedding table into
// `out` as raw bf16 bit patterns. Host-side because only `L` of 151936 rows are
// ever read; uploading the table would cost 1.56 GB for nothing.
void gather_embedding_rows(const TensorView& embed, const std::vector<int32_t>& ids,
                           std::vector<uint16_t>& out);

}  // namespace vidfab::text

// --- device-side pieces -----------------------------------------------------
//
// Declared here rather than in a private header because this module owns no
// header of its own; the guard keeps CUDA types out of every host translation
// unit that includes this file. Defined in src/cuda/encoder_kernels.cu.
#if defined(__CUDACC__)

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/workspace.cuh"

namespace vidfab::text {

// Causal grouped-query attention.
//
// `vidfab/cuda/attention.cuh` deliberately has no mask — the H3 DiT has none —
// so the encoder brings its own. Same blocked online-softmax algorithm, with
// key blocks past the diagonal skipped entirely and the diagonal block masked
// per row.
struct CausalAttentionConfig {
  int seq_len = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  float scale = 0.0f;     // 0 selects 1/sqrt(head_dim)
  int query_block = 0;    // 0 selects the default; results must not depend on it
};

float causal_attention_scale(const CausalAttentionConfig& cfg);
size_t causal_attention_workspace_bytes(const CausalAttentionConfig& cfg);

// out[seq, num_heads*head_dim] = softmax(q k^T * scale + causal) v.
// `k` and `v` are `[seq, num_kv_heads*head_dim]`; query head `h` reads kv head
// `h / (num_heads / num_kv_heads)` — contiguous blocks, not interleaved
// (spec section 4.2). `out` may not alias q, k or v.
void causal_attention_forward(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                              const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                              const CausalAttentionConfig& cfg, vidfab::cuda::Workspace& ws);

// out[i] = silu(gate[i]) * up[i]. Qwen3-VL ships gate_proj and up_proj as
// separate tensors, so unlike the H3 DiT's fused fc1 there is no halves
// ambiguity — but the SiLU still goes on the gate, never on `up`.
// `out` may alias `gate` or `up`.
void launch_swiglu_split(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                         size_t n, cudaStream_t stream);

// x[i] += branch[i], accumulated in fp32 and rounded once. The residual is
// never normalised, gated or scaled (spec section 4.4).
void launch_residual_add(__nv_bfloat16* x, const __nv_bfloat16* branch, size_t n,
                         cudaStream_t stream);

// One decoder layer's weights as device pointers. Nothing is owned here.
struct LayerWeights {
  vidfab::cuda::QuantWeight q_proj;
  vidfab::cuda::QuantWeight k_proj;
  vidfab::cuda::QuantWeight v_proj;
  vidfab::cuda::QuantWeight o_proj;
  vidfab::cuda::QuantWeight gate_proj;
  vidfab::cuda::QuantWeight up_proj;
  vidfab::cuda::QuantWeight down_proj;
  const __nv_bfloat16* input_layernorm = nullptr;
  const __nv_bfloat16* post_attention_layernorm = nullptr;
  const __nv_bfloat16* q_norm = nullptr;
  const __nv_bfloat16* k_norm = nullptr;
};

struct LayerDims {
  int num_tokens = 0;
  int hidden = 5120;
  int num_heads = 64;
  int num_kv_heads = 8;
  int head_dim = 128;
  int intermediate = 25600;
  float rms_norm_eps = 1e-6f;
  int attn_query_block = 0;
};

// Interprets one packed layer blob (see `LayerLayout`) as device pointers.
LayerWeights layer_weights_from_blob(const uint8_t* base, const LayerLayout& layout,
                                     const EncoderConfig& config);

size_t layer_workspace_bytes(const LayerDims& dims);

// x <- x + Attention(RMSNorm_in(x));  x <- x + MLP(RMSNorm_post(x))
//
// In place on the `[num_tokens, hidden]` bf16 residual stream. `cos`/`sin` are
// `[num_tokens, head_dim]` fp32, built once per request and shared by every
// layer and by both q and k.
void encoder_layer_forward(cublasHandle_t handle, cudaStream_t stream,
                           vidfab::cuda::LinearRunner& linear, const LayerWeights& w,
                           const LayerDims& dims, const float* cos, const float* sin,
                           __nv_bfloat16* x, vidfab::cuda::Workspace& ws);

}  // namespace vidfab::text

#endif  // __CUDACC__
