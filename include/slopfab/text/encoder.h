// Qwen3-VL-32B as MiniMax H3's text conditioner.
//
// H3 does not use a text encoder in the usual sense. It runs the *decoder*
// stack of a 32B VLM for exactly 50 layers and takes the raw residual stream —
// no final norm, no LM head, no chat template, no special tokens. The shipped
// checkpoint is already truncated: its metadata reads
// `{"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"}`
// and it contains `model.layers.0` through `.49` and nothing after.
//
// Ref2VA also runs the checkpoint's complete `visual.*` tower, replaces the
// image-pad embeddings, and injects its three DeepStack outputs at decoder
// vision layers 8, 16 and 24. Canonical decoder injection is after decoder
// layers 0, 1 and 2 respectively; those similarly numbered stages must not be
// conflated.
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
//   - Two builds of this model ship, and both are supported. The int8+ConvRot
//     one is ~24.4 GB resident; the nvfp4+AWQ one is ~12.6 GB. Neither is a
//     flag: `WeightFormat` below is read out of the file. 24.4 GB does not
//     co-exist on one 32 GB card with the 19.3 GB fp8 transformer, so `encode`
//     is expected to run once and then `unload` before the denoiser loads. The
//     pipeline enforces the ordering. The streaming residency mode below
//     removes even that constraint.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "slopfab/dtype.h"
#include "slopfab/text/limits.h"
#include "slopfab/safetensors.h"
#include "slopfab/text/tokenizer.h"
#include "slopfab/text/qwen_vision.h"

namespace slopfab::text {

// Where the layer weights live between `load` and `encode`.
//
// Both modes compute the same thing; only the schedule differs. Streaming
// exists because text encoding runs *once per request* while the transformer
// that follows needs 19.3 GB in the same process, so a conditioner that fits in
// ~2 GB is worth having even though it is slower. The nvfp4 build halves the
// resident figure, which changes when that trade is worth making but not the
// fact that it exists.
enum class Residency {
  // Free VRAM decides: resident when the card has room for the weights plus
  // working set, streaming otherwise. The default.
  kAuto,
  // All 50 layers uploaded once. Fastest encode; 24.4 GB held until unload()
  // on the int8 build, 12.6 GB on the nvfp4 one. load() also reserves the
  // maximum-request graph footprint and a driver budget; explicit resident
  // mode fails synchronously when that complete peak does not fit.
  kResident,
  // Each layer staged through pinned host memory and uploaded just before use,
  // double-buffered against compute. ~1 GB of layer buffers.
  kStreaming,
};

// How the 350 quantised linears of this checkpoint are stored. Detected from
// the file's own `comfy_quant` blobs, never from a flag: the two builds differ
// in tensor count, dtype and shape, so a mismatch is caught at validation
// rather than becoming wrong numbers.
enum class WeightFormat {
  // Read it out of the checkpoint. The default, and what `load` always uses.
  kAuto,
  // qwen3vl_32b_int8_convrot: int8 + per-output-channel F32 scale, ConvRot on
  // the contraction axis at group 256. 25 tensors per layer.
  kI8ConvRot,
  // qwen3vl_32b_minimax_h3_nvfp4_awq: E2M1 nibbles, an e4m3 scale per 16
  // contracted elements, one F32 global scale per tensor, and an AWQ
  // per-input-channel activation scale on the two linears whose input does not
  // come straight from a norm. Not rotated. 34 tensors per layer.
  kNVFP4Awq,
};

// The shipped path preserves the original high-throughput cuBLAS arithmetic.
// Exact uses the canonical scalar/materialized layer contract shared with the
// Vulkan conditioner. It is opt-in so existing callers do not silently trade
// throughput for cross-backend byte identity.
enum class EncoderArithmetic {
  kShipped,
  kExact,
};

// Reads the format out of `model.layers.0.*.comfy_quant`. Never returns kAuto;
// throws if the blobs do not describe either shipped build.
WeightFormat detect_weight_format(const SafeTensors& checkpoint);

// Validates that a reference request has the complete Qwen visual tower rather
// than silently running the ordinary text-only conditioner.
void require_reference_vision_support(const SafeTensors& checkpoint, size_t reference_count);

struct EncoderConfig {
  // Resolved from the checkpoint by `load`. Left kAuto here so that nothing but
  // the file can decide it; the host helpers below require it resolved.
  WeightFormat format = WeightFormat::kAuto;

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
  EncoderArithmetic arithmetic = EncoderArithmetic::kShipped;

  // The reference truncates nothing (spec section 10.6), so neither do we: a
  // silently shortened prompt is worse than a hard error. Buffers are sized for
  // this bound; raising it is safe but costs O(L^2) attention.
  int max_prompt_tokens = kMaxPromptTokens;
};

// The semantic output contract is independent of weight quantization and backend
// kernel support. Metadata selects only implemented conditioner families.
struct ConditionerDescriptor {
  int version = 1;
  std::string family = "qwen3_vl";
  std::string tokenizer = "qwen_byte_bpe";
  int output_width = 5120;
  int output_layer = 49;
  bool final_normalization = false;
  bool vision = false;
  bool explicit_metadata = false;
  std::string fingerprint() const;
};
ConditionerDescriptor resolve_conditioner_descriptor(const SafeTensors& checkpoint,
                                                     const EncoderConfig& config = {});

// `[num_tokens, hidden_size]` fp32, host side. A few thousand rows at most, so
// there is no reason to keep it on the device between stages.
struct PromptEmbedding {
  int num_tokens = 0;
  int hidden_size = 5120;
  std::vector<float> data;
  // H3 AdaLN modality tag per Qwen output row: text=1, video=0. For each
  // reference, vision_start, every image_pad, and vision_end are video.
  std::vector<int32_t> modality_tags;
};

// Optional exact-mode diagnostic. All 50 BF16 residual boundaries are copied
// device-to-device while the graph runs and downloaded together after layer
// 49, so enabling it does not introduce a per-layer host seam.
struct EncoderTrace {
  int num_tokens = 0;
  int hidden_size = 5120;
  std::vector<uint16_t> layer_residual_bf16;
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
  Residency residency() const;    // the mode actually chosen, never kAuto
  WeightFormat format() const;    // the format detected in the file, never kAuto
  const EncoderStats& stats() const;

  // Frees all device memory. Call before loading the transformer.
  void unload();

  // Runs the 50 layers over `token_ids` and returns the unnormalised residual
  // stream. `token_ids` must come from `Tokenizer::encode` with no special
  // tokens added.
  PromptEmbedding encode(const std::vector<int32_t>& token_ids,
                         EncoderTrace* trace = nullptr);
  PromptEmbedding encode(const std::vector<int32_t>& token_ids,
                         const std::vector<QwenPixelValues>& images,
                         EncoderTrace* trace = nullptr);

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

// The per-layer tensors this port loads into the device blob, as the union over
// both formats. Which of them exist, and with what dtype and shape, depends on
// `EncoderConfig::format`; `layer_tensor_spec` returns an absent spec for the
// ones a given format does not have, and those occupy no bytes in the blob.
//
// The seven `comfy_quant` blobs are validated and discarded. The seven nvfp4
// `weight_scale_2` values are single floats read at load time into
// `QuantWeight::global_scale`, so they are not in this list either — a scalar
// per tensor is not worth a device allocation (spec section 8.3).
enum class LayerTensor {
  kQWeight,
  kQScale,
  kKWeight,
  kKScale,
  kVWeight,
  kVScale,
  kOWeight,
  kOScale,
  kOPreQuantScale,  // nvfp4 only
  kGateWeight,
  kGateScale,
  kUpWeight,
  kUpScale,
  kDownWeight,
  kDownScale,
  kDownPreQuantScale,  // nvfp4 only
  kInputLayerNorm,
  kPostAttentionLayerNorm,
  kQNorm,
  kKNorm,
  kCount,
};

constexpr int kLayerTensorCount = static_cast<int>(LayerTensor::kCount);

struct TensorSpec {
  const char* suffix;  // appended to "model.layers.<i>."; null when absent
  DType dtype;
  int64_t dim0;
  int64_t dim1;  // 0 marks a 1-D tensor

  // False for a tensor this format does not ship. Callers must check: an
  // absent entry has no name to look up and no bytes in the blob.
  bool present() const { return suffix != nullptr; }
};

// Expected name, dtype and shape of one per-layer tensor, resolved from
// `config` — which must have a resolved `format`.
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
// quantised weights declare the format `config.format` names, and that no
// `model.norm` or `lm_head` is present — the last because their absence is what
// makes "output of the last layer, unnormalised" the right answer (spec
// section 1.4).
//
// `config.format` may be kAuto, in which case it is detected first; pass a
// resolved one to assert that the file is the build you expect.
void validate_checkpoint(const SafeTensors& checkpoint, const EncoderConfig& config);

// The seven `weight_scale_2` values of one layer, in the order of the seven
// quantised linears (q, k, v, o, gate, up, down). Single floats, so they are
// read straight into `QuantWeight::global_scale` rather than into the blob.
// All ones for a format that has no such scale.
struct LayerGlobalScales {
  float value[7] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
};

LayerGlobalScales read_global_scales(const SafeTensors& checkpoint, const EncoderConfig& config,
                                     int layer);

// Copies one layer's present tensors out of the mapping into `dst`, which must
// hold `layout.total_bytes`.
void pack_layer(const SafeTensors& checkpoint, const EncoderConfig& config, int layer,
                const LayerLayout& layout, uint8_t* dst);

// Issues one layer's present tensors as asynchronous DMAs straight out of the
// checkpoint mapping into `dst`, with no host copy. Only valid once the
// mapping has been page-locked; otherwise the copies degrade to synchronous
// staged transfers inside the driver and the point is lost.
void upload_layer_direct(const SafeTensors& checkpoint, const EncoderConfig& config, int layer,
                         const LayerLayout& layout, uint8_t* dst, void* stream);

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

// Gathers `ids.size()` rows of a `[vocab, hidden]` embedding table into `out`
// as raw bf16 bit patterns. Host-side because only `L` of 151936 rows are ever
// read; uploading the table would cost 1.56 GB on the bf16 build and 0.78 GB on
// the int8 one for nothing.
//
// Two storage forms, and which one is in front of you is not implied by the
// checkpoint's *weight* format: the nvfp4 build stores this table as I8 with an
// F32 per-row `weight_scale` while its linears are E2M1, and the int8+ConvRot
// build stores it as plain BF16 while its linears are int8. So `weight_scale`
// is passed separately and must be non-null exactly when `embed` is I8. The
// row scale multiplies — verified against the bf16 table of the other build at
// 0.94% relative L2 over a 153-row sample, where dividing is off by seven
// orders of magnitude.
void gather_embedding_rows(const TensorView& embed, const TensorView* weight_scale,
                           const std::vector<int32_t>& ids, std::vector<uint16_t>& out);

}  // namespace slopfab::text

// --- device-side pieces -----------------------------------------------------
//
// Declared here rather than in a private header because this module owns no
// header of its own; the guard keeps CUDA types out of every host translation
// unit that includes this file. Defined in src/cuda/encoder_kernels.cu.
#if defined(__CUDACC__)

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "slopfab/cuda/linear.cuh"
#include "slopfab/cuda/workspace.cuh"

namespace slopfab::text {

// Causal grouped-query attention.
//
// `slopfab/cuda/attention.cuh` deliberately has no mask — the H3 DiT has none —
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
                              const CausalAttentionConfig& cfg, slopfab::cuda::Workspace& ws);

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

// Canonical exact-mode variants shared with the Vulkan text-stage authority.
// Inputs are canonicalized at the BF16 boundary (subnormals -> signed zero,
// all NaNs -> one quiet NaN) and results use one RNE BF16 conversion.
void launch_swiglu_split_exact(const __nv_bfloat16* gate,
                               const __nv_bfloat16* up,
                               __nv_bfloat16* out, size_t n,
                               cudaStream_t stream);
void launch_residual_add_exact(__nv_bfloat16* x,
                               const __nv_bfloat16* branch, size_t n,
                               cudaStream_t stream);

// One decoder layer's weights as device pointers. Nothing is owned here.
struct LayerWeights {
  slopfab::cuda::QuantWeight q_proj;
  slopfab::cuda::QuantWeight k_proj;
  slopfab::cuda::QuantWeight v_proj;
  slopfab::cuda::QuantWeight o_proj;
  slopfab::cuda::QuantWeight gate_proj;
  slopfab::cuda::QuantWeight up_proj;
  slopfab::cuda::QuantWeight down_proj;
  const __nv_bfloat16* input_layernorm = nullptr;
  const __nv_bfloat16* post_attention_layernorm = nullptr;
  const __nv_bfloat16* q_norm = nullptr;
  const __nv_bfloat16* k_norm = nullptr;
};

struct LayerDims {
  // Only `layer_workspace_bytes` needs this: the forward pass dispatches on
  // each `QuantWeight::format` instead, so a wrong value here is a sizing bug
  // rather than an arithmetic one.
  WeightFormat format = WeightFormat::kI8ConvRot;
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
// `globals` supplies the seven nvfp4 `weight_scale_2` values; it is ignored for
// a format that has none.
LayerWeights layer_weights_from_blob(const uint8_t* base, const LayerLayout& layout,
                                     const EncoderConfig& config,
                                     const LayerGlobalScales& globals = {});

size_t layer_workspace_bytes(const LayerDims& dims);

// x <- x + Attention(RMSNorm_in(x));  x <- x + MLP(RMSNorm_post(x))
//
// In place on the `[num_tokens, hidden]` bf16 residual stream. `cos`/`sin` are
// `[num_tokens, head_dim]` fp32, built once per request and shared by every
// layer and by both q and k.
void encoder_layer_forward(cublasHandle_t handle, cudaStream_t stream,
                           slopfab::cuda::LinearRunner& linear, const LayerWeights& w,
                           const LayerDims& dims, const float* cos, const float* sin,
                           __nv_bfloat16* x, slopfab::cuda::Workspace& ws);

struct ExactLayerTaps {
  __nv_bfloat16* input_norm = nullptr;
  __nv_bfloat16* query = nullptr;
  __nv_bfloat16* key = nullptr;
  __nv_bfloat16* value = nullptr;
  __nv_bfloat16* attention = nullptr;
  __nv_bfloat16* attention_residual = nullptr;
  __nv_bfloat16* post_attention_norm = nullptr;
  __nv_bfloat16* gate = nullptr;
  __nv_bfloat16* up = nullptr;
  __nv_bfloat16* activation = nullptr;
  __nv_bfloat16* final_residual = nullptr;
};

// Canonical exact layer used as the CUDA authority for the device-resident
// Vulkan stage. It sequentially materializes every compressed projection into
// one caller-owned dense slot and never invokes cuBLAS/vendor attention.
size_t exact_layer_workspace_bytes(const LayerWeights& weights,
                                   const LayerDims& dims);
void encoder_layer_forward_exact(cudaStream_t stream, const LayerWeights& weights,
                                 const LayerDims& dims, const float* cosine,
                                 const float* sine, __nv_bfloat16* tokens,
                                 slopfab::cuda::Workspace& workspace,
                                 const ExactLayerTaps* taps = nullptr);

}  // namespace slopfab::text

#endif  // __CUDACC__
