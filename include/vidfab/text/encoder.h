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
//   - The prompt is tokenised with `add_special_tokens=False`. No BOS, no EOS,
//     no `<|im_start|>`, no system prompt. Spec section 1.2.
//   - Weights are int8 with ConvRot and per-output-channel scales, ~24.4 GB
//     resident. That does not co-exist on one 32 GB card with the 19.3 GB
//     transformer, so `encode` is expected to run once and then `unload`
//     before the denoiser loads. The pipeline enforces the ordering.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vidfab/safetensors.h"
#include "vidfab/text/tokenizer.h"

namespace vidfab::text {

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
};

// `[num_tokens, hidden_size]` fp32, host side. A few thousand rows at most, so
// there is no reason to keep it on the device between stages.
struct PromptEmbedding {
  int num_tokens = 0;
  int hidden_size = 5120;
  std::vector<float> data;
};

class Encoder {
 public:
  Encoder();
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  // Uploads the 50 decoder layers to the device. Skips `visual.*` entirely.
  // The embedding table stays on the host: only `L` of its 151936 rows are
  // ever read, so gathering on the host saves 1.6 GB of device memory.
  void load(const SafeTensors& checkpoint, const EncoderConfig& config = {});

  const EncoderConfig& config() const;
  size_t weight_bytes() const;

  // Frees all device weights. Call before loading the transformer.
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

}  // namespace vidfab::text
