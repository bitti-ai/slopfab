#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vidfab/safetensors.h"
#include "vidfab/text/encoder.h"
#include "vidfab/vulkan/tensor.h"

namespace vidfab::vulkan {

struct ExactQwenTextEncoderStats {
  double load_seconds = 0.0;
  double last_encode_seconds = 0.0;
  uint32_t last_num_tokens = 0;
  uint64_t max_layer_weight_bytes = 0;
  uint64_t scratch_bytes = 0;
  uint64_t activation_bytes = 0;
  uint64_t peak_device_bytes = 0;
  uint64_t allocator_baseline_bytes = 0;
  uint64_t allocator_peak_used_bytes = 0;
  uint64_t allocator_used_bytes = 0;
  uint64_t allocator_reserved_bytes = 0;
  uint64_t descriptor_set_allocations = 0;
};

// Complete text-only Qwen3-VL conditioner. Embedding gather and exact 1-D
// RoPE construction happen on the host, followed by one activation upload.
// The BF16 residual remains device-resident through all 50 exact layers and is
// widened/downloaded once after layer 49. No final norm or LM head is run.
// The checkpoint must outlive this object while loaded.
class ExactQwenTextEncoder {
 public:
  ExactQwenTextEncoder();
  ~ExactQwenTextEncoder();
  ExactQwenTextEncoder(ExactQwenTextEncoder&&) noexcept;
  ExactQwenTextEncoder& operator=(ExactQwenTextEncoder&&) noexcept;
  ExactQwenTextEncoder(const ExactQwenTextEncoder&) = delete;
  ExactQwenTextEncoder& operator=(const ExactQwenTextEncoder&) = delete;

  static ExactQwenTextEncoder create(TensorContext& context);
  void load(const SafeTensors& checkpoint,
            const text::EncoderConfig& config = {});
  void unload() noexcept;
  bool loaded() const noexcept;
  const text::EncoderConfig& config() const;
  text::WeightFormat format() const noexcept;

  text::PromptEmbedding encode(const std::vector<int32_t>& token_ids,
                               text::EncoderTrace* trace = nullptr);
  text::PromptEmbedding encode(const text::Tokenizer& tokenizer,
                               const std::string& prompt);
  const ExactQwenTextEncoderStats& stats() const noexcept;

 private:
  struct Impl;
  explicit ExactQwenTextEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
