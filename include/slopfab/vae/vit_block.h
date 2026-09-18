// Backend-neutral exact execution contract for one video-VAE transformer block.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "slopfab/device_tensor.h"
#include "slopfab/safetensors.h"

namespace slopfab::vae {

struct ViTBlockConfig {
  uint32_t sequence = 0;
  uint32_t num_patches = 0;
  uint32_t dim = 2048;
  uint32_t heads = 32;
  uint32_t head_dim = 64;
  uint32_t ffn_inner = 8192;
  uint32_t rope_dim = 48;
  float epsilon = 1.0e-5f;
};

struct ViTRopeTables {
  std::vector<float> cosine;
  std::vector<float> sine;
};

// Canonical video-VAE T/H/W rotary tables, including identity suffix rows.
ViTRopeTables build_vit_rope_tables(uint32_t time, uint32_t height,
                                    uint32_t width, uint32_t suffix,
                                    uint32_t rope_dim, float theta);

// Non-owning host views. Linear matrices are row-major fp16 in unrotated
// coordinates [out_features,in_features]; all other arrays are fp32.
struct ViTBlockWeightsView {
  const float* norm1 = nullptr;
  const float* norm2 = nullptr;
  const float* scale1 = nullptr;
  const float* scale2 = nullptr;
  const uint16_t* qkv_weight = nullptr;
  const float* qkv_bias = nullptr;
  const uint16_t* out_weight = nullptr;
  const float* out_bias = nullptr;
  const uint16_t* w1_weight = nullptr;
  const float* w1_bias = nullptr;
  const uint16_t* w2_weight = nullptr;
  const float* w2_bias = nullptr;
};

// Owning host representation used by checkpoint loaders and parity tools.
// It retains fp16 matrices instead of widening a second copy. Checkpoint fp16
// subnormals are canonically flushed to signed zero because CUDA/Vulkan matrix
// instructions do not share a portable subnormal-input contract.
struct ViTBlockWeights {
  std::vector<float> norm1, norm2, scale1, scale2;
  std::vector<uint16_t> qkv_weight;
  std::vector<float> qkv_bias;
  std::vector<uint16_t> out_weight;
  std::vector<float> out_bias;
  std::vector<uint16_t> w1_weight;
  std::vector<float> w1_bias;
  std::vector<uint16_t> w2_weight;
  std::vector<float> w2_bias;
  ViTBlockWeightsView view() const noexcept;
  uint64_t bytes() const noexcept;
};

// Loads one real decoder block using the shipped checkpoint naming/layout
// contract. Matrices must be rank-2 fp16 or Comfy INT8 in [out,in] order.
// INT8 scales and ConvRot are undone before narrowing to fp16; all fp16
// subnormals are canonicalized to signed zero. Affine tensors may
// be fp16/fp32 and are converted to the exact fp32 stage boundary.
ViTBlockWeights load_vit_block_weights(const SafeTensors& checkpoint,
                                       uint32_t layer,
                                       const ViTBlockConfig& config);

// The host boundary is intentional for this first vertical slice: it makes a
// CUDA/Vulkan comparison unambiguous and cannot hide a CUDA resource behind a
// Vulkan label. The later 36-block graph can add a device-resident chaining
// interface without changing this weight/config contract.
class ExactViTBlockStage {
 public:
  virtual ~ExactViTBlockStage() = default;
  virtual DeviceBackend backend() const noexcept = 0;
  virtual const ViTBlockConfig& config() const noexcept = 0;
  virtual void load(const ViTBlockWeightsView& weights) = 0;
  // tokens/output: fp32 [sequence,dim]. Cos/sin: fp32
  // [sequence,rope_dim], including identity rows for suffix tokens.
  virtual void forward(const float* tokens, const float* cosine,
                       const float* sine, float* output) = 0;
  virtual uint64_t persistent_bytes() const noexcept = 0;
  virtual uint64_t peak_device_bytes() const noexcept = 0;
};

}  // namespace slopfab::vae
