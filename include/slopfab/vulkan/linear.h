#pragma once

#include <cstdint>
#include <memory>

#include "slopfab/linear_weight.h"
#include "slopfab/vulkan/tensor.h"

namespace slopfab::vulkan {

// Persistent compressed/device-native weight storage. Construction copies the
// immutable checkpoint bytes and auxiliaries once. Dense expansion is always
// supplied by the caller and can therefore be cached per active/chunked weight
// without expanding every resident quantized matrix.
class LinearWeight {
public:
  LinearWeight();
  ~LinearWeight();
  LinearWeight(LinearWeight&&) noexcept;
  LinearWeight& operator=(LinearWeight&&) noexcept;
  LinearWeight(const LinearWeight&) = delete;
  LinearWeight& operator=(const LinearWeight&) = delete;

  static LinearWeight upload(TensorContext& context, const LinearWeightUpload& source);

  LinearWeightFormat format() const;
  uint32_t out_features() const;
  uint32_t in_features() const;
  uint64_t stored_bytes() const noexcept;
  // Sum of logical persistent tensor bytes. Allocator page/block rounding is
  // reported by TensorContext::reserved_bytes(), not charged per weight here.
  uint64_t resident_bytes() const noexcept;
  bool has_fp8_input_scale() const noexcept;
  float fp8_input_scale() const;
  bool full_precision_matrix_mult() const noexcept;
  bool has_pre_quant_scale() const noexcept;
  bool applies_convrot() const noexcept;
  uint32_t convrot_group() const noexcept;

  // Writes [out,in] BF16 or FP16 into caller-owned persistent workspace.
  // BF16 storage copies exactly; other formats preserve CUDA materialization
  // order and rounding. FP16 is the VAE/NF4 materialization contract.
  void materialize_bf16(TensorBatch& batch, DeviceTensor& dense) const;
  void materialize_f16(TensorBatch& batch, DeviceTensor& dense) const;

  // AWQ scale and regular-H4 ConvRot apply to [rows,in] activations. They are
  // independent so a caller with both can retain the two required buffers.
  void apply_pre_quant_scale(TensorBatch& batch, DeviceTensor& input, DeviceTensor& output) const;
  void apply_convrot(TensorBatch& batch, DeviceTensor& input, DeviceTensor& output) const;

private:
  struct Impl;
  explicit LinearWeight(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class TensorBatch;
  friend class StreamedNVFP4WeightCache;
};

} // namespace slopfab::vulkan
