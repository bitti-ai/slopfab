#pragma once

#include <cstdint>
#include <memory>

#include "vidfab/gemm.h"
#include "vidfab/vulkan/tensor.h"

namespace vidfab::vulkan {

class PreparedF16ActivationView;

// Reusable device-resident fp16 activation storage. Preparation is recorded
// once for a chunk and the returned batch-scoped view can feed any number of
// compatible VAE GEMM plans without repeating fp32-to-fp16 conversion.
class PreparedF16Activation {
 public:
  PreparedF16Activation();
  ~PreparedF16Activation();
  PreparedF16Activation(PreparedF16Activation&&) noexcept;
  PreparedF16Activation& operator=(PreparedF16Activation&&) noexcept;
  PreparedF16Activation(const PreparedF16Activation&) = delete;
  PreparedF16Activation& operator=(const PreparedF16Activation&) = delete;

  static PreparedF16Activation create(TensorContext& context,
                                      uint32_t max_rows,
                                      uint32_t in_features);
  PreparedF16ActivationView prepare(TensorBatch& batch, DeviceTensor& input,
                                    uint32_t rows,
                                    uint32_t input_row_offset = 0);
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit PreparedF16Activation(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class DenseGemmPlan;
  friend class PreparedF16ActivationView;
};

class PreparedF16ActivationView {
 public:
  PreparedF16ActivationView();
  ~PreparedF16ActivationView();
  PreparedF16ActivationView(PreparedF16ActivationView&&) noexcept;
  PreparedF16ActivationView& operator=(PreparedF16ActivationView&&) noexcept;
  PreparedF16ActivationView(const PreparedF16ActivationView&) = delete;
  PreparedF16ActivationView& operator=(const PreparedF16ActivationView&) = delete;
  uint32_t rows() const noexcept;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit PreparedF16ActivationView(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class DenseGemmPlan;
  friend class PreparedF16Activation;
};

// Immutable deterministic dense-GEMM plan. Pipelines and bounded descriptor
// slots belong to TensorContext; a plan holds only validated dimensions and
// retains that context identity. Rows and source/destination row offsets are
// supplied at record time so one prepared g1 weight is reused across chunks.
class DenseGemmPlan {
 public:
  DenseGemmPlan();
  ~DenseGemmPlan();
  DenseGemmPlan(DenseGemmPlan&&) noexcept;
  DenseGemmPlan& operator=(DenseGemmPlan&&) noexcept;
  DenseGemmPlan(const DenseGemmPlan&) = delete;
  DenseGemmPlan& operator=(const DenseGemmPlan&) = delete;

  static DenseGemmPlan create(TensorContext& context,
                              const DenseGemmPlanDesc& desc);

  const DenseGemmPlanDesc& description() const;
  void record(TensorBatch& batch, DeviceTensor& input,
              DeviceTensor& prepared_weight, DeviceTensor& output,
              uint32_t rows, uint32_t input_row_offset = 0,
              uint32_t output_row_offset = 0,
              DeviceTensor* bias = nullptr) const;
  void record(TensorBatch& batch, PreparedF16ActivationView& input,
              DeviceTensor& prepared_weight, DeviceTensor& output,
              uint32_t output_row_offset = 0,
              DeviceTensor* bias = nullptr) const;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit DenseGemmPlan(std::shared_ptr<Impl> impl);
  void record_impl(TensorBatch& batch, DeviceTensor& input,
                   DeviceTensor& prepared_weight, DeviceTensor& output,
                   uint32_t rows, uint32_t input_row_offset,
                   uint32_t output_row_offset, DeviceTensor* bias,
                   bool input_is_prepared_f16) const;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
