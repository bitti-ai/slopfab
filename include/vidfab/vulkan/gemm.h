#pragma once

#include <cstdint>
#include <memory>

#include "vidfab/gemm.h"
#include "vidfab/vulkan/tensor.h"

namespace vidfab::vulkan {

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
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit DenseGemmPlan(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
