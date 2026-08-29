#pragma once

#include <cstdint>
#include <memory>

#include "vidfab/gemm.h"
#include "vidfab/vulkan/tensor.h"

namespace vidfab::vulkan {

class LinearWeight;
class PreparedF16ActivationView;
class PreparedNVFP4WeightView;

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
  explicit PreparedF16ActivationView(std::shared_ptr<void> slot,
                                     uintptr_t batch_id, uint32_t rows,
                                     uint64_t generation) noexcept;
  std::shared_ptr<void> slot_;
  uintptr_t batch_id_ = 0;
  uint32_t rows_ = 0;
  uint64_t generation_ = 0;
  friend class DenseGemmPlan;
  friend class PreparedF16Activation;
};

// One bounded BF16 expansion slot shared by streamed NVFP4 weights. Preparing
// a weight records g1 materialization into the slot and returns a view valid
// only for the current batch/generation. A later prepare may overwrite the
// slot after all previously recorded reads; TensorBatch access transitions
// provide the queue-ordered W->R/R->W barriers without a device-wide wait.
class StreamedNVFP4WeightCache {
 public:
  StreamedNVFP4WeightCache();
  ~StreamedNVFP4WeightCache();
  StreamedNVFP4WeightCache(StreamedNVFP4WeightCache&&) noexcept;
  StreamedNVFP4WeightCache& operator=(StreamedNVFP4WeightCache&&) noexcept;
  StreamedNVFP4WeightCache(const StreamedNVFP4WeightCache&) = delete;
  StreamedNVFP4WeightCache& operator=(const StreamedNVFP4WeightCache&) = delete;

  static StreamedNVFP4WeightCache create(TensorContext& context,
                                         uint64_t max_weight_elements);
  PreparedNVFP4WeightView prepare(TensorBatch& batch,
                                  const LinearWeight& weight,
                                  const DenseGemmPlan& plan);
  uint64_t capacity_elements() const noexcept;
  uint64_t dense_bytes() const noexcept;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit StreamedNVFP4WeightCache(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class DenseGemmPlan;
  friend class PreparedNVFP4WeightView;
};

class PreparedNVFP4WeightView {
 public:
  PreparedNVFP4WeightView();
  ~PreparedNVFP4WeightView();
  PreparedNVFP4WeightView(PreparedNVFP4WeightView&&) noexcept;
  PreparedNVFP4WeightView& operator=(PreparedNVFP4WeightView&&) noexcept;
  PreparedNVFP4WeightView(const PreparedNVFP4WeightView&) = delete;
  PreparedNVFP4WeightView& operator=(const PreparedNVFP4WeightView&) = delete;
  uint32_t out_features() const noexcept;
  uint32_t in_features() const noexcept;
  bool full_precision_matrix_mult() const noexcept;
  explicit operator bool() const noexcept;

 private:
  explicit PreparedNVFP4WeightView(std::shared_ptr<void> cache,
                                   uintptr_t batch_id, uint64_t generation,
                                   uint32_t out_features,
                                   uint32_t in_features,
                                   bool full_precision) noexcept;
  std::shared_ptr<void> cache_;
  uintptr_t batch_id_ = 0;
  uint64_t generation_ = 0;
  uint32_t out_features_ = 0;
  uint32_t in_features_ = 0;
  bool full_precision_ = false;
  friend class DenseGemmPlan;
  friend class StreamedNVFP4WeightCache;
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
  void record(TensorBatch& batch, DeviceTensor& input,
              PreparedNVFP4WeightView& prepared_weight,
              DeviceTensor& output, uint32_t rows,
              uint32_t input_row_offset = 0,
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
  friend class StreamedNVFP4WeightCache;
};

}  // namespace vidfab::vulkan
