#pragma once
#include <map>
#include <utility>
#include <vector>
#include "slopfab/lora.h"
#include "slopfab/vulkan/gemm.h"

namespace slopfab::vulkan {
struct LoraScratch {
  // Share output scratch even when different layers have different ranks.
  std::map<uint32_t, DeviceTensor> hidden, delta;
  uint64_t reserved_bytes() const;
};

class LoraProjection {
 public:
  void load(TensorContext& context, const std::vector<LoraFactors>& factors,
            uint32_t row_offset, uint32_t out, uint32_t rows);
  void prepare(TensorContext& context, LoraScratch& scratch) const;
  void record(TensorBatch& batch, DeviceTensor& input, DeviceTensor& output,
              uint32_t rows, LoraScratch& scratch);
  uint64_t resident_bytes() const;
  uint32_t operators(uint32_t rows) const;
 private:
  struct Factors {
    uint32_t rank = 0, out = 0, rows = 0;
    DeviceTensor a, b;
    DenseGemmPlan down, up;
  };
  std::vector<Factors> factors_;
};
}  // namespace slopfab::vulkan
