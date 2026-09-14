#include "slopfab/vulkan/lora.h"
#include <algorithm>
#include "slopfab/dtype.h"

namespace slopfab::vulkan {
namespace {
TensorLayout matrix(uint64_t rows, uint64_t cols) {
  const uint64_t shape[] = {rows, cols};
  return TensorLayout::contiguous(shape, 2);
}
uint64_t bytes(const DeviceTensor& t) { return t ? t.layout().bytes(t.type()) : 0; }
}

uint64_t LoraScratch::reserved_bytes() const {
  uint64_t n = 0;
  for (const auto& entry : hidden) n += bytes(entry.second);
  for (const auto& entry : delta) n += bytes(entry.second);
  return n;
}

void LoraProjection::load(TensorContext& context, const std::vector<LoraFactors>& host,
                           uint32_t offset, uint32_t out, uint32_t rows) {
  auto upload = [&](const float* p, uint32_t rows, uint32_t cols) {
    std::vector<uint16_t> bf(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < bf.size(); ++i) bf[i] = f32_to_bf16(p[i]);
    auto result = context.allocate(matrix(rows, cols), ScalarType::kBFloat16);
    context.upload_transient_bytes(result, bf.data(), bf.size() * 2);
    return result;
  };
  for (const auto& h : host) {
    if (out == 0 || offset > static_cast<uint32_t>(h.out) || out > h.out - offset)
      throw std::runtime_error("LoRA: invalid Vulkan output slice");
    Factors f;
    f.rank = h.rank; f.out = out; f.rows = rows;
    f.a = upload(h.a.data(), h.rank, h.in);
    f.b = upload(h.b.data() + static_cast<size_t>(offset) * h.rank, out, h.rank);
    f.down = DenseGemmPlan::create(context, {rows, f.rank, static_cast<uint32_t>(h.in),
        DenseGemmMode::kBFloat16, DenseGemmBias::kNone, false});
    f.up = DenseGemmPlan::create(context, {rows, out, f.rank,
        DenseGemmMode::kBFloat16, DenseGemmBias::kNone, false});
    factors_.push_back(std::move(f));
  }
}

void LoraProjection::prepare(TensorContext& context, LoraScratch& scratch) const {
  for (const auto& f : factors_) {
    auto allocate = [&](DeviceTensor& tensor, uint32_t cols) {
      if (tensor) {
        if (tensor.layout().extent[0] != f.rows)
          throw std::invalid_argument("LoRA: scratch belongs to a different sequence");
        return;
      }
      tensor = context.allocate(matrix(f.rows, cols), ScalarType::kBFloat16);
    };
    allocate(scratch.hidden[f.rank], f.rank);
    allocate(scratch.delta[f.out], f.out);
  }
}

void LoraProjection::record(TensorBatch& batch, DeviceTensor& input,
                             DeviceTensor& output, uint32_t rows, LoraScratch& scratch) {
  for (auto& f : factors_) {
    auto& hidden = scratch.hidden.at(f.rank);
    auto& delta = scratch.delta.at(f.out);
    auto gemm = [&](DenseGemmPlan& plan, DeviceTensor& x, DeviceTensor& w, DeviceTensor& y) {
      const uint32_t tiled = rows / 64 * 64;
      if (tiled) plan.record(batch, x, w, y, tiled);
      if (tiled != rows) plan.record(batch, x, w, y, rows - tiled, tiled, tiled);
    };
    gemm(f.down, input, f.a, hidden);
    gemm(f.up, hidden, f.b, delta);
    batch.text_add_residual_bf16(output, delta);
  }
}

uint64_t LoraProjection::resident_bytes() const {
  uint64_t n = 0;
  for (const auto& f : factors_) n += bytes(f.a) + bytes(f.b);
  return n;
}
uint32_t LoraProjection::operators(uint32_t rows) const {
  return static_cast<uint32_t>(factors_.size()) *
      (2 * ((rows >= 64 ? 1u : 0u) + (rows % 64 ? 1u : 0u)) + 1);
}
}  // namespace slopfab::vulkan
