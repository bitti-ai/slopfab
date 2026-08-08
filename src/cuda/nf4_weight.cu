#include "vidfab/cuda/nf4_weight.cuh"

#include <stdexcept>
#include <vector>

#include "vidfab/cuda/linear.cuh"
#include "vidfab/nf4.h"
#include "vidfab/tensor_convert.h"

namespace vidfab::cuda {

void F16Weight::load(const SafeTensors& checkpoint, const std::string& name,
                     size_t expected_elements, cudaStream_t stream, const char* consumer) {
  const TensorView& view = checkpoint.at(name);
  elements_ = expected_elements;
  if (is_nf4_weight(checkpoint, name)) {
    const NF4State state = read_nf4_state(checkpoint, name, consumer);
    size_t logical = 1;
    for (int64_t dim : state.shape) logical *= static_cast<size_t>(dim);
    if (logical != expected_elements || view.dtype != DType::kU8 || view.nbytes * 2 != logical)
      throw std::runtime_error(std::string(consumer) + ": NF4 shape mismatch for '" + name + "'");
    const TensorView& a = checkpoint.at(name + ".absmax");
    const TensorView& q = checkpoint.at(name + ".quant_map");
    const TensorView& nq = checkpoint.at(name + ".nested_quant_map");
    const TensorView& na = checkpoint.at(name + ".nested_absmax");
    if (a.dtype != DType::kU8 || q.dtype != DType::kF32 || nq.dtype != DType::kF32 ||
        na.dtype != DType::kF32)
      throw std::runtime_error(std::string(consumer) + ": malformed NF4 auxiliaries for '" + name + "'");
    block_size_ = state.block_size;
    nested_block_size_ = state.nested_block_size;
    nested_offset_ = state.nested_offset;
    codes_.allocate(view.nbytes);
    absmax_.allocate(a.nbytes);
    quant_map_.allocate(static_cast<size_t>(q.numel()));
    nested_quant_map_.allocate(static_cast<size_t>(nq.numel()));
    nested_absmax_.allocate(static_cast<size_t>(na.numel()));
    codes_.copy_from_host(static_cast<const uint8_t*>(view.data), codes_.size(), stream);
    absmax_.copy_from_host(static_cast<const uint8_t*>(a.data), absmax_.size(), stream);
    quant_map_.copy_from_host(static_cast<const float*>(q.data), quant_map_.size(), stream);
    nested_quant_map_.copy_from_host(static_cast<const float*>(nq.data), nested_quant_map_.size(), stream);
    nested_absmax_.copy_from_host(static_cast<const float*>(na.data), nested_absmax_.size(), stream);
  } else {
    if (static_cast<size_t>(view.numel()) != expected_elements)
      throw std::runtime_error(std::string(consumer) + ": shape mismatch for '" + name + "'");
    dense_.allocate(expected_elements);
    if (view.dtype == DType::kF16) {
      dense_.copy_from_host(static_cast<const __half*>(view.data), expected_elements, stream);
    } else {
      const std::vector<float> f = to_f32(view);
      std::vector<__half> h(f.size());
      for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half_rn(f[i]);
      dense_.copy_from_host(h.data(), h.size(), stream);
    }
  }
}

const __half* F16Weight::materialize(__half* workspace, size_t workspace_elements,
                                     cudaStream_t stream) const {
  if (!packed_nf4()) return dense_.get();
  if (!workspace || workspace_elements < elements_)
    throw std::runtime_error("NF4 weight workspace is too small");
  launch_dequant_nf4_f16(codes_.get(), absmax_.get(), quant_map_.get(), nested_quant_map_.get(),
                          nested_absmax_.get(), block_size_, nested_block_size_, nested_offset_,
                          workspace, elements_, stream);
  return workspace;
}

size_t F16Weight::stored_bytes() const {
  return dense_.nbytes() + codes_.nbytes() + absmax_.nbytes() + quant_map_.nbytes() +
         nested_quant_map_.nbytes() + nested_absmax_.nbytes();
}

}  // namespace vidfab::cuda
