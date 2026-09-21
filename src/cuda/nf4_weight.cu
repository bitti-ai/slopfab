#include "slopfab/cuda/nf4_weight.cuh"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "slopfab/cuda/linear.cuh"
#include "slopfab/cuda/w4a8.cuh"
#include "slopfab/nf4.h"
#include "slopfab/int8_weight.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/w4a8.h"

namespace slopfab::cuda {
namespace {

// Undo W_rot = W H using H = kron(H4, ..., H4) / sqrt(group).
// H4 is the regular symmetric Hadamard, with its negative anti-diagonal.
// The integer butterflies are exact; apply the row scale only once at the end.
__global__ void unpack_int8_kernel(const int8_t* codes, const float* scales, __half* output,
                                   size_t count, int columns, int group, bool canonicalize) {
  __shared__ float values[256];
  const int lane = threadIdx.x;
  const size_t index = static_cast<size_t>(blockIdx.x) * 256 + lane;
  values[lane] = index < count ? static_cast<float>(codes[index]) : 0.0f;
  __syncthreads();
  for (int stride = 1; stride < group; stride *= 4) {
    const int base = lane / (4 * stride) * (4 * stride) + lane % stride;
    const int digit = lane / stride % 4;
    const float a = values[base], b = values[base + stride];
    const float c = values[base + 2 * stride], d = values[base + 3 * stride];
    const float value = digit == 0   ? a + b + c - d
                        : digit == 1 ? a + b - c + d
                        : digit == 2 ? a - b + c + d
                                     : -a + b + c + d;
    __syncthreads();
    values[lane] = value;
    __syncthreads();
  }
  if (index < count) {
    const float factor = scales[index / columns] * rsqrtf(static_cast<float>(group));
    __half value = __float2half_rn(values[lane] * factor);
    if (canonicalize) {
      unsigned short bits = __half_as_ushort(value);
      if ((bits & 0x7c00u) == 0)
        bits &= 0x8000u;
      value = __ushort_as_half(bits);
    }
    output[index] = value;
  }
}

} // namespace

void F16Weight::load(const SafeTensors& checkpoint, const std::string& name,
                     size_t expected_elements, cudaStream_t stream, const char* consumer,
                     bool canonicalize_f16_subnormals) {
  // Reloading a weight must not leave a previous packed representation active.
  *this = F16Weight{};
  const TensorView& view = checkpoint.at(name);
  elements_ = expected_elements;
  if (is_w4a8_weight(checkpoint, name)) {
    const W4A8State state = read_w4a8_state(checkpoint, name, consumer);
    if (view.dtype != DType::kI8 || view.shape.size() != 2 ||
        static_cast<size_t>(view.numel()) * 2 != expected_elements) {
      throw std::runtime_error(std::string(consumer) + ": W4A8 shape mismatch for '" + name + "'");
    }
    w4_out_features_ = static_cast<int>(view.shape[0]);
    w4_in_features_ = static_cast<int>(view.shape[1] * 2);
    w4_group_size_ = state.group_size;
    if (w4_out_features_ <= 0 || w4_in_features_ <= 0 ||
        w4_in_features_ % state.convrot_group_size != 0) {
      throw std::runtime_error(std::string(consumer) + ": invalid W4A8 matrix shape for '" + name +
                               "'");
    }
    auto auxiliary = [&](const char* suffix) {
      constexpr size_t weight_suffix_size = 7;
      return name.substr(0, name.size() - weight_suffix_size) + suffix;
    };
    const TensorView& group_scale = checkpoint.at(auxiliary(".weight_s_rel"));
    const TensorView& channel_scale = checkpoint.at(auxiliary(".weight_s_channel"));
    const TensorView& codebook = checkpoint.at(auxiliary(".weight_codebook"));
    const int groups = w4_in_features_ / w4_group_size_;
    if (group_scale.dtype != DType::kF8E4M3 || group_scale.shape.size() != 2 ||
        group_scale.shape[0] != w4_out_features_ || group_scale.shape[1] != groups ||
        channel_scale.dtype != DType::kF32 ||
        channel_scale.shape != std::vector<int64_t>{w4_out_features_} ||
        codebook.dtype != DType::kF32 || codebook.shape != std::vector<int64_t>{16}) {
      throw std::runtime_error(std::string(consumer) + ": malformed W4A8 auxiliaries for '" + name +
                               "'");
    }
    if (checkpoint.find(auxiliary(".weight_correction"))) {
      throw std::runtime_error(std::string(consumer) +
                               ": asymmetric W4A8 correction is unsupported for '" + name + "'");
    }
    w4_codes_.allocate(view.nbytes);
    w4_group_scale_.allocate(group_scale.nbytes);
    w4_channel_scale_.allocate(static_cast<size_t>(channel_scale.numel()));
    w4_codebook_.allocate(static_cast<size_t>(codebook.numel()));
    w4_codes_.copy_from_host(static_cast<const int8_t*>(view.data), w4_codes_.size(), stream);
    w4_group_scale_.copy_from_host(static_cast<const uint8_t*>(group_scale.data),
                                   w4_group_scale_.size(), stream);
    w4_channel_scale_.copy_from_host(static_cast<const float*>(channel_scale.data),
                                     w4_channel_scale_.size(), stream);
    w4_codebook_.copy_from_host(static_cast<const float*>(codebook.data), w4_codebook_.size(),
                                stream);
  } else if (is_nf4_weight(checkpoint, name)) {
    const NF4State state = read_nf4_state(checkpoint, name, consumer);
    size_t logical = 1;
    for (int64_t dim : state.shape)
      logical *= static_cast<size_t>(dim);
    if (logical != expected_elements || view.dtype != DType::kU8 || view.nbytes * 2 != logical)
      throw std::runtime_error(std::string(consumer) + ": NF4 shape mismatch for '" + name + "'");
    const TensorView& a = checkpoint.at(name + ".absmax");
    const TensorView& q = checkpoint.at(name + ".quant_map");
    const TensorView& nq = checkpoint.at(name + ".nested_quant_map");
    const TensorView& na = checkpoint.at(name + ".nested_absmax");
    if (a.dtype != DType::kU8 || q.dtype != DType::kF32 || nq.dtype != DType::kF32 ||
        na.dtype != DType::kF32)
      throw std::runtime_error(std::string(consumer) + ": malformed NF4 auxiliaries for '" + name +
                               "'");
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
    nested_quant_map_.copy_from_host(static_cast<const float*>(nq.data), nested_quant_map_.size(),
                                     stream);
    nested_absmax_.copy_from_host(static_cast<const float*>(na.data), nested_absmax_.size(),
                                  stream);
  } else if (view.dtype == DType::kI8) {
    const Int8WeightState state = read_int8_weight(checkpoint, name, consumer);
    if (static_cast<size_t>(view.numel()) != expected_elements)
      throw std::runtime_error(std::string(consumer) + ": INT8 shape mismatch for '" + name + "'");
    int8_columns_ = state.columns;
    int8_rotation_group_ = state.rotation_group;
    int8_canonicalize_ = canonicalize_f16_subnormals;
    int8_codes_.allocate(expected_elements);
    int8_scales_.allocate(state.rows);
    int8_codes_.copy_from_host(state.codes, expected_elements, stream);
    int8_scales_.copy_from_host(state.scales, state.rows, stream);
  } else {
    if (static_cast<size_t>(view.numel()) != expected_elements)
      throw std::runtime_error(std::string(consumer) + ": shape mismatch for '" + name + "'");
    dense_.allocate(expected_elements);
    if (view.dtype == DType::kF16) {
      if (canonicalize_f16_subnormals) {
        std::vector<uint16_t> canonical(expected_elements);
        std::memcpy(canonical.data(), view.data, expected_elements * 2);
        for (uint16_t& word : canonical) {
          if ((word & 0x7c00u) == 0 && (word & 0x03ffu) != 0)
            word &= 0x8000u;
        }
        dense_.copy_from_host(reinterpret_cast<const __half*>(canonical.data()), expected_elements,
                              stream);
        SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream));
      } else {
        dense_.copy_from_host(static_cast<const __half*>(view.data), expected_elements, stream);
      }
    } else {
      const std::vector<float> f = to_f32(view);
      std::vector<__half> h(f.size());
      for (size_t i = 0; i < f.size(); ++i)
        h[i] = __float2half_rn(f[i]);
      dense_.copy_from_host(h.data(), h.size(), stream);
    }
  }
}

const __half* F16Weight::materialize(__half* workspace, size_t workspace_elements,
                                     cudaStream_t stream) const {
  if (packed_w4a8())
    throw std::runtime_error("W4A8 weight requires the INT8 materialization path");
  if (!packed_nf4() && !packed_int8())
    return dense_.get();
  if (!workspace || workspace_elements < elements_)
    throw std::runtime_error("Packed weight workspace is too small");
  if (packed_int8()) {
    unpack_int8_kernel<<<static_cast<unsigned>((elements_ + 255) / 256), 256, 0, stream>>>(
        int8_codes_.get(), int8_scales_.get(), workspace, elements_, int8_columns_,
        int8_rotation_group_, int8_canonicalize_);
    SLOPFAB_CUDA_CHECK(cudaGetLastError());
    return workspace;
  }
  launch_dequant_nf4_f16(codes_.get(), absmax_.get(), quant_map_.get(), nested_quant_map_.get(),
                         nested_absmax_.get(), block_size_, nested_block_size_, nested_offset_,
                         workspace, elements_, stream);
  return workspace;
}

const int8_t* F16Weight::materialize_w4a8(int8_t* workspace, size_t workspace_elements,
                                          cudaStream_t stream) const {
  if (!packed_w4a8())
    throw std::runtime_error("INT8 materialization requested for a non-W4A8 weight");
  if (!workspace || workspace_elements < elements_)
    throw std::runtime_error("W4A8 weight workspace is too small");
  launch_dequant_w4a8_weight(w4_codes_.get(), w4_group_scale_.get(), w4_codebook_.get(), workspace,
                             w4_out_features_, w4_in_features_, w4_group_size_, stream);
  return workspace;
}

size_t F16Weight::stored_bytes() const {
  return dense_.nbytes() + int8_codes_.nbytes() + int8_scales_.nbytes() + codes_.nbytes() +
         absmax_.nbytes() + quant_map_.nbytes() + nested_quant_map_.nbytes() +
         nested_absmax_.nbytes() + w4_codes_.nbytes() + w4_group_scale_.nbytes() +
         w4_channel_scale_.nbytes() + w4_codebook_.nbytes();
}

} // namespace slopfab::cuda
