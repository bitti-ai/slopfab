#pragma once

#include <cuda_fp16.h>

#include <string>

#include "slopfab/cuda/device.h"
#include "slopfab/safetensors.h"

namespace slopfab::cuda {

// Keeps bitsandbytes NF4 packed on device and expands only when a consumer is
// about to use the matrix. Ordinary F16/BF16 matrices use `dense_` directly.
class F16Weight {
 public:
  F16Weight() = default;
  F16Weight(F16Weight&&) noexcept = default;
  F16Weight& operator=(F16Weight&&) noexcept = default;
  F16Weight(const F16Weight&) = delete;
  F16Weight& operator=(const F16Weight&) = delete;

  void load(const SafeTensors& checkpoint, const std::string& name, size_t expected_elements,
            cudaStream_t stream, const char* consumer,
            bool canonicalize_f16_subnormals = false);
  const __half* materialize(__half* workspace, size_t workspace_elements,
                            cudaStream_t stream) const;
  const int8_t* materialize_w4a8(int8_t* workspace,
                                 size_t workspace_elements,
                                 cudaStream_t stream) const;
  size_t elements() const { return elements_; }
  size_t stored_bytes() const;
  bool packed_nf4() const { return codes_.size() != 0; }
  bool packed_w4a8() const { return w4_codes_.size() != 0; }
  const float* w4a8_channel_scale() const { return w4_channel_scale_.get(); }

 private:
  size_t elements_ = 0;
  int block_size_ = 0;
  int nested_block_size_ = 0;
  float nested_offset_ = 0.0f;
  DeviceBuffer<__half> dense_;
  DeviceBuffer<uint8_t> codes_, absmax_;
  DeviceBuffer<float> quant_map_, nested_quant_map_, nested_absmax_;
  int w4_out_features_ = 0;
  int w4_in_features_ = 0;
  int w4_group_size_ = 0;
  DeviceBuffer<int8_t> w4_codes_;
  DeviceBuffer<uint8_t> w4_group_scale_;
  DeviceBuffer<float> w4_channel_scale_, w4_codebook_;
};

}  // namespace slopfab::cuda
