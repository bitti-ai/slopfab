#pragma once
#include "slopfab/cuda/attention.cuh"
#include "slopfab/cuda/linear.cuh"
#include "slopfab/safetensors.h"
#include "buffer.cuh"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace slopfab::moss {
using BFloat = __nv_bfloat16;

struct Tensor {
  int rows = 0, cols = 0;
  TemporaryBuffer<BFloat> data;
  Tensor() = default;

  Tensor(int r, int c) : rows(r), cols(c), data(size_t(r) * c) {
  }

  BFloat* ptr() {
    return data.get();
  }

  const BFloat* ptr() const {
    return data.get();
  }
};

// Reusable checkpoint-backed CUDA operations. No model names or dimensions
// occur in weight loading, normalization, activation or attention dispatch.
class Runtime {
public:
  explicit Runtime(const std::string& path);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  SafeTensors checkpoint;
  cublasHandle_t blas = nullptr;
  cuda::Workspace scratch;
  cuda::LinearRunner linear;
  // Autoregressive models reuse each matrix for thousands of single-token calls.
  bool cache_dense_weights = false;
  const BFloat* weight(const std::string& name);
  const float* float_weight(const std::string& name);
  Tensor project(const Tensor& x, const std::string& name, bool fused_bias = true);
  Tensor norm(const Tensor& x, const std::string& name, int width = 0, bool unit_offset = false,
              const Tensor* modulation = nullptr, int shift = 0, const Tensor* gate = nullptr,
              float epsilon = 1e-6f);
  // float_probabilities keeps softmax/PV in FP32 for the sensitive music flow stack.
  Tensor attend(const Tensor& q, const Tensor& k, const Tensor& v, int heads, int kv_heads,
                bool causal = false, bool float_probabilities = false);
  void release_prefix(const std::string& prefix);

private:
  std::map<std::string, cuda::DeviceBuffer<uint8_t>> storage_;
  std::map<std::string, cuda::QuantWeight> linears_;
  std::map<std::string, cuda::DeviceBuffer<BFloat>> prepared_;
  const void* upload(const std::string& key, const void* data, size_t bytes);
};

Tensor from_host(const std::vector<float>& values, int rows, int cols);
std::vector<float> to_host(const Tensor& x);
Tensor slice(const Tensor& x, int offset, int cols, int groups = 1);
Tensor concatenate(const Tensor& a, const Tensor& b);
void residual(Tensor& x, const Tensor& update, const Tensor* modulation = nullptr, int gate = 0);
Tensor activate(const Tensor& x, const Tensor* gate = nullptr, bool situ = false,
                bool sigmoid_gate = false);
void rotary(Tensor& x, int heads, const Tensor& sincos, int rotary_dim);
Tensor image_rope(int height, int width, int text_length, bool text_only = false);
Tensor text_rope(int length);
Tensor delta_attention(Runtime& rt, const Tensor& x, const std::string& prefix);
Tensor encode_text(Runtime& rt, const std::vector<int32_t>& ids);
Tensor image_prediction(Runtime& rt, const Tensor& latent, const Tensor& text, int height,
                        int width, float timestep, const std::vector<int>& skip);
std::vector<float> decode_image(const std::string& path, const std::vector<float>& latent,
                                int height, int width);
std::vector<float> encode_flux_image(const std::string& path, const std::vector<uint8_t>& pixels,
                                     int height, int width);
} // namespace slopfab::moss
