#pragma once
#include "slopfab/seedvr2.h"
#include "slopfab/safetensors.h"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/attention.cuh"
#include <cuda_bf16.h>
#include <map>

namespace slopfab::seedvr2 {
using BFloat = __nv_bfloat16;
struct Tensor {
  int t = 1, h = 1, w = 1, c = 1;
  cuda::DeviceBuffer<BFloat> data;
  Tensor() = default;
  Tensor(int t_, int h_, int w_, int c_) : t(t_), h(h_), w(w_), c(c_),
    data(size_t(t_) * h_ * w_ * c_) {}
  int rows() const { return t * h * w; }
  size_t size() const { return data.size(); }
};

class Runtime {
public:
  Runtime();
  ~Runtime();
  cublasHandle_t blas = nullptr;
  cuda::Workspace scratch;
  std::map<std::string, Tensor> weights;
  SafeTensors* file = nullptr;
  void clear(SafeTensors& f);
  const Tensor& weight(const std::string& name);
  Tensor upload(const std::vector<float>& values, int t, int h, int w, int c);
  std::vector<float> download(const Tensor& x);
  Tensor linear(const Tensor& x, const std::string& name);
  Tensor conv(const Tensor& x, const std::string& name, bool down = false, int temporal_stride = 1);
  Tensor groupnorm(const Tensor& x, const std::string& name, bool silu = true);
  Tensor rms(const Tensor& x, const std::string& name = "");
  void activation(Tensor& x, const Tensor* gate = nullptr);
  void add(Tensor& x, const Tensor& y);
  void modulate(Tensor& x, const Tensor& emb, const std::string& name, int layer, bool gate);
  Tensor upsample(const Tensor& x, int temporal_ratio);
  Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int heads, int dim);
  void window_attention(const Tensor& vqkv, const Tensor& tqkv, Tensor& video, Tensor& text,
                        const std::string& prefix, const std::string& vb, const std::string& tb,
                        const std::vector<Window>& windows);
};
} // namespace slopfab::seedvr2
