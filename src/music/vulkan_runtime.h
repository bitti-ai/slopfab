#pragma once
#include "slopfab/safetensors.h"
#include "slopfab/vulkan/gemm.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace slopfab::music_gpu {
namespace vk = slopfab::vulkan;
struct Engine {
  vk::Instance instance;
  vk::Device device;
  vk::TensorContext context;
  vk::TensorBatch pending;
  Engine();
  ~Engine();
  vk::TensorBatch& batch();
  void flush();
  vk::DeviceTensor allocate(int rows,int cols,ScalarType type=ScalarType::kBFloat16);
  vk::DeviceTensor upload(const std::vector<float>&,int rows,int cols,bool fp32=false,bool vector=false);
  void op(vk::DeviceTensor&,vk::DeviceTensor&,vk::DeviceTensor&,vk::DeviceTensor&,
          std::initializer_list<uint32_t>,uint32_t groups);
};
struct Tensor {
  int rows=0,cols=0;
  std::shared_ptr<Engine> engine;
  mutable vk::DeviceTensor data;
  Tensor()=default;
  Tensor(int,int);
};
class Runtime {
public:
  explicit Runtime(const std::string& path);
  ~Runtime();
  Runtime(const Runtime&)=delete;
  Runtime& operator=(const Runtime&)=delete;
  SafeTensors checkpoint;
  std::shared_ptr<Engine> engine;
  bool cache_dense_weights=false;
  vk::DeviceTensor& weight(const std::string&);
  vk::DeviceTensor& float_weight(const std::string&);
  Tensor project(const Tensor&,const std::string&,bool fused_bias=true);
  Tensor norm(const Tensor&,const std::string&,int width=0);
  Tensor attend(const Tensor&,const Tensor&,const Tensor&,int heads,int kv_heads,
                bool causal=false,bool float_probabilities=false);
  void release_prefix(const std::string&);
private:
  struct Matrix { int group=1; std::vector<vk::DeviceTensor> tiles,biases; };
  std::map<std::string,vk::DeviceTensor> weights_;
  std::map<std::string,Matrix> matrices_;
  Runtime* previous_=nullptr;
};
float round_bf(float);
Tensor from_host(const std::vector<float>&,int,int);
std::vector<float> to_host(const Tensor&);
Tensor slice(const Tensor&,int offset,int cols,int groups=1);
Tensor concatenate(const Tensor&,const Tensor&);
void residual(Tensor&,const Tensor&);
Tensor activate(const Tensor&,const Tensor* gate=nullptr);
void rotary(Tensor&,int heads,const Tensor&,int rotary_dim);
}
