#pragma once
#include "../music/vulkan_runtime.h"
#include <functional>

namespace slopfab::krea_vulkan {
namespace vk = slopfab::vulkan;
using music_gpu::Tensor;
using music_gpu::from_host;
using music_gpu::to_host;
using music_gpu::round_bf;

class Runtime : public music_gpu::Runtime {
public:
  explicit Runtime(const std::string& path, int device_index = -1);
  Tensor attend(const Tensor& q, const Tensor& k, const Tensor& v, int heads, int kv_heads,
                bool causal = false);
};

// Shader operations, shared with tensor_krea.comp. Operands stay on the device.
enum Operation {
  Rms,
  Modulate,
  Residual,
  Activation,
  Rotary,
  Collapse,
  LayerAttention,
  RepeatHeads,
  CaptureText,
  Patches,
  ImageRms,
  Nearest,
  ImageResidual,
  FloatColumns,
  Softmax
};

void dispatch(music_gpu::Engine& engine, vk::DeviceTensor& x, vk::DeviceTensor& w,
              vk::DeviceTensor& b, vk::DeviceTensor& y, std::initializer_list<uint32_t> parameters,
              uint32_t groups);
Tensor norm(Runtime& rt, const Tensor& x, const std::string& name, int width = 0);
void gpu_modulate(Tensor& x, const Tensor& mod, int offset);
Tensor gpu_layer_attention(const Tensor& q, const Tensor& k, const Tensor& v);
Tensor gpu_collapse(Runtime& rt, const Tensor& x);
Tensor encode_prompt(const std::string& path, const std::string& tokenizer,
                     const std::string& prompt, const std::function<void()>& hook);

namespace ops {
Tensor rows(const Tensor& x, int first, int count);
Tensor join(const Tensor& a, const Tensor& b);
Tensor gelu(const Tensor& x, bool approximate);
Tensor activate(const Tensor& x, const Tensor* gate = nullptr, bool situ = false,
                bool sigmoid = false);
void residual(Tensor& x, const Tensor& y, const Tensor* mod = nullptr, int gate = 0);
void rope(Tensor& x, int heads, const std::vector<float>& cosine, const std::vector<float>& sine,
          bool adjacent);
}
}
