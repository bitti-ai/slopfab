#pragma once
#include "../krea/vulkan_runtime.h"

namespace slopfab::moss_vulkan {
namespace vk = slopfab::vulkan;
using music_gpu::Tensor;
using music_gpu::from_host;
using music_gpu::to_host;
using music_gpu::slice;
using music_gpu::concatenate;
using music_gpu::rotary;
using krea_vulkan::dispatch;

// Shared image-model operations in tensor_krea.comp (0..14 belong to Krea).
enum Operation {
  Rms = 16,
  Modulate,
  Activation,
  CausalConv,
  DeltaParameters,
  RecurrentDelta,
  LayerNorm,
  Gelu,
  CopyColumns,
  Scatter,
  FiboModulate,
  Scale,
  GatedResidual,
  FluxPatches,
  FluxGroupNorm
};

uint32_t float_bits(float value);

class Runtime : public krea_vulkan::Runtime {
public:
  using krea_vulkan::Runtime::Runtime;
  Tensor norm(const Tensor&, const std::string&, int width = 0, bool unit_offset = false,
              const Tensor* modulation = nullptr, int shift = 0, const Tensor* gate = nullptr,
              float epsilon = 1e-6f);
  Tensor attend(const Tensor&, const Tensor&, const Tensor&, int heads, int kv_heads,
                bool causal = false, bool float_probabilities = false);
};

void residual(Tensor&, const Tensor&, const Tensor* modulation = nullptr, int gate = 0);
Tensor activate(const Tensor&, const Tensor* gate = nullptr, bool situ = false,
                bool sigmoid_gate = false);
Tensor image_rope(int height, int width, int text_length, bool text_only = false);
Tensor text_rope(int length);
Tensor delta_attention(Runtime&, const Tensor&, const std::string&);
Tensor encode_text(Runtime&, const std::vector<int32_t>&);
Tensor image_prediction(Runtime&, const Tensor&, const Tensor&, int height, int width,
                        float timestep, const std::vector<int>& skip);
std::vector<float> decode_image(const std::string&, const std::vector<float>&, int, int);
std::vector<float> encode_flux_image(const std::string&, const std::vector<uint8_t>&, int, int);
}
