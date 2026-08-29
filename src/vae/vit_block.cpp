#include "vidfab/vae/vit_block.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "vidfab/tensor_convert.h"

namespace vidfab::vae {
namespace {

void require_shape(const TensorView& tensor, int64_t rows, int64_t columns) {
  if (tensor.shape.size() != 2 || tensor.shape[0] != rows ||
      tensor.shape[1] != columns) {
    throw std::runtime_error("video VAE block: tensor '" + tensor.name +
                             "' has an unexpected matrix shape");
  }
}

std::vector<uint16_t> load_f16_matrix(const SafeTensors& checkpoint,
                                      const std::string& name, int64_t rows,
                                      int64_t columns) {
  const TensorView& tensor = checkpoint.at(name);
  require_shape(tensor, rows, columns);
  if (tensor.dtype != DType::kF16 ||
      tensor.nbytes != static_cast<size_t>(rows * columns) * sizeof(uint16_t)) {
    throw std::runtime_error("video VAE block: tensor '" + name +
                             "' must be checkpoint-native fp16");
  }
  std::vector<uint16_t> result(static_cast<size_t>(rows * columns));
  std::memcpy(result.data(), tensor.data, tensor.nbytes);
  for (uint16_t& bits : result) {
    if ((bits & 0x7c00u) == 0 && (bits & 0x03ffu) != 0)
      bits &= 0x8000u;
  }
  return result;
}

std::vector<float> load_vector(const SafeTensors& checkpoint,
                               const std::string& name, int64_t count) {
  const TensorView& tensor = checkpoint.at(name);
  if (tensor.shape.size() != 1 || tensor.shape[0] != count) {
    throw std::runtime_error("video VAE block: tensor '" + name +
                             "' has an unexpected vector shape");
  }
  if (tensor.dtype != DType::kF16 && tensor.dtype != DType::kF32) {
    throw std::runtime_error("video VAE block: tensor '" + name +
                             "' must be fp16 or fp32");
  }
  return to_f32(tensor);
}

}  // namespace

ViTBlockWeightsView ViTBlockWeights::view() const noexcept {
  return {norm1.data(), norm2.data(), scale1.data(), scale2.data(),
          qkv_weight.data(), qkv_bias.data(), out_weight.data(), out_bias.data(),
          w1_weight.data(), w1_bias.data(), w2_weight.data(), w2_bias.data()};
}

uint64_t ViTBlockWeights::bytes() const noexcept {
  return static_cast<uint64_t>(norm1.size() + norm2.size() + scale1.size() +
      scale2.size() + qkv_bias.size() + out_bias.size() + w1_bias.size() +
      w2_bias.size()) * sizeof(float) +
      static_cast<uint64_t>(qkv_weight.size() + out_weight.size() +
      w1_weight.size() + w2_weight.size()) * sizeof(uint16_t);
}

ViTBlockWeights load_vit_block_weights(const SafeTensors& checkpoint,
                                       uint32_t layer,
                                       const ViTBlockConfig& config) {
  if (!checkpoint.is_open())
    throw std::invalid_argument("video VAE block: checkpoint is not open");
  if (config.dim == 0 || config.ffn_inner == 0) {
    throw std::invalid_argument("video VAE block: invalid load configuration");
  }
  const int64_t d = config.dim, inner = config.ffn_inner;
  const std::string p = "decoder.transformer_blocks." + std::to_string(layer) + ".";
  ViTBlockWeights result;
  result.norm1 = load_vector(checkpoint, p + "norm1.weight", d);
  result.norm2 = load_vector(checkpoint, p + "norm2.weight", d);
  result.scale1 = load_vector(checkpoint, p + "scale1", d);
  result.scale2 = load_vector(checkpoint, p + "scale2", d);
  result.qkv_weight = load_f16_matrix(checkpoint, p + "attn.to_qkv.weight", 3 * d, d);
  result.qkv_bias = load_vector(checkpoint, p + "attn.to_qkv.bias", 3 * d);
  result.out_weight = load_f16_matrix(checkpoint, p + "attn.to_out.weight", d, d);
  result.out_bias = load_vector(checkpoint, p + "attn.to_out.bias", d);
  result.w1_weight = load_f16_matrix(checkpoint, p + "ff.w1.weight", 2 * inner, d);
  result.w1_bias = load_vector(checkpoint, p + "ff.w1.bias", 2 * inner);
  result.w2_weight = load_f16_matrix(checkpoint, p + "ff.w2.weight", d, inner);
  result.w2_bias = load_vector(checkpoint, p + "ff.w2.bias", d);
  return result;
}

}  // namespace vidfab::vae
