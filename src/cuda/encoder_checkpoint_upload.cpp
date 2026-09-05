#include "slopfab/text/encoder.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace slopfab::text {

void upload_layer_direct(const SafeTensors& checkpoint,
                         const EncoderConfig& config, int layer,
                         const LayerLayout& layout, uint8_t* dst,
                         void* stream) {
  if (layer < 0 || layer >= config.num_layers) {
    throw std::runtime_error("text encoder: upload_layer_direct layer out of range");
  }
  const std::string prefix = "model.layers." + std::to_string(layer) + ".";
  for (int i = 0; i < kLayerTensorCount; ++i) {
    const TensorSpec spec = layer_tensor_spec(
        config, static_cast<LayerTensor>(i));
    if (!spec.present()) continue;
    const std::string name = prefix + spec.suffix;
    const TensorView& view = checkpoint.at(name);
    if (view.nbytes != layout.bytes[i]) {
      throw std::runtime_error("text encoder: " + name +
                               " byte count changed after validation");
    }
    const cudaError_t status = cudaMemcpyAsync(
        dst + layout.offset[i], view.data, view.nbytes,
        cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) {
      throw std::runtime_error("text encoder: uploading " + name +
                               " failed: " + cudaGetErrorString(status));
    }
  }
}

}  // namespace slopfab::text
