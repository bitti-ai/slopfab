#pragma once
#include "../moss/runtime.cuh"
#include "slopfab/fibo.h"
#include "slopfab/text/qwen_vision.h"
#include <functional>

namespace slopfab::fibo {
RGBImage render_text(const std::string&);
}
namespace slopfab::fibo_vulkan {
using moss_vulkan::Runtime;
using moss_vulkan::Tensor;
using moss_vulkan::from_host;
using moss_vulkan::to_host;
using fibo::gist_count;
using fibo::edit_prompt;
using fibo::resize_vision_image;
using fibo::crop_image;
using fibo::render_text;

struct ImageLatent {
  int height = 0, width = 0;
  std::vector<float> values;
};
Tensor rows(const Tensor&, int start, int count);
Tensor join(const Tensor&, const Tensor&);
Tensor reshape(const Tensor&, int rows, int cols);
void copy_columns(const Tensor&, Tensor&, int offset);
Tensor layer_norm(Runtime&, const Tensor&, const std::string&, bool rms = false, float eps = 1e-6f);
Tensor gelu(const Tensor&, bool approximate);
void rope(Tensor&, int, const std::vector<float>&, const std::vector<float>&, bool);
void scatter(Tensor&, const Tensor&, const std::vector<int32_t>&, bool);
void modulate(Tensor&, const Tensor&, int start, int end, int field, bool gate);
void scale(Tensor&, const Tensor&);
void gated_residual(Tensor&, const Tensor&, float);
Tensor vision(Runtime&, const text::QwenPixelValues&, std::vector<Tensor>&);
std::vector<Tensor> encode_prompt(const std::string&, const std::string&, const std::string&,
                                  const std::vector<RGBImage>&, const std::function<void()>&);
Tensor perceiver(Runtime&, const Tensor&, bool streaming);
Tensor predict(Runtime&, const Tensor&, int, int, const std::vector<ImageLatent>&,
               const std::vector<Tensor>&, const Tensor&, float, bool,
               const std::function<void()>&);
ImageLatent encode_image(const std::string&, const RGBImage&);
}
