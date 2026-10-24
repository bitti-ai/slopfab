#pragma once
#include "../moss/runtime.cuh"
#include "slopfab/fibo.h"
#include "slopfab/text/qwen_vision.h"
#include <functional>

namespace slopfab::fibo {
using moss::Tensor;
using moss::Runtime;
using moss::BFloat;
using moss::from_host;
using moss::to_host;

struct ImageLatent {
  int height = 0, width = 0;
  std::vector<float> values;
};

Tensor rows(const Tensor& x, int start, int count);
Tensor join(const Tensor& a, const Tensor& b);
Tensor layer_norm(Runtime& rt, const Tensor& x, const std::string& prefix, bool rms = false,
                  float eps = 1e-6f);
Tensor gelu(const Tensor& x, bool approximate);
void rope(Tensor& x, int heads, const std::vector<float>& cosine, const std::vector<float>& sine,
          bool adjacent);
void scatter(Tensor& x, const Tensor& source, const std::vector<int32_t>& indices, bool add);
Tensor vision(Runtime& rt, const text::QwenPixelValues& image, std::vector<Tensor>& deep);
std::vector<Tensor> encode_prompt(const std::string& path, const std::string& tokenizer_path,
                                  const std::string& prompt, const std::vector<RGBImage>& images,
                                  const std::function<void()>& check_cancel);
Tensor perceiver(Runtime& rt, const Tensor& context, bool streaming);
Tensor predict(Runtime& rt, const Tensor& latent, int height, int width,
               const std::vector<ImageLatent>& references, const std::vector<Tensor>& context,
               const Tensor& gist, float sigma, bool streaming,
               const std::function<void()>& check_cancel);
ImageLatent encode_image(const std::string& path, const RGBImage& image);
RGBImage render_text(const std::string& text);
}
