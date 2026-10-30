#pragma once
#include "../fibo/runtime.cuh"

namespace slopfab::krea {
using moss::Tensor;
using moss::Runtime;
using moss::BFloat;
using moss::from_host;
using moss::to_host;
Tensor fuse_text(Runtime& rt, const Tensor& context);
Tensor predict(Runtime& rt, const Tensor& latent, const Tensor& text, int height, int width,
               float sigma, bool streaming, const std::function<void()>& check_cancel);
std::vector<float> decode_image(const std::string& path, const std::vector<float>& latent,
                                int height, int width);
}
