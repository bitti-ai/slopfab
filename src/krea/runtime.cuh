#pragma once
#ifdef SLOPFAB_KREA_VULKAN
#include "vulkan_runtime.h"
#define SLOPFAB_KREA_NS krea_vulkan
#else
#include "../fibo/runtime.cuh"
#define SLOPFAB_KREA_NS krea
#endif

namespace slopfab::SLOPFAB_KREA_NS {
#ifndef SLOPFAB_KREA_VULKAN
using moss::Tensor;
using moss::Runtime;
using moss::BFloat;
using moss::from_host;
using moss::to_host;

inline float round_bf(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

inline Tensor encode_prompt(const std::string& path, const std::string& tokenizer,
                            const std::string& prompt, const std::function<void()>& hook) {
  return fibo::encode_krea_prompt(path, tokenizer, prompt, hook);
}

namespace ops {
using fibo::rows;
using fibo::join;
using fibo::rope;
using fibo::gelu;
using moss::activate;
using moss::residual;
}
#endif
Tensor fuse_text(Runtime& rt, const Tensor& context);
Tensor predict(Runtime& rt, const Tensor& latent, const Tensor& text, int height, int width,
               float sigma, bool streaming, const std::function<void()>& check_cancel);
std::vector<float> decode_image(const std::string& path, const std::vector<float>& latent,
                                int height, int width);
}
