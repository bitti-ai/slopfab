#include "slopfab/text/prompt_embedding.h"
#include "slopfab/generate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "slopfab/audio/wav.h"
#include "slopfab/image.h"
#include "slopfab/cuda/profile.h"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/dit/denoise.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/dit/packing.h"
#include "slopfab/dit/ref2va.h"
#include "slopfab/dit/transformer.h"
#include "slopfab/text/encoder.h"
#include "slopfab/text/tokenizer.h"
#include "slopfab/sampler/scheduler.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sampler/noise.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vae/audio_decoder.h"
#include "slopfab/vae/vit_decoder.h"
#include "slopfab/vae/keyframe_encoder.h"
#include "slopfab/vae/audio_encoder.h"
#include "slopfab/reference_conditioning.h"
#include "slopfab/video/mux.h"
#include "slopfab/video/y4m.h"
#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/audio_decoder.h"
#include "slopfab/vulkan/dit_denoise.h"
#include "slopfab/vulkan/keyframe_encoder.h"
#include "slopfab/vulkan/reference_encoder.h"
#include "slopfab/vulkan/text_encoder.h"
#include "slopfab/vulkan/vae_decoder.h"
#endif

#include "helpers.h"

namespace slopfab::generation {
std::vector<uint8_t> resize_rgb_bilinear(const RGBImage& in, int width, int height) {
  std::vector<uint8_t> out(static_cast<size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      const float sy = (y + .5f) * in.height / height - .5f;
      const float sx = (x + .5f) * in.width / width - .5f;
      const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, in.height - 1);
      const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, in.width - 1);
      const int y1 = std::min(y0 + 1, in.height - 1), x1 = std::min(x0 + 1, in.width - 1);
      const float fy = std::clamp(sy - std::floor(sy), 0.0f, 1.0f);
      const float fx = std::clamp(sx - std::floor(sx), 0.0f, 1.0f);
      for (int c = 0; c < 3; ++c) {
        auto at = [&](int yy, int xx) {
          return in.pixels[(static_cast<size_t>(yy) * in.width + xx) * 3 + c];
        };
        const float v = (1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x1)) +
                        fy * ((1 - fx) * at(y1, x0) + fx * at(y1, x1));
        out[(static_cast<size_t>(y) * width + x) * 3 + c] =
            static_cast<uint8_t>(std::clamp(std::lround(v), 0l, 255l));
      }
    }
  return out;
}

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string strip_extension(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  const size_t dot = path.find_last_of('.');
  if (dot == std::string::npos)
    return path;
  if (slash != std::string::npos && dot < slash)
    return path;
  return path.substr(0, dot);
}

// The two VAEs both ship per-channel latent statistics as tensors. Prefer them
// over the copies in the config JSON: the tensors are what the checkpoint
// actually carries, and a config file can drift from the weights beside it.
std::vector<float> read_stat(const SafeTensors& st, const char* name, int expect) {
  const TensorView* found = st.find(name);
  if (found == nullptr && expect == 24) {
    if (std::strcmp(name, "latents_mean") == 0)
      return vae::default_video_latents_mean();
    if (std::strcmp(name, "latents_std") == 0)
      return vae::default_video_latents_std();
  }
  const TensorView& view = found ? *found : st.at(name);
  std::vector<float> out = to_f32(view);
  if (static_cast<int>(out.size()) != expect) {
    throw std::runtime_error(std::string("vae: ") + name + " has " + std::to_string(out.size()) +
                             " entries, expected " + std::to_string(expect));
  }
  return out;
}

// Same spelling as safetensors.cpp's, and file-local for the same reason:
// `std::getenv` is C4996 under /W4 on MSVC.
bool env_flag(const char* name) {
#ifdef _MSC_VER
  size_t len = 0;
  char buf[8] = {};
  if (getenv_s(&len, buf, sizeof(buf), name) != 0)
    return false;
  return len != 0 && buf[0] == '1';
#else
  const char* v = std::getenv(name);
  return v != nullptr && v[0] == '1';
#endif
}

#if SLOPFAB_WITH_VULKAN
vulkan::Device create_vulkan_inference_device(bool exact_h3, bool sage_attention) {
  if (!vulkan::Instance::available())
    throw std::runtime_error("Vulkan inference: no Vulkan loader is available");
  vulkan::Instance instance = vulkan::Instance::create();
  const std::vector<vulkan::PhysicalDevice> physical = instance.enumerate_devices();
  if (physical.empty())
    throw std::runtime_error("Vulkan inference: no compute device is available");
  const vulkan::DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 ||
      (exact_h3 && (!info.shader_float16 || !info.storage_buffer_16bit ||
                    !info.cooperative_matrix_bf16_f32_16x16x16)))
    throw std::runtime_error("Vulkan inference: device lacks required exact neural features");
  vulkan::DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = exact_h3;
  options.enable_storage_buffer_16bit = exact_h3;
  options.enable_cooperative_matrix = exact_h3;
  options.enable_shader_int8 = sage_attention && info.shader_int8;
  return physical.front().create_device(options);
}
#endif

}
