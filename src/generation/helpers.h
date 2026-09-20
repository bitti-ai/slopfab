#pragma once
#include <chrono>
#include <string>
#include <vector>
#include "slopfab/image.h"
#include "slopfab/safetensors.h"
#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/runtime.h"
#endif
namespace slopfab::generation {
using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point start);
std::string strip_extension(const std::string& path);
std::vector<float> read_stat(const SafeTensors&, const char* name, int expect);
bool env_flag(const char* name);
std::vector<uint8_t> resize_rgb_bilinear(const RGBImage&, int width, int height);
#if SLOPFAB_WITH_VULKAN
vulkan::Device create_vulkan_inference_device(bool exact_h3 = false, bool sage_attention = false);
#endif
}
