#pragma once
#include <chrono>
#include <optional>
#include "slopfab/attention_mode.h"
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
// Pure selection is shared by generation and its hardware-independent tests.
size_t select_vulkan_inference_device(const std::vector<vulkan::DeviceInfo>& devices,
                                      bool portable_arithmetic, bool cooperative,
                                      bool sage_attention,
                                      std::optional<AttentionMode> attention = std::nullopt);
vulkan::Device create_vulkan_inference_device(bool portable_arithmetic, bool cooperative = false,
                                               bool sage_attention = false,
                                               std::optional<AttentionMode> attention = std::nullopt);
#endif
}
