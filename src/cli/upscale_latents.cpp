#include "commands.h"
#include "slopfab/continuation.h"
#include "slopfab/latent_upscale.h"
#include <cstdio>
#include <stdexcept>

namespace slopfab::cli {
int cmd_upscale_latents(int argc, char** argv) {
  std::string input, output, model = kDefaultLatentUpscaleModel;
  LatentUpscaleOptions options;
#if SLOPFAB_WITH_CUDA
  DeviceBackend backend = DeviceBackend::kCuda;
#else
  DeviceBackend backend = DeviceBackend::kVulkan;
#endif
  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc)
        throw std::invalid_argument(arg + " requires a value");
      return argv[i];
    };
    if (arg == "--input") input = next();
    else if (arg == "--out") output = next();
    else if (arg == "--model") model = next();
    else if (arg == "--scale") {
      const auto value = next();
      size_t used = 0;
      options.scale = std::stof(value, &used);
      if (used != value.size()) throw std::invalid_argument("invalid --scale");
    } else if (arg == "--no-chunking") options.temporal_chunking = false;
    else if (arg == "--inference-backend") {
      const auto value = next();
      if (value == "cuda") backend = DeviceBackend::kCuda;
      else if (value == "vulkan") backend = DeviceBackend::kVulkan;
      else throw std::invalid_argument("backend must be cuda or vulkan");
    } else throw std::invalid_argument("unknown upscale-latents option: " + arg);
  }
  if (input.empty() || output.empty() || model.empty())
    throw std::invalid_argument("upscale-latents requires --input, --out and a nonempty model path");
  validate_latent_upscale_options(options);
#if !SLOPFAB_WITH_CUDA
  if (backend == DeviceBackend::kCuda) throw std::invalid_argument("CUDA backend not compiled in");
#endif
#if !SLOPFAB_WITH_VULKAN
  if (backend == DeviceBackend::kVulkan) throw std::invalid_argument("Vulkan backend not compiled in");
#endif
#if SLOPFAB_WITH_CUDA || SLOPFAB_WITH_VULKAN
  const auto source = LatentClip::load(input);
  const auto enlarged = upscale_latent_clip(*source, model, backend, options);
  enlarged->save(output);
  std::printf("saved %s: %dx%d -> %dx%d, %d frames; audio preserved\n", output.c_str(),
              source->width, source->height, enlarged->width, enlarged->height, enlarged->frames);
  return 0;
#else
  throw std::runtime_error("latent upscaling requires a CUDA or Vulkan build");
#endif
}
}
