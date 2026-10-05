#include "slopfab/upscale.h"
#include "slopfab/seedvr2.h"
#include "../upscale/backend.h"

#include <limits>

namespace slopfab::upscale_detail {
std::vector<ConvSpec> model_convolutions() {
  std::vector<ConvSpec> specs{{"conv_first", 3, 64}};
  for (int block = 0; block < 23; ++block)
    for (int rdb = 1; rdb <= 3; ++rdb)
      for (int conv = 1; conv <= 5; ++conv)
        specs.push_back({"body." + std::to_string(block) + ".rdb" + std::to_string(rdb) + ".conv" +
                             std::to_string(conv),
                         64 + (conv - 1) * 32, conv == 5 ? 64 : 32});
  for (const char* name : {"conv_body", "conv_up1", "conv_up2", "conv_hr"})
    specs.push_back({name, 64, 64});
  specs.push_back({"conv_last", 64, 3});
  return specs;
}
} // namespace slopfab::upscale_detail

namespace slopfab {
UpscaleMethod parse_upscale_method(std::string_view name) {
  if (name == "realesrgan")
    return UpscaleMethod::kRealEsrgan;
  if (name == "seedvr2")
    return UpscaleMethod::kSeedVr2;
  throw std::invalid_argument("unknown upscale method '" + std::string(name) +
                              "'; supported: realesrgan, seedvr2");
}

const char* upscale_method_name(UpscaleMethod method) {
  switch (method) {
  case UpscaleMethod::kRealEsrgan:
    return "realesrgan";
  case UpscaleMethod::kSeedVr2:
    return "seedvr2";
  }
  throw std::invalid_argument("unknown upscale method");
}

int upscale_scale_factor(UpscaleMethod method) {
  switch (method) {
  case UpscaleMethod::kRealEsrgan:
  case UpscaleMethod::kSeedVr2:
    return 4;
  }
  throw std::invalid_argument("unknown upscale method");
}

const char* default_upscale_model(UpscaleMethod method) {
  switch (method) {
  case UpscaleMethod::kRealEsrgan:
    return "weights/upscaler/RealESRGAN_x4plus.safetensors";
  case UpscaleMethod::kSeedVr2:
    return "weights/seedvr2/seedvr2_3b_fp16.safetensors";
  }
  throw std::invalid_argument("unknown upscale method");
}

void validate_upscale_checkpoint(const SafeTensors& checkpoint, UpscaleMethod method) {
  switch (method) {
  case UpscaleMethod::kRealEsrgan:
    return validate_realesrgan_checkpoint(checkpoint);
  case UpscaleMethod::kSeedVr2:
    if (checkpoint.at("vid_in.proj.weight").shape != std::vector<int64_t>{2560, 132} ||
        checkpoint.at("vid_out.proj.weight").shape != std::vector<int64_t>{64, 2560})
      throw std::invalid_argument("SeedVR2 requires the Comfy-Org 3B checkpoint");
    return;
  }
  throw std::invalid_argument("unknown upscale method");
}

void validate_upscale_options(const UpscaleOptions& o, UpscaleMethod method) {
  upscale_method_name(method);
  if (method == UpscaleMethod::kSeedVr2) {
    seedvr2::Options seed;
    if ((o.width == 0) != (o.height == 0))
      throw std::invalid_argument("SeedVR2 target width and height must both be specified");
    seed.width = o.width ? o.width : 16;
    seed.height = o.height ? o.height : 16;
    seed.segment_frames = o.segment_frames;
    seed.vae_tile = o.tile_size;
    seed.device = o.device;
    seedvr2::validate(seed);
    if (o.vae_path.empty())
      throw std::invalid_argument("SeedVR2 needs a VAE path");
    return;
  }
  if (o.width || o.height)
    throw std::invalid_argument("Real-ESRGAN has a fixed 4x output size");
  if (o.tile_size < 0 || o.tile_size > 512 || o.tile_pad < 0 || o.tile_pad > 256 || o.pre_pad < 0 ||
      o.pre_pad > 256)
    throw std::invalid_argument("upscale: tile size must be 0..512; padding must be 0..256");
}

std::pair<int, int> upscale_dimensions(int height, int width, UpscaleMethod method,
                                       const UpscaleOptions& options) {
  validate_upscale_options(options, method);
  const int scale = upscale_scale_factor(method);
  if (height <= 0 || width <= 0 || height > std::numeric_limits<int>::max() / scale ||
      width > std::numeric_limits<int>::max() / scale)
    throw std::invalid_argument("upscale: invalid or excessive image dimensions");
  if (method == UpscaleMethod::kSeedVr2) {
    auto resolved = options;
    resolved.height = options.height ? options.height : height * scale;
    resolved.width = options.width ? options.width : width * scale;
    validate_upscale_options(resolved, method);
    return {resolved.height, resolved.width};
  }
  return {height * scale, width * scale};
}

size_t upscale_output_elements(int frames, int height, int width, UpscaleMethod method,
                               const UpscaleOptions& options) {
  if (frames <= 0)
    throw std::invalid_argument("upscale: invalid frame count");
  const auto dims = upscale_dimensions(height, width, method, options);
  size_t count = 3;
  for (size_t axis : {size_t(frames), size_t(dims.first), size_t(dims.second)}) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(float) / axis)
      throw std::overflow_error("upscale: output dimensions overflow");
    count *= axis;
  }
  return count;
}

seedvr2::Options seedvr2_options(const UpscaleOptions& o, int height, int width) {
  const auto dims = upscale_dimensions(height, width, UpscaleMethod::kSeedVr2, o);
  seedvr2::Options result;
  result.width = dims.second;
  result.height = dims.first;
  result.vae = o.vae_path;
  result.vae_tile = o.tile_size;
  result.segment_frames = o.segment_frames;
  result.seed = o.seed;
  result.device = o.device;
  result.color_match = o.color_match;
  return result;
}

void validate_realesrgan_checkpoint(const SafeTensors& checkpoint) {
  for (const auto& spec : upscale_detail::model_convolutions()) {
    auto require = [&](const std::string& name, const std::vector<int64_t>& shape) {
      const auto* t = checkpoint.find(name);
      if (!t || t->shape != shape ||
          (t->dtype != DType::kF32 && t->dtype != DType::kF16 && t->dtype != DType::kBF16))
        throw std::invalid_argument("Real-ESRGAN x4plus: missing or incompatible tensor " + name);
    };
    require(spec.name + ".weight", {spec.output, spec.input, 3, 3});
    require(spec.name + ".bias", {spec.output});
  }
  if (checkpoint.find("body.23.rdb1.conv1.weight"))
    throw std::invalid_argument("Real-ESRGAN x4plus requires exactly 23 RRDB blocks");
}
} // namespace slopfab
