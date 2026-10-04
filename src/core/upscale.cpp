#include "slopfab/upscale.h"
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
  throw std::invalid_argument("unknown upscale method '" + std::string(name) +
                              "'; supported: realesrgan");
}

const char* upscale_method_name(UpscaleMethod method) {
  switch (method) {
  case UpscaleMethod::kRealEsrgan:
    return "realesrgan";
  }
  throw std::invalid_argument("unknown upscale method");
}

int upscale_scale_factor(UpscaleMethod method) {
  switch (method) {
  case UpscaleMethod::kRealEsrgan:
    return 4;
  }
  throw std::invalid_argument("unknown upscale method");
}

void validate_upscale_checkpoint(const SafeTensors& checkpoint, UpscaleMethod method) {
  switch (method) {
  case UpscaleMethod::kRealEsrgan:
    return validate_realesrgan_checkpoint(checkpoint);
  }
  throw std::invalid_argument("unknown upscale method");
}

void validate_upscale_options(const UpscaleOptions& o) {
  if (o.tile_size < 0 || o.tile_size > 512 || o.tile_pad < 0 || o.tile_pad > 256 || o.pre_pad < 0 ||
      o.pre_pad > 256)
    throw std::invalid_argument("upscale: tile size must be 0..512; padding must be 0..256");
}

size_t upscale_output_elements(int frames, int height, int width, UpscaleMethod method) {
  const int scale = upscale_scale_factor(method);
  if (frames <= 0 || height <= 0 || width <= 0 ||
      height > std::numeric_limits<int>::max() / scale ||
      width > std::numeric_limits<int>::max() / scale)
    throw std::invalid_argument("upscale: invalid or excessive image dimensions");
  size_t count = 3;
  for (size_t axis : {size_t(frames), size_t(height) * scale, size_t(width) * scale}) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(float) / axis)
      throw std::overflow_error("upscale: output dimensions overflow");
    count *= axis;
  }
  return count;
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
