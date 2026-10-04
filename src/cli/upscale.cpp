#include "commands.h"
#include "slopfab/image.h"
#include "slopfab/upscale.h"
#include "slopfab/safetensors_write.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace slopfab::cli {
int cmd_upscale(int argc, char** argv) {
  std::string input, output, dump;
  std::string model = "weights/upscaler/RealESRGAN_x4plus.safetensors";
#if SLOPFAB_WITH_CUDA
  auto backend = DeviceBackend::kCuda;
#else
  auto backend = DeviceBackend::kVulkan;
#endif
  UpscaleOptions options;
  auto method = UpscaleMethod::kRealEsrgan;
  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];
    if (i + 1 >= argc)
      throw std::invalid_argument(arg + " needs a value");
    const std::string value = argv[++i];
    if (arg == "--input")
      input = value;
    else if (arg == "--out")
      output = value;
    else if (arg == "--dump")
      dump = value;
    else if (arg == "--upscale-model")
      model = value;
    else if (arg == "--upscale-method")
      method = parse_upscale_method(value);
    else if (arg == "--inference-backend") {
      if (value == "cuda")
        backend = DeviceBackend::kCuda;
      else if (value == "vulkan")
        backend = DeviceBackend::kVulkan;
      else
        throw std::invalid_argument("--inference-backend requires cuda or vulkan");
    } else if (arg == "--upscale-tile" || arg == "--upscale-tile-pad" ||
               arg == "--upscale-pre-pad") {
      size_t used = 0;
      const int number = std::stoi(value, &used);
      if (used != value.size())
        throw std::invalid_argument(arg + " requires an integer");
      if (arg == "--upscale-tile")
        options.tile_size = number;
      else if (arg == "--upscale-tile-pad")
        options.tile_pad = number;
      else
        options.pre_pad = number;
    } else
      throw std::invalid_argument("upscale: unknown option " + arg);
  }
  validate_upscale_options(options);
  if (input.empty() || output.empty() || model.empty())
    throw std::invalid_argument("upscale requires --input, --out and a model path");
  if (std::filesystem::path(output).extension() != ".ppm")
    throw std::invalid_argument("upscale: --out must have a .ppm extension");
  auto image = load_reference_image(input);
  const size_t plane = size_t(image.height) * image.width;
  PixelBuffer pixels(plane * 3);
  for (size_t i = 0; i < plane; ++i)
    for (size_t c = 0; c < 3; ++c)
      pixels[c * plane + i] = image.pixels[i * 3 + c] / 255.0f;
  auto upscaler = make_upscaler(method, model, backend);
  auto up = upscaler->upscale(pixels, 1, image.height, image.width, options);
  const int scale = upscale_scale_factor(method);
  const int h = image.height * scale, w = image.width * scale;
  const size_t out_plane = size_t(h) * w;
  std::vector<uint8_t> rgb(up.size());
  for (size_t i = 0; i < out_plane; ++i)
    for (size_t c = 0; c < 3; ++c)
      rgb[i * 3 + c] = uint8_t(std::lround(up[c * out_plane + i] * 255));
  std::ofstream file(std::filesystem::u8path(output), std::ios::binary);
  file << "P6\n" << w << ' ' << h << "\n255\n";
  file.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
  file.close();
  if (!file)
    throw std::runtime_error("upscale: cannot write " + output);
  if (!dump.empty())
    write_safetensors(dump, {{"pixels", {3, 1, h, w}, std::vector<float>(up.begin(), up.end())}});
  std::printf("upscaled %dx%d -> %dx%d: %s\n", image.width, image.height, w, h, output.c_str());
  return 0;
}
} // namespace slopfab::cli
