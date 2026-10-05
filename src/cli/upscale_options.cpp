#include "upscale_options.h"

namespace slopfab::cli {
bool UpscaleArguments::parse(int argc, char** argv, int& i) {
  const std::string a = argv[i];
  if (a.compare(0, 10, "--upscale-") != 0)
    return false;
  if (a == "--upscale-no-color-match") {
    options.color_match = false;
    return true;
  }
  if (++i >= argc)
    throw std::invalid_argument(a + " needs a value");
  const std::string value = argv[i];
  auto integer = [](const std::string& text) {
    size_t end = 0;
    int n = std::stoi(text, &end);
    if (end != text.size())
      throw std::invalid_argument("invalid upscale integer: " + text);
    return n;
  };
  if (a == "--upscale-method") {
    method = parse_upscale_method(value);
    enabled = true;
  } else if (a == "--upscale-model") {
    if (value.empty())
      throw std::invalid_argument(a + " needs a path");
    model = value;
    enabled = true;
  } else if (a == "--upscale-tile") {
    options.tile_size = integer(value);
    tile_set = true;
  } else if (a == "--upscale-tile-pad")
    options.tile_pad = integer(value);
  else if (a == "--upscale-pre-pad")
    options.pre_pad = integer(value);
  else if (a == "--upscale-vae")
    options.vae_path = value;
  else if (a == "--upscale-segment-frames")
    options.segment_frames = integer(value);
  else if (a == "--upscale-device")
    options.device = integer(value);
  else if (a == "--upscale-resolution") {
    const auto x = value.find('x');
    if (x == std::string::npos)
      throw std::invalid_argument(a + " requires WIDTHxHEIGHT");
    options.width = integer(value.substr(0, x));
    options.height = integer(value.substr(x + 1));
  } else if (a == "--upscale-seed") {
    size_t end = 0;
    if (value.empty() || value[0] == '-')
      throw std::invalid_argument("upscale seed must be nonnegative");
    options.seed = std::stoull(value, &end);
    if (end != value.size())
      throw std::invalid_argument("invalid upscale seed");
  } else
    throw std::invalid_argument("unknown upscale option " + a);
  return true;
}

void UpscaleArguments::finish() {
  if (method == UpscaleMethod::kSeedVr2 && !tile_set)
    options.tile_size = 256;
  validate_upscale_options(options, method);
  if (enabled && model.empty())
    model = default_upscale_model(method);
}
}
