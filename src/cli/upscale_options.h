#pragma once
#include "slopfab/upscale.h"

namespace slopfab::cli {
struct UpscaleArguments {
  UpscaleMethod method = UpscaleMethod::kRealEsrgan;
  UpscaleOptions options;
  std::string model;
  bool enabled = false, tile_set = false;
  bool parse(int argc, char** argv, int& i);
  void finish();
};
}
