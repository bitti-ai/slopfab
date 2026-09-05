#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "slopfab/dtype.h"
#include "slopfab/safetensors.h"
#include "slopfab/text/qwen_vision.h"

namespace {
double rms(const std::vector<uint16_t>& x) {
  long double sum = 0;
  for (uint16_t b : x) {
    const float v = slopfab::bf16_to_f32(b);
    if (!std::isfinite(v)) throw std::runtime_error("non-finite visual output");
    sum += static_cast<long double>(v) * v;
  }
  return std::sqrt(static_cast<double>(sum / x.size()));
}
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: slopfab_qwenvisionprobe <qwen.safetensors>\n");
    return 2;
  }
  try {
    std::vector<uint8_t> rgb(256 * 256 * 3);
    for (int y = 0; y < 256; ++y) for (int x = 0; x < 256; ++x) {
      const size_t i = (static_cast<size_t>(y) * 256 + x) * 3;
      rgb[i] = static_cast<uint8_t>((x * 17 + y * 3) & 255);
      rgb[i + 1] = static_cast<uint8_t>((x * 5 + y * 11) & 255);
      rgb[i + 2] = static_cast<uint8_t>((x ^ y) & 255);
    }
    const auto pixels = slopfab::text::qwen3vl_patchify_resized_rgb(rgb, 256, 256);
    slopfab::SafeTensors checkpoint; checkpoint.open(argv[1]);
    slopfab::text::QwenVisionEncoder encoder;
    const auto t0 = std::chrono::steady_clock::now(); encoder.load(checkpoint);
    const auto t1 = std::chrono::steady_clock::now(); const auto out = encoder.encode({pixels});
    const auto t2 = std::chrono::steady_clock::now();
    if (out.tokens != 64 || out.hidden != 5120 || out.main.size() != 64ull * 5120)
      throw std::runtime_error("unexpected main visual output shape");
    std::printf("main       [64,5120] rms %.9g\n", rms(out.main));
    for (int i = 0; i < 3; ++i) {
      if (out.deepstack[i].size() != out.main.size())
        throw std::runtime_error("unexpected DeepStack output shape");
      std::printf("deepstack%d [64,5120] rms %.9g\n", i, rms(out.deepstack[i]));
    }
    size_t free_b = 0, total_b = 0; cudaMemGetInfo(&free_b, &total_b);
    std::printf("load %.3f s  forward %.3f s  device used %.3f GB\n",
      std::chrono::duration<double>(t1-t0).count(), std::chrono::duration<double>(t2-t1).count(),
      static_cast<double>(total_b-free_b)/1e9);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwenvisionprobe: %s\n", e.what()); return 1;
  }
}
