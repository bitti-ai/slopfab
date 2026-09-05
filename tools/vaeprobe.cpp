#include <cstdio>
#include <exception>

#include "slopfab/safetensors.h"
#include "slopfab/vae/keyframe_encoder.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: slopfab_vaeprobe <minimax_h3_video_vae_fp16.safetensors>\n");
    return 2;
  }
  try {
    slopfab::SafeTensors checkpoint;
    checkpoint.open(argv[1]);
    const auto summary = slopfab::vae::validate_keyframe_encoder_weights(checkpoint);
    std::printf("H3 image encoder: %zu tensors, %.3f GiB, graph valid\n", summary.tensors,
                static_cast<double>(summary.bytes) / (1024.0 * 1024.0 * 1024.0));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
