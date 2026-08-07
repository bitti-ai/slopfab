#include <cstdio>
#include <exception>

#include "vidfab/safetensors.h"
#include "vidfab/vae/keyframe_encoder.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: vidfab_vaecudaprobe <minimax_h3_video_vae_fp16.safetensors>\n");
    return 2;
  }
  try {
    vidfab::SafeTensors checkpoint;
    checkpoint.open(argv[1]);
    vidfab::vae::KeyframeEncoder encoder(checkpoint);
    std::puts("H3 image encoder weights uploaded to CUDA");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
