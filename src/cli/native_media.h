#pragma once
#include "slopfab/generate.h"
#include "slopfab/reference_media.h"
#include <string>
#include <vector>

namespace slopfab::cli {
// Native, dependency-free PNG encoding. Pixels are interleaved RGB8.
void write_png(const std::string& path, const uint8_t* rgb, int width, int height);

struct NativeVideo {
  int width = 0, height = 0;
  video::FrameRate fps;
  std::vector<std::vector<uint8_t>> frames;
};

NativeVideo read_y4m(const std::string& path);
ReferenceAudio read_wav(const std::string& path);
}
