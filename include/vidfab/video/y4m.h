// YUV4MPEG2 writer.
//
// Y4M is a trivial uncompressed container that ffmpeg, mpv and VLC all read
// directly, which makes it the right first output: it introduces no encoder
// between our pixels and what the eye sees, so a wrong decode looks wrong
// rather than being masked by compression. MP4/AAC muxing comes later via
// dynamically linked ffmpeg.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vidfab/pixel_buffer.h"

namespace vidfab::video {

struct FrameRate {
  int numerator = 24;
  int denominator = 1;
};

// Writes planar float RGB in [0,1], shaped [3][frames][height][width], as
// 8-bit YUV 4:2:0 progressive. Conversion is BT.709 limited range, which is
// what a 768p+ generated video is expected to be interpreted as.
void write_y4m(const std::string& path, const PixelBuffer& planar_rgb, int frames,
               int height, int width, FrameRate fps = {});

// Writes a single frame as a binary PPM, for eyeballing one image without a
// video player.
void write_ppm(const std::string& path, const PixelBuffer& planar_rgb, int frames,
               int height, int width, int frame_index);

}  // namespace vidfab::video
