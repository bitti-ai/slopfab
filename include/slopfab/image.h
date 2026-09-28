#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace slopfab {

struct RGBImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

// Binary PPM (P6) is read directly, and anything else is handed to whichever
// decoder this build has. With FFmpeg, that is a demux plus decode of the
// first video frame, so a video file is a legal reference image. Without it,
// the platform decoder below is used instead, so a build with no FFmpeg still
// reads ordinary still images.
RGBImage load_reference_image(const std::string& path);

// Decodes through the platform's own image stack: WIC on Windows, which covers
// PNG, JPEG, BMP, GIF, TIFF and whatever else has a codec installed. Linux
// uses libpng and libjpeg for PNG/JPEG. Throws on other platforms with no
// equivalent. Compiled in every configuration so that the two decoders
// can be compared, but only reached by `load_reference_image` when this build
// has no FFmpeg.
RGBImage load_platform_image(const std::string& path);

// Scale-adaptive separable Lanczos-3 with half-pixel pixel-center mapping.
RGBImage resize_reference_lanczos(const RGBImage& image, int width, int height);

} // namespace slopfab
