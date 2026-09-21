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

#include "slopfab/pixel_buffer.h"

namespace slopfab::video {

class FrameConverter {
public:
  virtual ~FrameConverter() = default;
  virtual void convert(const float* r, const float* g, const float* b, int height, int width,
                       uint8_t* y_plane, int y_stride, uint8_t* u_plane, int u_stride,
                       uint8_t* v_plane, int v_stride) = 0;
};

struct FrameRate {
  int numerator = 24;
  int denominator = 1;
};

// Writes planar float RGB in [0,1], shaped [3][frames][height][width], as
// 8-bit YUV 4:2:0 progressive. Conversion is BT.709 limited range, which is
// what a 768p+ generated video is expected to be interpreted as.
void write_y4m(const std::string& path, const PixelBuffer& planar_rgb, int frames, int height,
               int width, FrameRate fps = {}, FrameConverter* converter = nullptr);

// Converts one frame of planar float RGB in [0,1] to 8-bit YUV 4:2:0, BT.709
// limited range, into caller-supplied planes with arbitrary strides — which
// is what an AVFrame hands us, since ffmpeg pads every row for alignment.
//
// This is the *one* implementation of the transform. It used to have a second,
// byte-identical copy inside write_y4m, kept honest by a test that compared the
// two outputs; there is no way to keep two copies of a colour transform in step
// by hand, and a mismatch shows up as a gamma shift when a viewer switches
// between the .y4m and the .mp4. It lives here rather than in mux.h so the
// dependency runs the right way: the plain writer owns the colour transform and
// the muxer borrows it, not the reverse.
//
// Writes all `height` luma rows and `height/2` by `width/2` chroma samples, so
// it leaves no byte of the caller's planes untouched at any extent. 4:2:0 is
// not defined for odd extents — an odd final row or column contributes luma
// only and is not represented in chroma — and both container writers reject
// them before calling here.
void rgb_frame_to_yuv420(const float* r, const float* g, const float* b, int height, int width,
                         uint8_t* y_plane, int y_stride, uint8_t* u_plane, int u_stride,
                         uint8_t* v_plane, int v_stride);

// Writes a single frame as a binary PPM, for eyeballing one image without a
// video player.
void write_ppm(const std::string& path, const PixelBuffer& planar_rgb, int frames, int height,
               int width, int frame_index);

} // namespace slopfab::video
