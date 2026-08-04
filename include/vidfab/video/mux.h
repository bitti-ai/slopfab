// Container output: WAV, and MP4/AAC through a dynamically loaded ffmpeg.
//
// ffmpeg is linked **dynamically and at runtime**, never statically and never
// at link time. Two reasons, and both are requirements rather than
// preferences: LGPL compliance depends on the user being able to substitute
// their own build, and a binary that hard-fails on a missing DLL is a worse
// product than one that writes a .y4m and says so.
//
// So the contract is: try to load ffmpeg's shared libraries; if any part of
// that fails — not installed, wrong major version, missing symbol — report it
// and let the caller fall back to the uncompressed `.y4m` + `.wav` pair, which
// mpv, VLC and ffmpeg itself all read directly. `generate` never fails because
// of muxing.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vidfab/video/y4m.h"

namespace vidfab::video {

// Why an MP4 could not be produced. `kOk` means the file was written.
enum class MuxStatus {
  kOk,
  kLibraryNotFound,   // no avcodec/avformat/avutil/swscale on the system
  kSymbolMissing,     // found, but not the version we know how to drive
  kEncoderMissing,    // no H.264 or AAC encoder in this build
  kWriteFailed,
};

const char* mux_status_message(MuxStatus s);

struct MuxRequest {
  std::string path;

  // Planar float RGB in [0,1], shaped [3][frames][height][width] — the same
  // layout `DecodedVideo::data` and `write_y4m` use.
  const std::vector<float>* video = nullptr;
  int frames = 0;
  int height = 0;
  int width = 0;
  FrameRate fps;

  // Interleaved float samples in [-1, 1]. Optional: a video-only MP4 is
  // written when this is null or empty.
  const std::vector<float>* audio = nullptr;
  int audio_channels = 2;
  int audio_sample_rate = 32000;

  int video_bitrate = 12'000'000;
  int audio_bitrate = 192'000;
};

// True when ffmpeg's shared libraries loaded and carry the symbols we need.
// Safe to call repeatedly; the probe happens once. `detail` receives a
// human-readable explanation of a failure, including which library was tried.
bool ffmpeg_available(std::string* detail = nullptr);

// Version string of the loaded libraries, or an empty string.
std::string ffmpeg_version();

// Encodes H.264 + AAC into an MP4. Returns kOk only when the file is complete
// and closed.
MuxStatus write_mp4(const MuxRequest& request);

}  // namespace vidfab::video
