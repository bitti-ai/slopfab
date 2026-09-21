#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Stated rather than defaulted, because `#if UNDEFINED` is a silent 0 that no
// warning level diagnoses. A consumer compiling against these headers without
// the slopfab_core target's definitions would therefore be told the demuxer
// below does not exist — in a build where it does — and would find out as a
// confusing error in their own code. One line naming the cause is better.
#if !defined(SLOPFAB_WITH_FFMPEG)
#error                                                                                             \
    "slopfab: SLOPFAB_WITH_FFMPEG is not defined. Define it as 1 or 0 to match the library you are linking (in this build tree the slopfab_core target defines it for you)."
#endif

namespace slopfab::video {

struct DecodedVideoFrame {
  int width = 0;
  int height = 0;
  int64_t timestamp = 0;
  std::vector<uint8_t> rgb24;
};

// Demuxes media, selects its best video stream and returns its first decoded
// frame as tightly packed RGB24. The demux/decode boundary intentionally keeps
// a timestamp in the result for future time-based sampling.
//
// Declared only where it exists. A build without FFmpeg has no demuxer at all,
// so this is a compile error there rather than a link error or — worse — a
// stub that throws at the moment a user finally points at an MP4. Reference
// images go through `load_platform_image` in that configuration; see image.h.
#if SLOPFAB_WITH_FFMPEG
DecodedVideoFrame decode_first_video_frame(const std::string& path);
#endif

} // namespace slopfab::video
