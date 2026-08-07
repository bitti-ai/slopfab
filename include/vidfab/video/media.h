#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vidfab::video {

struct DecodedVideoFrame {
  int width = 0;
  int height = 0;
  int64_t timestamp = 0;
  std::vector<uint8_t> rgb24;
};

// Demuxes media, selects its best video stream and returns its first decoded
// frame as tightly packed RGB24. The demux/decode boundary intentionally keeps
// a timestamp in the result for future time-based sampling.
DecodedVideoFrame decode_first_video_frame(const std::string& path);

}  // namespace vidfab::video
