#include "mux_internal.h"

namespace slopfab::video::mux_detail {
const AVCodec* find_video_encoder(const Api& api, std::string* name) {
  // libx264 first because it is what everything plays; then whatever this
  // build calls its H.264 encoder; then MPEG-4 part 2, which every ffmpeg
  // ships and every MP4 player still decodes. A build with none of these is a
  // legitimate configuration, not a bug — hence kEncoderMissing.
  struct Candidate {
    const char* by_name;
    int by_id;
  };
  const Candidate candidates[] = {
      {"libx264", 0}, {"libopenh264", 0}, {nullptr, kCodecIdH264}, {"mpeg4", 0},
      {nullptr, kCodecIdMpeg4}};
  for (const Candidate& c : candidates) {
    const AVCodec* codec =
        c.by_name != nullptr ? api.avcodec_find_encoder_by_name(c.by_name)
                             : api.avcodec_find_encoder(c.by_id);
    if (codec == nullptr) continue;
    bool all = false;
    const std::vector<int> pix = supported_config<int>(api, codec, kConfigPixFormat, &all);
    if (!all && !contains(pix, kPixFmtYuv420p)) continue;
    *name = c.by_name != nullptr ? c.by_name : (c.by_id == kCodecIdH264 ? "h264" : "mpeg4");
    return codec;
  }
  return nullptr;
}

const AVCodec* find_audio_encoder(const Api& api, std::string* name) {
  const char* names[] = {"aac", "libfdk_aac"};
  for (const char* n : names) {
    if (const AVCodec* codec = api.avcodec_find_encoder_by_name(n)) {
      *name = n;
      return codec;
    }
  }
  if (const AVCodec* codec = api.avcodec_find_encoder(kCodecIdAac)) {
    *name = "aac";
    return codec;
  }
  return nullptr;
}

}  // namespace slopfab::video::mux_detail
