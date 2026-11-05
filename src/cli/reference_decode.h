#pragma once
#include <string>
#include "slopfab/reference_media.h"

namespace slopfab::cli {
// CLI-only native Y4M/WAV adapter, never linked into slopfab.dll.
ReferenceMedia decode_reference_file(const std::string& path, bool video, const char* executable);

struct DrivingAudioAttachment {
  int channels = 0, sample_rate = 0;
  size_t source_frames = 0, attached_frames = 0;
  size_t used_frames = 0, trimmed_frames = 0, padded_frames = 0;
};

// Attach WAV at clip time zero. Preserve rate/channels, trim its end or pad
// silence to floor(video duration * sample rate) complete sample frames.
ReferenceMedia decode_driving_reference(const std::string& video_path,
                                        const std::string& audio_path, const char* executable,
                                        DrivingAudioAttachment& attachment);
}
