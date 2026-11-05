#include "reference_decode.h"
#include "native_media.h"
#include <algorithm>
#include <cmath>

namespace slopfab::cli {
ReferenceMedia decode_reference_file(const std::string& path, bool video, const char*) {
  ReferenceMedia result;
  if (video) {
    auto input = read_y4m(path);
    const double dt = double(input.fps.denominator) / input.fps.numerator;
    result = ReferenceMedia::video(input.frames.size() * dt);
    for (size_t i = 0; i < input.frames.size(); ++i)
      result.append_frame(input.frames[i].data(), input.frames[i].size(), input.width, input.height,
                          size_t(input.width) * 3, 3, i * dt);
  } else {
    auto input = read_wav(path);
    result = ReferenceMedia::audio(input.interleaved.data(), input.interleaved.size(),
                                   input.channels, input.sample_rate);
  }
  result.validate();
  return result;
}

ReferenceMedia decode_driving_reference(const std::string& video_path,
                                        const std::string& audio_path, const char* executable,
                                        DrivingAudioAttachment& attachment) {
  auto video = decode_reference_file(video_path, true, executable);
  // The ReferenceMedia contract validates range as well as finiteness. Do not
  // silently clip a float WAV; an out-of-range soundtrack must be corrected.
  auto audio = read_wav(audio_path, false);
  for (float value : audio.interleaved)
    if (value < -1 || value > 1)
      throw std::invalid_argument("driving audio: WAV samples must be within [-1,1]");
  DrivingAudioAttachment info;
  info.channels = audio.channels;
  info.sample_rate = audio.sample_rate;
  info.source_frames = audio.interleaved.size() / audio.channels;
  // The epsilon only compensates floating-point representation of an exact
  // integer frame count. It is much smaller than ReferenceMedia's tolerance.
  info.attached_frames = size_t(std::floor(video.duration_seconds() * audio.sample_rate + 1e-7));
  info.used_frames = std::min(info.source_frames, info.attached_frames);
  info.trimmed_frames = info.source_frames - info.used_frames;
  info.padded_frames = info.attached_frames - info.used_frames;
  audio.interleaved.resize(info.attached_frames * audio.channels, 0);
  video.set_audio(audio.interleaved.data(), audio.interleaved.size(), audio.channels,
                  audio.sample_rate, 0);
  video.validate();
  attachment = info;
  return video;
}
}
