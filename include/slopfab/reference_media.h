#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "slopfab/image.h"
#include "slopfab/sha256.h"

namespace slopfab {

// Decoded reference input. No decoder, file format or accelerator dependency.
// Immutable payloads make request snapshots cheap without borrowing host memory.
struct ReferenceFrame {
  RGBImage image;
  double timestamp_seconds = 0;
  Sha256Digest pixel_digest{};
};

struct ReferenceAudio {
  std::vector<float> interleaved;
  int channels = 0;
  int sample_rate = 0;
  double start_seconds = 0;
  Sha256Digest sample_digest{};

  double duration_seconds() const;
};

class ReferenceMedia {
public:
  // A video's duration is explicit: the last frame timestamp alone cannot
  // determine how long that frame is displayed. All times are clip-relative.
  static ReferenceMedia video(double duration_seconds);
  static ReferenceMedia audio(const float* samples, size_t float_count, int channels,
                              int sample_rate);

  // Copies visible pixels, ignores RGBA alpha and row padding. Timestamps must
  // start at zero, increase strictly, and lie before the clip's duration.
  void append_frame(const uint8_t* pixels, size_t buffer_bytes, int width, int height,
                    size_t row_stride_bytes, int pixel_channels, double timestamp_seconds);

  // Replaces the soundtrack transactionally. Mono and stereo float PCM in
  // [-1,1] are accepted at the native sample rate. No resampling occurs here.
  void set_audio(const float* samples, size_t float_count, int channels, int sample_rate,
                 double start_seconds = 0);

  bool is_video() const {
    return video_;
  }

  double duration_seconds() const {
    return duration_;
  }

  const std::vector<std::shared_ptr<const ReferenceFrame>>& frames() const {
    return frames_;
  }

  const std::shared_ptr<const ReferenceAudio>& soundtrack() const {
    return audio_;
  }

  // Checks completeness as well as the public 2..15 second clip contract.
  void validate() const;

private:
  bool video_ = false;
  double duration_ = 0;
  std::vector<std::shared_ptr<const ReferenceFrame>> frames_;
  std::shared_ptr<const ReferenceAudio> audio_;
};

// Counts standalone audio separately from a video's attached soundtrack.
void validate_reference_media(size_t image_count,
                              const std::vector<std::shared_ptr<const ReferenceMedia>>& references);

// Content and timing identity, independent of row padding, caller addresses,
// and the file/decoder used to obtain the buffers.
std::string reference_media_identity(const ReferenceMedia& reference);

} // namespace slopfab
