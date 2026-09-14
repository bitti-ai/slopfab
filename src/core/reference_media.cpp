#include "slopfab/reference_media.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "slopfab/sha256.h"

namespace slopfab {
namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

size_t multiply(size_t a, size_t b) {
  require(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
          "reference media: buffer size overflow");
  return a * b;
}

}  // namespace

double ReferenceAudio::duration_seconds() const {
  return channels > 0 && sample_rate > 0
      ? static_cast<double>(interleaved.size() / channels) / sample_rate : 0;
}

ReferenceMedia ReferenceMedia::video(double duration_seconds) {
  require(std::isfinite(duration_seconds) && duration_seconds >= 2 && duration_seconds <= 15,
          "reference video: duration must be between 2 and 15 seconds");
  ReferenceMedia result;
  result.video_ = true;
  result.duration_ = duration_seconds;
  return result;
}

ReferenceMedia ReferenceMedia::audio(const float* samples, size_t float_count,
                                      int channels, int sample_rate) {
  ReferenceMedia result;
  result.set_audio(samples, float_count, channels, sample_rate);
  result.validate();
  return result;
}

void ReferenceMedia::append_frame(const uint8_t* pixels, size_t buffer_bytes,
    int width, int height, size_t row_stride_bytes, int pixel_channels,
    double timestamp_seconds) {
  require(video_, "reference media: frames require a video reference");
  require(pixels && width > 0 && height > 0, "reference video: invalid pixel buffer or dimensions");
  require(pixel_channels == 3 || pixel_channels == 4, "reference video: expected RGB24 or RGBA8");
  require(static_cast<double>(width) / height >= .25 &&
          static_cast<double>(width) / height <= 4, "reference video: aspect must be within 1:4 and 4:1");
  require(std::isfinite(timestamp_seconds) && timestamp_seconds >= 0 && timestamp_seconds < duration_,
          "reference video: timestamp must be finite and within the clip");
  if (frames_.empty()) {
    require(timestamp_seconds == 0, "reference video: first frame timestamp must be zero");
  } else {
    require(timestamp_seconds > frames_.back()->timestamp_seconds,
            "reference video: frame timestamps must increase strictly");
    require(width == frames_.front()->image.width && height == frames_.front()->image.height,
            "reference video: all frames must have the same dimensions");
  }
  const size_t row_bytes = multiply(static_cast<size_t>(width), pixel_channels);
  require(row_stride_bytes >= row_bytes, "reference video: row stride is too small");
  const size_t last_row = multiply(static_cast<size_t>(height - 1), row_stride_bytes);
  require(last_row <= buffer_bytes && row_bytes <= buffer_bytes - last_row,
          "reference video: pixel buffer is too small");
  auto frame = std::make_shared<ReferenceFrame>();
  frame->timestamp_seconds = timestamp_seconds == 0 ? 0 : timestamp_seconds;
  frame->image.width = width;
  frame->image.height = height;
  frame->image.pixels.resize(multiply(multiply(static_cast<size_t>(width), height), 3));
  for (int y = 0; y < height; ++y) {
    const uint8_t* src = pixels + static_cast<size_t>(y) * row_stride_bytes;
    uint8_t* dst = frame->image.pixels.data() + static_cast<size_t>(y) * width * 3;
    if (pixel_channels == 3) std::memcpy(dst, src, row_bytes);
    else for (int x = 0; x < width; ++x) std::memcpy(dst + size_t(x) * 3, src + size_t(x) * 4, 3);
  }
  frame->pixel_digest = sha256_bytes(frame->image.pixels.data(), frame->image.pixels.size());
  frames_.push_back(std::move(frame));
}

void ReferenceMedia::set_audio(const float* samples, size_t float_count,
    int channels, int sample_rate, double start_seconds) {
  require(samples && float_count > 0, "reference audio: empty sample buffer");
  require(channels == 1 || channels == 2, "reference audio: expected mono or stereo PCM");
  require(sample_rate > 0, "reference audio: sample rate must be positive");
  require(float_count % channels == 0, "reference audio: incomplete interleaved sample frame");
  (void)multiply(float_count, sizeof(float));
  require(std::isfinite(start_seconds) && start_seconds >= 0,
          "reference audio: start time must be finite and nonnegative");
  const double duration = static_cast<double>(float_count / channels) / sample_rate;
  require(duration <= 15, "reference audio: duration exceeds 15 seconds");
  if (video_) require(start_seconds + duration <= duration_ + 1e-9,
                      "reference audio: soundtrack extends beyond the video");
  else require(start_seconds == 0 && duration >= 2,
               "reference audio: standalone clip must start at zero and last 2 to 15 seconds");
  for (size_t i = 0; i < float_count; ++i) {
    require(std::isfinite(samples[i]) && samples[i] >= -1 && samples[i] <= 1,
            "reference audio: samples must be finite and within [-1,1]");
  }
  auto audio = std::make_shared<ReferenceAudio>();
  audio->channels = channels;
  audio->sample_rate = sample_rate;
  audio->start_seconds = start_seconds == 0 ? 0 : start_seconds;
  audio->interleaved.assign(samples, samples + float_count);
  audio->sample_digest = sha256_bytes(samples, float_count * sizeof(float));
  audio_ = std::move(audio);
  if (!video_) duration_ = duration;
}

void ReferenceMedia::validate() const {
  require(duration_ >= 2 && duration_ <= 15, "reference media: duration must be 2 to 15 seconds");
  require(video_ ? !frames_.empty() : bool(audio_), "reference media: reference has no frames or audio");
}

void validate_reference_media(size_t image_count,
    const std::vector<std::shared_ptr<const ReferenceMedia>>& references) {
  require(image_count <= 9, "MiniMax-H3 Ref2VA accepts at most 9 reference images");
  require(references.size() <= 12 - image_count, "MiniMax-H3 Ref2VA accepts at most 12 references in total");
  size_t videos = 0, audios = 0;
  double video_seconds = 0, audio_seconds = 0;
  for (const auto& reference : references) {
    require(bool(reference), "reference media: null reference");
    reference->validate();
    if (reference->is_video()) { ++videos; video_seconds += reference->duration_seconds(); }
    else { ++audios; audio_seconds += reference->duration_seconds(); }
  }
  require(videos <= 3 && audios <= 3, "MiniMax-H3 Ref2VA accepts at most 3 videos and 3 standalone audio references");
  require(video_seconds <= 15 + 1e-9 && audio_seconds <= 15 + 1e-9,
          "MiniMax-H3 Ref2VA reference videos and standalone audio each have a 15 second total limit");
}

std::string reference_media_identity(const ReferenceMedia& reference) {
  reference.validate();
  std::string key = "decoded-reference-v1";
  auto append = [&](const auto& value) {
    key.append(reinterpret_cast<const char*>(&value), sizeof(value));
  };
  append(reference.is_video());
  append(reference.duration_seconds());
  append(static_cast<uint64_t>(reference.frames().size()));
  for (const auto& frame : reference.frames()) {
    append(frame->image.width);
    append(frame->image.height);
    append(frame->timestamp_seconds);
    append(frame->pixel_digest);
  }
  append(bool(reference.soundtrack()));
  if (const auto& audio = reference.soundtrack()) {
    append(audio->channels);
    append(audio->sample_rate);
    append(audio->start_seconds);
    append(static_cast<uint64_t>(audio->interleaved.size()));
    append(audio->sample_digest);
  }
  const auto digest = sha256_bytes(key.data(), key.size());
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

}  // namespace slopfab
