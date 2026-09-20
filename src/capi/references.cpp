#include "internal.h"
SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_create(
    double duration_seconds, slopfab_reference_video** out_video) {
  if (!out_video) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference video: null output pointer");
  *out_video = nullptr;
  return reference_input_guarded([&] {
    auto video = std::make_unique<slopfab_reference_video>();
    video->media = slopfab::ReferenceMedia::video(duration_seconds);
    *out_video = video.release();
  });
}

SLOPFAB_C_API void SLOPFAB_CALL slopfab_reference_video_destroy(slopfab_reference_video* video) {
  delete video;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_append_rgb24(
    slopfab_reference_video* video, const uint8_t* pixels, size_t buffer_bytes,
    int32_t width, int32_t height, size_t row_stride_bytes, double timestamp_seconds) {
  if (!video) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference video: null handle");
  return reference_input_guarded([&] {
    video->media.append_frame(pixels, buffer_bytes, width, height, row_stride_bytes, 3, timestamp_seconds);
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_append_rgba8(
    slopfab_reference_video* video, const uint8_t* pixels, size_t buffer_bytes,
    int32_t width, int32_t height, size_t row_stride_bytes, double timestamp_seconds) {
  if (!video) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference video: null handle");
  return reference_input_guarded([&] {
    video->media.append_frame(pixels, buffer_bytes, width, height, row_stride_bytes, 4, timestamp_seconds);
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_set_audio_f32(
    slopfab_reference_video* video, const float* samples, size_t float_count,
    int32_t channels, int32_t sample_rate, double start_seconds) {
  if (!video) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference video: null handle");
  return reference_input_guarded([&] {
    video->media.set_audio(samples, float_count, channels, sample_rate, start_seconds);
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_reference_video(
    slopfab_request* request, const slopfab_reference_video* video) {
  if (!request || !video) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference video: null handle");
  return reference_input_guarded([&] { attach_reference(request, video->media); });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_reference_audio_f32(
    slopfab_request* request, const float* samples, size_t float_count,
    int32_t channels, int32_t sample_rate) {
  if (!request) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference audio: null request");
  return reference_input_guarded([&] {
    attach_reference(request, slopfab::ReferenceMedia::audio(samples, float_count, channels, sample_rate));
  });
}

