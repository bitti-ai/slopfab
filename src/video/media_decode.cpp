#include "mux_internal.h"

namespace slopfab::video {
using namespace mux_detail;
DecodedVideoFrame decode_first_video_frame(const std::string& path) {
  const Loaded& state = loaded();
  if (!state.ok) throw std::runtime_error("FFmpeg media decoder unavailable: " + state.detail);
  const Api& api = state.api;
  const Layout& l = state.layout;
  AVFormatContext* format = nullptr;
  AVCodecContext* decoder = nullptr;
  AVFrame* frame = nullptr;
  AVPacket* packet = nullptr;
  SwsContext* sws = nullptr;
  auto cleanup = [&] {
    if (sws != nullptr) api.sws_freeContext(sws);
    if (packet != nullptr) api.av_packet_free(&packet);
    if (frame != nullptr) api.av_frame_free(&frame);
    if (decoder != nullptr) api.avcodec_free_context(&decoder);
    if (format != nullptr) api.avformat_close_input(&format);
  };
  auto fail = [&](const std::string& reason) -> DecodedVideoFrame {
    cleanup();
    throw std::runtime_error("reference media '" + path + "': " + reason);
  };

  int rc = api.avformat_open_input(&format, path.c_str(), nullptr, nullptr);
  if (rc < 0) return fail("cannot open input: " + err_text(api, rc));
  rc = api.avformat_find_stream_info(format, nullptr);
  if (rc < 0) return fail("cannot read stream information: " + err_text(api, rc));
  const AVCodec* codec = nullptr;
  const int stream_index =
      api.av_find_best_stream(format, kMediaTypeVideo, -1, -1, &codec, 0);
  if (stream_index < 0 || codec == nullptr) return fail("input has no decodable video stream");
  const unsigned count = fld<unsigned>(format, l.format_nb_streams);
  AVStream** streams = fld<AVStream**>(format, l.format_streams);
  if (streams == nullptr || static_cast<unsigned>(stream_index) >= count)
    return fail("FFmpeg returned an invalid video stream index");
  AVCodecParameters* par = fld<AVCodecParameters*>(streams[stream_index], l.stream_codecpar);
  if (par == nullptr) return fail("video stream has no codec parameters");
  decoder = api.avcodec_alloc_context3(codec);
  if (decoder == nullptr) return fail("cannot allocate video decoder");
  rc = api.avcodec_parameters_to_context(decoder, par);
  if (rc < 0) return fail("cannot configure video decoder: " + err_text(api, rc));
  rc = api.avcodec_open2(decoder, codec, nullptr);
  if (rc < 0) return fail("cannot open video decoder: " + err_text(api, rc));
  frame = api.av_frame_alloc();
  packet = api.av_packet_alloc();
  if (frame == nullptr || packet == nullptr) return fail("cannot allocate decode buffers");

  bool input_done = false;
  for (;;) {
    rc = api.avcodec_receive_frame(decoder, frame);
    if (rc == 0) break;
    if (rc != kErrAgain && rc != kErrEof)
      return fail("video decode failed: " + err_text(api, rc));
    if (input_done) return fail("video stream contains no decoded frames");
    rc = api.av_read_frame(format, packet);
    if (rc < 0) {
      input_done = true;
      rc = api.avcodec_send_packet(decoder, nullptr);
      if (rc < 0 && rc != kErrEof) return fail("cannot flush video decoder: " + err_text(api, rc));
      continue;
    }
    if (fld<int>(packet, l.packet_stream_index) == stream_index) {
      rc = api.avcodec_send_packet(decoder, packet);
      api.av_packet_unref(packet);
      if (rc < 0 && rc != kErrAgain)
        return fail("cannot submit video packet: " + err_text(api, rc));
    } else {
      api.av_packet_unref(packet);
    }
  }

  const int width = fld<int>(frame, l.frame_width);
  const int height = fld<int>(frame, l.frame_height);
  const int pixel_format = fld<int>(frame, l.frame_format);
  if (width <= 0 || height <= 0 ||
      static_cast<uint64_t>(width) * height > std::numeric_limits<size_t>::max() / 3)
    return fail("decoded frame has invalid dimensions");
  DecodedVideoFrame result;
  result.width = width;
  result.height = height;
  result.timestamp = fld<int64_t>(frame, l.frame_pts);
  result.rgb24.resize(static_cast<size_t>(width) * height * 3);
  sws = api.sws_getContext(width, height, pixel_format, width, height, kPixFmtRgb24,
                           kSwsBicubic, nullptr, nullptr, nullptr);
  if (sws == nullptr) return fail("cannot create RGB24 converter");
  const uint8_t* const* src = &fld<uint8_t*>(frame, l.frame_data);
  const int* src_stride = &fld<int>(frame, l.frame_linesize);
  uint8_t* dst[] = {result.rgb24.data(), nullptr, nullptr, nullptr};
  const int dst_stride[] = {width * 3, 0, 0, 0};
  rc = api.sws_scale(sws, src, src_stride, 0, height, dst, dst_stride);
  if (rc != height) return fail("RGB24 conversion failed");
  cleanup();
  return result;
}

}  // namespace slopfab::video
