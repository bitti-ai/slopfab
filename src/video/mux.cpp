#include "mux_internal.h"

namespace slopfab::video {
using namespace mux_detail;

namespace {
struct Track {
  AVStream* stream = nullptr;
  AVCodecContext* ctx = nullptr;
  AVFrame* frame = nullptr;
  AVRational codec_tb{1, 1};  // what the encoder stamps packets in
  AVRational stream_tb{1, 1}; // what the muxer decided on, known after the header
  // AVStream::index is assigned in creation order, which is documented
  // behaviour — so the video stream, created first, is 0 and the audio stream
  // is 1. Relying on that is what keeps AVStream::index out of the offset
  // table.
  int index = 0;
  bool finished = false;
};

} // namespace

MuxStatus write_mp4(const MuxRequest& request) {
  const Loaded& s = loaded();
  if (!s.ok)
    return s.status;
  const Api& api = s.api;
  const Layout& l = s.layout;

  if (request.path.empty() || request.video == nullptr || request.frames <= 0 ||
      request.height <= 0 || request.width <= 0 || (request.width % 2) != 0 ||
      (request.height % 2) != 0 || request.fps.numerator <= 0 || request.fps.denominator <= 0) {
    return MuxStatus::kWriteFailed;
  }
  const size_t frame_pixels = static_cast<size_t>(request.height) * request.width;
  if (request.video->size() != 3 * static_cast<size_t>(request.frames) * frame_pixels) {
    return MuxStatus::kWriteFailed;
  }

  // A null or empty audio buffer means "video only"; a buffer whose length
  // disagrees with the channel count means the caller has a layout bug, and
  // silently dropping the audio track would hide it.
  const bool want_audio = request.audio != nullptr && !request.audio->empty();
  if (want_audio && (request.audio_channels <= 0 || request.audio_sample_rate <= 0 ||
                     (request.audio->size() % static_cast<size_t>(request.audio_channels)) != 0)) {
    return MuxStatus::kWriteFailed;
  }

  std::string video_name;
  const AVCodec* vcodec = find_video_encoder(api, &video_name);
  if (vcodec == nullptr)
    return MuxStatus::kEncoderMissing;

  std::string audio_name;
  const AVCodec* acodec = want_audio ? find_audio_encoder(api, &audio_name) : nullptr;
  if (want_audio && acodec == nullptr)
    return MuxStatus::kEncoderMissing;

  // Everything below owns ffmpeg objects, so it runs under one cleanup path.
  AVFormatContext* oc = nullptr;
  AVPacket* pkt = nullptr;
  Track video;
  Track audio;
  bool header_written = false;
  bool file_opened = false;
  MuxStatus status = MuxStatus::kWriteFailed;

  auto cleanup = [&]() {
    if (video.frame != nullptr)
      api.av_frame_free(&video.frame);
    if (audio.frame != nullptr)
      api.av_frame_free(&audio.frame);
    if (video.ctx != nullptr)
      api.avcodec_free_context(&video.ctx);
    if (audio.ctx != nullptr)
      api.avcodec_free_context(&audio.ctx);
    if (pkt != nullptr)
      api.av_packet_free(&pkt);
    if (oc != nullptr) {
      AVIOContext*& pb = fld<AVIOContext*>(oc, l.format_pb);
      if (pb != nullptr)
        api.avio_closep(&pb);
      api.avformat_free_context(oc);
      oc = nullptr;
    }
    // A half-written MP4 is worse than no MP4: the caller falls back to the
    // .y4m pair, and a truncated file sitting next to it will be opened by
    // someone eventually.
    if (status != MuxStatus::kOk && file_opened)
      std::remove(request.path.c_str());
    return status;
  };

  if (api.avformat_alloc_output_context2(&oc, nullptr, "mp4", request.path.c_str()) < 0 ||
      oc == nullptr) {
    return cleanup();
  }

  // --- video stream ---
  video.index = 0;
  video.stream = api.avformat_new_stream(oc, nullptr);
  if (video.stream == nullptr)
    return cleanup();
  video.codec_tb = AVRational{request.fps.denominator, request.fps.numerator};
  fld<AVRational>(video.stream, l.stream_time_base) = video.codec_tb;

  video.ctx = api.avcodec_alloc_context3(vcodec);
  if (video.ctx == nullptr)
    return cleanup();
  fld<int>(video.ctx, l.codec_width) = request.width;
  fld<int>(video.ctx, l.codec_height) = request.height;
  fld<int>(video.ctx, l.codec_pix_fmt) = kPixFmtYuv420p;
  fld<AVRational>(video.ctx, l.codec_time_base) = video.codec_tb;
  fld<AVRational>(video.ctx, l.codec_framerate) =
      AVRational{request.fps.numerator, request.fps.denominator};
  api.av_opt_set_int(video.ctx, "b", request.video_bitrate, 0);
  // One keyframe per second: seeking granularity a viewer will accept without
  // inflating a short clip.
  api.av_opt_set_int(video.ctx, "g",
                     std::max(1, request.fps.numerator / std::max(1, request.fps.denominator)), 0);
  api.av_opt_set_int(video.ctx, "threads", 0, 0);
  // Tag the stream as BT.709 limited range, which is what rgb_frame_to_yuv420
  // actually produced. Without this a player guesses from the resolution and a
  // 128x128 test clip guesses BT.601.
  api.av_opt_set_int(video.ctx, "color_range", kColorRangeMpeg, 0);
  api.av_opt_set_int(video.ctx, "colorspace", kColorSpaceBt709, 0);
  api.av_opt_set_int(video.ctx, "color_primaries", kColorPrimariesBt709, 0);
  api.av_opt_set_int(video.ctx, "color_trc", kColorTrcBt709, 0);
  // MP4 always wants the extradata in the container rather than in-band. We
  // ask for "mp4" explicitly when allocating the output context, so this is
  // unconditional instead of read from AVOutputFormat::flags — one less struct
  // whose layout we would have to know.
  {
    int64_t flags = 0;
    api.av_opt_get_int(video.ctx, "flags", 0, &flags);
    api.av_opt_set_int(video.ctx, "flags", flags | kCodecFlagGlobalHeader, 0);
  }

  if (api.avcodec_open2(video.ctx, vcodec, nullptr) < 0)
    return cleanup();
  if (api.avcodec_parameters_from_context(fld<AVCodecParameters*>(video.stream, l.stream_codecpar),
                                          video.ctx) < 0) {
    return cleanup();
  }

  video.frame = api.av_frame_alloc();
  if (video.frame == nullptr)
    return cleanup();
  fld<int>(video.frame, l.frame_width) = request.width;
  fld<int>(video.frame, l.frame_height) = request.height;
  fld<int>(video.frame, l.frame_format) = kPixFmtYuv420p;
  if (api.av_frame_get_buffer(video.frame, 32) != 0)
    return cleanup();

  // --- audio stream ---
  int audio_sample_fmt = kSampleFmtFltp;
  int audio_rate = request.audio_sample_rate;
  int audio_frame_samples = 1024;
  std::vector<float> audio_samples;
  if (want_audio) {
    bool all = false;
    const std::vector<int> fmts = supported_config<int>(api, acodec, kConfigSampleFormat, &all);
    if (!all) {
      const int preference[] = {kSampleFmtFltp, kSampleFmtFlt, kSampleFmtS16p, kSampleFmtS16};
      audio_sample_fmt = -1;
      for (int candidate : preference) {
        if (contains(fmts, candidate)) {
          audio_sample_fmt = candidate;
          break;
        }
      }
      if (audio_sample_fmt < 0)
        return (status = MuxStatus::kEncoderMissing, cleanup());
    }

    const std::vector<int> rates = supported_config<int>(api, acodec, kConfigSampleRate, &all);
    if (!all && !contains(rates, audio_rate)) {
      // Nearest supported rate rather than the highest: resampling further
      // than necessary is pure loss.
      int best = 0;
      for (int candidate : rates) {
        if (best == 0 || std::abs(candidate - audio_rate) < std::abs(best - audio_rate)) {
          best = candidate;
        }
      }
      if (best == 0)
        return (status = MuxStatus::kEncoderMissing, cleanup());
      audio_rate = best;
    }
    audio_samples = resample_linear(*request.audio, request.audio_channels,
                                    request.audio_sample_rate, audio_rate);

    audio.index = 1;
    audio.stream = api.avformat_new_stream(oc, nullptr);
    if (audio.stream == nullptr)
      return cleanup();
    audio.codec_tb = AVRational{1, audio_rate};
    fld<AVRational>(audio.stream, l.stream_time_base) = audio.codec_tb;

    audio.ctx = api.avcodec_alloc_context3(acodec);
    if (audio.ctx == nullptr)
      return cleanup();
    fld<int>(audio.ctx, l.codec_sample_fmt) = audio_sample_fmt;
    fld<AVRational>(audio.ctx, l.codec_time_base) = audio.codec_tb;
    api.av_opt_set_int(audio.ctx, "ar", audio_rate, 0);
    api.av_opt_set_int(audio.ctx, "b", request.audio_bitrate, 0);
    // Channel layout goes through the option API, so its offset never has to
    // be known: "2c" is the generic spelling ffmpeg accepts for any count.
    const std::string layout_name = request.audio_channels == 1 ? "mono"
                                    : request.audio_channels == 2
                                        ? "stereo"
                                        : std::to_string(request.audio_channels) + "c";
    if (api.av_opt_set(audio.ctx, "ch_layout", layout_name.c_str(), 0) < 0) {
      return (status = MuxStatus::kEncoderMissing, cleanup());
    }
    {
      int64_t flags = 0;
      api.av_opt_get_int(audio.ctx, "flags", 0, &flags);
      api.av_opt_set_int(audio.ctx, "flags", flags | kCodecFlagGlobalHeader, 0);
    }

    if (api.avcodec_open2(audio.ctx, acodec, nullptr) < 0)
      return cleanup();
    if (api.avcodec_parameters_from_context(
            fld<AVCodecParameters*>(audio.stream, l.stream_codecpar), audio.ctx) < 0) {
      return cleanup();
    }

    // The encoder's frame size is only known after opening it, and it is read
    // through the option API rather than a struct offset.
    int64_t frame_size = 0;
    if (api.av_opt_get_int(audio.ctx, "frame_size", 0, &frame_size) == 0 && frame_size > 0) {
      audio_frame_samples = static_cast<int>(frame_size);
    }

    audio.frame = api.av_frame_alloc();
    if (audio.frame == nullptr)
      return cleanup();
    fld<int>(audio.frame, l.frame_nb_samples) = audio_frame_samples;
    fld<int>(audio.frame, l.frame_format) = audio_sample_fmt;
    api.av_channel_layout_default(reinterpret_cast<char*>(audio.frame) + l.frame_ch_layout,
                                  request.audio_channels);
    if (api.av_frame_get_buffer(audio.frame, 0) != 0)
      return cleanup();
  }

  // --- open the file and write the header ---
  {
    AVIOContext*& pb = fld<AVIOContext*>(oc, l.format_pb);
    if (api.avio_open(&pb, request.path.c_str(), kAvioFlagWrite) < 0)
      return cleanup();
    file_opened = true;
  }

  AVDictionary* opts = nullptr;
  // Rewriting the index to the front costs one extra pass over the file and
  // makes the result streamable, which is what a viewer expects of an MP4.
  api.av_dict_set(&opts, "movflags", "+faststart", 0);
  const int header = api.avformat_write_header(oc, &opts);
  api.av_dict_free(&opts);
  if (header < 0)
    return cleanup();
  header_written = true;

  // The muxer rewrites the stream time bases — mov picks its own timescale —
  // so every packet has to be rescaled from the encoder's base into whatever
  // it decided on. Getting this wrong is the classic silent a/v drift.
  video.stream_tb = fld<AVRational>(video.stream, l.stream_time_base);
  if (video.stream_tb.num <= 0 || video.stream_tb.den <= 0)
    return cleanup();
  if (want_audio) {
    audio.stream_tb = fld<AVRational>(audio.stream, l.stream_time_base);
    if (audio.stream_tb.num <= 0 || audio.stream_tb.den <= 0)
      return cleanup();
  }

  pkt = api.av_packet_alloc();
  if (pkt == nullptr)
    return cleanup();

  auto drain = [&](Track& track) -> bool {
    for (;;) {
      const int ret = api.avcodec_receive_packet(track.ctx, pkt);
      if (ret == kErrAgain || ret == kErrEof)
        return true;
      if (ret < 0)
        return false;
      api.av_packet_rescale_ts(pkt, track.codec_tb, track.stream_tb);
      fld<int>(pkt, l.packet_stream_index) = track.index;
      // Takes ownership of the packet's reference and blanks it, so there is
      // no unref to pair with this.
      if (api.av_interleaved_write_frame(oc, pkt) < 0)
        return false;
    }
  };

  const size_t plane = static_cast<size_t>(request.frames) * frame_pixels;
  const float* r_plane = request.video->data();
  const float* g_plane = r_plane + plane;
  const float* b_plane = r_plane + 2 * plane;

  const size_t audio_frames_total =
      want_audio ? audio_samples.size() / static_cast<size_t>(request.audio_channels) : 0;
  int64_t video_pts = 0;
  int64_t audio_pts = 0;
  size_t audio_cursor = 0;
  int video_index = 0;

  while (!video.finished || (want_audio && !audio.finished)) {
    // Feed whichever track is behind in wall-clock terms. Doing this rather
    // than encoding all video then all audio keeps av_interleaved_write_frame's
    // reordering buffer small and, more importantly, keeps the interleaving in
    // the file itself tight enough for a player to stream it.
    bool do_video = !video.finished;
    if (do_video && want_audio && !audio.finished) {
      do_video = api.av_compare_ts(video_pts, video.codec_tb, audio_pts, audio.codec_tb) <= 0;
    }

    if (do_video) {
      AVFrame* frame = nullptr;
      if (video_index < request.frames) {
        if (api.av_frame_make_writable(video.frame) < 0)
          return cleanup();
        uint8_t** data = &fld<uint8_t*>(video.frame, l.frame_data);
        const int* linesize = &fld<int>(video.frame, l.frame_linesize);
        const size_t base = static_cast<size_t>(video_index) * frame_pixels;
        if (request.frame_converter != nullptr) {
          try {
            request.frame_converter->convert(r_plane + base, g_plane + base, b_plane + base,
                                             request.height, request.width, data[0], linesize[0],
                                             data[1], linesize[1], data[2], linesize[2]);
          } catch (...) {
            cleanup();
            throw;
          }
        } else {
          rgb_frame_to_yuv420(r_plane + base, g_plane + base, b_plane + base, request.height,
                              request.width, data[0], linesize[0], data[1], linesize[1], data[2],
                              linesize[2]);
        }
        fld<int64_t>(video.frame, l.frame_pts) = video_pts;
        frame = video.frame;
        ++video_index;
        ++video_pts;
      } else {
        video.finished = true; // null frame flushes the encoder
      }
      if (api.avcodec_send_frame(video.ctx, frame) < 0)
        return cleanup();
      if (!drain(video))
        return cleanup();
    } else {
      AVFrame* frame = nullptr;
      if (audio_cursor < audio_frames_total) {
        const int count = static_cast<int>(
            std::min(static_cast<size_t>(audio_frame_samples), audio_frames_total - audio_cursor));
        // Shrinking nb_samples for the final partial frame is allowed and is
        // how the encoder learns not to emit a block of trailing silence.
        fld<int>(audio.frame, l.frame_nb_samples) = count;
        if (api.av_frame_make_writable(audio.frame) < 0)
          return cleanup();
        uint8_t** data = &fld<uint8_t*>(audio.frame, l.frame_data);
        fill_audio_frame(audio_samples, audio_cursor, count, request.audio_channels,
                         audio_sample_fmt, data);
        fld<int64_t>(audio.frame, l.frame_pts) = audio_pts;
        audio_cursor += static_cast<size_t>(count);
        audio_pts += count;
        frame = audio.frame;
      } else {
        audio.finished = true;
      }
      if (api.avcodec_send_frame(audio.ctx, frame) < 0)
        return cleanup();
      if (!drain(audio))
        return cleanup();
    }
  }

  if (header_written && api.av_write_trailer(oc) < 0)
    return cleanup();

  status = MuxStatus::kOk;
  return cleanup();
}

} // namespace slopfab::video
