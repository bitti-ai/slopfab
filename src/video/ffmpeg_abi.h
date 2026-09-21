// The slice of ffmpeg's ABI that mux.cpp drives, declared here rather than
// included from ffmpeg's headers.
//
// Nothing in this file may reference an ffmpeg header, and the build must
// never gain an ffmpeg include directory or link library. That is an LGPL
// requirement, not a style choice: the user has to be able to drop in their
// own build of ffmpeg, which only works if we bind to it at runtime.
//
// Two rules keep that from turning into memory corruption:
//
//  1. **No ffmpeg struct layout is declared.** Every ffmpeg object is an
//     incomplete type behind a pointer, allocated by ffmpeg itself. We never
//     take a sizeof, never allocate one, never copy one. So fields we do not
//     touch cannot hurt us however they move between releases.
//
//  2. **The few fields we must touch are reached through the byte offsets in
//     `Layout` below, and every one of them is checked at load time** (see
//     `validate_layout` in mux.cpp) by writing through an ffmpeg API and
//     reading the result back through our offset, or by handing ffmpeg a
//     structure we filled and confirming it accepted it. A version whose
//     layout we do not know therefore fails the probe and reports
//     `kSymbolMissing`; it never silently reads garbage.
//
// Fields still have to be touched because ffmpeg exposes no accessors for
// them. Where an AVOption exists we use `av_opt_set*`/`av_opt_get*` instead
// and need no offset at all — that is why, for instance, bit rate, GOP size,
// channel layout and the encoder's frame size are absent from `Layout`.
//
// The offsets below are for **ffmpeg 8.x** (libavcodec 62, libavformat 62,
// libavutil 60), which is the only ABI this port claims to drive. They were
// not eyeballed from the headers: they were produced by compiling
// `offsetof(...)` against ffmpeg 8.0.1's own headers. To add another major
// version, do the same against that version's headers, add a `Layout` row and
// widen `is_supported_version`; the load-time validation will tell you at once
// if a number is wrong.
#pragma once

#include <cstddef>
#include <cstdint>

namespace slopfab::video::ff {

// --- opaque handles ---------------------------------------------------------
//
// Incomplete types on purpose: a pointer to one is all we ever hold, and an
// incomplete type cannot be accidentally allocated, copied or sized.
struct AVFormatContext;
struct AVCodecContext;
struct AVCodecParameters;
struct AVStream;
struct AVFrame;
struct AVPacket;
struct AVCodec;
struct AVOutputFormat;
struct AVIOContext;
struct AVDictionary;
struct SwsContext;

// The one exception, and it is not an exception at all: AVRational is two ints
// passed by value across the ABI, and has been since libavutil existed. It is
// not a struct we allocate, it is part of several function signatures.
struct AVRational {
  int num;
  int den;
};

// --- constants --------------------------------------------------------------
//
// Enumerator values, not layout, so they are safe to hard-code: ffmpeg only
// ever appends to these enums, and the ones used here are the oldest members.
constexpr int kPixFmtYuv420p = 0;
constexpr int kPixFmtRgb24 = 2; // used only by the load-time layout check

constexpr int kSampleFmtS16 = 1;
constexpr int kSampleFmtFlt = 3;
constexpr int kSampleFmtS16p = 6;
constexpr int kSampleFmtFltp = 8;

constexpr int kCodecIdMpeg4 = 12;
constexpr int kCodecIdH264 = 27;
constexpr int kCodecIdAac = 86018;

constexpr int kColorRangeMpeg = 1; // limited range, 16..235
constexpr int kColorPrimariesBt709 = 1;
constexpr int kColorTrcBt709 = 1;
constexpr int kColorSpaceBt709 = 1;

constexpr int kMediaTypeUnknown = -1;
constexpr int kMediaTypeVideo = 0;
constexpr int kSwsBicubic = 4;

constexpr int kCodecFlagGlobalHeader = 0x00400000;
constexpr int kAvioFlagWrite = 2;

constexpr int kLogError = 16;
constexpr int kLogVerbose = 40;

constexpr int64_t kNoPts = INT64_MIN;

// AVERROR(EAGAIN) and AVERROR_EOF. EAGAIN is 11 on every platform this builds
// for; AVERROR_EOF is the fixed tag -MKTAG('E','O','F',' ').
constexpr int kErrAgain = -11;
constexpr int kErrEof = -541478725;

// enum AVCodecConfig, used with avcodec_get_supported_config.
constexpr int kConfigPixFormat = 0;
constexpr int kConfigSampleRate = 2;
constexpr int kConfigSampleFormat = 3;

// --- field offsets ----------------------------------------------------------

struct Layout {
  // AVFormatContext: only `pb`, which has to be assigned after avio_open
  // because there is no setter for it. It has sat behind
  // {av_class, iformat, oformat, priv_data} since libavformat 52.
  size_t format_pb;
  size_t format_nb_streams;
  size_t format_streams;

  // AVStream: the muxer rewrites `time_base` during avformat_write_header and
  // we must read it back to rescale packet timestamps, and `codecpar` is where
  // avcodec_parameters_from_context has to deposit the stream description.
  // Note `codecpar` moved forward in ffmpeg 8 (it now sits before priv_data),
  // which is exactly the kind of change the load-time check exists to catch.
  size_t stream_codecpar;
  size_t stream_time_base;

  // AVCodecParameters, read only, and only to recognise a freshly allocated
  // one during the layout check.
  size_t par_codec_type;
  size_t par_codec_id;
  size_t par_format;

  // AVCodecContext. Everything reachable through an AVOption is set that way
  // instead; what is left has no option entry.
  size_t codec_bit_rate;    // validated against av_opt_set_int(ctx, "b")
  size_t codec_gop_size;    // validated against av_opt_set_int(ctx, "g")
  size_t codec_sample_rate; // validated against av_opt_set_int(ctx, "ar")
  size_t codec_time_base;
  size_t codec_framerate;
  size_t codec_width;
  size_t codec_height;
  size_t codec_pix_fmt;
  size_t codec_sample_fmt;

  // AVFrame. data/linesize/width/height/nb_samples/format form a prefix that
  // has not moved since 2012; pts and ch_layout have both shifted with
  // deprecation removals, so both are checked at load time.
  //
  // `extended_data` is deliberately absent: it only differs from `data` above
  // eight planes, and nothing here exceeds three. `sample_rate` is absent for
  // a better reason — the encoder reads its rate from the AVCodecContext, so
  // setting it on the frame would mean writing through an offset with nothing
  // to validate it against, for no effect.
  size_t frame_data;
  size_t frame_linesize;
  size_t frame_width;
  size_t frame_height;
  size_t frame_nb_samples;
  size_t frame_format;
  size_t frame_pts;
  size_t frame_ch_layout;

  // AVPacket, unchanged since libavcodec 58.
  size_t packet_pts;
  size_t packet_dts;
  size_t packet_data;
  size_t packet_size;
  size_t packet_stream_index;
};

// ffmpeg 8.x: libavcodec 62, libavformat 62, libavutil 60.
constexpr Layout kLayoutFfmpeg8 = {
    /* format_pb */ 32,
    /* format_nb_streams */ 44,
    /* format_streams */ 48,
    /* stream_codecpar */ 16,
    /* stream_time_base */ 32,
    /* par_codec_type */ 0,
    /* par_codec_id */ 4,
    /* par_format */ 44,
    /* codec_bit_rate */ 56,
    /* codec_gop_size */ 332,
    /* codec_sample_rate */ 344,
    /* codec_time_base */ 84,
    /* codec_framerate */ 100,
    /* codec_width */ 112,
    /* codec_height */ 116,
    /* codec_pix_fmt */ 136,
    /* codec_sample_fmt */ 348,
    /* frame_data */ 0,
    /* frame_linesize */ 64,
    /* frame_width */ 104,
    /* frame_height */ 108,
    /* frame_nb_samples */ 112,
    /* frame_format */ 116,
    /* frame_pts */ 136,
    /* frame_ch_layout */ 384,
    /* packet_pts */ 8,
    /* packet_dts */ 16,
    /* packet_data */ 24,
    /* packet_size */ 32,
    /* packet_stream_index */ 36,
};

// Reads or writes a field at a known byte offset of an ffmpeg-allocated
// object. Deliberately noisy at the call site: every use is a place where we
// depend on a layout, and they should be easy to grep for.
template <typename T, typename Obj> inline T& fld(Obj* obj, size_t offset) {
  return *reinterpret_cast<T*>(reinterpret_cast<char*>(obj) + offset);
}

template <typename T, typename Obj> inline const T& fld(const Obj* obj, size_t offset) {
  return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(obj) + offset);
}

// --- function pointer types -------------------------------------------------
//
// Every one of these is resolved by name and checked for null before the
// loader reports success, so no call site has to guard again.

// libavutil
using AvutilVersionFn = unsigned (*)(void);
using AvLogSetLevelFn = void (*)(int);
using AvFrameAllocFn = AVFrame* (*)(void);
using AvFrameFreeFn = void (*)(AVFrame**);
using AvFrameGetBufferFn = int (*)(AVFrame*, int);
using AvFrameMakeWritableFn = int (*)(AVFrame*);
using AvChannelLayoutDefaultFn = void (*)(void*, int);
using AvOptSetFn = int (*)(void*, const char*, const char*, int);
using AvOptSetIntFn = int (*)(void*, const char*, int64_t, int);
using AvOptGetIntFn = int (*)(void*, const char*, int, int64_t*);
using AvDictSetFn = int (*)(AVDictionary**, const char*, const char*, int);
using AvDictFreeFn = void (*)(AVDictionary**);
using AvStrerrorFn = int (*)(int, char*, size_t);
using AvCompareTsFn = int (*)(int64_t, AVRational, int64_t, AVRational);

// libavcodec
using AvcodecVersionFn = unsigned (*)(void);
using AvcodecFindEncoderByNameFn = const AVCodec* (*)(const char*);
using AvcodecFindEncoderFn = const AVCodec* (*)(int);
using AvcodecFindDecoderFn = const AVCodec* (*)(int);
using AvcodecAllocContext3Fn = AVCodecContext* (*)(const AVCodec*);
using AvcodecFreeContextFn = void (*)(AVCodecContext**);
using AvcodecOpen2Fn = int (*)(AVCodecContext*, const AVCodec*, AVDictionary**);
using AvcodecParametersFromContextFn = int (*)(AVCodecParameters*, const AVCodecContext*);
using AvcodecParametersToContextFn = int (*)(AVCodecContext*, const AVCodecParameters*);
using AvcodecSendFrameFn = int (*)(AVCodecContext*, const AVFrame*);
using AvcodecSendPacketFn = int (*)(AVCodecContext*, const AVPacket*);
using AvcodecReceiveFrameFn = int (*)(AVCodecContext*, AVFrame*);
using AvcodecReceivePacketFn = int (*)(AVCodecContext*, AVPacket*);
using AvcodecGetSupportedConfigFn = int (*)(const AVCodecContext*, const AVCodec*, int, unsigned,
                                            const void**, int*);
using AvPacketAllocFn = AVPacket* (*)(void);
using AvPacketFreeFn = void (*)(AVPacket**);
using AvPacketRescaleTsFn = void (*)(AVPacket*, AVRational, AVRational);
using AvPacketUnrefFn = void (*)(AVPacket*);

// libavformat
using AvformatVersionFn = unsigned (*)(void);
using AvformatAllocOutputContext2Fn = int (*)(AVFormatContext**, const AVOutputFormat*, const char*,
                                              const char*);
using AvformatFreeContextFn = void (*)(AVFormatContext*);
using AvformatNewStreamFn = AVStream* (*)(AVFormatContext*, const AVCodec*);
using AvformatWriteHeaderFn = int (*)(AVFormatContext*, AVDictionary**);
using AvInterleavedWriteFrameFn = int (*)(AVFormatContext*, AVPacket*);
using AvWriteTrailerFn = int (*)(AVFormatContext*);
using AvioOpenFn = int (*)(AVIOContext**, const char*, int);
using AvioClosepFn = int (*)(AVIOContext**);
using AvformatOpenInputFn = int (*)(AVFormatContext**, const char*, const void*, AVDictionary**);
using AvformatFindStreamInfoFn = int (*)(AVFormatContext*, AVDictionary**);
using AvFindBestStreamFn = int (*)(AVFormatContext*, int, int, int, const AVCodec**, int);
using AvReadFrameFn = int (*)(AVFormatContext*, AVPacket*);
using AvformatCloseInputFn = void (*)(AVFormatContext**);

using SwscaleVersionFn = unsigned (*)(void);
using SwsGetContextFn = SwsContext* (*)(int, int, int, int, int, int, int, void*, void*,
                                        const double*);
using SwsScaleFn = int (*)(SwsContext*, const uint8_t* const[], const int[], int, int,
                           uint8_t* const[], const int[]);
using SwsFreeContextFn = void (*)(SwsContext*);

// The whole bound surface. Anything added here must also be added to the
// resolve list in mux.cpp, which is what makes a missing symbol a clean
// kSymbolMissing rather than a null call.
struct Api {
  AvutilVersionFn avutil_version;
  AvLogSetLevelFn av_log_set_level;
  AvFrameAllocFn av_frame_alloc;
  AvFrameFreeFn av_frame_free;
  AvFrameGetBufferFn av_frame_get_buffer;
  AvFrameMakeWritableFn av_frame_make_writable;
  AvChannelLayoutDefaultFn av_channel_layout_default;
  AvOptSetFn av_opt_set;
  AvOptSetIntFn av_opt_set_int;
  AvOptGetIntFn av_opt_get_int;
  AvDictSetFn av_dict_set;
  AvDictFreeFn av_dict_free;
  AvStrerrorFn av_strerror;
  AvCompareTsFn av_compare_ts;

  AvcodecVersionFn avcodec_version;
  AvcodecFindEncoderByNameFn avcodec_find_encoder_by_name;
  AvcodecFindEncoderFn avcodec_find_encoder;
  AvcodecFindDecoderFn avcodec_find_decoder;
  AvcodecAllocContext3Fn avcodec_alloc_context3;
  AvcodecFreeContextFn avcodec_free_context;
  AvcodecOpen2Fn avcodec_open2;
  AvcodecParametersFromContextFn avcodec_parameters_from_context;
  AvcodecParametersToContextFn avcodec_parameters_to_context;
  AvcodecSendFrameFn avcodec_send_frame;
  AvcodecSendPacketFn avcodec_send_packet;
  AvcodecReceiveFrameFn avcodec_receive_frame;
  AvcodecReceivePacketFn avcodec_receive_packet;
  AvcodecGetSupportedConfigFn avcodec_get_supported_config;
  AvPacketAllocFn av_packet_alloc;
  AvPacketFreeFn av_packet_free;
  AvPacketRescaleTsFn av_packet_rescale_ts;
  AvPacketUnrefFn av_packet_unref;

  AvformatVersionFn avformat_version;
  AvformatAllocOutputContext2Fn avformat_alloc_output_context2;
  AvformatFreeContextFn avformat_free_context;
  AvformatNewStreamFn avformat_new_stream;
  AvformatWriteHeaderFn avformat_write_header;
  AvInterleavedWriteFrameFn av_interleaved_write_frame;
  AvWriteTrailerFn av_write_trailer;
  AvioOpenFn avio_open;
  AvioClosepFn avio_closep;
  AvformatOpenInputFn avformat_open_input;
  AvformatFindStreamInfoFn avformat_find_stream_info;
  AvFindBestStreamFn av_find_best_stream;
  AvReadFrameFn av_read_frame;
  AvformatCloseInputFn avformat_close_input;

  SwscaleVersionFn swscale_version;
  SwsGetContextFn sws_getContext;
  SwsScaleFn sws_scale;
  SwsFreeContextFn sws_freeContext;
};

} // namespace slopfab::video::ff
