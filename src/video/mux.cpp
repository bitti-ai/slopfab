#include "vidfab/video/mux.h"
#include "vidfab/video/media.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ffmpeg_abi.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace vidfab::video {
namespace {

using namespace ff;

// --- dynamic loading --------------------------------------------------------

#if defined(_WIN32)
using LibHandle = HMODULE;

LibHandle lib_open(const std::string& path) {
  // A bare name goes through the normal search order, which is what finds an
  // ffmpeg that is already on PATH. A name with a directory in it has to be
  // opened with LOAD_WITH_ALTERED_SEARCH_PATH, otherwise Windows resolves
  // *that* library's own dependencies against the process search path rather
  // than against the directory it came from — so avcodec-62.dll loads and then
  // fails to find the avutil-60.dll sitting next to it. Without this flag the
  // VIDFAB_FFMPEG_DIR override only works when the directory is on PATH too,
  // which would make it pointless.
  if (path.find('\\') != std::string::npos || path.find('/') != std::string::npos) {
    return ::LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  }
  return ::LoadLibraryA(path.c_str());
}

void* lib_sym(LibHandle h, const char* name) {
  return reinterpret_cast<void*>(::GetProcAddress(h, name));
}

// Windows ships the majors in the file name: avcodec-62.dll.
std::string lib_file(const char* base, int major) {
  return std::string(base) + "-" + std::to_string(major) + ".dll";
}

std::string lib_file_unversioned(const char* base) { return std::string(base) + ".dll"; }

constexpr char kPathSep = '\\';
#else
using LibHandle = void*;

LibHandle lib_open(const std::string& path) { return ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL); }

void* lib_sym(LibHandle h, const char* name) { return ::dlsym(h, name); }

#if defined(__APPLE__)
std::string lib_file(const char* base, int major) {
  return std::string("lib") + base + "." + std::to_string(major) + ".dylib";
}
std::string lib_file_unversioned(const char* base) { return std::string("lib") + base + ".dylib"; }
#else
std::string lib_file(const char* base, int major) {
  return std::string("lib") + base + ".so." + std::to_string(major);
}
std::string lib_file_unversioned(const char* base) { return std::string("lib") + base + ".so"; }
#endif

constexpr char kPathSep = '/';
#endif

// The soname carries the major version, so there is no single file name to
// open. Probing downwards from a version that does not exist yet means a
// future ffmpeg is found and reported by version — "libavcodec 64, we drive
// 62" is a far better message than "not installed" — rather than missed.
constexpr int kProbeMajorHigh = 70;
constexpr int kProbeMajorLow = 55;

// MSVC deprecates getenv and this build carries no _CRT_SECURE_NO_WARNINGS,
// so Windows goes through the Win32 call the CRT wrapper would have used.
bool env_value(const char* name, std::string* out) {
#if defined(_WIN32)
  char buf[1024];
  const DWORD n = ::GetEnvironmentVariableA(name, buf, static_cast<DWORD>(sizeof(buf)));
  if (n == 0 || n >= sizeof(buf)) return false;
  out->assign(buf, n);
  return true;
#else
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') return false;
  out->assign(v);
  return true;
#endif
}

// Set VIDFAB_FFMPEG_DIR to point at a directory of ffmpeg shared libraries.
// This exists for the LGPL substitution case: swapping in your own build must
// not require rebuilding vidfab, and on Windows it must not require editing
// PATH either.
std::string ffmpeg_dir() {
  std::string s;
  if (!env_value("VIDFAB_FFMPEG_DIR", &s)) return {};
  if (s.back() != '/' && s.back() != '\\') s.push_back(kPathSep);
  return s;
}

struct OpenedLib {
  LibHandle handle = nullptr;
  std::string name;
};

// `preferred` is the major this build actually drives. It is tried first, then
// the downward sweep runs as before.
//
// Only the pinned major is usable — anything else is rejected on version a few
// dozen lines below — so on a correct installation the sweep was ~15 failed
// LoadLibrary calls per library before reaching the one that works, and about
// 41 across the four of them. Trying the pinned major first costs one call and
// changes nothing else: the sweep still finds a newer or older ffmpeg when the
// pinned one is absent, which is what keeps the "found libavcodec 64, but
// vidfab drives 62" diagnostic working instead of a bare "not installed".
//
// It does reorder one case: a machine with both the pinned major and a newer
// one now loads the pinned one and works, where before it found the newer one
// and refused. That is the better answer.
OpenedLib open_library(const char* base, const std::string& dir, int preferred,
                       int high = kProbeMajorHigh, int low = kProbeMajorLow) {
  const auto try_major = [&](int major) -> OpenedLib {
    const std::string file = lib_file(base, major);
    if (!dir.empty()) {
      if (LibHandle h = lib_open(dir + file)) return {h, dir + file};
    }
    if (LibHandle h = lib_open(file)) return {h, file};
    return {};
  };
  if (preferred >= low && preferred <= high) {
    const OpenedLib hit = try_major(preferred);
    if (hit.handle != nullptr) return hit;
  }
  for (int major = high; major >= low; --major) {
    if (major == preferred) continue;  // just tried
    const OpenedLib hit = try_major(major);
    if (hit.handle != nullptr) return hit;
  }
  // Unversioned last: on Linux this is the -dev symlink, on Windows it is a
  // hand-renamed build. Either is a deliberate act by the user, so honour it,
  // but never in preference to a properly versioned library.
  const std::string plain = lib_file_unversioned(base);
  if (!dir.empty()) {
    if (LibHandle h = lib_open(dir + plain)) return {h, dir + plain};
  }
  if (LibHandle h = lib_open(plain)) return {h, plain};
  return {};
}

// The one and only place a symbol address is turned into something callable.
// Records the first failure so the caller can name it.
template <typename Fn>
bool bind(LibHandle lib, Fn* out, const char* name, std::string* missing) {
  void* sym = lib_sym(lib, name);
  if (sym == nullptr) {
    if (missing->empty()) *missing = name;
    return false;
  }
  *out = reinterpret_cast<Fn>(sym);
  return true;
}

// --- the ABI we accept ------------------------------------------------------

// ffmpeg 8.x. See ffmpeg_abi.h for why the accepted set is pinned and how to
// widen it.
constexpr unsigned kRequiredAvcodecMajor = 62;
constexpr unsigned kRequiredAvformatMajor = 62;
constexpr unsigned kRequiredAvutilMajor = 60;
// libswscale is not version-gated — nothing below reads a struct of its — so
// this is only the major open_library reaches for first, not a requirement.
constexpr int kPreferredSwscaleMajor = 9;

std::string version_triple(unsigned v) {
  return std::to_string(v >> 16) + "." + std::to_string((v >> 8) & 0xFF) + "." +
         std::to_string(v & 0xFF);
}

// --- load-time layout validation --------------------------------------------

// Confirms that every byte offset in `layout` addresses the field we think it
// does, by driving ffmpeg's own allocators and setters and reading the result
// back through the offset. A version whose layout has moved fails here, with a
// name for the structure that disagreed, instead of corrupting memory two
// hundred lines later.
bool validate_layout(const Api& api, const Layout& l, std::string* why) {
  // AVPacket: a fresh packet has both timestamps at AV_NOPTS_VALUE and an
  // empty payload. Four fields agreeing on that is not a coincidence.
  if (AVPacket* pkt = api.av_packet_alloc()) {
    const bool ok = fld<int64_t>(pkt, l.packet_pts) == kNoPts &&
                    fld<int64_t>(pkt, l.packet_dts) == kNoPts &&
                    fld<void*>(pkt, l.packet_data) == nullptr &&
                    fld<int>(pkt, l.packet_size) == 0 &&
                    fld<int>(pkt, l.packet_stream_index) == 0;
    api.av_packet_free(&pkt);
    if (!ok) {
      *why = "AVPacket";
      return false;
    }
  } else {
    *why = "av_packet_alloc";
    return false;
  }

  // AVFrame, part one: pts. av_frame_alloc zeroes the struct and then sets
  // exactly three timestamps to AV_NOPTS_VALUE, of which pts is the first. So
  // requiring that our offset holds AV_NOPTS_VALUE *and* that no earlier
  // 8-aligned slot does pins the offset rather than merely agreeing with it.
  // pts is the field most disturbed by deprecation removals — `key_frame`
  // sitting in front of it disappeared in ffmpeg 8 — so it earns the scan.
  if (AVFrame* f = api.av_frame_alloc()) {
    bool ok = fld<int64_t>(f, l.frame_pts) == kNoPts;
    for (size_t off = (l.frame_format + 4 + 7) & ~size_t{7}; off + 8 <= l.frame_pts; off += 8) {
      if (fld<int64_t>(f, off) == kNoPts) ok = false;
    }
    api.av_frame_free(&f);
    if (!ok) {
      *why = "AVFrame.pts";
      return false;
    }
  } else {
    *why = "av_frame_alloc";
    return false;
  }

  // AVFrame, part two: hand ffmpeg a description written through our offsets
  // and see whether it allocates the buffers that description implies. RGB24
  // is packed and YUV420P is three planes, so getting one plane for the first
  // and three for the second checks width, height, format, data and linesize
  // at once. RGB24 is used precisely because AV_PIX_FMT_YUV420P is zero: a
  // wrong `format` offset would leave the field zeroed and pass a YUV-only
  // check by accident.
  struct PixCase {
    int format;
    int planes;
    const char* label;
  };
  const PixCase pix_cases[] = {{kPixFmtRgb24, 1, "AVFrame(rgb24)"},
                               {kPixFmtYuv420p, 3, "AVFrame(yuv420p)"}};
  for (const PixCase& c : pix_cases) {
    AVFrame* f = api.av_frame_alloc();
    if (f == nullptr) {
      *why = "av_frame_alloc";
      return false;
    }
    fld<int>(f, l.frame_width) = 64;
    fld<int>(f, l.frame_height) = 32;
    fld<int>(f, l.frame_format) = c.format;
    bool ok = api.av_frame_get_buffer(f, 32) == 0;
    if (ok) {
      uint8_t** data = &fld<uint8_t*>(f, l.frame_data);
      const int* linesize = &fld<int>(f, l.frame_linesize);
      for (int p = 0; p < 4; ++p) {
        const bool want = p < c.planes;
        if ((data[p] != nullptr) != want) ok = false;
      }
      if (linesize[0] < 64) ok = false;
    }
    api.av_frame_free(&f);
    if (!ok) {
      *why = c.label;
      return false;
    }
  }

  // AVFrame, part three: the audio side, which also validates nb_samples and
  // ch_layout. ch_layout is written by ffmpeg itself through our offset, so if
  // the offset were wrong the layout would be unset and the allocation would
  // fail rather than quietly producing one plane.
  {
    AVFrame* f = api.av_frame_alloc();
    if (f == nullptr) {
      *why = "av_frame_alloc";
      return false;
    }
    fld<int>(f, l.frame_nb_samples) = 1024;
    fld<int>(f, l.frame_format) = kSampleFmtFltp;
    api.av_channel_layout_default(reinterpret_cast<char*>(f) + l.frame_ch_layout, 2);
    bool ok = api.av_frame_get_buffer(f, 0) == 0;
    if (ok) {
      uint8_t** data = &fld<uint8_t*>(f, l.frame_data);
      const int* linesize = &fld<int>(f, l.frame_linesize);
      ok = data[0] != nullptr && data[1] != nullptr && data[2] == nullptr &&
           linesize[0] == 1024 * 4;
    }
    api.av_frame_free(&f);
    if (!ok) {
      *why = "AVFrame(audio)";
      return false;
    }
  }

  // AVCodecContext: write through the option API — which addresses fields by
  // name and so cannot be wrong — and read back through our offsets. The three
  // options span the struct from byte 56 to byte 344, which is the whole
  // region we touch. The `-1` defaults for the two formats catch an offset
  // that landed on a neighbouring integer field.
  {
    AVCodecContext* c = api.avcodec_alloc_context3(nullptr);
    if (c == nullptr) {
      *why = "avcodec_alloc_context3";
      return false;
    }
    bool ok = fld<int>(c, l.codec_pix_fmt) == -1 && fld<int>(c, l.codec_sample_fmt) == -1 &&
              fld<int>(c, l.codec_width) == 0 && fld<int>(c, l.codec_height) == 0 &&
              fld<AVRational>(c, l.codec_time_base).num == 0 &&
              fld<AVRational>(c, l.codec_framerate).num == 0;
    ok = ok && api.av_opt_set_int(c, "b", 1234567, 0) == 0 &&
         fld<int64_t>(c, l.codec_bit_rate) == 1234567;
    ok = ok && api.av_opt_set_int(c, "g", 37, 0) == 0 && fld<int>(c, l.codec_gop_size) == 37;
    ok = ok && api.av_opt_set_int(c, "ar", 44100, 0) == 0 &&
         fld<int>(c, l.codec_sample_rate) == 44100;
    // frame_size is read back through the option API too, so it needs no
    // offset — but the option has to exist for that to work.
    int64_t frame_size = -1;
    ok = ok && api.av_opt_get_int(c, "frame_size", 0, &frame_size) == 0;
    api.avcodec_free_context(&c);
    if (!ok) {
      *why = "AVCodecContext";
      return false;
    }
  }

  // AVFormatContext and AVStream. A freshly created stream owns a freshly
  // allocated AVCodecParameters, whose {unknown type, no codec id, no format}
  // triple is a signature nothing else in the struct matches — which is what
  // makes it safe to follow the pointer at all.
  {
    AVFormatContext* oc = nullptr;
    if (api.avformat_alloc_output_context2(&oc, nullptr, "mp4", nullptr) < 0 || oc == nullptr) {
      *why = "no mp4 muxer";
      return false;
    }
    bool ok = fld<void*>(oc, l.format_pb) == nullptr;
    AVStream* st = api.avformat_new_stream(oc, nullptr);
    if (st == nullptr) {
      ok = false;
    } else {
      AVCodecParameters* par = fld<AVCodecParameters*>(st, l.stream_codecpar);
      ok = ok && par != nullptr && fld<int>(par, l.par_codec_type) == kMediaTypeUnknown &&
           fld<int>(par, l.par_codec_id) == 0 && fld<int>(par, l.par_format) == -1;
    }
    api.avformat_free_context(oc);
    if (!ok) {
      *why = "AVStream";
      return false;
    }
  }

  return true;
}

// --- the probe --------------------------------------------------------------

struct Loaded {
  bool ok = false;
  MuxStatus status = MuxStatus::kLibraryNotFound;
  std::string detail;
  std::string version;
  Api api{};
  Layout layout{};
};

Loaded probe() {
  Loaded s;
  const std::string dir = ffmpeg_dir();

  const OpenedLib avutil =
      open_library("avutil", dir, static_cast<int>(kRequiredAvutilMajor));
  const OpenedLib avcodec =
      open_library("avcodec", dir, static_cast<int>(kRequiredAvcodecMajor));
  const OpenedLib avformat =
      open_library("avformat", dir, static_cast<int>(kRequiredAvformatMajor));
  // libswscale has its own much smaller major sequence (9 in FFmpeg 8).
  const OpenedLib swscale = open_library("swscale", dir, kPreferredSwscaleMajor, 20, 1);
  if (avutil.handle == nullptr || avcodec.handle == nullptr || avformat.handle == nullptr ||
      swscale.handle == nullptr) {
    s.status = MuxStatus::kLibraryNotFound;
    s.detail = "could not load ";
    if (avutil.handle == nullptr) s.detail += "libavutil ";
    if (avcodec.handle == nullptr) s.detail += "libavcodec ";
    if (avformat.handle == nullptr) s.detail += "libavformat ";
    if (swscale.handle == nullptr) s.detail += "libswscale ";
    s.detail += "(tried majors " + std::to_string(kProbeMajorLow) + "-" +
                std::to_string(kProbeMajorHigh) + " on the library search path";
    if (!dir.empty()) s.detail += " and in VIDFAB_FFMPEG_DIR=" + dir;
    s.detail += ")";
    return s;
  }

  // Version first: a mismatched major means every offset below is wrong, so
  // there is no point resolving symbols we would not dare call.
  std::string missing;
  Api& api = s.api;
  bool bound = bind(avutil.handle, &api.avutil_version, "avutil_version", &missing) &&
               bind(avcodec.handle, &api.avcodec_version, "avcodec_version", &missing) &&
               bind(avformat.handle, &api.avformat_version, "avformat_version", &missing);
  bound = bound && bind(swscale.handle, &api.swscale_version, "swscale_version", &missing);
  if (!bound) {
    s.status = MuxStatus::kSymbolMissing;
    s.detail = "loaded " + avutil.name + ", " + avcodec.name + ", " + avformat.name +
               " but one of them has no " + missing;
    return s;
  }

  const unsigned util_v = api.avutil_version();
  const unsigned codec_v = api.avcodec_version();
  const unsigned format_v = api.avformat_version();
  s.version = "libavcodec " + version_triple(codec_v) + ", libavformat " +
              version_triple(format_v) + ", libavutil " + version_triple(util_v);

  if ((util_v >> 16) != kRequiredAvutilMajor || (codec_v >> 16) != kRequiredAvcodecMajor ||
      (format_v >> 16) != kRequiredAvformatMajor) {
    s.status = MuxStatus::kSymbolMissing;
    s.detail = "found " + s.version + ", but vidfab drives libavcodec " +
               std::to_string(kRequiredAvcodecMajor) + " / libavformat " +
               std::to_string(kRequiredAvformatMajor) + " / libavutil " +
               std::to_string(kRequiredAvutilMajor) + " (ffmpeg 8.x)";
    return s;
  }

  bound =
      bind(avutil.handle, &api.av_log_set_level, "av_log_set_level", &missing) &&
      bind(avutil.handle, &api.av_frame_alloc, "av_frame_alloc", &missing) &&
      bind(avutil.handle, &api.av_frame_free, "av_frame_free", &missing) &&
      bind(avutil.handle, &api.av_frame_get_buffer, "av_frame_get_buffer", &missing) &&
      bind(avutil.handle, &api.av_frame_make_writable, "av_frame_make_writable", &missing) &&
      bind(avutil.handle, &api.av_channel_layout_default, "av_channel_layout_default", &missing) &&
      bind(avutil.handle, &api.av_opt_set, "av_opt_set", &missing) &&
      bind(avutil.handle, &api.av_opt_set_int, "av_opt_set_int", &missing) &&
      bind(avutil.handle, &api.av_opt_get_int, "av_opt_get_int", &missing) &&
      bind(avutil.handle, &api.av_dict_set, "av_dict_set", &missing) &&
      bind(avutil.handle, &api.av_dict_free, "av_dict_free", &missing) &&
      bind(avutil.handle, &api.av_strerror, "av_strerror", &missing) &&
      bind(avutil.handle, &api.av_compare_ts, "av_compare_ts", &missing) &&
      bind(avcodec.handle, &api.avcodec_find_encoder_by_name, "avcodec_find_encoder_by_name",
           &missing) &&
      bind(avcodec.handle, &api.avcodec_find_encoder, "avcodec_find_encoder", &missing) &&
      bind(avcodec.handle, &api.avcodec_find_decoder, "avcodec_find_decoder", &missing) &&
      bind(avcodec.handle, &api.avcodec_alloc_context3, "avcodec_alloc_context3", &missing) &&
      bind(avcodec.handle, &api.avcodec_free_context, "avcodec_free_context", &missing) &&
      bind(avcodec.handle, &api.avcodec_open2, "avcodec_open2", &missing) &&
      bind(avcodec.handle, &api.avcodec_parameters_from_context, "avcodec_parameters_from_context",
           &missing) &&
      bind(avcodec.handle, &api.avcodec_parameters_to_context, "avcodec_parameters_to_context",
           &missing) &&
      bind(avcodec.handle, &api.avcodec_send_frame, "avcodec_send_frame", &missing) &&
      bind(avcodec.handle, &api.avcodec_send_packet, "avcodec_send_packet", &missing) &&
      bind(avcodec.handle, &api.avcodec_receive_frame, "avcodec_receive_frame", &missing) &&
      bind(avcodec.handle, &api.avcodec_receive_packet, "avcodec_receive_packet", &missing) &&
      bind(avcodec.handle, &api.avcodec_get_supported_config, "avcodec_get_supported_config",
           &missing) &&
      bind(avcodec.handle, &api.av_packet_alloc, "av_packet_alloc", &missing) &&
      bind(avcodec.handle, &api.av_packet_free, "av_packet_free", &missing) &&
      bind(avcodec.handle, &api.av_packet_rescale_ts, "av_packet_rescale_ts", &missing) &&
      bind(avcodec.handle, &api.av_packet_unref, "av_packet_unref", &missing) &&
      bind(avformat.handle, &api.avformat_alloc_output_context2, "avformat_alloc_output_context2",
           &missing) &&
      bind(avformat.handle, &api.avformat_free_context, "avformat_free_context", &missing) &&
      bind(avformat.handle, &api.avformat_new_stream, "avformat_new_stream", &missing) &&
      bind(avformat.handle, &api.avformat_write_header, "avformat_write_header", &missing) &&
      bind(avformat.handle, &api.av_interleaved_write_frame, "av_interleaved_write_frame",
           &missing) &&
      bind(avformat.handle, &api.av_write_trailer, "av_write_trailer", &missing) &&
      bind(avformat.handle, &api.avio_open, "avio_open", &missing) &&
      bind(avformat.handle, &api.avio_closep, "avio_closep", &missing);
  bound = bound &&
      bind(avformat.handle, &api.avformat_open_input, "avformat_open_input", &missing) &&
      bind(avformat.handle, &api.avformat_find_stream_info, "avformat_find_stream_info", &missing) &&
      bind(avformat.handle, &api.av_find_best_stream, "av_find_best_stream", &missing) &&
      bind(avformat.handle, &api.av_read_frame, "av_read_frame", &missing) &&
      bind(avformat.handle, &api.avformat_close_input, "avformat_close_input", &missing) &&
      bind(swscale.handle, &api.sws_getContext, "sws_getContext", &missing) &&
      bind(swscale.handle, &api.sws_scale, "sws_scale", &missing) &&
      bind(swscale.handle, &api.sws_freeContext, "sws_freeContext", &missing);
  if (!bound) {
    s.status = MuxStatus::kSymbolMissing;
    s.detail = s.version + " is missing " + missing;
    return s;
  }

  // ffmpeg logs to stderr at AV_LOG_INFO by default, which turns a successful
  // encode into a wall of text on a CLI that has its own progress output.
  std::string verbose;
  api.av_log_set_level(env_value("VIDFAB_FFMPEG_VERBOSE", &verbose) ? kLogVerbose : kLogError);

  s.layout = kLayoutFfmpeg8;
  std::string why;
  if (!validate_layout(api, s.layout, &why)) {
    s.status = MuxStatus::kSymbolMissing;
    s.detail = s.version + " has an unexpected " + why +
               " layout; refusing to write through offsets we cannot verify";
    return s;
  }

  s.ok = true;
  s.status = MuxStatus::kOk;
  s.detail = "loaded " + avutil.name + ", " + avcodec.name + ", " + avformat.name;
  // The handles are intentionally never freed. They live for the process, the
  // cached Api points into them, and unloading avcodec while a static
  // destructor elsewhere still holds an AVFrame is a much worse outcome than
  // leaking three module references.
  return s;
}

const Loaded& loaded() {
  // Function-local static: the probe runs exactly once, thread-safely, on the
  // first call, which is what lets the CLI ask `ffmpeg_available()` before any
  // work starts without paying for it twice.
  static const Loaded state = probe();
  return state;
}

std::string err_text(const Api& api, int code) {
  char buf[256] = {0};
  if (api.av_strerror(code, buf, sizeof(buf)) < 0) return std::to_string(code);
  return std::string(buf) + " (" + std::to_string(code) + ")";
}

// --- encoder selection ------------------------------------------------------

// A list terminated by a config-specific sentinel, or null when the encoder
// accepts everything. `all` distinguishes those two, because "no restrictions"
// and "nothing supported" must not be confused.
template <typename T>
std::vector<T> supported_config(const Api& api, const AVCodec* codec, int config, bool* all) {
  const void* list = nullptr;
  int count = 0;
  *all = false;
  if (api.avcodec_get_supported_config(nullptr, codec, config, 0, &list, &count) < 0) {
    *all = true;  // query unsupported for this codec: assume no restriction
    return {};
  }
  if (list == nullptr) {
    *all = true;
    return {};
  }
  const T* typed = static_cast<const T*>(list);
  return std::vector<T>(typed, typed + count);
}

template <typename T>
bool contains(const std::vector<T>& v, T value) {
  return std::find(v.begin(), v.end(), value) != v.end();
}

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

// --- audio conversion -------------------------------------------------------

// Linear interpolation, used only when the encoder refuses the requested rate.
// Every AAC encoder in existence accepts 32 kHz, so in practice this is dead
// code that exists so an exotic build degrades in quality rather than failing.
std::vector<float> resample_linear(const std::vector<float>& in, int channels, int in_rate,
                                   int out_rate) {
  if (in_rate == out_rate || in.empty()) return in;
  const size_t in_frames = in.size() / static_cast<size_t>(channels);
  const size_t out_frames =
      static_cast<size_t>(static_cast<double>(in_frames) * out_rate / in_rate);
  std::vector<float> out(out_frames * static_cast<size_t>(channels));
  for (size_t i = 0; i < out_frames; ++i) {
    const double src = static_cast<double>(i) * in_rate / out_rate;
    const size_t i0 = static_cast<size_t>(src);
    const size_t i1 = std::min(i0 + 1, in_frames - 1);
    const float t = static_cast<float>(src - static_cast<double>(i0));
    for (int c = 0; c < channels; ++c) {
      const float a = in[i0 * channels + c];
      const float b = in[i1 * channels + c];
      out[i * channels + c] = a + (b - a) * t;
    }
  }
  return out;
}

int16_t to_s16(float v) {
  const float clamped = std::min(1.0f, std::max(-1.0f, v));
  return static_cast<int16_t>(std::lround(clamped * 32767.0f));
}

// Writes `count` interleaved frames starting at `first` into an AVFrame's
// buffers in whichever of the four layouts the encoder asked for.
void fill_audio_frame(const std::vector<float>& interleaved, size_t first, int count, int channels,
                      int sample_fmt, uint8_t** data) {
  for (int i = 0; i < count; ++i) {
    for (int c = 0; c < channels; ++c) {
      const size_t src = (first + static_cast<size_t>(i)) * channels + c;
      const float v = src < interleaved.size() ? interleaved[src] : 0.0f;
      switch (sample_fmt) {
        case kSampleFmtFltp:
          reinterpret_cast<float*>(data[c])[i] = v;
          break;
        case kSampleFmtFlt:
          reinterpret_cast<float*>(data[0])[i * channels + c] = v;
          break;
        case kSampleFmtS16p:
          reinterpret_cast<int16_t*>(data[c])[i] = to_s16(v);
          break;
        default:  // kSampleFmtS16
          reinterpret_cast<int16_t*>(data[0])[i * channels + c] = to_s16(v);
          break;
      }
    }
  }
}

// --- encode state -----------------------------------------------------------

struct Track {
  AVStream* stream = nullptr;
  AVCodecContext* ctx = nullptr;
  AVFrame* frame = nullptr;
  AVRational codec_tb{1, 1};   // what the encoder stamps packets in
  AVRational stream_tb{1, 1};  // what the muxer decided on, known after the header
  // AVStream::index is assigned in creation order, which is documented
  // behaviour — so the video stream, created first, is 0 and the audio stream
  // is 1. Relying on that is what keeps AVStream::index out of the offset
  // table.
  int index = 0;
  bool finished = false;
};

}  // namespace

// --- public -----------------------------------------------------------------

const char* mux_status_message(MuxStatus s) {
  switch (s) {
    case MuxStatus::kOk:
      return "ok";
    case MuxStatus::kLibraryNotFound:
      return "ffmpeg shared libraries not found";
    case MuxStatus::kSymbolMissing:
      return "ffmpeg found but not a version vidfab knows how to drive";
    case MuxStatus::kEncoderMissing:
      return "this ffmpeg build has no usable H.264 or AAC encoder";
    case MuxStatus::kWriteFailed:
      return "writing the MP4 failed";
  }
  return "unknown";
}

bool ffmpeg_available(std::string* detail) {
  const Loaded& s = loaded();
  if (detail != nullptr) *detail = s.detail;
  return s.ok;
}

std::string ffmpeg_version() { return loaded().version; }

MuxStatus write_mp4(const MuxRequest& request) {
  const Loaded& s = loaded();
  if (!s.ok) return s.status;
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
  if (want_audio &&
      (request.audio_channels <= 0 || request.audio_sample_rate <= 0 ||
       (request.audio->size() % static_cast<size_t>(request.audio_channels)) != 0)) {
    return MuxStatus::kWriteFailed;
  }

  std::string video_name;
  const AVCodec* vcodec = find_video_encoder(api, &video_name);
  if (vcodec == nullptr) return MuxStatus::kEncoderMissing;

  std::string audio_name;
  const AVCodec* acodec = want_audio ? find_audio_encoder(api, &audio_name) : nullptr;
  if (want_audio && acodec == nullptr) return MuxStatus::kEncoderMissing;

  // Everything below owns ffmpeg objects, so it runs under one cleanup path.
  AVFormatContext* oc = nullptr;
  AVPacket* pkt = nullptr;
  Track video;
  Track audio;
  bool header_written = false;
  bool file_opened = false;
  MuxStatus status = MuxStatus::kWriteFailed;

  auto cleanup = [&]() {
    if (video.frame != nullptr) api.av_frame_free(&video.frame);
    if (audio.frame != nullptr) api.av_frame_free(&audio.frame);
    if (video.ctx != nullptr) api.avcodec_free_context(&video.ctx);
    if (audio.ctx != nullptr) api.avcodec_free_context(&audio.ctx);
    if (pkt != nullptr) api.av_packet_free(&pkt);
    if (oc != nullptr) {
      AVIOContext*& pb = fld<AVIOContext*>(oc, l.format_pb);
      if (pb != nullptr) api.avio_closep(&pb);
      api.avformat_free_context(oc);
      oc = nullptr;
    }
    // A half-written MP4 is worse than no MP4: the caller falls back to the
    // .y4m pair, and a truncated file sitting next to it will be opened by
    // someone eventually.
    if (status != MuxStatus::kOk && file_opened) std::remove(request.path.c_str());
    return status;
  };

  if (api.avformat_alloc_output_context2(&oc, nullptr, "mp4", request.path.c_str()) < 0 ||
      oc == nullptr) {
    return cleanup();
  }

  // --- video stream ---
  video.index = 0;
  video.stream = api.avformat_new_stream(oc, nullptr);
  if (video.stream == nullptr) return cleanup();
  video.codec_tb = AVRational{request.fps.denominator, request.fps.numerator};
  fld<AVRational>(video.stream, l.stream_time_base) = video.codec_tb;

  video.ctx = api.avcodec_alloc_context3(vcodec);
  if (video.ctx == nullptr) return cleanup();
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

  if (api.avcodec_open2(video.ctx, vcodec, nullptr) < 0) return cleanup();
  if (api.avcodec_parameters_from_context(fld<AVCodecParameters*>(video.stream, l.stream_codecpar),
                                          video.ctx) < 0) {
    return cleanup();
  }

  video.frame = api.av_frame_alloc();
  if (video.frame == nullptr) return cleanup();
  fld<int>(video.frame, l.frame_width) = request.width;
  fld<int>(video.frame, l.frame_height) = request.height;
  fld<int>(video.frame, l.frame_format) = kPixFmtYuv420p;
  if (api.av_frame_get_buffer(video.frame, 32) != 0) return cleanup();

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
      if (audio_sample_fmt < 0) return (status = MuxStatus::kEncoderMissing, cleanup());
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
      if (best == 0) return (status = MuxStatus::kEncoderMissing, cleanup());
      audio_rate = best;
    }
    audio_samples = resample_linear(*request.audio, request.audio_channels,
                                    request.audio_sample_rate, audio_rate);

    audio.index = 1;
    audio.stream = api.avformat_new_stream(oc, nullptr);
    if (audio.stream == nullptr) return cleanup();
    audio.codec_tb = AVRational{1, audio_rate};
    fld<AVRational>(audio.stream, l.stream_time_base) = audio.codec_tb;

    audio.ctx = api.avcodec_alloc_context3(acodec);
    if (audio.ctx == nullptr) return cleanup();
    fld<int>(audio.ctx, l.codec_sample_fmt) = audio_sample_fmt;
    fld<AVRational>(audio.ctx, l.codec_time_base) = audio.codec_tb;
    api.av_opt_set_int(audio.ctx, "ar", audio_rate, 0);
    api.av_opt_set_int(audio.ctx, "b", request.audio_bitrate, 0);
    // Channel layout goes through the option API, so its offset never has to
    // be known: "2c" is the generic spelling ffmpeg accepts for any count.
    const std::string layout_name = request.audio_channels == 1   ? "mono"
                                    : request.audio_channels == 2 ? "stereo"
                                    : std::to_string(request.audio_channels) + "c";
    if (api.av_opt_set(audio.ctx, "ch_layout", layout_name.c_str(), 0) < 0) {
      return (status = MuxStatus::kEncoderMissing, cleanup());
    }
    {
      int64_t flags = 0;
      api.av_opt_get_int(audio.ctx, "flags", 0, &flags);
      api.av_opt_set_int(audio.ctx, "flags", flags | kCodecFlagGlobalHeader, 0);
    }

    if (api.avcodec_open2(audio.ctx, acodec, nullptr) < 0) return cleanup();
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
    if (audio.frame == nullptr) return cleanup();
    fld<int>(audio.frame, l.frame_nb_samples) = audio_frame_samples;
    fld<int>(audio.frame, l.frame_format) = audio_sample_fmt;
    api.av_channel_layout_default(reinterpret_cast<char*>(audio.frame) + l.frame_ch_layout,
                                  request.audio_channels);
    if (api.av_frame_get_buffer(audio.frame, 0) != 0) return cleanup();
  }

  // --- open the file and write the header ---
  {
    AVIOContext*& pb = fld<AVIOContext*>(oc, l.format_pb);
    if (api.avio_open(&pb, request.path.c_str(), kAvioFlagWrite) < 0) return cleanup();
    file_opened = true;
  }

  AVDictionary* opts = nullptr;
  // Rewriting the index to the front costs one extra pass over the file and
  // makes the result streamable, which is what a viewer expects of an MP4.
  api.av_dict_set(&opts, "movflags", "+faststart", 0);
  const int header = api.avformat_write_header(oc, &opts);
  api.av_dict_free(&opts);
  if (header < 0) return cleanup();
  header_written = true;

  // The muxer rewrites the stream time bases — mov picks its own timescale —
  // so every packet has to be rescaled from the encoder's base into whatever
  // it decided on. Getting this wrong is the classic silent a/v drift.
  video.stream_tb = fld<AVRational>(video.stream, l.stream_time_base);
  if (video.stream_tb.num <= 0 || video.stream_tb.den <= 0) return cleanup();
  if (want_audio) {
    audio.stream_tb = fld<AVRational>(audio.stream, l.stream_time_base);
    if (audio.stream_tb.num <= 0 || audio.stream_tb.den <= 0) return cleanup();
  }

  pkt = api.av_packet_alloc();
  if (pkt == nullptr) return cleanup();

  auto drain = [&](Track& track) -> bool {
    for (;;) {
      const int ret = api.avcodec_receive_packet(track.ctx, pkt);
      if (ret == kErrAgain || ret == kErrEof) return true;
      if (ret < 0) return false;
      api.av_packet_rescale_ts(pkt, track.codec_tb, track.stream_tb);
      fld<int>(pkt, l.packet_stream_index) = track.index;
      // Takes ownership of the packet's reference and blanks it, so there is
      // no unref to pair with this.
      if (api.av_interleaved_write_frame(oc, pkt) < 0) return false;
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
        if (api.av_frame_make_writable(video.frame) < 0) return cleanup();
        uint8_t** data = &fld<uint8_t*>(video.frame, l.frame_data);
        const int* linesize = &fld<int>(video.frame, l.frame_linesize);
        const size_t base = static_cast<size_t>(video_index) * frame_pixels;
        rgb_frame_to_yuv420(r_plane + base, g_plane + base, b_plane + base, request.height,
                            request.width, data[0], linesize[0], data[1], linesize[1], data[2],
                            linesize[2]);
        fld<int64_t>(video.frame, l.frame_pts) = video_pts;
        frame = video.frame;
        ++video_index;
        ++video_pts;
      } else {
        video.finished = true;  // null frame flushes the encoder
      }
      if (api.avcodec_send_frame(video.ctx, frame) < 0) return cleanup();
      if (!drain(video)) return cleanup();
    } else {
      AVFrame* frame = nullptr;
      if (audio_cursor < audio_frames_total) {
        const int count = static_cast<int>(
            std::min(static_cast<size_t>(audio_frame_samples), audio_frames_total - audio_cursor));
        // Shrinking nb_samples for the final partial frame is allowed and is
        // how the encoder learns not to emit a block of trailing silence.
        fld<int>(audio.frame, l.frame_nb_samples) = count;
        if (api.av_frame_make_writable(audio.frame) < 0) return cleanup();
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
      if (api.avcodec_send_frame(audio.ctx, frame) < 0) return cleanup();
      if (!drain(audio)) return cleanup();
    }
  }

  if (header_written && api.av_write_trailer(oc) < 0) return cleanup();

  status = MuxStatus::kOk;
  return cleanup();
}

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

}  // namespace vidfab::video
