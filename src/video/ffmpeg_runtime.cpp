#include "mux_internal.h"
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

namespace slopfab::video::mux_detail {
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
  // SLOPFAB_FFMPEG_DIR override only works when the directory is on PATH too,
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

std::string lib_file_unversioned(const char* base) {
  return std::string(base) + ".dll";
}

constexpr char kPathSep = '\\';
#else
using LibHandle = void*;

LibHandle lib_open(const std::string& path) {
  return ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}

void* lib_sym(LibHandle h, const char* name) {
  return ::dlsym(h, name);
}

#if defined(__APPLE__)
std::string lib_file(const char* base, int major) {
  return std::string("lib") + base + "." + std::to_string(major) + ".dylib";
}

std::string lib_file_unversioned(const char* base) {
  return std::string("lib") + base + ".dylib";
}
#else
std::string lib_file(const char* base, int major) {
  return std::string("lib") + base + ".so." + std::to_string(major);
}

std::string lib_file_unversioned(const char* base) {
  return std::string("lib") + base + ".so";
}
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
  if (n == 0 || n >= sizeof(buf))
    return false;
  out->assign(buf, n);
  return true;
#else
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0')
    return false;
  out->assign(v);
  return true;
#endif
}

// Set SLOPFAB_FFMPEG_DIR to point at a directory of ffmpeg shared libraries.
// This exists for the LGPL substitution case: swapping in your own build must
// not require rebuilding slopfab, and on Windows it must not require editing
// PATH either.
std::string ffmpeg_dir() {
  std::string s;
  if (!env_value("SLOPFAB_FFMPEG_DIR", &s))
    return {};
  if (s.back() != '/' && s.back() != '\\')
    s.push_back(kPathSep);
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
// slopfab drives 62" diagnostic working instead of a bare "not installed".
//
// It does reorder one case: a machine with both the pinned major and a newer
// one now loads the pinned one and works, where before it found the newer one
// and refused. That is the better answer.
OpenedLib open_library(const char* base, const std::string& dir, int preferred,
                       int high = kProbeMajorHigh, int low = kProbeMajorLow) {
  const auto try_major = [&](int major) -> OpenedLib {
    const std::string file = lib_file(base, major);
    if (!dir.empty()) {
      if (LibHandle h = lib_open(dir + file))
        return {h, dir + file};
    }
    if (LibHandle h = lib_open(file))
      return {h, file};
    return {};
  };
  if (preferred >= low && preferred <= high) {
    const OpenedLib hit = try_major(preferred);
    if (hit.handle != nullptr)
      return hit;
  }
  for (int major = high; major >= low; --major) {
    if (major == preferred)
      continue; // just tried
    const OpenedLib hit = try_major(major);
    if (hit.handle != nullptr)
      return hit;
  }
  // Unversioned last: on Linux this is the -dev symlink, on Windows it is a
  // hand-renamed build. Either is a deliberate act by the user, so honour it,
  // but never in preference to a properly versioned library.
  const std::string plain = lib_file_unversioned(base);
  if (!dir.empty()) {
    if (LibHandle h = lib_open(dir + plain))
      return {h, dir + plain};
  }
  if (LibHandle h = lib_open(plain))
    return {h, plain};
  return {};
}

// The one and only place a symbol address is turned into something callable.
// Records the first failure so the caller can name it.
template <typename Fn> bool bind(LibHandle lib, Fn* out, const char* name, std::string* missing) {
  void* sym = lib_sym(lib, name);
  if (sym == nullptr) {
    if (missing->empty())
      *missing = name;
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
                    fld<int>(pkt, l.packet_size) == 0 && fld<int>(pkt, l.packet_stream_index) == 0;
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
      if (fld<int64_t>(f, off) == kNoPts)
        ok = false;
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
        if ((data[p] != nullptr) != want)
          ok = false;
      }
      if (linesize[0] < 64)
        ok = false;
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
      ok =
          data[0] != nullptr && data[1] != nullptr && data[2] == nullptr && linesize[0] == 1024 * 4;
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

Loaded probe() {
  Loaded s;
  const std::string dir = ffmpeg_dir();

  const OpenedLib avutil = open_library("avutil", dir, static_cast<int>(kRequiredAvutilMajor));
  const OpenedLib avcodec = open_library("avcodec", dir, static_cast<int>(kRequiredAvcodecMajor));
  const OpenedLib avformat =
      open_library("avformat", dir, static_cast<int>(kRequiredAvformatMajor));
  // libswscale has its own much smaller major sequence (9 in FFmpeg 8).
  const OpenedLib swscale = open_library("swscale", dir, kPreferredSwscaleMajor, 20, 1);
  if (avutil.handle == nullptr || avcodec.handle == nullptr || avformat.handle == nullptr ||
      swscale.handle == nullptr) {
    s.status = MuxStatus::kLibraryNotFound;
    s.detail = "could not load ";
    if (avutil.handle == nullptr)
      s.detail += "libavutil ";
    if (avcodec.handle == nullptr)
      s.detail += "libavcodec ";
    if (avformat.handle == nullptr)
      s.detail += "libavformat ";
    if (swscale.handle == nullptr)
      s.detail += "libswscale ";
    s.detail += "(tried majors " + std::to_string(kProbeMajorLow) + "-" +
                std::to_string(kProbeMajorHigh) + " on the library search path";
    if (!dir.empty())
      s.detail += " and in SLOPFAB_FFMPEG_DIR=" + dir;
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
    s.detail = "found " + s.version + ", but slopfab drives libavcodec " +
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
          bind(avformat.handle, &api.avformat_find_stream_info, "avformat_find_stream_info",
               &missing) &&
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
  api.av_log_set_level(env_value("SLOPFAB_FFMPEG_VERBOSE", &verbose) ? kLogVerbose : kLogError);

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
  if (api.av_strerror(code, buf, sizeof(buf)) < 0)
    return std::to_string(code);
  return std::string(buf) + " (" + std::to_string(code) + ")";
}

} // namespace slopfab::video::mux_detail

namespace slopfab::video {
using namespace mux_detail;

const char* mux_status_message(MuxStatus s) {
  switch (s) {
  case MuxStatus::kOk:
    return "ok";
  case MuxStatus::kLibraryNotFound:
    return "ffmpeg shared libraries not found";
  case MuxStatus::kSymbolMissing:
    return "ffmpeg found but not a version slopfab knows how to drive";
  case MuxStatus::kEncoderMissing:
    return "this ffmpeg build has no usable H.264 or AAC encoder";
  case MuxStatus::kWriteFailed:
    return "writing the MP4 failed";
  }
  return "unknown";
}

bool ffmpeg_available(std::string* detail) {
  const Loaded& s = loaded();
  if (detail != nullptr)
    *detail = s.detail;
  return s.ok;
}

std::string ffmpeg_version() {
  return loaded().version;
}

} // namespace slopfab::video
