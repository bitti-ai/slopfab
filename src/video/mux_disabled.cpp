// The muxer's interface, for builds compiled without FFmpeg.
//
// This is a whole translation unit rather than an `#if` around mux.cpp's 1400
// lines, because the two configurations share no code at all: everything in
// that file exists to find, load and drive libavcodec.
//
// The point of it is that no call site has to know. `run_generate` already
// asks `ffmpeg_available` before muxing and falls back to `.y4m` + `.wav` when
// the answer is no — a path that exists for the machine that has no FFmpeg
// installed, and works identically for the build that has none compiled in.
// So a build with SLOPFAB_WITH_FFMPEG off writes raw output through the same
// branch, and the C API, which takes the pixels before either branch runs, is
// unaffected either way.
//
// `decode_first_video_frame` is deliberately absent rather than stubbed: it is
// declared in media.h behind the same switch, and its one caller — the
// reference-image loader — uses the platform decoder in this configuration.
// A stub would turn a compile error into a runtime one for no benefit.
#include "slopfab/video/mux.h"

#include <string>

namespace slopfab::video {

// Defined here as well as in mux.cpp because `run_generate` calls it on the
// mux-failed branch. That branch is unreachable in this configuration —
// nothing gets past `ffmpeg_available` — but it is still compiled, so the
// symbol has to exist or the link fails.
const char* mux_status_message(MuxStatus s) {
  switch (s) {
  case MuxStatus::kOk:
    return "ok";
  case MuxStatus::kLibraryNotFound:
    return "this build has no FFmpeg support compiled in";
  case MuxStatus::kSymbolMissing:
    return "ffmpeg found but not a version slopfab knows how to drive";
  case MuxStatus::kEncoderMissing:
    return "this ffmpeg build has no usable H.264 or AAC encoder";
  case MuxStatus::kWriteFailed:
    return "writing the MP4 failed";
  }
  return "unknown mux status";
}

bool ffmpeg_available(std::string* detail) {
  if (detail != nullptr) {
    *detail = "this build was compiled without FFmpeg support (SLOPFAB_WITH_FFMPEG=OFF)";
  }
  return false;
}

std::string ffmpeg_version() {
  return std::string();
}

MuxStatus write_mp4(const MuxRequest&) {
  // Reported as "not found" rather than as a distinct status, because that is
  // what it is from every caller's point of view: there is no library to drive
  // and the fallback is the same one a missing DLL takes. Adding a status for
  // it would oblige every consumer of MuxStatus to handle a case that means
  // exactly what an existing one already does.
  return MuxStatus::kLibraryNotFound;
}

} // namespace slopfab::video
