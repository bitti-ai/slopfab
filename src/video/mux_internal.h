#pragma once
#include "slopfab/video/mux.h"
#include "slopfab/video/media.h"

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

namespace slopfab::video::mux_detail {
using namespace ff;

struct Loaded {
  bool ok = false;
  MuxStatus status = MuxStatus::kLibraryNotFound;
  std::string detail;
  std::string version;
  Api api{};
  Layout layout{};
};

const Loaded& loaded();
std::string err_text(const Api&, int code);
const AVCodec* find_video_encoder(const Api&, std::string* name);
const AVCodec* find_audio_encoder(const Api&, std::string* name);
std::vector<float> resample_linear(const std::vector<float>&, int channels, int in_rate,
                                   int out_rate);
void fill_audio_frame(const std::vector<float>&, size_t first, int count, int channels,
                      int sample_fmt, uint8_t** data);

// A list terminated by a config-specific sentinel, or null when the encoder
// accepts everything. `all` distinguishes those two, because "no restrictions"
// and "nothing supported" must not be confused.
template <typename T>
std::vector<T> supported_config(const Api& api, const AVCodec* codec, int config, bool* all) {
  const void* list = nullptr;
  int count = 0;
  *all = false;
  if (api.avcodec_get_supported_config(nullptr, codec, config, 0, &list, &count) < 0) {
    *all = true; // query unsupported for this codec: assume no restriction
    return {};
  }
  if (list == nullptr) {
    *all = true;
    return {};
  }
  const T* typed = static_cast<const T*>(list);
  return std::vector<T>(typed, typed + count);
}

template <typename T> bool contains(const std::vector<T>& v, T value) {
  return std::find(v.begin(), v.end(), value) != v.end();
}

} // namespace slopfab::video::mux_detail
