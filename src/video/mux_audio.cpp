#include "mux_internal.h"

namespace slopfab::video::mux_detail {
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

}  // namespace slopfab::video::mux_detail
