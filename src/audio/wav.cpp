#include "slopfab/audio/wav.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace slopfab::audio {
namespace {

// RIFF is little-endian everywhere, including on big-endian hosts, so the
// integers are serialised byte by byte rather than memcpy'd. The cost is
// nothing next to the sample conversion and it removes a whole class of
// "works on my machine" bug.
void put_u16(std::vector<uint8_t>& out, uint16_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

void put_tag(std::vector<uint8_t>& out, const char (&tag)[5]) {
  out.insert(out.end(), tag, tag + 4);
}

// Appends a complete chunk: FourCC, 32-bit payload size, payload, and — when
// the payload length is odd — the RIFF pad byte.
//
// The pad byte is the trap in this format: it is *not* counted in the chunk's
// own size field but *is* counted in the enclosing RIFF size, and a reader
// that walks chunks by `8 + size` without rounding up lands mid-word on the
// next header. None of the formats written here produce an odd payload (16-
// and 32-bit samples are both even multiples), but the padding is applied by
// construction so adding a chunk later cannot silently break the file.
void put_chunk(std::vector<uint8_t>& out, const char (&tag)[5],
               const std::vector<uint8_t>& payload) {
  put_tag(out, tag);
  put_u32(out, static_cast<uint32_t>(payload.size()));
  out.insert(out.end(), payload.begin(), payload.end());
  if ((payload.size() % 2) != 0) out.push_back(0);
}

constexpr uint16_t kFormatPcm = 1;        // WAVE_FORMAT_PCM
constexpr uint16_t kFormatIeeeFloat = 3;  // WAVE_FORMAT_IEEE_FLOAT

}  // namespace

void write_wav(const std::string& path, const std::vector<float>& interleaved, int channels,
               int sample_rate, SampleFormat format) {
  if (channels <= 0) throw std::runtime_error("wav: channel count must be positive");
  if (sample_rate <= 0) throw std::runtime_error("wav: sample rate must be positive");
  if ((interleaved.size() % static_cast<size_t>(channels)) != 0) {
    throw std::runtime_error("wav: sample count " + std::to_string(interleaved.size()) +
                             " is not a multiple of " + std::to_string(channels) + " channels");
  }

  const bool is_float = format == SampleFormat::kFloat32;
  const uint16_t bits = is_float ? 32 : 16;
  const uint16_t bytes_per_sample = static_cast<uint16_t>(bits / 8);
  const uint16_t block_align = static_cast<uint16_t>(bytes_per_sample * channels);
  const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;
  const size_t frames = interleaved.size() / static_cast<size_t>(channels);

  // `fmt ` is 16 bytes for plain PCM and 18 for everything else: non-PCM
  // formats must carry the cbSize field, and a reader that trusts the format
  // tag but not the chunk size will look for it. The extension is empty, so
  // cbSize is zero.
  std::vector<uint8_t> fmt;
  fmt.reserve(18);
  put_u16(fmt, is_float ? kFormatIeeeFloat : kFormatPcm);
  put_u16(fmt, static_cast<uint16_t>(channels));
  put_u32(fmt, static_cast<uint32_t>(sample_rate));
  put_u32(fmt, byte_rate);
  put_u16(fmt, block_align);
  put_u16(fmt, bits);
  if (is_float) put_u16(fmt, 0);  // cbSize

  std::vector<uint8_t> data;
  data.resize(interleaved.size() * bytes_per_sample);
  if (is_float) {
    // Pass through unchanged, including out-of-range values: a float WAV is
    // the debugging format, and silently squashing a decode that overshot
    // would hide exactly the bug it exists to expose.
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32 bits");
    for (size_t i = 0; i < interleaved.size(); ++i) {
      const float v = interleaved[i];
      uint32_t bits32 = 0;
      std::memcpy(&bits32, &v, sizeof(bits32));
      uint8_t* p = data.data() + i * 4;
      p[0] = static_cast<uint8_t>(bits32 & 0xFF);
      p[1] = static_cast<uint8_t>((bits32 >> 8) & 0xFF);
      p[2] = static_cast<uint8_t>((bits32 >> 16) & 0xFF);
      p[3] = static_cast<uint8_t>((bits32 >> 24) & 0xFF);
    }
  } else {
    for (size_t i = 0; i < interleaved.size(); ++i) {
      // Clamp before scaling, and scale by 32767 rather than 32768: the
      // asymmetric range is what makes +1.0 and -1.0 map to equal magnitudes,
      // so a symmetric waveform stays symmetric instead of clipping one lobe
      // a sample earlier than the other.
      const float v = std::min(1.0f, std::max(-1.0f, interleaved[i]));
      const int32_t s = static_cast<int32_t>(std::lround(v * 32767.0f));
      const uint16_t u = static_cast<uint16_t>(static_cast<int16_t>(s));
      uint8_t* p = data.data() + i * 2;
      p[0] = static_cast<uint8_t>(u & 0xFF);
      p[1] = static_cast<uint8_t>((u >> 8) & 0xFF);
    }
  }

  std::vector<uint8_t> body;
  body.reserve(data.size() + 64);
  put_tag(body, "WAVE");
  put_chunk(body, "fmt ", fmt);
  if (is_float) {
    // A `fact` chunk carrying the sample-frame count is mandatory for every
    // non-PCM format. Most readers ignore it; the ones that do not refuse the
    // file without it.
    std::vector<uint8_t> fact;
    put_u32(fact, static_cast<uint32_t>(frames));
    put_chunk(body, "fact", fact);
  }
  put_chunk(body, "data", data);

  std::vector<uint8_t> file;
  file.reserve(body.size() + 8);
  put_tag(file, "RIFF");
  put_u32(file, static_cast<uint32_t>(body.size()));
  file.insert(file.end(), body.begin(), body.end());

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("wav: cannot open " + path + " for writing");
  out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
  if (!out) throw std::runtime_error("wav: write failed for " + path);
}

}  // namespace slopfab::audio
