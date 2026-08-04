// Container output tests: RIFF/WAVE, and MP4 muxing through whatever ffmpeg
// happens to be installed.
//
// The WAV cases parse the bytes back with a reader written here rather than
// trusting the writer's own view of them. That duplication is the point: a
// writer checked only by "the file is non-empty" passes with the chunk sizes
// off by two, and the failure then shows up as a player refusing the file
// long after the commit that caused it.
//
// The MP4 case is gated on `ffmpeg_available()` and skips with a printed note
// otherwise, because a machine without ffmpeg is a supported configuration —
// the whole point of loading it at runtime is that its absence is not a
// build error and must not be a test failure either.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "harness.h"
#include "vidfab/audio/wav.h"
#include "vidfab/video/mux.h"
#include "vidfab/video/y4m.h"

namespace {

std::filesystem::path temp_path(const char* name) {
  return std::filesystem::temp_directory_path() / name;
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
}

uint16_t rd_u16(const std::vector<uint8_t>& b, size_t off) {
  return static_cast<uint16_t>(b[off] | (b[off + 1] << 8));
}

uint32_t rd_u32(const std::vector<uint8_t>& b, size_t off) {
  return static_cast<uint32_t>(b[off]) | (static_cast<uint32_t>(b[off + 1]) << 8) |
         (static_cast<uint32_t>(b[off + 2]) << 16) | (static_cast<uint32_t>(b[off + 3]) << 24);
}

bool tag_at(const std::vector<uint8_t>& b, size_t off, const char* tag) {
  return b.size() >= off + 4 && std::memcmp(b.data() + off, tag, 4) == 0;
}

// A deliberately literal RIFF walker: it follows the `8 + size, rounded up to
// even` rule rather than assuming the chunk order the writer happens to use,
// so a missing pad byte or a wrong size shows up as a failed lookup instead
// of being absorbed.
struct RiffChunk {
  std::string id;
  size_t payload_off = 0;
  uint32_t payload_size = 0;
};

std::vector<RiffChunk> walk_riff(const std::vector<uint8_t>& b) {
  std::vector<RiffChunk> chunks;
  if (b.size() < 12) return chunks;
  size_t off = 12;  // past "RIFF" + size + "WAVE"
  while (off + 8 <= b.size()) {
    RiffChunk c;
    c.id.assign(reinterpret_cast<const char*>(b.data() + off), 4);
    c.payload_size = rd_u32(b, off + 4);
    c.payload_off = off + 8;
    if (c.payload_off + c.payload_size > b.size()) break;
    chunks.push_back(c);
    off = c.payload_off + c.payload_size + (c.payload_size % 2);
  }
  return chunks;
}

const RiffChunk* find_chunk(const std::vector<RiffChunk>& chunks, const char* id) {
  for (const RiffChunk& c : chunks) {
    if (c.id == id) return &c;
  }
  return nullptr;
}

// --- shared fixtures --------------------------------------------------------

// A moving diagonal gradient plus a colour ramp. Deterministic, has content in
// all three channels, and moves between frames so a muxer that repeats or
// drops a frame produces a visibly different file.
std::vector<float> make_clip(int frames, int height, int width) {
  const size_t n = static_cast<size_t>(frames) * height * width;
  std::vector<float> v(3 * n);
  for (int f = 0; f < frames; ++f) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const size_t i = (static_cast<size_t>(f) * height + y) * width + x;
        const float fx = static_cast<float>(x) / static_cast<float>(width);
        const float fy = static_cast<float>(y) / static_cast<float>(height);
        const float t = static_cast<float>(f) / static_cast<float>(frames);
        v[i] = std::fmod(fx + t, 1.0f);
        v[n + i] = fy;
        v[2 * n + i] = 0.5f * (1.0f + std::sin(6.2831853f * (fx + fy + t)));
      }
    }
  }
  return v;
}

std::vector<float> make_tone(int channels, int sample_rate, float seconds, float hz) {
  const int frames = static_cast<int>(sample_rate * seconds);
  std::vector<float> v(static_cast<size_t>(frames) * channels);
  for (int i = 0; i < frames; ++i) {
    const float s = 0.4f * std::sin(6.2831853f * hz * static_cast<float>(i) /
                                    static_cast<float>(sample_rate));
    for (int c = 0; c < channels; ++c) {
      // Second channel inverted so a swapped or duplicated channel is audible
      // and, more usefully, detectable.
      v[static_cast<size_t>(i) * channels + c] = (c == 0) ? s : -s;
    }
  }
  return v;
}

}  // namespace

// --- wav --------------------------------------------------------------------

VIDFAB_TEST(wav_pcm16_header_and_samples) {
  using namespace vidfab::audio;

  const int channels = 2;
  const int rate = 32000;
  const int frames = 257;  // odd, so a frames/2 bug in the header shows up
  std::vector<float> pcm(static_cast<size_t>(frames) * channels);
  for (int i = 0; i < frames; ++i) {
    pcm[static_cast<size_t>(i) * 2 + 0] = std::sin(0.01f * static_cast<float>(i));
    pcm[static_cast<size_t>(i) * 2 + 1] = -0.5f + static_cast<float>(i) / 1024.0f;
  }

  const std::filesystem::path path = temp_path("vidfab_pcm16.wav");
  write_wav(path.string(), pcm, channels, rate, SampleFormat::kPcm16);
  const std::vector<uint8_t> b = read_file(path);

  CHECK(tag_at(b, 0, "RIFF"));
  CHECK(tag_at(b, 8, "WAVE"));
  // The RIFF size counts everything after the size field itself.
  CHECK(rd_u32(b, 4) == b.size() - 8);

  const std::vector<RiffChunk> chunks = walk_riff(b);
  const RiffChunk* fmt = find_chunk(chunks, "fmt ");
  const RiffChunk* data = find_chunk(chunks, "data");
  CHECK(fmt != nullptr);
  CHECK(data != nullptr);
  // PCM carries no cbSize, so the chunk is exactly 16 bytes.
  CHECK(fmt != nullptr && fmt->payload_size == 16);
  CHECK(find_chunk(chunks, "fact") == nullptr);

  if (fmt != nullptr) {
    const size_t o = fmt->payload_off;
    CHECK(rd_u16(b, o + 0) == 1);                             // WAVE_FORMAT_PCM
    CHECK(rd_u16(b, o + 2) == 2);                             // channels
    CHECK(rd_u32(b, o + 4) == 32000u);                        // sample rate
    CHECK(rd_u32(b, o + 8) == 32000u * 2u * 2u);              // byte rate
    CHECK(rd_u16(b, o + 12) == 4);                            // block align
    CHECK(rd_u16(b, o + 14) == 16);                           // bits per sample
  }

  if (data != nullptr) {
    CHECK(data->payload_size == static_cast<uint32_t>(frames) * 2u * 2u);
    // Quantisation error of a 16-bit signed sample scaled by 32767 is at most
    // half a step; anything larger means a scale or rounding mistake.
    double worst = 0.0;
    for (size_t i = 0; i < pcm.size(); ++i) {
      const int16_t s = static_cast<int16_t>(rd_u16(b, data->payload_off + i * 2));
      const double back = static_cast<double>(s) / 32767.0;
      worst = std::max(worst, std::abs(back - static_cast<double>(pcm[i])));
    }
    CHECK_MSG(worst <= 0.5 / 32767.0 + 1e-9, "pcm16 worst round-trip error %.3e", worst);
  }

  std::filesystem::remove(path);
}

VIDFAB_TEST(wav_float32_header_and_samples) {
  using namespace vidfab::audio;

  const int channels = 1;
  const int rate = 48000;
  std::vector<float> pcm(129);
  for (size_t i = 0; i < pcm.size(); ++i) {
    pcm[i] = static_cast<float>(i) / 128.0f - 0.5f;
  }

  const std::filesystem::path path = temp_path("vidfab_f32.wav");
  write_wav(path.string(), pcm, channels, rate, SampleFormat::kFloat32);
  const std::vector<uint8_t> b = read_file(path);

  CHECK(tag_at(b, 0, "RIFF"));
  CHECK(tag_at(b, 8, "WAVE"));
  CHECK(rd_u32(b, 4) == b.size() - 8);

  const std::vector<RiffChunk> chunks = walk_riff(b);
  const RiffChunk* fmt = find_chunk(chunks, "fmt ");
  const RiffChunk* fact = find_chunk(chunks, "fact");
  const RiffChunk* data = find_chunk(chunks, "data");
  CHECK(fmt != nullptr);
  CHECK(fact != nullptr);
  CHECK(data != nullptr);
  // Non-PCM must carry cbSize, so `fmt ` grows to 18 and a `fact` chunk with
  // the sample-frame count appears. Finding `data` at all after them is what
  // proves the walker's `8 + size` arithmetic agrees with the writer's.
  CHECK(fmt != nullptr && fmt->payload_size == 18);

  if (fmt != nullptr) {
    const size_t o = fmt->payload_off;
    CHECK(rd_u16(b, o + 0) == 3);                 // WAVE_FORMAT_IEEE_FLOAT
    CHECK(rd_u16(b, o + 2) == 1);                 // channels
    CHECK(rd_u32(b, o + 4) == 48000u);            // sample rate
    CHECK(rd_u32(b, o + 8) == 48000u * 4u);       // byte rate
    CHECK(rd_u16(b, o + 12) == 4);                // block align
    CHECK(rd_u16(b, o + 14) == 32);               // bits per sample
    CHECK(rd_u16(b, o + 16) == 0);                // cbSize
  }
  if (fact != nullptr) {
    CHECK(fact->payload_size == 4);
    CHECK(rd_u32(b, fact->payload_off) == 129u);
  }

  if (data != nullptr) {
    CHECK(data->payload_size == 129u * 4u);
    // Float is lossless, so the comparison is exact, bit for bit.
    int mismatches = 0;
    for (size_t i = 0; i < pcm.size(); ++i) {
      const uint32_t bits = rd_u32(b, data->payload_off + i * 4);
      float back = 0.0f;
      std::memcpy(&back, &bits, sizeof(back));
      if (std::memcmp(&back, &pcm[i], sizeof(back)) != 0) ++mismatches;
    }
    CHECK_MSG(mismatches == 0, "float32 samples differ in %d of %zu slots", mismatches,
              pcm.size());
  }

  std::filesystem::remove(path);
}

VIDFAB_TEST(wav_clamps_pcm16_and_passes_float) {
  using namespace vidfab::audio;

  const std::vector<float> hot = {2.0f, -2.0f, 1.0f, -1.0f, 0.0f, 1e9f};

  const std::filesystem::path p16 = temp_path("vidfab_clip16.wav");
  write_wav(p16.string(), hot, 1, 8000, SampleFormat::kPcm16);
  const std::vector<uint8_t> b16 = read_file(p16);
  const RiffChunk* d16 = find_chunk(walk_riff(b16), "data");
  CHECK(d16 != nullptr);
  if (d16 != nullptr) {
    const size_t o = d16->payload_off;
    CHECK(static_cast<int16_t>(rd_u16(b16, o + 0)) == 32767);   // +2.0 clamps
    CHECK(static_cast<int16_t>(rd_u16(b16, o + 2)) == -32767);  // -2.0 clamps
    CHECK(static_cast<int16_t>(rd_u16(b16, o + 4)) == 32767);   // +1.0 is full scale
    CHECK(static_cast<int16_t>(rd_u16(b16, o + 6)) == -32767);  // and symmetric
    CHECK(static_cast<int16_t>(rd_u16(b16, o + 8)) == 0);
    CHECK(static_cast<int16_t>(rd_u16(b16, o + 10)) == 32767);
  }
  std::filesystem::remove(p16);

  const std::filesystem::path p32 = temp_path("vidfab_clip32.wav");
  write_wav(p32.string(), hot, 1, 8000, SampleFormat::kFloat32);
  const std::vector<uint8_t> b32 = read_file(p32);
  const RiffChunk* d32 = find_chunk(walk_riff(b32), "data");
  CHECK(d32 != nullptr);
  if (d32 != nullptr) {
    // Out-of-range values survive untouched: clipping is a bug worth hearing.
    for (size_t i = 0; i < hot.size(); ++i) {
      const uint32_t bits = rd_u32(b32, d32->payload_off + i * 4);
      float back = 0.0f;
      std::memcpy(&back, &bits, sizeof(back));
      CHECK(back == hot[i]);
    }
  }
  std::filesystem::remove(p32);
}

VIDFAB_TEST(wav_rejects_malformed_requests) {
  using namespace vidfab::audio;
  const std::filesystem::path path = temp_path("vidfab_bad.wav");

  CHECK(vidfab::test::throws([] {
    write_wav((std::filesystem::temp_directory_path() / "vidfab_bad.wav").string(),
              std::vector<float>(4), 0, 8000);
  }));
  CHECK(vidfab::test::throws([] {
    write_wav((std::filesystem::temp_directory_path() / "vidfab_bad.wav").string(),
              std::vector<float>(4), 2, 0);
  }));
  // 5 samples cannot be an integral number of stereo frames.
  CHECK(vidfab::test::throws([] {
    write_wav((std::filesystem::temp_directory_path() / "vidfab_bad.wav").string(),
              std::vector<float>(5), 2, 8000);
  }));
  std::filesystem::remove(path);
}
