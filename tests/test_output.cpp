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
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.h"
#include "vidfab/audio/wav.h"
#include "vidfab/video/mux.h"
#include "vidfab/video/y4m.h"
#include "vidfab/video/y4m_compare.h"

namespace {

struct CountingConverter final : vidfab::video::FrameConverter {
  int calls = 0;
  bool saw_padded_stride = false;
  void convert(const float* r, const float* g, const float* b, int height, int width,
               uint8_t* y, int ys, uint8_t* u, int us, uint8_t* v, int vs) override {
    ++calls;
    saw_padded_stride = saw_padded_stride || ys > width || us > width / 2 || vs > width / 2;
    vidfab::video::rgb_frame_to_yuv420(r, g, b, height, width, y, ys, u, us, v, vs);
  }
};

struct ThrowingConverter final : vidfab::video::FrameConverter {
  int calls = 0;
  int throw_after = 1;
  void convert(const float* r, const float* g, const float* b, int height, int width,
               uint8_t* y, int ys, uint8_t* u, int us, uint8_t* v, int vs) override {
    if (++calls > throw_after) throw std::runtime_error("converter sentinel failure");
    vidfab::video::rgb_frame_to_yuv420(r, g, b, height, width, y, ys, u, us, v, vs);
  }
};

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
vidfab::PixelBuffer make_clip(int frames, int height, int width) {
  const size_t n = static_cast<size_t>(frames) * height * width;
  vidfab::PixelBuffer v(3 * n);
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

// --- colour conversion ------------------------------------------------------

// The .y4m fallback and the .mp4 are two views of the same frame, so their
// colour transforms have to agree exactly — not approximately, since a
// half-step difference in the limited-range scaling is the signature of a
// full-range mix-up and would look like a gamma shift when switching between
// the two files. Comparing against the bytes actually in a .y4m, rather than
// against a formula repeated here, is what keeps the two implementations
// pinned together as either changes.
// The RGB -> YUV 4:2:0 conversion as it was: one thread, all luma then all
// chroma. Copied verbatim from the version of src/video/mux.cpp that shipped it
// (the copy in y4m.cpp was byte-identical to it by hand, which is exactly the
// arrangement the shared implementation replaces).
void rgb_frame_to_yuv420_serial(const float* r, const float* g, const float* b, int height,
                                int width, uint8_t* y_plane, int y_stride, uint8_t* u_plane,
                                int u_stride, uint8_t* v_plane, int v_stride) {
  auto clamp_u8 = [](float v) -> uint8_t {
    return static_cast<uint8_t>(std::lround(std::min(255.0f, std::max(0.0f, v))));
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t i = static_cast<size_t>(y) * width + x;
      const float luma = 0.2126f * r[i] + 0.7152f * g[i] + 0.0722f * b[i];
      y_plane[static_cast<size_t>(y) * y_stride + x] = clamp_u8(16.0f + 219.0f * luma);
    }
  }
  const int chroma_w = width / 2;
  const int chroma_h = height / 2;
  for (int cy = 0; cy < chroma_h; ++cy) {
    for (int cx = 0; cx < chroma_w; ++cx) {
      float rs = 0.0f;
      float gs = 0.0f;
      float bs = 0.0f;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const size_t i = static_cast<size_t>(cy * 2 + dy) * width + (cx * 2 + dx);
          rs += r[i];
          gs += g[i];
          bs += b[i];
        }
      }
      rs *= 0.25f;
      gs *= 0.25f;
      bs *= 0.25f;
      const float luma = 0.2126f * rs + 0.7152f * gs + 0.0722f * bs;
      const float u = (bs - luma) / 1.8556f;
      const float v = (rs - luma) / 1.5748f;
      u_plane[static_cast<size_t>(cy) * u_stride + cx] = clamp_u8(128.0f + 224.0f * u);
      v_plane[static_cast<size_t>(cy) * v_stride + cx] = clamp_u8(128.0f + 224.0f * v);
    }
  }
}

VIDFAB_TEST(rgb_to_yuv_threading_is_bit_identical) {
  using namespace vidfab::video;

  // Above the parallel floor (256k pixels) so the threaded path really runs,
  // and below it so the serial path is exercised too. A row count that is not
  // a multiple of the worker count is deliberate: the last worker's short
  // range is where an off-by-one in the split would land.
  struct Case {
    int height, width;
  };
  const Case cases[] = {
      {482, 640}, {8, 16}, {2, 2}, {768, 400},
      // Odd extents. 4:2:0 is not defined for them and both container writers
      // reject them, but this is a public function now and a chroma-row split
      // silently skips the final luma row at an odd height — which is a buffer
      // the caller sized and we left uninitialised. The frozen serial copy
      // wrote it, so these cases are what holds the split to that.
      {9, 16}, {7, 16}, {1, 16}, {483, 640}, {8, 15}, {1, 1},
  };

  for (const Case& c : cases) {
    const vidfab::PixelBuffer clip = make_clip(1, c.height, c.width);
    const size_t plane = static_cast<size_t>(c.height) * c.width;
    // Over-wide strides again: the split must not confuse stride with extent.
    const int y_stride = c.width + 7;
    const int c_stride = c.width / 2 + 3;
    const size_t y_bytes = static_cast<size_t>(y_stride) * c.height;
    const size_t c_bytes = static_cast<size_t>(c_stride) * (c.height / 2);

    std::vector<uint8_t> y_a(y_bytes, 0xAB), u_a(c_bytes, 0xAB), v_a(c_bytes, 0xAB);
    std::vector<uint8_t> y_b(y_bytes, 0xAB), u_b(c_bytes, 0xAB), v_b(c_bytes, 0xAB);

    rgb_frame_to_yuv420_serial(clip.data(), clip.data() + plane, clip.data() + 2 * plane, c.height,
                               c.width, y_a.data(), y_stride, u_a.data(), c_stride, v_a.data(),
                               c_stride);
    rgb_frame_to_yuv420(clip.data(), clip.data() + plane, clip.data() + 2 * plane, c.height,
                        c.width, y_b.data(), y_stride, u_b.data(), c_stride, v_b.data(), c_stride);

    // Byte equality, not a tolerance: threading reorders independent stores and
    // must not change one bit of any of them. The padding bytes are compared
    // too, so a worker writing past its rows is caught.
    CHECK_MSG(y_a == y_b, "%dx%d: threaded luma differs from serial", c.width, c.height);
    CHECK_MSG(u_a == u_b, "%dx%d: threaded Cb differs from serial", c.width, c.height);
    CHECK_MSG(v_a == v_b, "%dx%d: threaded Cr differs from serial", c.width, c.height);
  }
}

VIDFAB_TEST(y4m_frame_split_writes_frames_in_order) {
  using namespace vidfab::video;

  // write_y4m converts frames in parallel batches and writes each batch in
  // order once it has joined. The risk that buys is a file whose frames are
  // transposed, duplicated or torn — none of which a per-frame conversion test
  // can see, and none of which changes the file's length.
  //
  // Big enough per frame to clear the parallel floor (256k pixels) so the split
  // really runs, and a frame count that is deliberately not a multiple of the
  // worker count, so the last batch is short.
  const int frames = 19;
  const int height = 512;
  const int width = 520;
  const vidfab::PixelBuffer clip = make_clip(frames, height, width);

  const std::filesystem::path path = temp_path("vidfab_frame_split.y4m");
  write_y4m(path.string(), clip, frames, height, width, FrameRate{24, 1});
  const std::vector<uint8_t> file = read_file(path);
  std::filesystem::remove(path);

  size_t off = 0;
  while (off < file.size() && file[off] != '\n') ++off;
  ++off;

  const size_t frame_pixels = static_cast<size_t>(height) * width;
  const size_t chroma_pixels = frame_pixels / 4;
  const size_t plane = static_cast<size_t>(frames) * frame_pixels;

  std::vector<uint8_t> y(frame_pixels), u(chroma_pixels), v(chroma_pixels);
  size_t mismatched_frames = 0;
  for (int f = 0; f < frames; ++f) {
    CHECK(tag_at(file, off, "FRAM"));
    while (off < file.size() && file[off] != '\n') ++off;
    ++off;

    // Frame f of the file must be frame f of the input, converted. A batching
    // bug that swapped two frames still produces a well-formed file of the
    // right size, and only this comparison catches it.
    const size_t base = static_cast<size_t>(f) * frame_pixels;
    rgb_frame_to_yuv420(clip.data() + base, clip.data() + plane + base,
                        clip.data() + 2 * plane + base, height, width, y.data(), width, u.data(),
                        width / 2, v.data(), width / 2);

    const bool luma_ok = std::memcmp(&file[off], y.data(), frame_pixels) == 0;
    const bool cb_ok = std::memcmp(&file[off + frame_pixels], u.data(), chroma_pixels) == 0;
    const bool cr_ok =
        std::memcmp(&file[off + frame_pixels + chroma_pixels], v.data(), chroma_pixels) == 0;
    if (!luma_ok || !cb_ok || !cr_ok) ++mismatched_frames;
    off += frame_pixels + 2 * chroma_pixels;
  }
  CHECK_MSG(mismatched_frames == 0, "%zu of %d frames in the .y4m are not their own conversion",
            mismatched_frames, frames);
  CHECK(off == file.size());
}

VIDFAB_TEST(rgb_to_yuv_matches_y4m_bytes) {
  using namespace vidfab::video;

  const int frames = 3;
  const int height = 8;
  const int width = 16;
  const vidfab::PixelBuffer clip = make_clip(frames, height, width);

  const std::filesystem::path path = temp_path("vidfab_convert.y4m");
  write_y4m(path.string(), clip, frames, height, width, FrameRate{24, 1});
  const std::vector<uint8_t> file = read_file(path);

  // Skip the header line, then walk FRAME markers.
  size_t off = 0;
  while (off < file.size() && file[off] != '\n') ++off;
  ++off;

  const size_t frame_pixels = static_cast<size_t>(height) * width;
  const size_t chroma_pixels = frame_pixels / 4;
  const size_t plane = static_cast<size_t>(frames) * frame_pixels;

  // The colour transform half of this is now true by construction — write_y4m
  // and the muxer call one function — so what it still earns its keep for is
  // the *container*: that the .y4m header, FRAME markers and plane order put
  // those bytes where a player expects them, at strides that are not the
  // extents. rgb_to_yuv_threading_is_bit_identical is what pins the arithmetic.
  //
  // Deliberately over-wide strides: ffmpeg pads every AVFrame row for
  // alignment, so a converter that assumes stride == width writes a sheared
  // image into a real encoder while still passing a naive test.
  const int y_stride = width + 13;
  const int c_stride = width / 2 + 5;
  std::vector<uint8_t> y(static_cast<size_t>(y_stride) * height, 0xAB);
  std::vector<uint8_t> u(static_cast<size_t>(c_stride) * (height / 2), 0xAB);
  std::vector<uint8_t> v(static_cast<size_t>(c_stride) * (height / 2), 0xAB);

  int y_diff = 0;
  int u_diff = 0;
  int v_diff = 0;
  int pad_touched = 0;
  for (int f = 0; f < frames; ++f) {
    CHECK(tag_at(file, off, "FRAM"));
    while (off < file.size() && file[off] != '\n') ++off;
    ++off;

    const size_t base = static_cast<size_t>(f) * frame_pixels;
    rgb_frame_to_yuv420(clip.data() + base, clip.data() + plane + base,
                        clip.data() + 2 * plane + base, height, width, y.data(), y_stride, u.data(),
                        c_stride, v.data(), c_stride);

    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        if (y[static_cast<size_t>(row) * y_stride + col] !=
            file[off + static_cast<size_t>(row) * width + col]) {
          ++y_diff;
        }
      }
      for (int col = width; col < y_stride; ++col) {
        if (y[static_cast<size_t>(row) * y_stride + col] != 0xAB) ++pad_touched;
      }
    }
    const size_t u_off = off + frame_pixels;
    const size_t v_off = u_off + chroma_pixels;
    for (int row = 0; row < height / 2; ++row) {
      for (int col = 0; col < width / 2; ++col) {
        const size_t src = static_cast<size_t>(row) * (width / 2) + col;
        if (u[static_cast<size_t>(row) * c_stride + col] != file[u_off + src]) ++u_diff;
        if (v[static_cast<size_t>(row) * c_stride + col] != file[v_off + src]) ++v_diff;
      }
    }
    off = v_off + chroma_pixels;
  }

  CHECK_MSG(y_diff == 0, "luma differs from write_y4m in %d of %zu samples", y_diff,
            frame_pixels * frames);
  CHECK_MSG(u_diff == 0, "Cb differs from write_y4m in %d samples", u_diff);
  CHECK_MSG(v_diff == 0, "Cr differs from write_y4m in %d samples", v_diff);
  CHECK_MSG(pad_touched == 0, "%d stride pad bytes were overwritten", pad_touched);

  std::filesystem::remove(path);
}

VIDFAB_TEST(y4m_uses_frame_converter_hook) {
  using namespace vidfab::video;
  const int frames = 3;
  const int width = 10;
  const int height = 6;
  const vidfab::PixelBuffer clip = make_clip(frames, height, width);
  const auto cpu_path = temp_path("vidfab_y4m_cpu.y4m");
  const auto hook_path = temp_path("vidfab_y4m_hook.y4m");
  write_y4m(cpu_path.string(), clip, frames, height, width);
  CountingConverter converter;
  write_y4m(hook_path.string(), clip, frames, height, width, {}, &converter);
  CHECK(converter.calls == frames);
  CHECK(read_file(cpu_path) == read_file(hook_path));
  std::filesystem::remove(cpu_path);
  std::filesystem::remove(hook_path);
}

VIDFAB_TEST(y4m_removes_partial_file_when_converter_throws) {
  using namespace vidfab::video;
  const auto path = temp_path("vidfab_y4m_converter_failure.y4m");
  std::filesystem::remove(path);
  const vidfab::PixelBuffer clip = make_clip(3, 6, 10);
  ThrowingConverter converter;
  bool preserved = false;
  try {
    write_y4m(path.string(), clip, 3, 6, 10, {}, &converter);
  } catch (const std::runtime_error& error) {
    preserved = std::string(error.what()) == "converter sentinel failure";
  }
  CHECK(preserved);
  CHECK(converter.calls == 2);
  CHECK(!std::filesystem::exists(path));
}

VIDFAB_TEST(y4m_exact_comparison_reports_first_byte) {
  using namespace vidfab::video;
  const auto expected_path = temp_path("vidfab_compare_expected.y4m");
  const auto actual_path = temp_path("vidfab_compare_actual.y4m");
  const vidfab::PixelBuffer clip = make_clip(2, 6, 10);
  write_y4m(expected_path.string(), clip, 2, 6, 10);
  write_y4m(actual_path.string(), clip, 2, 6, 10);
  ExactY4mComparison comparison =
      compare_y4m_exact(expected_path.string(), actual_path.string());
  CHECK(comparison.equal());
  CHECK(comparison.expected_size == comparison.actual_size);
  CHECK(comparison.expected_header == comparison.actual_header);

  bool same_file_rejected = false;
  try {
    (void)compare_y4m_exact(expected_path.string(), expected_path.string());
  } catch (const std::invalid_argument& error) {
    same_file_rejected = std::string(error.what()).find("same file") != std::string::npos;
  }
  CHECK(same_file_rejected);

  {
    std::fstream actual(actual_path, std::ios::binary | std::ios::in | std::ios::out);
    actual.seekg(-1, std::ios::end);
    char byte = 0;
    actual.get(byte);
    byte ^= 0x01;
    actual.seekp(-1, std::ios::end);
    actual.put(byte);
  }
  comparison = compare_y4m_exact(expected_path.string(), actual_path.string());
  CHECK(!comparison.equal());
  CHECK(comparison.first_difference + 1 == comparison.expected_size);
  CHECK(comparison.expected_byte >= 0);
  CHECK(comparison.actual_byte >= 0);

  write_y4m(actual_path.string(), clip, 2, 6, 10);
  std::filesystem::resize_file(actual_path, std::filesystem::file_size(actual_path) - 1);
  comparison = compare_y4m_exact(expected_path.string(), actual_path.string());
  CHECK(!comparison.equal());
  CHECK(comparison.first_difference == comparison.actual_size);
  CHECK(comparison.expected_byte >= 0);
  CHECK(comparison.actual_byte == -1);

  {
    std::ofstream malformed(actual_path, std::ios::binary | std::ios::trunc);
    malformed << "not a y4m\n";
  }
  bool header_rejected = false;
  try {
    (void)compare_y4m_exact(expected_path.string(), actual_path.string());
  } catch (const std::runtime_error& error) {
    header_rejected = std::string(error.what()).find("header") != std::string::npos;
  }
  CHECK(header_rejected);

  std::filesystem::remove(expected_path);
  std::filesystem::remove(actual_path);
}

// --- ffmpeg ----------------------------------------------------------------

VIDFAB_TEST(ffmpeg_probe_is_coherent) {
  using namespace vidfab::video;

  std::string first;
  std::string second;
  const bool a = ffmpeg_available(&first);
  const bool b = ffmpeg_available(&second);
  // The probe is cached, so repeated calls must agree — and must be callable
  // before anything else has run.
  CHECK(a == b);
  CHECK(first == second);
  CHECK(ffmpeg_available(nullptr) == a);
  // A failure that does not say why is useless to whoever has to install it.
  CHECK(!first.empty());

  if (a) {
    CHECK(!ffmpeg_version().empty());
    std::printf("  ffmpeg: %s\n", ffmpeg_version().c_str());
    // The library search now tries the pinned majors before sweeping downwards
    // from 70, because only the pinned ones survive the version gate. That
    // shortcut is correct exactly as long as a *usable* ffmpeg is always one of
    // them, so pin it: if these majors ever change, the preference in
    // open_library has to change with them or the sweep silently comes back.
    const std::string version = ffmpeg_version();
    CHECK_MSG(version.find("libavcodec 62.") != std::string::npos,
              "usable ffmpeg is not libavcodec 62: %s", version.c_str());
    CHECK_MSG(version.find("libavformat 62.") != std::string::npos,
              "usable ffmpeg is not libavformat 62: %s", version.c_str());
    CHECK_MSG(version.find("libavutil 60.") != std::string::npos,
              "usable ffmpeg is not libavutil 60: %s", version.c_str());
  } else {
    // No assertion on the version string here: it is empty when nothing
    // loaded and populated when something loaded but was the wrong major, and
    // both are legitimate outcomes of an unavailable ffmpeg.
    std::printf("  ffmpeg unavailable: %s\n", first.c_str());
  }

  // Every status has a message, including whatever the probe returned.
  const MuxStatus all[] = {MuxStatus::kOk, MuxStatus::kLibraryNotFound, MuxStatus::kSymbolMissing,
                           MuxStatus::kEncoderMissing, MuxStatus::kWriteFailed};
  for (MuxStatus st : all) {
    CHECK(mux_status_message(st) != nullptr && mux_status_message(st)[0] != '\0');
  }
}

VIDFAB_TEST(mp4_video_and_audio_end_to_end) {
  using namespace vidfab::video;

  std::string detail;
  if (!ffmpeg_available(&detail)) {
    std::printf("  skipped: no usable ffmpeg (%s)\n", detail.c_str());
    return;
  }

  const int frames = 24;
  const int width = 130;
  const int height = 128;
  const int rate = 32000;
  const vidfab::PixelBuffer clip = make_clip(frames, height, width);
  const std::vector<float> tone = make_tone(2, rate, 1.0f, 440.0f);

  const std::filesystem::path path = temp_path("vidfab_muxed.mp4");
  std::filesystem::remove(path);

  MuxRequest req;
  req.path = path.string();
  req.video = &clip;
  req.frames = frames;
  req.height = height;
  req.width = width;
  req.fps = FrameRate{24, 1};
  req.audio = &tone;
  req.audio_channels = 2;
  req.audio_sample_rate = rate;
  req.video_bitrate = 2'000'000;
  req.audio_bitrate = 128'000;
  CountingConverter converter;
  req.frame_converter = &converter;

  const MuxStatus status = write_mp4(req);
  CHECK_MSG(status == MuxStatus::kOk, "write_mp4 returned %s", mux_status_message(status));
  if (status != MuxStatus::kOk) return;
  CHECK(converter.calls == frames);
  CHECK(converter.saw_padded_stride);

  const std::vector<uint8_t> b = read_file(path);
  // An MP4 opens with an `ftyp` box: a big-endian size, the tag, then a brand.
  // Checking the tag rather than the extension is what catches a truncated or
  // never-finalised file, which is exactly what a missing av_write_trailer
  // produces.
  CHECK(b.size() > 4096);
  CHECK(tag_at(b, 4, "ftyp"));
  if (b.size() >= 8) {
    const uint32_t box = (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
                         (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
    CHECK(box >= 8 && box <= b.size());
  }
  // A finalised MP4 carries its index; without a trailer there is no `moov`.
  bool has_moov = false;
  for (size_t i = 0; i + 4 <= b.size(); ++i) {
    if (tag_at(b, i, "moov")) {
      has_moov = true;
      break;
    }
  }
  CHECK(has_moov);
  std::printf("  wrote %s (%zu bytes)\n", path.string().c_str(), b.size());

  std::filesystem::remove(path);
}

VIDFAB_TEST(mp4_video_only_and_bad_requests) {
  using namespace vidfab::video;

  if (!ffmpeg_available(nullptr)) {
    std::printf("  skipped: no usable ffmpeg\n");
    return;
  }

  const int frames = 8;
  const int size = 64;
  const vidfab::PixelBuffer clip = make_clip(frames, size, size);

  const std::filesystem::path path = temp_path("vidfab_silent.mp4");
  std::filesystem::remove(path);

  MuxRequest req;
  req.path = path.string();
  req.video = &clip;
  req.frames = frames;
  req.height = size;
  req.width = size;
  req.fps = FrameRate{24, 1};
  req.video_bitrate = 1'000'000;
  // No audio at all: a video-only MP4 is a supported request, not an error.
  CHECK(write_mp4(req) == MuxStatus::kOk);
  const std::vector<uint8_t> b = read_file(path);
  CHECK(b.size() > 1024);
  CHECK(tag_at(b, 4, "ftyp"));
  std::filesystem::remove(path);

  // Malformed requests are rejected without writing anything, and without
  // reaching ffmpeg: odd dimensions cannot be 4:2:0 subsampled, and a sample
  // count that disagrees with the shape means the caller has a layout bug.
  MuxRequest odd = req;
  odd.height = size + 1;
  CHECK(write_mp4(odd) == MuxStatus::kWriteFailed);

  MuxRequest short_data = req;
  short_data.frames = frames + 1;
  CHECK(write_mp4(short_data) == MuxStatus::kWriteFailed);

  MuxRequest no_video = req;
  no_video.video = nullptr;
  CHECK(write_mp4(no_video) == MuxStatus::kWriteFailed);

  // An audio buffer that is not a whole number of frames is a caller layout
  // bug; dropping the track and writing a silent file would hide it.
  const std::vector<float> ragged(1001);
  MuxRequest bad_audio = req;
  bad_audio.audio = &ragged;
  bad_audio.audio_channels = 2;
  CHECK(write_mp4(bad_audio) == MuxStatus::kWriteFailed);

  const std::filesystem::path failed_path = temp_path("vidfab_converter_failure.mp4");
  std::filesystem::remove(failed_path);
  ThrowingConverter converter;
  converter.throw_after = 0;
  MuxRequest failed = req;
  failed.path = failed_path.string();
  failed.frame_converter = &converter;
  bool preserved = false;
  try {
    (void)write_mp4(failed);
  } catch (const std::runtime_error& error) {
    preserved = std::string(error.what()) == "converter sentinel failure";
  }
  CHECK(preserved);
  CHECK(!std::filesystem::exists(failed_path));
}
