#include "native_media.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace slopfab::cli {
namespace {
void be32(std::vector<uint8_t>& b, uint32_t n) {
  for (int i = 24; i >= 0; i -= 8)
    b.push_back(uint8_t(n >> i));
}

void chunk(std::ofstream& out, const char* type, const std::vector<uint8_t>& data) {
  std::vector<uint8_t> b;
  be32(b, uint32_t(data.size()));
  b.insert(b.end(), type, type + 4);
  b.insert(b.end(), data.begin(), data.end());
  uint32_t crc = ~uint32_t(0);
  for (size_t i = 4; i < b.size(); ++i) {
    crc ^= b[i];
    for (int k = 0; k < 8; ++k)
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
  }
  be32(b, ~crc);
  out.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
}

uint32_t le(const uint8_t* p, int n) {
  uint32_t v = 0;
  for (int i = 0; i < n; ++i)
    v |= uint32_t(p[i]) << (8 * i);
  return v;
}

void read(std::istream& in, void* p, size_t n) {
  if (!in.read(static_cast<char*>(p), std::streamsize(n)))
    throw std::invalid_argument("truncated native media file");
}
}

static void write_png_channels(const std::string& path, const uint8_t* rgb, int width, int height,
                               int channels) {
  if (!rgb || width <= 0 || height <= 0 || uint64_t(width) * height > (1ull << 28))
    throw std::invalid_argument("invalid PNG dimensions");
  std::ofstream out(std::filesystem::u8path(path), std::ios::binary);
  const uint8_t signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
  out.write(reinterpret_cast<const char*>(signature), 8);
  std::vector<uint8_t> header;
  be32(header, width);
  be32(header, height);
  header.insert(header.end(), {8, uint8_t(channels == 4 ? 6 : 2), 0, 0, 0});
  chunk(out, "IHDR", header);
  // zlib stream with stored DEFLATE blocks. No external codec or runtime.
  // Each row is emitted separately to bound scratch memory.
  uint32_t a = 1, b = 0;
  bool first = true;
  for (int y = 0; y < height; ++y) {
    std::vector<uint8_t> row(size_t(width) * channels + 1);
    std::memcpy(row.data() + 1, rgb + size_t(y) * width * channels, size_t(width) * channels);
    for (auto byte : row) {
      a = (a + byte) % 65521;
      b = (b + a) % 65521;
    }
    for (size_t pos = 0; pos < row.size();) {
      const size_t n = std::min(size_t(65535), row.size() - pos);
      const bool last = y + 1 == height && pos + n == row.size();
      std::vector<uint8_t> block;
      if (first) {
        block.insert(block.end(), {0x78, 0x01});
        first = false;
      }
      block.insert(block.end(),
                   {uint8_t(last), uint8_t(n), uint8_t(n >> 8), uint8_t(~n), uint8_t((~n) >> 8)});
      block.insert(block.end(), row.begin() + pos, row.begin() + pos + n);
      if (last)
        be32(block, (b << 16) | a);
      chunk(out, "IDAT", block);
      pos += n;
    }
  }
  chunk(out, "IEND", {});
  out.close();
  if (!out)
    throw std::runtime_error("cannot write PNG: " + path);
}

void write_png(const std::string& path, const uint8_t* rgb, int width, int height) {
  write_png_channels(path, rgb, width, height, 3);
}

void write_png_rgba(const std::string& path, const uint8_t* rgba, int width, int height) {
  write_png_channels(path, rgba, width, height, 4);
}

NativeVideo read_y4m(const std::string& path) {
  std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
  std::string line;
  std::getline(in, line);
  if (line.size() > 4096 || line.rfind("YUV4MPEG2 ", 0) != 0)
    throw std::invalid_argument("video input must be Y4M; convert it externally first");
  NativeVideo result;
  std::istringstream fields(line.substr(10));
  std::string f, chroma = "420jpeg";
  while (fields >> f) {
    if (f[0] == 'W')
      result.width = std::stoi(f.substr(1));
    if (f[0] == 'H')
      result.height = std::stoi(f.substr(1));
    if (f[0] == 'F') {
      const auto colon = f.find(':');
      if (colon == std::string::npos)
        throw std::invalid_argument("invalid Y4M rate");
      result.fps = {std::stoi(f.substr(1, colon - 1)), std::stoi(f.substr(colon + 1))};
    }
    if (f[0] == 'C')
      chroma = f.substr(1);
    if (f[0] == 'I' && f != "Ip")
      throw std::invalid_argument("Y4M must be progressive");
    if (f == "XCOLORRANGE=FULL")
      throw std::invalid_argument("Y4M must use limited range BT.709");
  }
  const int w = result.width, h = result.height;
  if (w <= 0 || h <= 0 || w > 8192 || h > 8192 || w % 2 || h % 2 || result.fps.numerator <= 0 ||
      result.fps.denominator <= 0 ||
      (chroma != "420" && chroma != "420jpeg" && chroma != "420mpeg2"))
    throw std::invalid_argument("Y4M requires even dimensions <=8192 and 8-bit 4:2:0");
  const size_t plane = size_t(w) * h;
  std::vector<uint8_t> yuv(plane * 3 / 2);
  while (std::getline(in, line)) {
    if (line.size() > 4096 || (line != "FRAME" && line.rfind("FRAME ", 0) != 0))
      throw std::invalid_argument("invalid Y4M frame header");
    if ((result.frames.size() + 1) * plane * 3 > 2ull * 1024 * 1024 * 1024)
      throw std::invalid_argument("decoded Y4M exceeds 2 GiB; trim or resize externally");
    read(in, yuv.data(), yuv.size());
    std::vector<uint8_t> rgb(plane * 3);
    auto byte = [](double v) {
      return uint8_t(std::clamp(std::lround(v), 0l, 255l));
    };
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        size_t p = size_t(y) * w + x, c = size_t(y / 2) * (w / 2) + x / 2;
        double yy = (yuv[p] - 16) * (255.0 / 219), u = (yuv[plane + c] - 128) * (255.0 / 224),
               v = (yuv[plane + plane / 4 + c] - 128) * (255.0 / 224);
        rgb[p * 3] = byte(yy + 1.5748 * v);
        rgb[p * 3 + 1] = byte(yy - .187324 * u - .468124 * v);
        rgb[p * 3 + 2] = byte(yy + 1.8556 * u);
      }
    result.frames.push_back(std::move(rgb));
  }
  if (result.frames.empty())
    throw std::invalid_argument("Y4M contains no frames");
  return result;
}

ReferenceAudio read_wav(const std::string& path) {
  std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
  uint8_t head[12];
  read(in, head, 12);
  if (std::memcmp(head, "RIFF", 4) || std::memcmp(head + 8, "WAVE", 4))
    throw std::invalid_argument(
        "audio input must be PCM16/24/32 or float32 WAV; convert externally");
  ReferenceAudio result;
  int format = 0, bits = 0, align = 0;
  std::vector<uint8_t> pcm;
  while (in.peek() != EOF) {
    uint8_t h[8];
    read(in, h, 8);
    const uint32_t n = le(h + 4, 4);
    if (n > 256u * 1024 * 1024)
      throw std::invalid_argument("WAV chunk exceeds 256 MiB");
    if (!std::memcmp(h, "fmt ", 4)) {
      if (n < 16 || n > 4096)
        throw std::invalid_argument("invalid WAV format chunk");
      std::vector<uint8_t> data(n);
      read(in, data.data(), n);
      format = le(data.data(), 2);
      result.channels = le(data.data() + 2, 2);
      result.sample_rate = int(le(data.data() + 4, 4));
      align = le(data.data() + 12, 2);
      bits = le(data.data() + 14, 2);
    } else if (!std::memcmp(h, "data", 4)) {
      if (!pcm.empty())
        throw std::invalid_argument("multiple WAV data chunks are unsupported");
      pcm.resize(n);
      read(in, pcm.data(), n);
    } else {
      in.seekg(n, std::ios::cur);
      if (!in)
        throw std::invalid_argument("truncated WAV chunk");
    }
    if (n & 1)
      in.get();
  }
  if ((result.channels != 1 && result.channels != 2) || result.sample_rate <= 0 ||
      result.sample_rate > 384000 ||
      !((format == 1 && (bits == 16 || bits == 24 || bits == 32)) || (format == 3 && bits == 32)) ||
      align != result.channels * (bits / 8) || pcm.empty() || pcm.size() % align)
    throw std::invalid_argument("unsupported WAV; use mono/stereo PCM16/24/32 or float32");
  result.interleaved.resize(pcm.size() / (bits / 8));
  for (size_t i = 0; i < result.interleaved.size(); ++i) {
    const auto p = pcm.data() + i * (bits / 8);
    float v;
    if (format == 3) {
      uint32_t raw = le(p, 4);
      std::memcpy(&v, &raw, 4);
    } else {
      uint32_t raw = le(p, bits / 8);
      int64_t value = raw;
      if (raw & (1u << (bits - 1)))
        value -= int64_t(1) << bits;
      v = float(double(value) / double(int64_t(1) << (bits - 1)));
    }
    if (!std::isfinite(v))
      throw std::invalid_argument("non-finite WAV sample");
    result.interleaved[i] = std::clamp(v, -1.0f, 1.0f);
  }
  return result;
}
}
