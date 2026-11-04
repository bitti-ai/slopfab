#include "frames.h"
#include "native_media.h"
#include "pipe_process.h"
#include "script_host.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cctype>
#include <fstream>
#include <sstream>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace slopfab::cli {
namespace fs = std::filesystem;

namespace {
[[noreturn]] void invalid(const std::string& message) {
  throw CliError("INVALID_REQUEST", message);
}

double number(const Value& o, const char* key, double fallback, double lo, double hi,
              bool integer = false) {
  const auto* v = o.find(key);
  const double n = v ? v->as_number() : fallback;
  if (!std::isfinite(n) || n < lo || n > hi || (integer && n != std::floor(n)))
    invalid(std::string(key) + " is out of range");
  return n;
}

fs::path path(const std::string& s) {
  if (s.empty() || s.find('\0') != std::string::npos)
    invalid("missing or invalid file path");
  return fs::absolute(fs::u8path(s)).lexically_normal();
}

uint8_t byte(double v) {
  return uint8_t(std::lround(std::clamp(std::isfinite(v) ? v : 0.0, 0.0, 255.0)));
}

void dimensions(int w, int h) {
  if (w < 1 || h < 1 || w > 16384 || h > 16384 || uint64_t(w) * h > 64000000)
    invalid("frame dimensions exceed 64 million pixels or 16384 per side");
}

struct Source {
  int width = 0, height = 0;
  std::vector<double> times;
  // Fill an already downscaled RGB frame, in chronological order.
  std::function<void(size_t, int, int, uint8_t*)> read;
};

// Box downsampling averages source pixels, including for planar generation samples.
template <class Pixel>
void downscale(int sw, int sh, int dw, int dh, uint8_t* out, const Pixel& pixel,
               const FrameCheck& check) {
  for (int y = 0; y < dh; ++y) {
    check();
    const int y0 = int(int64_t(y) * sh / dh), y1 = int(int64_t(y + 1) * sh / dh);
    for (int x = 0; x < dw; ++x) {
      const int x0 = int(int64_t(x) * sw / dw), x1 = int(int64_t(x + 1) * sw / dw);
      double sum[3]{};
      for (int sy = y0; sy < y1; ++sy)
        for (int sx = x0; sx < x1; ++sx)
          for (int c = 0; c < 3; ++c)
            sum[c] += pixel(sx, sy, c);
      const double count = double(x1 - x0) * (y1 - y0);
      for (int c = 0; c < 3; ++c)
        out[(size_t(y) * dw + x) * 3 + c] = byte(sum[c] / count);
    }
  }
}

void timeline(Source& s, size_t count, double fps) {
  dimensions(s.width, s.height);
  if (!count || count > 10000000 || !std::isfinite(fps) || fps <= 0)
    invalid("invalid frame count or frame rate");
  s.times.resize(count);
  for (size_t i = 0; i < count; ++i)
    s.times[i] = double(i) / fps;
}

size_t nearest(const Source& s, double time) {
  auto it = std::lower_bound(s.times.begin(), s.times.end(), time);
  if (it == s.times.end())
    return s.times.size() - 1;
  if (it != s.times.begin() && time - *(it - 1) <= *it - time)
    --it;
  return size_t(it - s.times.begin());
}

struct Selection {
  int columns = 1, rows = 1, width = 0, height = 0;
  std::vector<size_t> indices;
};

Selection select(const Source& s, const Value& o) {
  Selection a;
  const bool grid = o.find("columns") || o.find("rows");
  if ((o.find("frame") && o.find("time")) || (grid && (o.find("frame") || o.find("time"))) ||
      (!grid && (o.find("start") || o.find("end"))))
    invalid("choose frame, time, or a grid with optional start/end");
  a.columns = int(number(o, "columns", 4, 1, 32, true));
  a.rows = int(number(o, "rows", 3, 1, 32, true));
  if (!grid)
    a.columns = a.rows = 1;
  if (a.columns * a.rows > 256)
    invalid("grid exceeds 256 cells");
  a.width = std::min(s.width, int(number(o, "width", grid ? 320 : s.width, 1, 16384, true)));
  a.height = std::max(1, int(std::lround(double(s.height) * a.width / s.width)));
  dimensions(a.width * a.columns, a.height * a.rows);
  if (grid) {
    double begin = number(o, "start", 0, 0, s.times.back());
    double end = number(o, "end", s.times.back(), begin, s.times.back());
    int cells = a.columns * a.rows;
    for (int i = 0; i < cells; ++i)
      a.indices.push_back(nearest(s, begin + (end - begin) * i / std::max(1, cells - 1)));
  } else if (o.find("time"))
    a.indices.push_back(nearest(s, number(o, "time", 0, 0, s.times.back())));
  else
    a.indices.push_back(size_t(number(o, "frame", 0, 0, double(s.times.size() - 1), true)));
  return a;
}

struct Output {
  fs::path final, temp, lock;
  bool overwrite;

  Output(const Value& o)
      : final(path(script_string(o, "output"))), overwrite(script_bool(o, "overwrite")) {
    auto extension = final.extension().u8string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
      return char(std::tolower(c));
    });
    if (extension != ".png")
      invalid("frame output must be a PNG file");
    fs::create_directories(final.parent_path());
    auto reserved = final;
    reserved += ".slopfab-reserved";
    if (!fs::create_directory(reserved))
      throw CliError("OUTPUT_EXISTS", "frame output is reserved: " + final.u8string(), 6);
    lock = reserved;
    if (!overwrite && fs::exists(final)) {
      fs::remove(lock);
      lock.clear();
      throw CliError("OUTPUT_EXISTS", "frame output exists: " + final.u8string(), 6);
    }
    temp = final;
    temp += "." + unique_id() + ".tmp";
  }

  ~Output() {
    std::error_code ec;
    if (!temp.empty())
      fs::remove(temp, ec);
    if (!lock.empty())
      fs::remove(lock, ec);
  }

  void publish() {
#ifdef _WIN32
    if (!MoveFileExW(temp.c_str(), final.c_str(),
                     MOVEFILE_WRITE_THROUGH | (overwrite ? MOVEFILE_REPLACE_EXISTING : 0)))
      throw CliError("OUTPUT_FAILED", "cannot publish frame output", 6);
#else
    if (overwrite)
      fs::rename(temp, final);
    else {
      fs::create_hard_link(temp, final);
      fs::remove(temp);
    }
#endif
  }
};

Value render(Source& s, const Value& options, const FrameCheck& check, const std::string& source,
             const std::function<void(const Selection&)>& prepare = {}) {
  auto a = select(s, options);
  Output output(options);
  if (prepare)
    prepare(a);
  const int width = a.width * a.columns, height = a.height * a.rows;
  std::vector<uint8_t> sheet(size_t(width) * height * 3);
  std::vector<uint8_t> tile(size_t(a.width) * a.height * 3);
  Array cells;
  size_t previous = size_t(-1);
  for (size_t i = 0; i < a.indices.size(); ++i) {
    check();
    size_t index = a.indices[i];
    if (index != previous)
      s.read(index, a.width, a.height, tile.data());
    previous = index;
    int x = int(i % a.columns) * a.width, y = int(i / a.columns) * a.height;
    for (int row = 0; row < a.height; ++row)
      std::copy_n(tile.data() + size_t(row) * a.width * 3, size_t(a.width) * 3,
                  sheet.data() + (size_t(y + row) * width + x) * 3);
    cells.push_back(Value(Object{{"frame", num(double(index))},
                                 {"time", num(s.times[index])},
                                 {"x", num(x)},
                                 {"y", num(y)}}));
  }
  check();
  write_png(output.temp.u8string(), sheet.data(), width, height);
  check();
  const auto bytes = fs::file_size(output.temp);
  output.publish();
  return Value(Object{{"path", str(output.final.u8string())},
                      {"media_type", str("image/png")},
                      {"bytes", num(double(bytes))},
                      {"width", num(width)},
                      {"height", num(height)},
                      {"columns", num(a.columns)},
                      {"rows", num(a.rows)},
                      {"cell_width", num(a.width)},
                      {"cell_height", num(a.height)},
                      {"source", str(source)},
                      {"source_frames", num(double(s.times.size()))},
                      {"cells", Value(cells)}});
}

std::string line(std::istream& in) {
  std::string s;
  char c;
  while (in.get(c)) {
    if (c == '\n')
      return s;
    if (s.size() >= 4096)
      invalid("Y4M header exceeds 4096 bytes");
    s += c;
  }
  invalid("truncated Y4M header");
}

Value y4m(const fs::path& input, const Value& options, const FrameCheck& check) {
  std::ifstream in(input, std::ios::binary);
  std::istringstream header(line(in));
  std::string token, chroma = "420jpeg";
  Source s;
  double fps = 0;
  header >> token;
  if (token != "YUV4MPEG2")
    invalid("invalid Y4M signature");
  while (header >> token) {
    if (token[0] == 'W')
      s.width = std::stoi(token.substr(1));
    if (token[0] == 'H')
      s.height = std::stoi(token.substr(1));
    if (token[0] == 'C')
      chroma = token.substr(1);
    if (token[0] == 'I' && token != "Ip" && token != "I?")
      invalid("interlaced Y4M is unsupported");
    if (token == "XCOLORRANGE=FULL")
      invalid("full-range Y4M is unsupported; convert with FFmpeg");
    if (token[0] == 'F') {
      auto colon = token.find(':');
      if (colon == std::string::npos)
        invalid("invalid Y4M frame rate");
      const double denominator = std::stod(token.substr(colon + 1));
      fps = std::stod(token.substr(1, colon - 1)) / denominator;
    }
  }
  dimensions(s.width, s.height);
  if (s.width % 2 || s.height % 2 ||
      (chroma != "420jpeg" && chroma != "420mpeg2" && chroma != "420"))
    invalid("native extraction requires even-sized 8-bit 420 Y4M");
  const size_t plane = size_t(s.width) * s.height, bytes = plane * 3 / 2;
  const auto length = fs::file_size(input);
  std::vector<uint64_t> offsets;
  while (in.peek() != EOF) {
    check();
    auto frame = line(in);
    if (frame != "FRAME" && frame.rfind("FRAME ", 0) != 0)
      invalid("invalid Y4M frame header");
    auto offset = uint64_t(in.tellg());
    if (offset > length || bytes > length - offset)
      invalid("truncated Y4M frame");
    offsets.push_back(offset);
    if (offsets.size() > 10000000)
      invalid("video exceeds ten million frames");
    in.seekg(std::streamoff(bytes), std::ios::cur);
  }
  timeline(s, offsets.size(), fps);
  std::vector<uint8_t> yuv(bytes);
  s.read = [&](size_t index, int w, int h, uint8_t* rgb) {
    in.clear();
    in.seekg(std::streamoff(offsets[index]));
    in.read(reinterpret_cast<char*>(yuv.data()), std::streamsize(bytes));
    if (size_t(in.gcount()) != bytes)
      invalid("cannot read Y4M frame");
    downscale(
        s.width, s.height, w, h, rgb,
        [&](int x, int y, int c) {
          size_t uv = size_t(y / 2) * (s.width / 2) + x / 2;
          double yy = (int(yuv[size_t(y) * s.width + x]) - 16) * (255.0 / 219);
          double u = (int(yuv[plane + uv]) - 128) * (255.0 / 224);
          double v = (int(yuv[plane + plane / 4 + uv]) - 128) * (255.0 / 224);
          return double(byte(yy + (c == 0   ? 1.5748 * v
                                   : c == 1 ? -.187324 * u - .468124 * v
                                            : 1.8556 * u)));
        },
        check);
  };
  return render(s, options, check, input.u8string());
}

// The watchdog cancels blocking reads when a workflow times out or is interrupted.
struct Decoder {
  PipeProcess pipe;
  std::atomic<bool> stop{false};
  std::thread watcher;
  std::exception_ptr error;

  Decoder(const std::vector<std::string>& args, const FrameCheck& check) : pipe(args, false) {
    watcher = std::thread([this, check] {
      while (!stop) {
        try {
          check();
        } catch (...) {
          error = std::current_exception();
          pipe.cancel();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    });
  }

  ~Decoder() {
    stop = true;
    pipe.cancel();
    if (watcher.joinable())
      watcher.join();
  }

  void finish() {
    try {
      pipe.finish();
    } catch (...) {
      stop = true;
      watcher.join();
      if (error)
        std::rethrow_exception(error);
      throw;
    }
    stop = true;
    watcher.join();
    if (error)
      std::rethrow_exception(error);
  }
};

Value compressed(const fs::path& input, const Value& options, const FrameCheck& check) {
  ScriptProcesses processes(check);
  auto probe = script_string(options, "ffprobe", media_tool("ffprobe", agent().executable.c_str()));
  auto launched = processes.spawn({probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height:frame=best_effort_timestamp_time", "-of",
                                   "json", input.u8string()},
                                  Value(Object{{"maxOutputBytes", num(64 * 1024 * 1024)}}));
  auto process = processes.poll(int(launched.find("id")->as_int()), true);
  if (process.find("exitCode")->as_int() != 0)
    invalid("ffprobe failed: " + process.find("stderr")->as_string());
  auto metadata = json::parse(process.find("stdout")->as_string());
  const auto* streams = metadata.find("streams");
  if (!streams || streams->as_array().empty())
    invalid("input has no video stream");
  const auto& stream = streams->as_array().front();
  Source s;
  s.width = int(stream.find("width")->as_int());
  s.height = int(stream.find("height")->as_int());
  dimensions(s.width, s.height);
  const auto* frames = metadata.find("frames");
  if (!frames || frames->as_array().empty())
    invalid("input has no video frames");
  double origin = 0;
  for (const auto& f : frames->as_array()) {
    check();
    const auto* t = f.find("best_effort_timestamp_time");
    if (!t)
      invalid("video frame has no timestamp");
    double time = std::stod(t->as_string());
    if (s.times.empty())
      origin = time;
    time -= origin;
    if (!std::isfinite(time) || (!s.times.empty() && time < s.times.back()))
      invalid("video timestamps are not monotonic");
    s.times.push_back(time);
  }
  std::unique_ptr<Decoder> decoder;
  s.read = [&](size_t, int w, int h, uint8_t* out) {
    const size_t bytes = size_t(w) * h * 3;
    if (decoder->pipe.read(out, bytes) != bytes) {
      check();
      invalid("FFmpeg returned an incomplete frame");
    }
  };
  // Drain and verify the decoder before publishing the PNG.
  size_t reads = 0, expected = 0;
  auto read = s.read;
  s.read = [&](size_t index, int w, int h, uint8_t* out) {
    read(index, w, h, out);
    if (++reads == expected)
      decoder->finish();
  };
  return render(s, options, check, input.u8string(), [&](const Selection& a) {
    std::string filter = "select='";
    size_t previous = size_t(-1);
    for (size_t index : a.indices)
      if (index != previous) {
        if (expected++)
          filter += "+";
        filter += "eq(n," + std::to_string(index) + ")";
        previous = index;
      }
    filter += "',scale=" + std::to_string(a.width) + ":" + std::to_string(a.height) + ":flags=area";
    auto ffmpeg =
        script_string(options, "ffmpeg", media_tool("ffmpeg", agent().executable.c_str()));
    decoder = std::make_unique<Decoder>(std::vector<std::string>{ffmpeg,
                                                                 "-nostdin",
                                                                 "-v",
                                                                 "error",
                                                                 "-noautorotate",
                                                                 "-i",
                                                                 input.u8string(),
                                                                 "-map",
                                                                 "0:v:0",
                                                                 "-an",
                                                                 "-sn",
                                                                 "-dn",
                                                                 "-vf",
                                                                 filter,
                                                                 "-vsync",
                                                                 "0",
                                                                 "-frames:v",
                                                                 std::to_string(expected),
                                                                 "-pix_fmt",
                                                                 "rgb24",
                                                                 "-f",
                                                                 "rawvideo",
                                                                 "pipe:1"},
                                        check);
  });
}
}

Value extract_frames(const Value& options, const FrameCheck& check) {
  check();
  validate_frame_options(options);
  auto input = path(script_string(options, "input"));
  if (!fs::is_regular_file(input))
    throw CliError("INPUT_NOT_FOUND", "video does not exist", 3);
  auto output = path(script_string(options, "output"));
  if (input == output || (fs::exists(output) && fs::equivalent(input, output)))
    invalid("frame output must differ from input");
  std::ifstream in(input, std::ios::binary);
  char magic[9]{};
  in.read(magic, 9);
  if (std::string(magic, 9) == "YUV4MPEG2")
    return y4m(input, options, check);
  return compressed(input, options, check);
}

Value frames_from_rgb(const uint8_t* bytes, size_t size, const Value& options,
                      const FrameCheck& check) {
  validate_frame_options(options);
  Source s;
  s.width = int(number(options, "sourceWidth", 0, 1, 16384, true));
  s.height = int(number(options, "sourceHeight", 0, 1, 16384, true));
  size_t count = size_t(number(options, "count", 1, 1, 10000000, true));
  timeline(s, count, number(options, "fps", 30, .000001, 1000000));
  size_t plane = size_t(s.width) * s.height;
  if (!bytes || uint64_t(plane) * count * 3 != size)
    invalid("RGB buffer size must equal width * height * count * 3");
  s.read = [&](size_t index, int w, int h, uint8_t* out) {
    downscale(
        s.width, s.height, w, h, out,
        [&](int x, int y, int c) {
          return bytes[(index * plane + size_t(y) * s.width + x) * 3 + c];
        },
        check);
  };
  return render(s, options, check, "memory:rgb24");
}

Value frames_from_samples(const RunSamples& samples, video::FrameRate fps, const Value& options,
                          const FrameCheck& check) {
  validate_frame_options(options);
  Source s;
  s.width = samples.width;
  s.height = samples.height;
  timeline(s, size_t(samples.frames), double(fps.numerator) / fps.denominator);
  const size_t plane = size_t(s.width) * s.height, channel = plane * samples.frames;
  if (!samples.video || samples.channels != 3 || samples.video->size() != channel * 3)
    invalid("generation has no RGB video samples");
  s.read = [&](size_t index, int w, int h, uint8_t* out) {
    downscale(
        s.width, s.height, w, h, out,
        [&](int x, int y, int c) {
          return double(
              byte((*samples.video)[size_t(c) * channel + index * plane + size_t(y) * s.width + x] *
                   255.0));
        },
        check);
  };
  return render(s, options, check, "memory:generation");
}

void validate_frame_options(const Value& o) {
  if (!o.is_object())
    invalid("frame options must be an object");
  const bool grid = o.find("columns") || o.find("rows");
  if ((o.find("frame") && o.find("time")) || (grid && (o.find("frame") || o.find("time"))) ||
      (!grid && (o.find("start") || o.find("end"))))
    invalid("choose frame, time, or a grid with optional start/end");
  if (number(o, "columns", 4, 1, 32, true) * number(o, "rows", 3, 1, 32, true) > 256)
    invalid("grid exceeds 256 cells");
  number(o, "width", 320, 1, 16384, true);
  number(o, "frame", 0, 0, 9999999, true);
  number(o, "time", 0, 0, 1e12);
  auto start = number(o, "start", 0, 0, 1e12);
  number(o, "end", 1e12, start, 1e12);
  Output preflight(o); // Fail bad paths/collisions before decoding or loading a model.
}

}
