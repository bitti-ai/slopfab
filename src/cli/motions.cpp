#include "motions.h"
#include "script.h"
#include "script_host.h"
#include "pipe_process.h"
#include "native_media.h"
#include <thorvg.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <map>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace slopfab::cli {
namespace fs = std::filesystem;

namespace {
void tvg_check(tvg::Result result, const char* action) {
  if (result != tvg::Result::Success)
    throw std::runtime_error(std::string("ThorVG ") + action + " failed (" +
                             std::to_string(int(result)) + ")");
}

double number(const Value& options, const char* name, double fallback, double min, double max) {
  const auto* value = options.find(name);
  const double n = value ? value->as_number() : fallback;
  if (!std::isfinite(n) || n < min || n > max)
    throw std::runtime_error(std::string(name) + " is out of range");
  return n;
}

fs::path file_path(const std::string& text) {
  if (text.empty() || text.find('\0') != std::string::npos)
    throw std::runtime_error("invalid Motion path");
  return fs::absolute(fs::u8path(text)).lexically_normal();
}

std::string lower(std::string text) {
  for (auto& c : text)
    if (c >= 'A' && c <= 'Z')
      c += 'a' - 'A';
  return text;
}

void publish(const fs::path& temp, const fs::path& target, bool overwrite) {
#ifdef _WIN32
  if (!MoveFileExW(temp.c_str(), target.c_str(),
                   MOVEFILE_WRITE_THROUGH | (overwrite ? MOVEFILE_REPLACE_EXISTING : 0)))
    throw std::runtime_error("cannot publish Motion output (destination exists or is unwritable)");
#else
  if (overwrite)
    fs::rename(temp, target);
  else {
    if (::link(temp.c_str(), target.c_str()) != 0)
      throw std::runtime_error(
          "cannot publish Motion output (destination exists or is unwritable)");
    fs::remove(temp);
  }
#endif
}

struct Engine {
  Engine() {
    tvg_check(tvg::Initializer::init(0), "initialization");
  }

  ~Engine() {
    tvg::Initializer::term();
  }
};

struct Session {
  std::function<void()> check;
  fs::path output, pending;
  std::string extension;
  int width = 0, height = 0, frames = 0, written = 0;
  double fps = 30, duration = 0;
  bool overwrite = false, flatten = false, overlay = false;
  std::vector<uint32_t> pixels;
  std::unique_ptr<tvg::SwCanvas> canvas;
  std::unique_ptr<PipeProcess> pipe;
  std::atomic<bool> stop{false};
  std::thread watcher;

  ~Session() {
    stop = true;
    if (pipe)
      pipe->cancel();
    if (watcher.joinable())
      watcher.join();
    pipe.reset();
    canvas.reset();
    if (!pending.empty()) {
      std::error_code ec;
      fs::remove(pending, ec);
    }
  }

  void render(const std::string& svg) {
    check();
    if (written >= frames)
      throw std::runtime_error("too many Motion frames");
    if (svg.empty() || svg.size() > 16 * 1024 * 1024)
      throw std::runtime_error("Motion SVG must contain 1..16 MiB");
    tvg_check(canvas->remove(), "scene reset");
    std::unique_ptr<tvg::Picture> picture(tvg::Picture::gen());
    if (!picture)
      throw std::runtime_error("cannot allocate Motion picture");
    tvg_check(picture->load(svg.data(), uint32_t(svg.size()), "svg", nullptr, true), "SVG load");
    tvg_check(picture->size(float(width), float(height)), "SVG size");
    tvg_check(canvas->add(picture.get()), "scene add");
    picture.release(); // Canvas owns the picture from here.
    tvg_check(canvas->draw(true), "draw");
    tvg_check(canvas->sync(), "sync");
    check();
    // ThorVG ABGR8888S is straight RGBA on supported little-endian platforms.
    auto* rgba = reinterpret_cast<uint8_t*>(pixels.data());
    if (flatten)
      for (size_t i = 0; i < pixels.size(); ++i) {
        for (int c = 0; c < 3; ++c)
          rgba[i * 4 + c] = uint8_t((unsigned(rgba[i * 4 + c]) * rgba[i * 4 + 3] + 127) / 255);
        rgba[i * 4 + 3] = 255;
      }
    if (extension == ".svg") {
      std::ofstream stream(pending, std::ios::binary);
      stream.write(svg.data(), std::streamsize(svg.size()));
      stream.close();
      if (!stream)
        throw std::runtime_error("cannot write Motion SVG");
    } else if (extension == ".png")
      write_png_rgba(pending.u8string(), rgba, width, height);
    else
      pipe->write(rgba, pixels.size() * 4);
    ++written;
    check();
  }
};
}

struct Motions::Impl {
  std::function<void()> check;
  std::unique_ptr<Engine> engine;
  std::map<int, std::unique_ptr<Session>> sessions;
  std::vector<std::string> fonts;
  int next = 1;

  void initialize() {
    if (!engine)
      engine = std::make_unique<Engine>();
  }

  ~Impl() {
    sessions.clear();
    for (const auto& name : fonts)
      tvg::Text::load(name.c_str(), nullptr, 0);
  }
};

Motions::Motions(std::function<void()> check) : impl_(new Impl) {
  impl_->check = std::move(check);
}

Motions::~Motions() = default;

Value Motions::call(const std::string& operation, const Array& args) {
  impl_->check();
  impl_->initialize();
  if (operation == "font") {
    const auto name = args.at(0).as_string();
    if (name.empty() || name.find('\0') != std::string::npos)
      throw std::runtime_error("invalid font name");
    const auto path = file_path(args.at(1).as_string());
    const auto size = fs::file_size(path);
    if (!size || size > 64 * 1024 * 1024)
      throw std::runtime_error("font exceeds 64 MiB");
    std::string data(size_t(size), '\0');
    std::ifstream input(path, std::ios::binary);
    if (!input.read(data.data(), std::streamsize(size)))
      throw std::runtime_error("cannot read font");
    const auto type = lower(path.extension().u8string()) == ".otf" ? "otf" : "ttf";
    tvg_check(tvg::Text::load(name.c_str(), data.data(), uint32_t(size), type, true), "font load");
    impl_->fonts.push_back(name);
    return str(name);
  }
  if (operation == "start") {
    const auto& options = args.at(0);
    auto session = std::make_unique<Session>();
    auto& s = *session;
    s.check = impl_->check;
    const auto width = script_uint(options, "width", 1280),
               height = script_uint(options, "height", 720);
    if (!width || !height || width > 8192 || height > 8192 || width * height > 16777216)
      throw std::runtime_error("Motion dimensions must be 1..8192 and at most 16 megapixels");
    s.width = int(width);
    s.height = int(height);
    s.fps = number(options, "fps", 30, 1, 240);
    s.duration = number(options, "duration", 5, 0.001, 3600);
    s.output = file_path(script_string(options, "output"));
    s.extension = lower(s.output.extension().u8string());
    const bool still = s.extension == ".svg" || s.extension == ".png";
    if (!still && s.extension != ".mp4" && s.extension != ".mov" && s.extension != ".webm")
      throw std::runtime_error("Motion output must be .svg, .png, .mp4, .mov or .webm");
    s.frames = still ? 1 : int(std::ceil(s.fps * s.duration));
    s.overwrite = script_bool(options, "overwrite");
    if (!s.overwrite && fs::exists(s.output))
      throw std::runtime_error("Motion output exists; use overwrite");
    if (!fs::is_directory(s.output.parent_path()))
      throw std::runtime_error("Motion output directory does not exist");
    const auto over = script_string(options, "over"), audio = script_string(options, "audio");
    s.overlay = !over.empty();
    if (still && (!over.empty() || !audio.empty()))
      throw std::runtime_error("over/audio require a video output");
    for (const auto& path : {over, audio})
      if (!path.empty()) {
        const auto input = file_path(path);
        if (!fs::is_regular_file(input))
          throw std::runtime_error("Motion input not found: " + path);
        if (input == s.output || (fs::exists(s.output) && fs::equivalent(input, s.output)))
          throw std::runtime_error("Motion output must differ from its input videos/audio");
      }
    if (!still && !s.overlay && s.extension == ".mp4" && (width % 2 || height % 2))
      throw std::runtime_error("MP4 Motion dimensions must be even");
    s.flatten = s.extension == ".mp4" && !s.overlay;
    s.pending = s.output.parent_path() /
                ("." + s.output.stem().u8string() + "." + unique_id() + s.extension);
    s.pixels.resize(size_t(width * height));
    s.canvas.reset(tvg::SwCanvas::gen());
    if (!s.canvas)
      throw std::runtime_error("cannot allocate ThorVG canvas");
    tvg_check(
        s.canvas->target(s.pixels.data(), s.width, s.width, s.height, tvg::ColorSpace::ABGR8888S),
        "target");
    if (!still) {
      const auto tool =
          script_string(options, "ffmpeg", media_tool("ffmpeg", agent().executable.c_str()));
      std::vector<std::string> argv{tool,
                                    "-hide_banner",
                                    "-loglevel",
                                    "error",
                                    "-nostdin",
                                    "-n",
                                    "-f",
                                    "rawvideo",
                                    "-pixel_format",
                                    "rgba",
                                    "-video_size",
                                    std::to_string(width) + "x" + std::to_string(height),
                                    "-framerate",
                                    std::to_string(s.fps),
                                    "-i",
                                    "pipe:0"};
      if (s.overlay)
        argv.insert(argv.end(), {"-i", file_path(over).u8string()});
      if (!audio.empty())
        argv.insert(argv.end(), {"-i", file_path(audio).u8string()});
      if (s.overlay) {
        const auto x = number(options, "x", 0, -8192, 8192),
                   y = number(options, "y", 0, -8192, 8192);
        argv.insert(
            argv.end(),
            {"-filter_complex",
             "[1:v]setpts=PTS-STARTPTS[base];[0:v]setpts=PTS-STARTPTS[motion];[base][motion]overlay=x=" +
                 std::to_string(x) + ":y=" + std::to_string(y) +
                 ":eof_action=pass:repeatlast=0:format=auto:alpha=straight[v]",
             "-map", "[v]"});
      } else
        argv.insert(argv.end(), {"-map", "0:v:0"});
      if (!audio.empty()) {
        argv.insert(argv.end(),
                    {"-map", s.overlay ? "2:a:0" : "1:a:0", "-af", "apad", "-shortest"});
      } else if (s.overlay)
        argv.insert(argv.end(), {"-map", "1:a?"});
      if (!s.overlay)
        argv.insert(argv.end(), {"-t", std::to_string(s.frames / s.fps)});
      if (s.extension == ".mp4")
        argv.insert(argv.end(), {"-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags",
                                 "+faststart", "-c:a", "aac"});
      else if (s.extension == ".mov")
        argv.insert(argv.end(), {"-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le",
                                 "-c:a", "pcm_s16le"});
      else
        argv.insert(argv.end(), {"-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
                                 "-c:a", "libopus"});
      argv.push_back(s.pending.u8string());
      s.pipe = std::make_unique<PipeProcess>(argv, true);
      s.watcher = std::thread([&s] {
        while (!s.stop.load()) {
          try {
            s.check();
          } catch (...) {
            s.pipe->cancel();
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      });
    }
    const auto id = impl_->next++;
    const auto result = Value(Object{{"id", num(id)},
                                     {"width", num(s.width)},
                                     {"height", num(s.height)},
                                     {"frames", num(s.frames)},
                                     {"fps", num(s.fps)},
                                     {"duration", num(s.duration)}});
    impl_->sessions.emplace(id, std::move(session));
    return result;
  }
  const auto id = int(args.at(0).as_int());
  auto found = impl_->sessions.find(id);
  if (found == impl_->sessions.end())
    throw std::runtime_error("unknown Motion session");
  auto& s = *found->second;
  if (operation == "cancel") {
    impl_->sessions.erase(found);
    return Value();
  }
  if (operation == "frame") {
    s.render(args.at(1).as_string());
    return Value();
  }
  if (operation == "finish") {
    if (s.written != s.frames)
      throw std::runtime_error("incomplete Motion frame sequence");
    if (s.pipe)
      s.pipe->finish();
    s.stop = true;
    if (s.watcher.joinable())
      s.watcher.join();
    s.check();
    publish(s.pending, s.output, s.overwrite);
    s.pending.clear();
    auto result = Value(Object{{"kind", str("motion")},
                               {"renderer", str("ThorVG 1.1.3")},
                               {"output", str(s.output.u8string())},
                               {"width", num(s.width)},
                               {"height", num(s.height)},
                               {"frames", num(s.frames)},
                               {"fps", num(s.fps)},
                               {"motion_duration", num(s.frames / s.fps)},
                               {"overlay", Value(s.overlay)}});
    impl_->sessions.erase(found);
    return result;
  }
  throw std::runtime_error("unknown Motion operation");
}

int motions_command(const std::vector<std::string>& args) {
  if (args.size() < 2 || (args[0] != "render" && args[0] != "frame"))
    throw CliError("INVALID_REQUEST", "use motions render|frame FILE.js --out FILE [options]");
  const auto script = file_path(args[1]);
  if (!fs::is_regular_file(script))
    throw CliError("INPUT_NOT_FOUND", "Motion script not found", 3);
  Object options;
  std::vector<std::string> passed;
  uint64_t timeout = 0;
  size_t memory = 256;
  for (size_t i = 2; i < args.size(); ++i) {
    const auto key = args[i];
    if (key == "--") {
      passed.assign(args.begin() + i + 1, args.end());
      break;
    }
    if (key == "--overwrite") {
      options["overwrite"] = Value(true);
      continue;
    }
    if (key != "--out" && key != "--over" && key != "--audio" && key != "--width" &&
        key != "--height" && key != "--fps" && key != "--duration" && key != "--time" &&
        key != "--x" && key != "--y" && key != "--timeout-ms" && key != "--memory-mb" &&
        key != "--ffmpeg")
      throw CliError("INVALID_REQUEST", "unknown Motion option: " + key);
    if (++i == args.size())
      throw CliError("INVALID_REQUEST", "missing value for " + key);
    if (key == "--out" || key == "--over" || key == "--audio" || key == "--ffmpeg")
      options[key == "--out" ? "output" : key.substr(2)] = str(args[i]);
    else {
      const auto value = parse_cli_json(args[i]);
      if (key == "--timeout-ms" || key == "--memory-mb") {
        const auto n = script_uint(Value(Object{{"value", value}}), "value", 0);
        if (key == "--timeout-ms")
          timeout = n;
        else
          memory = size_t(n);
      } else
        options[key.substr(2)] = value;
    }
  }
  if (!options.count("output"))
    throw CliError("INVALID_REQUEST", "motions requires --out FILE");
  const auto ext = lower(fs::u8path(options.at("output").as_string()).extension().u8string());
  if ((args[0] == "frame") != (ext == ".svg" || ext == ".png"))
    throw CliError("INVALID_REQUEST", "frame writes PNG/SVG; render writes MP4/MOV/WebM");
  const auto source =
      "import definition from " + json::stringify(str(script.generic_u8string())) +
      "; export default async args => { const motion = typeof definition === 'function' ? await definition(args) : definition; "
      "if (!motion || typeof motion.frame !== 'function') throw Error('Motion must provide frame(context)'); "
      "return slopfab.motions.render({...motion,..." +
      json::stringify(Value(options)) + "}, motion.frame); };";
  return run_script(script.parent_path() / "<motion-entry>.js", passed, &source, memory, timeout);
}
}
