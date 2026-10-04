#include "commands.h"
#include "pipe_process.h"
#include "slopfab/seedvr2.h"
#include "slopfab/json.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace slopfab::cli {
namespace {
volatile std::sig_atomic_t interrupted = 0;

void interrupt(int) {
  interrupted = 1;
}

int integer(const std::string& s) {
  size_t end = 0;
  int value = std::stoi(s, &end);
  if (end != s.size())
    throw std::invalid_argument("invalid integer: " + s);
  return value;
}
}

int cmd_upscale(int argc, char** argv, const char* executable) {
  seedvr2::Options o;
  std::string input, output;
  bool dry = false;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&]() {
      if (++i >= argc)
        throw std::invalid_argument("missing value for " + a);
      return std::string(argv[i]);
    };
    if (a == "--transformer")
      o.transformer = value();
    else if (a == "--vae")
      o.vae = value();
    else if (a == "--out")
      output = value();
    else if (a == "--resolution") {
      auto s = value();
      auto x = s.find('x');
      if (x == std::string::npos)
        throw std::invalid_argument("resolution must be WIDTHxHEIGHT");
      o.width = integer(s.substr(0, x));
      o.height = integer(s.substr(x + 1));
    } else if (a == "--segment-frames")
      o.segment_frames = integer(value());
    else if (a == "--vae-tile")
      o.vae_tile = integer(value());
    else if (a == "--device")
      o.device = integer(value());
    else if (a == "--seed") {
      auto s = value();
      size_t end = 0;
      if (s.empty() || s[0] == '-')
        throw std::invalid_argument("seed must be nonnegative");
      o.seed = std::stoull(s, &end);
      if (end != s.size())
        throw std::invalid_argument("invalid seed");
    } else if (a == "--no-color-match")
      o.color_match = false;
    else if (a == "--dry-run")
      dry = true;
    else if (a.size() && a[0] != '-' && input.empty())
      input = a;
    else
      throw std::invalid_argument("unknown upscale argument: " + a);
  }
  seedvr2::validate(o);
  if (input.empty() || output.empty())
    throw std::invalid_argument("upscale requires an input and --out output.mp4");
  if (o.width % 2 || o.height % 2)
    throw std::invalid_argument("MP4 output requires even width and height");
  auto in = std::filesystem::absolute(std::filesystem::u8path(input));
  auto out = std::filesystem::absolute(std::filesystem::u8path(output));
  if (!std::filesystem::is_regular_file(in))
    throw std::invalid_argument("input file does not exist: " + input);
  if (std::filesystem::exists(out))
    throw std::invalid_argument("output already exists: " + output);
  if (out.extension() != ".mp4" && out.extension() != ".mkv")
    throw std::invalid_argument("upscale output must be .mp4 or .mkv");
  if (o.transformer.empty())
    o.transformer = "weights/seedvr2/seedvr2_3b_fp16.safetensors";
  if (o.vae.empty())
    o.vae = "weights/seedvr2/ema_vae_fp16.safetensors";
  std::printf("SeedVR2 3B CUDA: %dx%d, %d-frame segments, VAE tile %d, seed %llu\n", o.width,
              o.height, o.segment_frames, o.vae_tile, static_cast<unsigned long long>(o.seed));
  if (dry)
    return 0;
#if !SLOPFAB_WITH_CUDA
  (void)executable;
  throw std::runtime_error("SeedVR2 requires a CUDA-enabled build");
#else
  PipeProcess probe({media_tool("ffprobe", executable), "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=avg_frame_rate,r_frame_rate,start_time", "-of", "json",
                     in.u8string()},
                    false);
  std::string metadata;
  char buf[4096];
  size_t n;
  while ((n = probe.read(buf, sizeof(buf)))) {
    metadata.append(buf, n);
    if (metadata.size() > 65536)
      throw std::runtime_error("excessive ffprobe metadata");
  }
  probe.finish();
  auto info = json::parse(metadata);
  auto* streams = info.find("streams");
  if (!streams || streams->as_array().empty())
    throw std::runtime_error("input has no video stream");
  const auto& s = streams->as_array()[0];
  auto* rate = s.find("avg_frame_rate");
  if (!rate || rate->as_string() == "0/0")
    rate = s.find("r_frame_rate");
  if (!rate)
    throw std::runtime_error("input has no usable frame rate");
  std::string fps = rate->as_string();
  double video_start = 0;
  if (const auto* start = s.find("start_time")) {
    if (start->as_string() != "N/A") video_start = std::stod(start->as_string());
  }
  if (!std::isfinite(video_start)) throw std::runtime_error("invalid input video start time");
  auto slash = fps.find('/');
  int numerator = integer(fps.substr(0, slash)),
      denominator = slash == std::string::npos ? 1 : integer(fps.substr(slash + 1));
  if (numerator <= 0 || denominator <= 0 || double(numerator) / denominator > 1000)
    throw std::runtime_error("invalid input frame rate");
  std::printf("Output rate: %s fps (variable-rate inputs are normalized to this rate)\n",
              fps.c_str());
  seedvr2::Restorer model(o);
  auto previous = std::signal(SIGINT, interrupt);
  interrupted = 0;

  struct SignalRestore {
    decltype(previous) handler;

    ~SignalRestore() {
      std::signal(SIGINT, handler);
    }
  } signal_restore{previous};

  model.cancelled = [] {
    return interrupted != 0;
  };
  model.progress = [](const std::string& msg) {
    std::fprintf(stderr, "  %s\n", msg.c_str());
  };
  auto temporary =
      out.parent_path() / (out.stem().u8string() + ".seedvr2-part" + out.extension().u8string());
  if (std::filesystem::exists(temporary))
    throw std::runtime_error("temporary output already exists: " + temporary.u8string());
  try {
    PipeProcess reader({media_tool("ffmpeg", executable), "-nostdin", "-v", "error", "-i",
                        in.u8string(), "-map", "0:v:0", "-an", "-sn", "-vf",
                        "scale=" + std::to_string(o.width) + ":" + std::to_string(o.height) +
                            ":flags=lanczos,fps=" + fps,
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"},
                       false);
    PipeProcess writer({media_tool("ffmpeg", executable),
                        "-nostdin",
                        "-v",
                        "error",
                        "-n",
                        "-copyts",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        std::to_string(o.width) + "x" + std::to_string(o.height),
                        "-r",
                        fps,
                        "-i",
                        "pipe:0",
                        "-itsoffset",
                        std::to_string(-video_start),
                        "-i",
                        in.u8string(),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a?",
                        "-map_metadata",
                        "1",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "copy",
                        temporary.u8string()},
                       true);
    std::vector<uint8_t> bytes(size_t(o.width) * o.height * 3);
    auto start = std::chrono::steady_clock::now();
    uint64_t frames = seedvr2::stream(
        o,
        [&](seedvr2::Frame& frame) {
          if (interrupted)
            throw std::runtime_error("SeedVR2: cancelled");
          size_t got = reader.read(bytes.data(), bytes.size());
          if (!got)
            return false;
          if (got != bytes.size())
            throw std::runtime_error("truncated decoded frame");
          frame.resize(bytes.size());
          for (size_t i = 0; i < bytes.size(); ++i)
            frame[i] = bytes[i] / 255.0f;
          return true;
        },
        [&](const seedvr2::Frame& frame) {
          for (size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = uint8_t(std::lround(std::clamp(frame[i], 0.0f, 1.0f) * 255));
          writer.write(bytes.data(), bytes.size());
        },
        [&](const std::vector<seedvr2::Frame>& batch, uint64_t first) {
          std::fprintf(stderr, "Restoring segment at frame %llu\n",
                       static_cast<unsigned long long>(first));
          return model.restore(batch, first);
        });
    reader.finish();
    writer.finish();
    if (!frames)
      throw std::runtime_error("input contains no decoded frames");
    std::filesystem::rename(temporary, out);
    double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("Restored %llu frames in %.1f seconds: %s\n",
                static_cast<unsigned long long>(frames), seconds, output.c_str());
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
  return 0;
#endif
}
}
