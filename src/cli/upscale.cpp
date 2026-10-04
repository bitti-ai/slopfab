#include "commands.h"
#include "pipe_process.h"
#include "upscale_options.h"
#include "slopfab/image.h"
#include "slopfab/safetensors_write.h"
#include <fstream>
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
  UpscaleArguments args;
  args.enabled = true;
  std::string input, output, dump;
#if SLOPFAB_WITH_CUDA
  auto backend = DeviceBackend::kCuda;
#else
  auto backend = DeviceBackend::kVulkan;
#endif
  bool dry = false;
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() {
      if (++i >= argc)
        throw std::invalid_argument("missing value for " + a);
      return std::string(argv[i]);
    };
    if (args.parse(argc, argv, i)) {
    } else if (a == "--input")
      input = value();
    else if (a == "--out")
      output = value();
    else if (a == "--dump")
      dump = value();
    else if (a == "--inference-backend") {
      const auto v = value();
      if (v == "cuda")
        backend = DeviceBackend::kCuda;
      else if (v == "vulkan")
        backend = DeviceBackend::kVulkan;
      else
        throw std::invalid_argument("--inference-backend requires cuda or vulkan");
    } else if (a == "--dry-run")
      dry = true;
    else if (!a.empty() && a[0] != '-' && input.empty())
      input = a;
    else
      throw std::invalid_argument("unknown upscale argument: " + a);
  }
  args.finish();
  if (args.method == UpscaleMethod::kSeedVr2 && backend != DeviceBackend::kCuda)
    throw std::invalid_argument("SeedVR2 requires the CUDA backend");
  if (input.empty() || output.empty())
    throw std::invalid_argument("upscale requires --input and --out");
  auto in = std::filesystem::absolute(std::filesystem::u8path(input));
  auto out = std::filesystem::absolute(std::filesystem::u8path(output));
  if (!std::filesystem::is_regular_file(in))
    throw std::invalid_argument("input file does not exist: " + input);
  if (std::filesystem::exists(out))
    throw std::invalid_argument("output already exists: " + output);
  const bool still = out.extension() == ".ppm";
  if (!still && out.extension() != ".mp4" && out.extension() != ".mkv")
    throw std::invalid_argument("upscale output must be .ppm, .mp4 or .mkv");
  if (!still && !dump.empty())
    throw std::invalid_argument("--dump requires .ppm output");
  std::printf("Upscaler: %s, model %s, tile %d\n", upscale_method_name(args.method),
              args.model.c_str(), args.options.tile_size);
  if (dry)
    return 0;
#if !SLOPFAB_WITH_CUDA && !SLOPFAB_WITH_VULKAN
  (void)executable;
  throw std::runtime_error("upscaling requires a GPU-enabled build");
#else
  auto previous = std::signal(SIGINT, interrupt);
  interrupted = 0;

  struct SignalRestore {
    decltype(previous) handler;

    ~SignalRestore() {
      std::signal(SIGINT, handler);
    }
  } signal_restore{previous};

  if (still) {
    auto image = load_reference_image(input);
    const size_t plane = size_t(image.height) * image.width;
    PixelBuffer pixels(plane * 3);
    for (size_t p = 0; p < plane; ++p)
      for (size_t c = 0; c < 3; ++c)
        pixels[c * plane + p] = image.pixels[p * 3 + c] / 255.0f;
    auto upscaler = make_upscaler(args.method, args.model, backend);
    auto up = upscaler->upscale(pixels, 1, image.height, image.width, args.options, [](int, int) {
      return !interrupted;
    });
    const auto dims = upscale_dimensions(image.height, image.width, args.method, args.options);
    const size_t out_plane = size_t(dims.first) * dims.second;
    std::vector<uint8_t> rgb(up.size());
    for (size_t p = 0; p < out_plane; ++p)
      for (size_t c = 0; c < 3; ++c)
        rgb[p * 3 + c] = uint8_t(std::lround(up[c * out_plane + p] * 255));
    std::ofstream file(out, std::ios::binary);
    file << "P6\n" << dims.second << ' ' << dims.first << "\n255\n";
    file.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
    file.close();
    if (!file)
      throw std::runtime_error("upscale: cannot write " + output);
    if (!dump.empty())
      write_safetensors(
          dump,
          {{"pixels", {3, 1, dims.first, dims.second}, std::vector<float>(up.begin(), up.end())}});
    return 0;
  }
  PipeProcess probe({media_tool("ffprobe", executable), "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,start_time",
                     "-of", "json", in.u8string()},
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
    if (start->as_string() != "N/A")
      video_start = std::stod(start->as_string());
  }
  if (!std::isfinite(video_start))
    throw std::runtime_error("invalid input video start time");
  auto slash = fps.find('/');
  int numerator = integer(fps.substr(0, slash)),
      denominator = slash == std::string::npos ? 1 : integer(fps.substr(slash + 1));
  if (numerator <= 0 || denominator <= 0 || double(numerator) / denominator > 1000)
    throw std::runtime_error("invalid input frame rate");
  std::printf("Output rate: %s fps (variable-rate inputs are normalized to this rate)\n",
              fps.c_str());
  const int source_width = int(s.find("width")->as_number());
  const int source_height = int(s.find("height")->as_number());
  const auto dims = upscale_dimensions(source_height, source_width, args.method, args.options);
  const bool seed = args.method == UpscaleMethod::kSeedVr2;
  const int decode_width = seed ? dims.second : source_width;
  const int decode_height = seed ? dims.first : source_height;
  if (dims.first % 2 || dims.second % 2)
    throw std::invalid_argument("video output needs even dimensions");
  std::unique_ptr<Upscaler> upscaler;
#if SLOPFAB_WITH_CUDA
  std::unique_ptr<seedvr2::Restorer> model;
  seedvr2::Options o;
  if (seed) {
    o = seedvr2_options(args.options, source_height, source_width);
    o.transformer = args.model;
    model = std::make_unique<seedvr2::Restorer>(o);
    model->cancelled = [] {
      return interrupted != 0;
    };
    model->progress = [](const std::string& msg) {
      std::fprintf(stderr, "  %s\n", msg.c_str());
    };
  } else
#endif
    upscaler = make_upscaler(args.method, args.model, backend);
  auto temporary =
      out.parent_path() / (out.stem().u8string() + ".upscale-part" + out.extension().u8string());
  if (std::filesystem::exists(temporary))
    throw std::runtime_error("temporary output already exists: " + temporary.u8string());
  try {
    PipeProcess reader({media_tool("ffmpeg", executable), "-nostdin", "-v", "error", "-i",
                        in.u8string(), "-map", "0:v:0", "-an", "-sn", "-vf",
                        "scale=" + std::to_string(decode_width) + ":" +
                            std::to_string(decode_height) + ":flags=lanczos,fps=" + fps,
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
                        std::to_string(dims.second) + "x" + std::to_string(dims.first),
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
    std::vector<uint8_t> input_bytes(size_t(decode_width) * decode_height * 3);
    std::vector<uint8_t> output_bytes(size_t(dims.first) * dims.second * 3);
    auto start = std::chrono::steady_clock::now();
    auto read = [&](seedvr2::Frame& frame) {
      if (interrupted)
        throw UpscaleCancelled();
      const size_t got = reader.read(input_bytes.data(), input_bytes.size());
      if (!got)
        return false;
      if (got != input_bytes.size())
        throw std::runtime_error("truncated decoded frame");
      frame.resize(got);
      for (size_t i = 0; i < got; ++i)
        frame[i] = input_bytes[i] / 255.0f;
      return true;
    };
    auto write = [&](const seedvr2::Frame& frame) {
      if (interrupted)
        throw UpscaleCancelled();
      for (size_t i = 0; i < frame.size(); ++i)
        output_bytes[i] = uint8_t(std::lround(std::clamp(frame[i], 0.0f, 1.0f) * 255));
      writer.write(output_bytes.data(), output_bytes.size());
    };
    uint64_t frames = 0;
#if SLOPFAB_WITH_CUDA
    if (seed) {
      frames = seedvr2::stream(o, read, write,
                               [&](const std::vector<seedvr2::Frame>& batch, uint64_t first) {
                                 std::fprintf(stderr, "Restoring segment at frame %llu\n",
                                              static_cast<unsigned long long>(first));
                                 return model->restore(batch, first);
                               });
    } else
#endif
    {
      seedvr2::Frame frame, restored(output_bytes.size());
      const size_t plane = size_t(source_width) * source_height;
      const size_t out_plane = size_t(dims.first) * dims.second;
      PixelBuffer planar(plane * 3);
      while (read(frame)) {
        for (size_t p = 0; p < plane; ++p)
          for (size_t c = 0; c < 3; ++c)
            planar[c * plane + p] = frame[p * 3 + c];
        auto up =
            upscaler->upscale(planar, 1, source_height, source_width, args.options, [](int, int) {
              return !interrupted;
            });
        for (size_t p = 0; p < out_plane; ++p)
          for (size_t c = 0; c < 3; ++c)
            restored[p * 3 + c] = up[c * out_plane + p];
        write(restored);
        ++frames;
      }
    }
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
