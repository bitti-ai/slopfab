#include "commands.h"
#include "bounded_pipeline.h"
#include "native_media.h"
#include "agent.h"
#include "upscale_options.h"
#include "upscale_geometry.h"
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
// SIGINT may be delivered to either media worker. Lock-free atomics are safe
// in the handler and make its notification visible to the inference thread.
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> interrupted{false};

void interrupt(int) {
  interrupted.store(true, std::memory_order_relaxed);
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
  const bool still = out.extension() == ".ppm" || out.extension() == ".png";
  if (!still && out.extension() != ".y4m")
    throw std::invalid_argument(
        "upscale output must be .png, .ppm or .y4m; convert other formats externally");
  if (!still && !dump.empty())
    throw std::invalid_argument("--dump requires image output");
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
    Delivery delivery(output);
    delivery.reserve(delivery.output);
    RunSamples samples;
    samples.video = &up;
    samples.frames = 1;
    samples.height = dims.first;
    samples.width = dims.second;
    Delivery::samples(samples, &delivery);
    if (!dump.empty())
      write_safetensors(
          dump,
          {{"pixels", {3, 1, dims.first, dims.second}, std::vector<float>(up.begin(), up.end())}});
    return 0;
  }
  (void)executable;
  auto input_video = read_y4m(input);
  const int frames = static_cast<int>(input_video.frames.size());
  const size_t plane = size_t(input_video.width) * input_video.height;
  PixelBuffer pixels(plane * frames * 3);
  for (int f = 0; f < frames; ++f)
    for (size_t p = 0; p < plane; ++p)
      for (size_t c = 0; c < 3; ++c)
        pixels[c * plane * frames + size_t(f) * plane + p] =
            input_video.frames[f][p * 3 + c] / 255.0f;
  input_video.frames.clear();
  auto upscaler = make_upscaler(args.method, args.model, backend);
  auto up = upscaler->upscale(
      pixels, frames, input_video.height, input_video.width, args.options, [](int step, int total) {
        return !interrupted && progress(RunStage::kUpscaling, step, total, nullptr);
      });
  const auto dims =
      upscale_dimensions(input_video.height, input_video.width, args.method, args.options);
  Delivery delivery(output);
  delivery.reserve(delivery.output);
  delivery.fps = input_video.fps;
  RunSamples samples;
  samples.video = &up;
  samples.frames = frames;
  samples.height = dims.first;
  samples.width = dims.second;
  Delivery::samples(samples, &delivery);
  return 0;
#endif
}
}
