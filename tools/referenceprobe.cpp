#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <stdexcept>

#include "slopfab/reference_conditioning.h"
#include "slopfab/safetensors.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vae/audio_encoder.h"
#include "slopfab/vae/keyframe_encoder.h"
#ifdef SLOPFAB_REFERENCEPROBE_VULKAN
#include "slopfab/vulkan/reference_encoder.h"
#endif

void compare(const std::vector<float>& actual, const std::vector<float>& expected, const char* name,
             double absolute_tolerance = 5e-4, double relative_tolerance = 1e-4) {
  if (actual.size() != expected.size())
    throw std::runtime_error("reference probe: shape mismatch");
  double error = 0, energy = 0, maximum = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]))
      throw std::runtime_error("reference probe: nonfinite output");
    double d = actual[i] - expected[i];
    error += d * d;
    energy += double(expected[i]) * expected[i];
    maximum = std::max(maximum, std::abs(d));
  }
  const double relative = std::sqrt(error / std::max(energy, 1e-20));
  std::printf("%s: max_abs %.8g, relative_RMS %.8g\n", name, maximum, relative);
  if (relative > relative_tolerance || maximum > absolute_tolerance)
    throw std::runtime_error("reference encoder differs from CPU authority");
}

int main(int argc, char** argv) {
  try {
    const bool mixed = argc > 4 && std::strcmp(argv[argc - 1], "--fp16") == 0;
    const int arguments = argc - (mixed ? 1 : 0);
    const bool benchmark = arguments == 8 && std::strcmp(argv[4], "--bench") == 0;
    if (arguments != 4 && !benchmark)
      throw std::runtime_error(
          "usage: referenceprobe video-vae audio-vae golden.safetensors [--bench width height frames] [--fp16]");
#ifdef SLOPFAB_REFERENCEPROBE_VULKAN
    if (mixed)
      throw std::invalid_argument("FP16 probe requires CUDA");
    if (benchmark)
      throw std::invalid_argument("reference benchmark requires CUDA");
#endif
    slopfab::SafeTensors golden;
    golden.open(argv[3]);
#ifdef SLOPFAB_REFERENCEPROBE_VULKAN
    auto instance = slopfab::vulkan::Instance::create();
    auto physical = instance.enumerate_devices();
    if (physical.empty())
      throw std::runtime_error("No Vulkan device");
    slopfab::vulkan::DeviceOptions options;
    options.enable_timeline_semaphore = true;
    auto device = physical.front().create_device(options);
    std::printf("Vulkan maximum storage buffer: %llu bytes\n",
                static_cast<unsigned long long>(physical.front().info().max_storage_buffer_bytes));
#endif
    {
      auto input = slopfab::to_f32(golden.at("resample_input"));
      auto audio = slopfab::ReferenceMedia::audio(input.data(), input.size(), 1, 44100);
      compare(slopfab::prepare_reference_condition(audio, 2).audio,
              slopfab::to_f32(golden.at("resample_output")), "resample");
    }
    {
      slopfab::SafeTensors ck;
      ck.open(argv[1]);
#ifdef SLOPFAB_REFERENCEPROBE_VULKAN
      slopfab::vulkan::ReferenceEncoder encoder(device, ck, false);
#else
      slopfab::vae::KeyframeEncoder encoder(ck);
#endif
      auto input = slopfab::to_f32(golden.at("video_input"));
#ifdef SLOPFAB_REFERENCEPROBE_VULKAN
      compare(encoder.encode_temporal_moments(input.data(), 17, 32, 32),
              slopfab::to_f32(golden.at("video_moments")), "video");
#else
      // Mixed precision has its own explicit gate; the FP32 authority retains
      // its original tolerances. Never silently rebaseline the authority.
      const auto begin = std::chrono::steady_clock::now();
      compare(encoder.encode_temporal_moments(input.data(), 17, 32, 32, mixed),
              slopfab::to_f32(golden.at("video_moments")), "video", mixed ? .1 : 5e-4,
              mixed ? .005 : 1e-4);
      std::printf("video moments wall: %.3f s (%s)\n",
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(),
                  mixed ? "fp16" : "fp32");
#endif
      auto rgb = slopfab::to_f32(golden.at("video_rgb"));
      std::vector<slopfab::RGBImage> frames(22);
      for (int i = 0; i < 22; ++i) {
        frames[i].width = frames[i].height = 32;
        frames[i].pixels.resize(3072);
        for (size_t j = 0; j < 3072; ++j)
          frames[i].pixels[j] = static_cast<uint8_t>(rgb[size_t(i) * 3072 + j]);
      }
      compare(encoder.encode_reference_video(frames, 22,
                                             slopfab::to_f32(golden.at("video_latent_mean")),
                                             slopfab::to_f32(golden.at("video_latent_std"))
#ifndef SLOPFAB_REFERENCEPROBE_VULKAN
                                                 ,
                                             mixed
#endif
                                             ),
              slopfab::to_f32(golden.at("video_rows")), "video chunking/posterior/rows",
              mixed ? .1 : 5e-3, mixed ? .005 : 1e-4);
#ifndef SLOPFAB_REFERENCEPROBE_VULKAN
      if (benchmark) {
        const int width = std::stoi(argv[5]), height = std::stoi(argv[6]),
                  count = std::stoi(argv[7]);
        if (width < 32 || height < 32 || width > 2048 || height > 2048 || width % 32 ||
            height % 32 || count < 22 || count > 345 || (count - 5) % 17)
          throw std::invalid_argument(
              "benchmark: dimensions must be 32..2048 multiples of 32; frames must be 17*n+5, 22..345");
        std::vector<slopfab::RGBImage> clip(count);
        for (int t = 0; t < count; ++t) {
          auto& frame = clip[t];
          frame.width = width;
          frame.height = height;
          frame.pixels.resize(size_t(width) * height * 3);
          for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
              for (int c = 0; c < 3; ++c)
                frame.pixels[(size_t(y) * width + x) * 3 + c] =
                    static_cast<uint8_t>((x * 3 + y * 5 + t * 7 + c * 47) % 256);
        }
        std::vector<float> previous;
        for (int run = 0; run < 2; ++run) {
          const auto begin = std::chrono::steady_clock::now();
          auto rows = encoder.encode_reference_video(
              clip, count, slopfab::to_f32(golden.at("video_latent_mean")),
              slopfab::to_f32(golden.at("video_latent_std")), mixed);
          std::printf(
              "benchmark %s %dx%d %d frames run %d: %.3f s\n", mixed ? "fp16" : "fp32", width,
              height, count, run + 1,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
          for (float value : rows)
            if (!std::isfinite(value))
              throw std::runtime_error("benchmark: nonfinite reference rows");
          if (run && previous != rows)
            throw std::runtime_error("benchmark: repeated encode changed rows");
          previous = std::move(rows);
        }
        const auto path = std::string("reference-bench-") + (mixed ? "fp16" : "fp32") + ".bin";
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(previous.data()),
                     previous.size() * sizeof(float));
        if (!output)
          throw std::runtime_error("benchmark: failed to write rows");
        std::printf("benchmark rows: %s\n", path.c_str());
      }
#endif
    }
    {
      slopfab::SafeTensors ck;
      ck.open(argv[2]);
#ifdef SLOPFAB_REFERENCEPROBE_VULKAN
      slopfab::vulkan::ReferenceEncoder encoder(device, ck, true);
#else
      slopfab::vae::AudioEncoder encoder(ck);
#endif
      auto input = slopfab::to_f32(golden.at("audio_input"));
      compare(encoder.encode_mean(input.data(), int(input.size() / 2)),
              slopfab::to_f32(golden.at("audio_mean")), "audio");
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
