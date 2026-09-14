#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "slopfab/reference_conditioning.h"
#include "slopfab/safetensors.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vae/audio_encoder.h"
#include "slopfab/vae/keyframe_encoder.h"
#ifdef SLOPFAB_REFERENCEPROBE_VULKAN
#include "slopfab/vulkan/reference_encoder.h"
#endif

void compare(const std::vector<float>& actual,
             const std::vector<float>& expected, const char* name,
             double absolute_tolerance = 5e-4) {
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
  if (relative > 1e-4 || maximum > absolute_tolerance)
    throw std::runtime_error("reference encoder differs from CPU authority");
}
int main(int argc, char** argv) {
  try {
    if (argc != 4)
      throw std::runtime_error(
          "usage: referenceprobe video-vae audio-vae golden.safetensors");
    slopfab::SafeTensors golden;
    golden.open(argv[3]);
#ifdef SLOPFAB_REFERENCEPROBE_VULKAN
    auto instance = slopfab::vulkan::Instance::create();
    auto physical = instance.enumerate_devices();
    if (physical.empty()) throw std::runtime_error("No Vulkan device");
    slopfab::vulkan::DeviceOptions options;
    options.enable_timeline_semaphore = true;
    auto device = physical.front().create_device(options);
    std::printf("Vulkan maximum storage buffer: %llu bytes\n",
        static_cast<unsigned long long>(physical.front().info().max_storage_buffer_bytes));
#endif
    {
      auto input = slopfab::to_f32(golden.at("resample_input"));
      auto audio =
          slopfab::ReferenceMedia::audio(input.data(), input.size(), 1, 44100);
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
      compare(encoder.encode_temporal_moments(input.data(), 17, 32, 32),
              slopfab::to_f32(golden.at("video_moments")), "video");
      auto rgb = slopfab::to_f32(golden.at("video_rgb"));
      std::vector<slopfab::RGBImage> frames(22);
      for (int i = 0; i < 22; ++i) {
        frames[i].width = frames[i].height = 32;
        frames[i].pixels.resize(3072);
        for (size_t j = 0; j < 3072; ++j)
          frames[i].pixels[j] = static_cast<uint8_t>(rgb[size_t(i) * 3072 + j]);
      }
      compare(encoder.encode_reference_video(
                  frames, 22, slopfab::to_f32(golden.at("video_latent_mean")),
                  slopfab::to_f32(golden.at("video_latent_std"))),
              slopfab::to_f32(golden.at("video_rows")),
              "video chunking/posterior/rows", 5e-3);
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
