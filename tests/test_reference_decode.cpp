#include "../src/cli/reference_decode.h"
#include "slopfab/audio/wav.h"
#include "slopfab/pipeline.h"
#include "slopfab/video/y4m.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class F> void rejects(F function) {
  bool failed = false;
  try {
    function();
  } catch (const std::exception&) {
    failed = true;
  }
  check(failed, "invalid media accepted");
}

int main() {
  using namespace slopfab;
  const auto root = std::filesystem::temp_directory_path() /
                    ("slopfab-attachment-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root);

  struct Cleanup {
    std::filesystem::path path;

    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove_all(path, ec);
    }
  } cleanup{root};

  try {
    const auto movie = (root / "video & source.y4m").u8string();
    const auto sound = (root / "audio & source.wav").u8string();
    PixelBuffer pixels(3 * 48 * 4 * 4, .5f);
    video::write_y4m(movie, pixels, 48, 4, 4, {24, 1});
    const auto plain = cli::decode_reference_file(movie, true, nullptr);
    check(!plain.soundtrack() && plain.duration_seconds() == 2, "plain Y4M gained audio");
    for (int channels : {1, 2})
      for (int rate : {8000, 44100, 48000})
        for (int seconds : {1, 2, 3}) {
          std::vector<float> pcm(size_t(channels) * rate * seconds);
          for (size_t i = 0; i < pcm.size(); ++i)
            pcm[i] = float(int(i % 31) - 15) / 32;
          audio::write_wav(sound, pcm, channels, rate, audio::SampleFormat::kFloat32);
          cli::DrivingAudioAttachment info;
          auto attached = cli::decode_driving_reference(movie, sound, nullptr, info);
          const auto& track = *attached.soundtrack();
          check(attached.is_video() && attached.frames().size() == 48, "lost video frames");
          check(attached.frames()[0]->image.pixels == plain.frames()[0]->image.pixels,
                "changed video pixels");
          check(track.start_seconds == 0 && track.channels == channels && track.sample_rate == rate,
                "changed attachment format or synchronization");
          check(track.interleaved.size() == size_t(2 * rate * channels), "wrong aligned length");
          check(info.source_frames == size_t(seconds * rate) &&
                    info.attached_frames == size_t(2 * rate),
                "wrong source/aligned metadata");
          check(info.trimmed_frames == size_t(std::max(seconds - 2, 0) * rate) &&
                    info.padded_frames == size_t(std::max(2 - seconds, 0) * rate),
                "wrong trim/pad metadata");
          const auto used = size_t(channels) * info.used_frames;
          check(std::equal(pcm.begin(), pcm.begin() + used, track.interleaved.begin()),
                "PCM changed");
          check(std::all_of(track.interleaved.begin() + used, track.interleaved.end(),
                            [](float v) {
                              return v == 0;
                            }),
                "padding is not silence");
          check(reference_media_identity(attached) != reference_media_identity(plain),
                "attachment omitted from cache identity");
          GenerateRequest request;
          request.reference_media.push_back(std::make_shared<const ReferenceMedia>(attached));
          request.preserve_driving_audio = true;
          const auto plan = resolve_plan(request);
          check(plan.conditioning.pin_target_audio && !plan.conditioning.include_reference_audio,
                "pinning policy not resolved");
          check(describe_plan(request, plan).find("1 (1 with audio)") != std::string::npos,
                "soundtrack not part of the video reference");
        }
    // No fractional final sample may extend past the video's explicit duration.
    PixelBuffer fractional(3 * 60 * 4 * 4, .5f);
    video::write_y4m(movie, fractional, 60, 4, 4, {30000, 1001});
    std::vector<float> pcm(3 * 44100, .25f);
    audio::write_wav(sound, pcm, 1, 44100, audio::SampleFormat::kFloat32);
    cli::DrivingAudioAttachment info;
    auto attached = cli::decode_driving_reference(movie, sound, nullptr, info);
    check(info.attached_frames == 88288 &&
              attached.soundtrack()->duration_seconds() <= attached.duration_seconds(),
          "fractional duration must round down");
    auto standalone = cli::decode_reference_file(sound, false, nullptr);
    check(!standalone.is_video() && standalone.soundtrack()->interleaved == pcm,
          "standalone audio behavior changed");
    for (float invalid : {1.1f, -1.1f, std::numeric_limits<float>::quiet_NaN()}) {
      pcm.back() = invalid;
      audio::write_wav(sound, pcm, 1, 44100, audio::SampleFormat::kFloat32);
      rejects([&] {
        cli::decode_driving_reference(movie, sound, nullptr, info);
      });
    }
    rejects([&] {
      cli::decode_driving_reference(sound, sound, nullptr, info);
    });
    rejects([&] {
      cli::decode_driving_reference(movie, movie, nullptr, info);
    });
    std::cout
        << "PASS: Y4M/WAV attachment, native PCM, trim/pad, fractional timing and pinned plans\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
