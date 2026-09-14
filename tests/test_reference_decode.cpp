#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "../src/cli/reference_decode.h"
#include "slopfab/pipeline.h"

namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}

// Exercises the exact CLI adapter without running the application's interactive
// startup or loading any model. Fixtures are made by the adjacent CMake script.
int main(int argc, char** argv) {
  try {
    check(argc == 4, "expected video, audio and delayed-video fixture paths");
    auto video = slopfab::cli::decode_reference_file(argv[1], true, argv[0]);
    auto audio = slopfab::cli::decode_reference_file(argv[2], false, argv[0]);
    check(video.frames().size() == 48, "expected two seconds at 24 fps");
    check(video.frames()[0]->image.width == 4 && video.frames()[0]->image.height == 4,
          "incorrect decoded dimensions");
    check(video.frames()[0]->image.pixels[0] >= 250 && video.frames()[0]->image.pixels[1] <= 5,
          "incorrect decoded RGB pixels");
    check(video.frames().back()->timestamp_seconds == 47.0 / 24, "incorrect video timestamp");
    check(bool(video.soundtrack()), "missing video soundtrack");
    check(video.soundtrack()->channels == 1 && video.soundtrack()->sample_rate == 44100,
          "decoder changed the native audio rate or mono channel count");
    check(video.soundtrack()->interleaved.size() == 88200, "incorrect native audio sample count");
    check(video.soundtrack()->interleaved == audio.soundtrack()->interleaved,
          "standalone and embedded PCM disagree");
    double energy = 0;
    for (float sample : audio.soundtrack()->interleaved) energy += double(sample) * sample;
    check(energy / 88200 > .007 && energy / 88200 < .0085, "decoded audio has incorrect gain");
    auto delayed = slopfab::cli::decode_reference_file(argv[3], true, argv[0]);
    check(delayed.frames().size() == 72, "incorrect delayed clip length");
    check(delayed.soundtrack()->start_seconds == 1, "lost the video/audio offset");
    check(delayed.soundtrack()->interleaved == audio.soundtrack()->interleaved,
          "delayed soundtrack changed samples");
    slopfab::GenerateRequest request;
    request.reference_media.push_back(std::make_shared<const slopfab::ReferenceMedia>(video));
    request.reference_media.push_back(std::make_shared<const slopfab::ReferenceMedia>(audio));
    const auto plan = slopfab::resolve_plan(request);
    check(slopfab::describe_plan(request, plan).find("1 (1 with audio)") != std::string::npos,
          "plan omitted reference soundtrack");
    std::puts("Video, native PCM, soundtrack synchronization and request planning passed.");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
