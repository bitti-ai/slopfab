#include "../src/cli/agent.h"
#include "../src/cli/native_media.h"
#include "slopfab/image.h"
#include <fstream>
#include <iostream>

using namespace slopfab;
using namespace slopfab::cli;

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

int main() {
  auto root = std::filesystem::temp_directory_path() / ("slopfab-delivery-" + unique_id());
  try {
    start_request();
    PixelBuffer pixels(3 * 4 * 4, .25f);
    std::vector<float> sound(48000 * 2, .125f);
    RunSamples samples;
    samples.video = &pixels;
    samples.width = 4;
    samples.height = 4;
    samples.frames = 1;
    const auto path = (root / "image.png").u8string();
    {
      Delivery delivery(path);
      delivery.reserve(delivery.output);
      bool collision = false;
      try {
        Delivery duplicate(path);
        duplicate.reserve(duplicate.output);
      } catch (const CliError& e) {
        collision = e.code == "OUTPUT_EXISTS";
      }
      require(collision, "concurrent output reservation was accepted");
      Delivery::samples(samples, &delivery);
      require(delivery.artifacts.size() == 1, "missing PNG artifact");
      auto image = load_platform_image(path);
      require(image.width == 4 && image.pixels[0] == 64, "PNG pixel conversion mismatch");
    }
    {
      bool collision = false;
      try {
        Delivery duplicate(path);
        duplicate.reserve(duplicate.output);
      } catch (const CliError& e) {
        collision = e.code == "OUTPUT_EXISTS";
      }
      require(collision, "existing output overwritten by default");
      require(!std::filesystem::exists(path + ".slopfab-reserved"),
              "failed reservation was leaked");
    }
    {
      agent().overwrite = true;
      Delivery replacement(path);
      replacement.reserve(replacement.output);
      Delivery::samples(samples, &replacement);
      agent().overwrite = false;
    }
    {
      Delivery delivery((root / "movie.y4m").u8string());
      delivery.reserve(delivery.output);
      delivery.reserve((root / "movie.wav").u8string());
      samples.audio = &sound;
      samples.audio_channels = 2;
      samples.audio_sample_rate = 48000;
      Delivery::samples(samples, &delivery);
      require(delivery.artifacts.size() == 2, "video soundtrack artifact missing");
      require(read_y4m(delivery.output).frames.size() == 1, "invalid video output");
      require(read_wav((root / "movie.wav").u8string()).interleaved.size() == sound.size(),
              "invalid audio output");
    }
    {
      Delivery delivery((root / "cancelled.png").u8string());
      delivery.reserve(delivery.output);
      agent().cancel_file = (root / "cancel").u8string();
      std::ofstream(agent().cancel_file).put('1');
      bool stopped = false;
      try {
        Delivery::samples(samples, &delivery);
      } catch (const CliError& e) {
        stopped = e.code == "CANCELLED";
      }
      require(stopped && delivery.error && !std::filesystem::exists(delivery.output),
              "cancellation did not stop publication");
      agent().cancel_file.clear();
    }
    for (const auto& entry : std::filesystem::directory_iterator(root))
      require(entry.path().extension() != ".tmp", "temporary artifact was leaked");
    std::filesystem::remove_all(root);
    std::cout << "PASS: PNG/Y4M/WAV delivery, reservations, overwrite, cancellation and cleanup\n";
    return 0;
  } catch (const std::exception& e) {
    std::filesystem::remove_all(root);
    std::cerr << e.what() << '\n';
    return 1;
  }
}
