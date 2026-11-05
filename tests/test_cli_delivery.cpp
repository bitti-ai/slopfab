#include "../src/cli/agent.h"
#include "../src/cli/native_media.h"
#include "../src/cli/projects.h"
#include "../src/cli/frames.h"
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
      // Real delivery callback with borrowed planar generation pixels, no models or GPU.
      PixelBuffer movie(3 * 3 * 4 * 4, 0.0f);
      for (int frame = 0; frame < 3; ++frame)
        for (int p = 0; p < 16; ++p)
          movie[frame * 48 + frame * 16 + p] = 1.0f;
      RunSamples video;
      video.video = &movie;
      video.frames = 3;
      video.width = video.height = 4;
      const auto review = (root / "review.png").u8string();
      bool observed = false;
      agent().samples_observer = [&](RunSamples& borrowed, Delivery& delivery) {
        require(borrowed.video == &movie, "sample observer copied the generation buffer");
        auto image = frames_from_samples(
            borrowed, delivery.fps,
            Value(Object{
                {"output", str(review)}, {"columns", num(3)}, {"rows", num(1)}, {"width", num(2)}}),
            [] {
            });
        require(image.find("source")->as_string() == "memory:generation", "wrong review source");
        require(image.find("cells")->as_array()[2].find("time")->as_number() == 1,
                "wrong sample timestamp");
        delivery.artifacts.push_back(image);
        observed = true;
      };
      Delivery delivery((root / "review-video.y4m").u8string());
      delivery.fps = {2, 1};
      delivery.reserve(delivery.output);
      Delivery::samples(video, &delivery);
      agent().samples_observer = {};
      require(observed && delivery.artifacts.size() == 2, "missing RAM review artifact");
      auto image = load_platform_image(review);
      require(image.width == 6 && image.height == 2, "wrong RAM grid size");
      require(image.pixels[0] == 255 && image.pixels[7] == 255 && image.pixels[14] == 255,
              "planar RGB generation frame selection is wrong");
    }
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
    {
      agent().config_path = (root / "config.json").u8string();
      agent().project = "review";
      agent().project_output = (root / "projects").u8string();
      projects_command({"create", "review", "--output-dir", agent().project_output});
      const auto directory = project_directory("review", agent().project_output);
      GenerateRequest request;
      request.out_path = "image.png";
      request.prompt = "Project image";
      request.negative_prompt = "Unwanted lettering";
      request.music.lyrics = "Original lyrics\nSecond line";
      request.seed = 42;
      agent().generation_arguments = {str("image"), str("--prompt"), str(request.prompt)};
      agent().reference_attachments = {Value(Object{{"video_path", str("clip.y4m")},
          {"audio_path", str("clip.wav")}, {"source_start_seconds", num(0)},
          {"source_end_seconds", num(2)}, {"padded_frames", num(8000)}})};
      samples.audio = nullptr;
      auto generate = [&]() {
        prepare_project_generation(request, false, 1);
        Delivery delivery(request.out_path);
        delivery.reserve(delivery.output);
        Delivery::samples(samples, &delivery);
        return record_project_generation(request,
                                         Value(Object{{"index", num(0)},
                                                      {"ok", Value(true)},
                                                      {"plan", plan_json(request, resolve_plan(request))},
                                                      {"artifacts", Value(delivery.artifacts)}}));
      };
      auto first = generate();
      const auto original = request.out_path;
      const auto id = first.find("id")->as_string();
      require(first.find("version")->as_number() == 1, "project registration version mismatch");
      agent().project_item = id;
      request.prompt = "Revised project image";
      request.music.lyrics.clear();
      auto second = generate();
      require(second.find("version")->as_number() == 2, "project regeneration version mismatch");
      require(request.out_path != original && std::filesystem::exists(original),
              "project regeneration did not preserve original output");
      auto manifest = read_json(directory / "project.json");
      const auto& item = manifest.find("items")->as_array()[0];
      require(manifest.find("items")->as_array().size() == 1 &&
                  item.find("history")->as_array().size() == 1,
              "project regeneration lost stable ID/history");
      require(item.find("generation")->find("plan")->find("seed")->as_string() == "42",
              "project recipe lost resolved seed");
      const auto& attachment = item.find("generation")->find("plan")
                                   ->find("reference_attachments")->as_array().at(0);
      require(attachment.find("audio_path")->as_string() == "clip.wav" &&
                  attachment.find("video_path")->as_string() == "clip.y4m" &&
                  attachment.find("source_end_seconds")->as_number() == 2 &&
                  attachment.find("padded_frames")->as_number() == 8000,
              "project recipe lost driving soundtrack provenance");
      const auto& prompts = *item.find("generation")->find("prompts");
      require(prompts.find("prompt")->as_string() == "Revised project image" &&
                  prompts.find("negative_prompt")->as_string() == "Unwanted lettering" &&
                  prompts.find("lyrics")->as_string().empty(),
              "project recipe did not snapshot resolved prompt inputs");
      const auto& previous =
          *item.find("history")->as_array()[0].find("generation")->find("prompts");
      require(previous.find("prompt")->as_string() == "Project image" &&
                  previous.find("lyrics")->as_string() == "Original lyrics\nSecond line",
              "project history lost previous prompt inputs");
      require(std::filesystem::exists(directory / "index.html"), "project page was not generated");
      // An edit after planning must not be replaced by a late generation.
      prepare_project_generation(request, false, 1);
      std::filesystem::create_directories(std::filesystem::u8path(request.out_path).parent_path());
      std::filesystem::copy_file(original, std::filesystem::u8path(request.out_path));
      projects_command({"update", "review", id, "--title", "A newer edit", "--output-dir",
                        agent().project_output});
      bool rejected = false;
      try {
        record_project_generation(request, Value(Object{{"index", num(0)},
                                                        {"plan", Value(Object{})},
                                                        {"artifacts", Value(Array{})}}));
      } catch (const CliError& error) {
        rejected = error.code == "PROJECT_CHANGED";
      }
      require(rejected && std::filesystem::exists(std::filesystem::u8path(request.out_path)),
              "late project registration overwrote an edit or lost output");
      agent().project.clear();
      agent().project_item.clear();
      agent().project_output.clear();
      agent().reference_attachments.clear();
    }
    for (const auto& entry : std::filesystem::directory_iterator(root))
      require(entry.path().extension() != ".tmp", "temporary artifact was leaked");
    std::filesystem::remove_all(root);
    std::cout
        << "PASS: PNG/Y4M/WAV delivery, reservations, cancellation, project registration and versions\n";
    return 0;
  } catch (const std::exception& e) {
    std::filesystem::remove_all(root);
    std::cerr << e.what() << '\n';
    return 1;
  }
}
