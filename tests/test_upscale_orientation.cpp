#include "../src/cli/pipe_process.h"
#include "../src/cli/upscale_geometry.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace {
using namespace slopfab::cli;

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::string run(const std::vector<std::string>& args) {
  PipeProcess child(args, false);
  std::string result;
  char buffer[4096];
  while (const auto n = child.read(buffer, sizeof(buffer)))
    result.append(buffer, n);
  child.finish();
  return result;
}

std::string header_field(const std::string& y4m, char key) {
  std::istringstream header(y4m.substr(0, y4m.find('\n')));
  std::string token;
  while (header >> token)
    if (token[0] == key)
      return token.substr(1);
  throw std::runtime_error("missing Y4M header field");
}

UpscaleVideoGeometry parse(const std::string& stream) {
  return upscale_video_geometry(slopfab::json::parse(stream));
}

struct Temporary {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("slopfab-orientation-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

  Temporary() {
    std::filesystem::create_directory(path);
  }

  ~Temporary() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};
}

int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("usage: orientation_tests FFMPEG FFPROBE");
    const std::string ffmpeg = argv[1], ffprobe = argv[2];
    require(parse(R"({"width":32,"height":16})").sar() == "1/1", "absent SAR");
    for (const auto* sar : {"N/A", "0:1", "0/1", "0:0"})
      require(parse(std::string(R"({"width":32,"height":16,"sample_aspect_ratio":")") + sar + "\"}")
                      .sar() == "1/1",
              "unknown SAR");
    for (const auto* invalid :
         {R"({"width":0,"height":16})", R"({"width":32.5,"height":16})",
          R"({"width":2147483648,"height":16})", R"({"height":16})",
          R"({"width":32,"height":16,"sample_aspect_ratio":"2:0"})",
          R"({"width":32,"height":16,"side_data_list":[{"displaymatrix":"bad"}]})",
          R"({"width":32,"height":16,"tags":{"rotate":"90oops"}})"}) {
      bool rejected = false;
      try {
        parse(invalid);
      } catch (const std::exception&) {
        rejected = true;
      }
      require(rejected, "malformed geometry accepted");
    }
    require(parse(R"({"width":32,"height":16,"tags":{"rotate":"-90"}})").width == 16,
            "legacy rotation tag");
    require(parse(R"({"width":32,"height":16,"side_data_list":[{"rotation":450}]})").width == 16,
            "rotation normalization");
    Temporary temporary;
    const auto base = (temporary.path / "base.mp4").u8string();
    run({ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=32x16:rate=1", "-vf",
         "setsar=2/1", "-frames:v", "1", "-c:v", "libx264", base});
    // FFmpeg 8 uses the input display option; older releases set the same
    // display matrix via stream metadata during remuxing.
    const bool display_option =
        run({ffmpeg, "-hide_banner", "-h", "full"}).find("-display_rotation") != std::string::npos;
    for (const double angle : {0., 90., -90., 180., 270., 45., 89.5, 90.5, 88.5, 269.5, 270.5}) {
      const auto fixture = (temporary.path / (std::to_string(angle) + ".mp4")).u8string();
      if (display_option)
        run({ffmpeg, "-v", "error", "-display_rotation", std::to_string(angle), "-i", base, "-c",
             "copy", fixture});
      else
        run({ffmpeg, "-v", "error", "-i", base, "-c", "copy", "-metadata:s:v:0",
             "rotate=" + std::to_string(angle), fixture});
      const auto metadata = slopfab::json::parse(
          run({ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
               kUpscaleVideoEntries, "-of", "json", fixture}));
      const auto geometry = upscale_video_geometry(metadata.find("streams")->as_array().at(0));
      const bool quarter = std::abs(std::abs(std::remainder(angle, 360.0)) - 90.0) < 1.0;
      require(geometry.width == (quarter ? 16 : 32) && geometry.height == (quarter ? 32 : 16),
              "rotation fixture did not produce expected canvas");
      const auto y4m = run({ffmpeg, "-v", "error", "-i", fixture, "-frames:v", "1", "-pix_fmt",
                            "yuv420p", "-f", "yuv4mpegpipe", "pipe:1"});
      require(std::stoi(header_field(y4m, 'W')) == geometry.width &&
                  std::stoi(header_field(y4m, 'H')) == geometry.height,
              "FFmpeg canvas mismatch");
      require(header_field(y4m, 'A') ==
                  std::to_string(geometry.sar_num) + ":" + std::to_string(geometry.sar_den),
              "FFmpeg autorotation SAR mismatch");
      const auto rgb = run({ffmpeg, "-v", "error", "-i", fixture, "-frames:v", "1", "-pix_fmt",
                            "rgb24", "-f", "rawvideo", "pipe:1"});
      const auto scaled = run({ffmpeg, "-v", "error", "-i", fixture, "-vf",
                               "scale=" + std::to_string(geometry.width) + ":" +
                                   std::to_string(geometry.height) + ":flags=lanczos,fps=1",
                               "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"});
      require(rgb == scaled, "CLI geometry distorts autorotated pixels");
      // Exercise the writer's SAR restoration after the raw-video boundary.
      const auto encoded = (temporary.path / (std::to_string(angle) + "-out.mp4")).u8string();
      run({ffmpeg, "-v", "error", "-f", "lavfi", "-i",
           "color=size=" + std::to_string(geometry.width * 4) + "x" +
               std::to_string(geometry.height * 4),
           "-frames:v", "1", "-vf", "setsar=" + geometry.sar() + ":max=2147483647", "-c:v",
           "libx264", encoded});
      const auto output = slopfab::json::parse(
          run({ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
               kUpscaleVideoEntries, "-of", "json", encoded}));
      const auto out = upscale_video_geometry(output.find("streams")->as_array().at(0));
      require(out.width == geometry.width * 4 && out.height == geometry.height * 4 &&
                  out.sar() == geometry.sar(),
              "encoded output lost display geometry");
      std::cout << "Rotation " << angle << ": " << geometry.width << 'x' << geometry.height
                << " SAR " << geometry.sar() << " passed\n";
    }
    const auto fractional = (temporary.path / "fractional-sar.mp4").u8string();
    run({ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=size=32x16", "-frames:v", "1", "-vf",
         "setsar=1000/1001:max=2147483647", "-c:v", "libx264", fractional});
    const auto metadata =
        slopfab::json::parse(run({ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  kUpscaleVideoEntries, "-of", "json", fractional}));
    require(upscale_video_geometry(metadata.find("streams")->as_array().at(0)).sar() == "1000/1001",
            "fractional SAR was approximated");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
