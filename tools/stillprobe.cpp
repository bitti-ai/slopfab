#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include "slopfab/generate.h"
#include "slopfab/image.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vae/keyframe_encoder.h"
#include "stillprobe_helpers.h"

namespace {

using Clock = std::chrono::steady_clock;

std::ofstream output_file(const std::filesystem::path& path) {
  std::ofstream output;
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.open(path, std::ios::binary);
  return output;
}

void write_ppm(const std::filesystem::path& path, const slopfab::RGBImage& image) {
  auto output = output_file(path);
  output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
  output.write(reinterpret_cast<const char*>(image.pixels.data()),
               static_cast<std::streamsize>(image.pixels.size()));
}

double luminance(const slopfab::vae::DecodedVideo& image, int row, int column) {
  const size_t pixel = static_cast<size_t>(row) * image.width + column;
  const size_t plane = image.frame_stride();
  return 255.0 * (0.2126 * image.data[pixel] + 0.7152 * image.data[plane + pixel] +
                  0.0722 * image.data[2 * plane + pixel]);
}

std::vector<double> row_profile(const slopfab::vae::DecodedVideo& image, bool edges) {
  std::vector<double> profile(static_cast<size_t>(image.height), 0.0);
  const int margin = std::max(1, image.width / 20);
  for (int row = 0; row < image.height; ++row) {
    int count = 0;
    for (int column = 0; column < image.width; ++column) {
      if (edges && column >= margin && column < image.width - margin) continue;
      profile[row] += luminance(image, row, column);
      ++count;
    }
    profile[row] /= count;
  }
  return profile;
}

double patch_amplitude(const std::vector<double>& profile) {
  double cosine = 0.0;
  double sine = 0.0;
  const double pi = std::acos(-1.0);
  const int count = static_cast<int>(profile.size());
  for (int row = 1; row + 1 < count; ++row) {
    const double residual = profile[row] - 0.5 * (profile[row - 1] + profile[row + 1]);
    const double phase = 2.0 * pi * row / 16.0;
    cosine += residual * std::cos(phase);
    sine += residual * std::sin(phase);
  }
  return 2.0 * std::hypot(cosine, sine) /
         ((count - 2) * (1.0 - std::cos(2.0 * pi / 16.0)));
}

void save_image(const std::filesystem::path& directory, const std::string& name,
                 const slopfab::vae::DecodedVideo& image) {
  for (float value : image.data) {
    if (!std::isfinite(value)) throw std::runtime_error("non-finite decoded pixel");
  }
  slopfab::write_safetensors((directory / (name + ".safetensors")).string(),
      {{"rgb", {3, image.height, image.width},
        std::vector<float>(image.data.begin(), image.data.end())}});
  slopfab::RGBImage rgb;
  rgb.width = image.width;
  rgb.height = image.height;
  rgb.pixels.resize(3 * image.frame_stride());
  for (size_t pixel = 0; pixel < image.frame_stride(); ++pixel) {
    for (int channel = 0; channel < 3; ++channel) {
      const float value = image.data[channel * image.frame_stride() + pixel];
      rgb.pixels[3 * pixel + channel] =
          static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
    }
  }
  write_ppm(directory / (name + ".ppm"), rgb);
  if (image.width % 3 == 0 && image.height % 3 == 0) {
    write_ppm(directory / (name + "-third.ppm"),
               slopfab::resize_reference_lanczos(rgb, image.width / 3, image.height / 3));
  }
}

}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    slopfab::GenerateRequest request;
    request.still_image = true;
    request.canvas_width = 768;
    request.canvas_height = 768;
    request.seed = 1234;
    std::string latent_path;
    std::string reference_path;
    std::filesystem::path directory;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--help") {
        std::puts("usage: slopfab_stillprobe --vae FILE --out NEW_DIRECTORY\n"
                  "  (--reference-image FILE | --latents FILE | --prompt TEXT --transformer FILE --text-encoder FILE --tokenizer FILE)\n"
                  "  [--width 768 --height 768 --steps 50 --seed 1234]\n"
                  "Saved latents must be F32 video_rows [height/32 * width/32, 96].\n"
                  "Checks default still decoding against seven-token video decoding.\n"
                  "Both decodes use identical latents, weights, spatial tiling and phase 3.");
        return 0;
      }
      if (++index >= argc) throw std::invalid_argument("missing value for " + argument);
      const std::string value = argv[index];
      if (argument == "--vae") request.video_vae_path = value;
      else if (argument == "--out") directory = value;
      else if (argument == "--latents") latent_path = value;
      else if (argument == "--reference-image") reference_path = value;
      else if (argument == "--prompt") request.prompt = value;
      else if (argument == "--transformer") request.transformer_path = value;
      else if (argument == "--text-encoder") request.text_encoder_path = value;
      else if (argument == "--tokenizer") request.tokenizer_path = value;
      else if (argument == "--width") request.canvas_width = std::stoi(value);
      else if (argument == "--height") request.canvas_height = std::stoi(value);
      else if (argument == "--steps") request.num_inference_steps = std::stoi(value);
      else if (argument == "--seed") request.seed = std::stoull(value);
      else throw std::invalid_argument("unknown argument " + argument);
    }
    if (directory.empty() || request.video_vae_path.empty() ||
        (latent_path.empty() && reference_path.empty() && (request.prompt.empty() || request.transformer_path.empty() ||
                                request.text_encoder_path.empty() || request.tokenizer_path.empty())))
      throw std::invalid_argument("missing required options; use --help");
    if (!reference_path.empty() && !latent_path.empty())
      throw std::invalid_argument("choose reference-image or latents, not both");
    const auto plan = slopfab::resolve_plan(request);
    if (std::filesystem::exists(directory))
      throw std::invalid_argument("output directory already exists; choose a new directory");
    std::filesystem::create_directories(directory);
    auto report = output_file(directory / "report.txt");
    report << std::setprecision(10)
           << "canvas=" << plan.canvas_width << 'x' << plan.canvas_height << '\n'
           << "vae=" << request.video_vae_path << '\n';
    if (latent_path.empty() && reference_path.empty()) {
      report << "prompt=" << request.prompt << '\n'
             << "seed=" << request.seed << " steps=" << request.num_inference_steps << '\n'
             << "transformer=" << request.transformer_path << '\n'
             << "text_encoder=" << request.text_encoder_path << '\n'
             << "tokenizer=" << request.tokenizer_path << '\n'
             << "Generation defaults: Euler, Flash2; no step/block cache.\n";
    } else {
      report << "Generation skipped; original prompt/seed/settings are not inferred from latents.\n";
    }
    report << "Patch amplitude: 16px row-luma fundamental, second-difference detrended,\n"
           << "in 0..255 luma units; scene content can contribute. Not an aesthetic score.\n"
           << "edges: outer 5% of columns on each side, not guaranteed background.\n";
    if (!reference_path.empty()) {
      std::puts("Encoding reference image; bypassing denoising...");
      slopfab::SafeTensors checkpoint;
      checkpoint.open(request.video_vae_path);
      const auto* mean_tensor = checkpoint.find("latents_mean");
      const auto* std_tensor = checkpoint.find("latents_std");
      const auto mean = mean_tensor ? slopfab::to_f32(*mean_tensor)
                                   : slopfab::vae::default_video_latents_mean();
      const auto stddev = std_tensor ? slopfab::to_f32(*std_tensor)
                                    : slopfab::vae::default_video_latents_std();
      const auto image = slopfab::resize_reference_lanczos(
          slopfab::load_reference_image(reference_path), plan.canvas_width, plan.canvas_height);
      slopfab::vae::KeyframeEncoder encoder(checkpoint);
      const auto encoded = encoder.encode_reference_image(image, mean, stddev);
      latent_path = (directory / "input.safetensors").string();
      slopfab::write_safetensors(latent_path,
          {{"video_rows", {plan.layout.num_video_rows, 96}, encoded}});
      report << "reference_image=" << reference_path << " (denoising bypassed)\n";
    }
    if (latent_path.empty()) {
      slopfab::RunOptions options;
      options.dump_latents_path = (directory / "input.safetensors").string();
      options.on_progress = [](slopfab::RunStage stage, int, int, void*) {
        return stage != slopfab::RunStage::kVideoDecode;
      };
      const auto result = slopfab::run_generate(request, plan, options);
      if (!result.cancelled || !std::filesystem::exists(options.dump_latents_path))
        throw std::runtime_error("generation failed before latent capture: " + result.message);
      latent_path = options.dump_latents_path;
    }
    report << "latent_source=" << latent_path << '\n';
    slopfab::SafeTensors latent_file;
    latent_file.open(latent_path);
    const auto& rows = latent_file.at("video_rows");
    if (rows.dtype != slopfab::DType::kF32 ||
        rows.shape != std::vector<int64_t>{plan.layout.num_video_rows, 96})
      throw std::invalid_argument("latent video_rows must be F32 and match the still canvas");
    const auto packed = slopfab::to_f32(rows);
    for (float value : packed) {
      if (!std::isfinite(value)) throw std::invalid_argument("non-finite latent");
    }
    std::vector<float> latent(static_cast<size_t>(24) * plan.layout.latent_height *
                               plan.layout.latent_width);
    slopfab::dit::unpatchify_video(packed.data(), plan.layout, latent.data());
    slopfab::write_safetensors((directory / "normalized-latent.safetensors").string(),
        {{"latent", {24, 1, plan.layout.latent_height, plan.layout.latent_width}, latent}});

    std::puts("Loading VAE once for both decodes...");
    slopfab::SafeTensors checkpoint;
    checkpoint.open(request.video_vae_path);
    const auto* mean_tensor = checkpoint.find("latents_mean");
    const auto* std_tensor = checkpoint.find("latents_std");
    const auto mean = mean_tensor ? slopfab::to_f32(*mean_tensor)
                                 : slopfab::vae::default_video_latents_mean();
    const auto stddev = std_tensor ? slopfab::to_f32(*std_tensor)
                                   : slopfab::vae::default_video_latents_std();
    slopfab::vae::ViTDecoder decoder;
    decoder.load(checkpoint);
    std::vector<std::vector<double>> profiles;
    auto record = [&](const std::string& name, const slopfab::vae::DecodedVideo& image,
                      Clock::time_point started) {
      const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
      save_image(directory, name, image);
      profiles.push_back(row_profile(image, false));
      profiles.push_back(row_profile(image, true));
      const double whole = patch_amplitude(profiles[profiles.size() - 2]);
      const double edges = patch_amplitude(profiles.back());
      report << name << " seconds=" << seconds << " patch_luma=" << whole
             << " edge_patch_luma=" << edges << '\n';
      report.flush();
      std::printf("%s: %.3fs, patch luma %.5f, edge patch luma %.5f\n",
                  name.c_str(), seconds, whole, edges);
    };

    std::puts("A: default seven-token still decoder");
    auto started = Clock::now();
    const auto original = slopfab::vae::decode_still_image(
        decoder, latent.data(), plan.layout.latent_height, plan.layout.latent_width, mean, stddev);
    record("still-default", original, started);

    std::puts("B: same latent repeated seven times, normal video decoder, first retained frame");
    const auto repeated = slopfab::probe::repeat_still_latent(latent, 24, 7);
    started = Clock::now();
    const auto video = decoder.decode(repeated.data(), 7, plan.layout.latent_height,
                                      plan.layout.latent_width, mean, stddev);
    const auto temporal = slopfab::probe::first_frame(video);
    record("video-reference", temporal, started);
    double squared = 0.0;
    double absolute = 0.0;
    for (size_t index = 0; index < original.data.size(); ++index) {
      const double difference = static_cast<double>(original.data[index]) - temporal.data[index];
      squared += difference * difference;
      absolute += std::abs(difference);
    }
    report << "pair_rgb_mae=" << absolute / original.data.size()
           << " pair_rgb_rmse=" << std::sqrt(squared / original.data.size()) << '\n';
    auto csv = output_file(directory / "row-luma.csv");
    csv << std::setprecision(10) << "row,still_all,still_edges,reference_all,reference_edges";
    csv << '\n';
    for (int row = 0; row < plan.canvas_height; ++row) {
      csv << row;
      for (const auto& profile : profiles) csv << ',' << profile[row];
      csv << '\n';
    }
    std::printf("Diagnostic saved to %s\n", directory.string().c_str());
    if (original.data != temporal.data)
      throw std::runtime_error("default still output differs from seven-token video reference");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "stillprobe: %s\n", error.what());
    return 1;
  }
}
