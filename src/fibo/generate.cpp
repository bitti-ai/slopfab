#include "runtime.cuh"
#include "slopfab/generate.h"
#include "slopfab/sampler/noise.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace slopfab::fibo {
namespace {
struct Cancelled {};

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void parent(const std::string& path) {
  auto p = std::filesystem::path(path).parent_path();
  if (!p.empty())
    std::filesystem::create_directories(p);
}

RGBImage masked_source(const ImageEdit& edit, int width, int height) {
  auto image = pad_edit_image(edit, width, height);
  const bool masked = edit.mask || edit.x || edit.y || edit.width != edit.image->width ||
                      edit.height != edit.image->height || edit.feather;
  if (!masked)
    return image;
  for (int y = 0; y < edit.image->height; ++y)
    for (int x = 0; x < edit.image->width; ++x) {
      float coverage = 0;
      if (edit.mask)
        coverage = (*edit.mask)[size_t(y) * edit.image->width + x] / 255.f;
      else if (x >= edit.x && y >= edit.y && x < edit.x + edit.width && y < edit.y + edit.height) {
        coverage = 1;
        if (edit.feather)
          coverage =
              std::min(1.f, (float(std::min({x - edit.x, y - edit.y, edit.x + edit.width - 1 - x,
                                             edit.y + edit.height - 1 - y})) +
                             .5f) /
                                edit.feather);
      }
      if (coverage >= 128.f / 255)
        for (int c = 0; c < 3; ++c)
          image.pixels[(size_t(y) * width + x) * 3 + c] = 128;
    }
  return image;
}

std::vector<Tensor> load_context(const std::string& path) {
  SafeTensors st;
  st.open(path);
  const auto& t = st.at("prompt_embedding");
  if (t.shape.size() != 2 || t.shape[0] < 1 || t.shape[0] > 4096 || t.shape[1] != 46080)
    throw std::invalid_argument("Fibo prompt_embedding must have shape [1..4096,46080]");
  auto packed = from_host(to_f32(t), int(t.shape[0]), 46080);
  std::vector<Tensor> out;
  for (int i = 0; i < 6; ++i)
    out.push_back(moss::slice(packed, i * 7680, 7680));
  return out;
}
}

ImageLatent encode_image(const std::string& path, const RGBImage& image) {
  return {image.height / 16, image.width / 16,
          moss::encode_flux_image(path, image.pixels, image.height, image.width)};
}

RunResult run(const GenerateRequest& request, const GeneratePlan& plan, const RunOptions& options) {
  RunResult result;
  auto notify = [&](RunStage stage, int step = -1, int steps = 0) {
    if (options.on_progress && !options.on_progress(stage, step, steps, options.hook_userdata))
      throw Cancelled{};
  };
  try {
    validate_generation_options(request, plan, options);
    if (!options.on_samples && std::filesystem::path(request.out_path).extension() != ".ppm")
      throw std::invalid_argument("Fibo file output requires .ppm; the C API returns RGB pixels");
    notify(RunStage::kStarting);
    const int h = plan.canvas_height / 16, w = plan.canvas_width / 16, n = h * w;
    std::vector<RGBImage> images;
    if (request.image_edit.image)
      images.push_back(masked_source(request.image_edit, plan.canvas_width, plan.canvas_height));
    else
      for (const auto& path : request.reference_image_paths)
        images.push_back(
            crop_image(load_reference_image(path), plan.canvas_width, plan.canvas_height));
    notify(RunStage::kConditioning);
    auto start = Clock::now();
    std::vector<ImageLatent> refs;
    for (const auto& image : images) {
      notify(RunStage::kConditioning);
      refs.push_back(encode_image(request.video_vae_path, image));
    }
    std::vector<float> latent(size_t(n) * 128);
    sampler::fill_normal(request.seed, sampler::NoiseStream::kVideoLatents, latent.data(),
                         latent.size());
    if (plan.video_sigmas.front() < 1) {
      auto original =
          encode_image(request.video_vae_path,
                       pad_edit_image(request.image_edit, plan.canvas_width, plan.canvas_height));
      float sigma = plan.video_sigmas.front();
      for (size_t i = 0; i < latent.size(); ++i)
        latent[i] = sigma * latent[i] + (1 - sigma) * original.values[i];
    }
    {
      auto context = options.prompt_embedding_path.empty()
                         ? encode_prompt(request.text_encoder_path, request.tokenizer_path,
                                         request.prompt, images,
                                         [&] {
                                           notify(RunStage::kConditioning);
                                         })
                         : load_context(options.prompt_embedding_path);
      result.conditioner_executed = options.prompt_embedding_path.empty();
      result.seconds_conditioning = elapsed(start);
      notify(RunStage::kTransformerLoad);
      start = Clock::now();
      size_t free = 0, total = 0;
      SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free, &total));
      const bool streaming = free < std::filesystem::file_size(request.transformer_path) +
                                        size_t(3) * 1024 * 1024 * 1024;
      Runtime transformer(request.transformer_path);
      auto gist = perceiver(transformer, context[0], streaming);
      const int steps = int(plan.video_timesteps.size());
      notify(RunStage::kDenoising, -1, steps);
      for (int step = 0; step < steps; ++step) {
        auto prediction = to_host(predict(transformer, from_host(latent, n, 128), h, w, refs,
                                          context, gist, plan.video_sigmas[step], streaming, [&] {
                                            notify(RunStage::kDenoising, step - 1, steps);
                                          }));
        float dt = plan.video_sigmas[step + 1] - plan.video_sigmas[step];
        for (size_t i = 0; i < latent.size(); ++i) {
          latent[i] += dt * prediction[i];
          if (!std::isfinite(latent[i]))
            throw std::runtime_error("non-finite Fibo latent");
        }
        ++result.steps_computed;
        notify(RunStage::kDenoising, step, steps);
        if (options.verbose)
          std::printf("Fibo step   %d/%d\n", step + 1, steps);
      }
      result.seconds_denoise = result.seconds_denoise_loop = elapsed(start);
    }
    if (!options.dump_latents_path.empty()) {
      parent(options.dump_latents_path);
      write_safetensors(options.dump_latents_path, {{"image_latents", {h, w, 128}, latent}},
                        {{"slopfab.model_family", "fibo2"}});
    }
    notify(RunStage::kVideoDecode);
    start = Clock::now();
    auto decoded = moss::decode_image(request.video_vae_path, latent, h, w);
    PixelBuffer pixels(decoded.begin(), decoded.end());
    int width = plan.canvas_width, height = plan.canvas_height;
    if (request.image_edit.image) {
      pixels = composite_image_edit(request.image_edit, pixels, width, height);
      width = request.image_edit.image->width;
      height = request.image_edit.image->height;
    }
    result.seconds_video_decode = elapsed(start);
    notify(RunStage::kDelivering);
    start = Clock::now();
    RunSamples samples;
    samples.frames = 1;
    samples.height = height;
    samples.width = width;
    samples.video = &pixels;
    bool taken = options.on_samples && options.on_samples(samples, options.hook_userdata);
    if (!taken) {
      if (std::filesystem::path(request.out_path).extension() != ".ppm")
        throw std::invalid_argument("Fibo file output requires .ppm");
      parent(request.out_path);
      std::ofstream file(request.out_path, std::ios::binary);
      file << "P6\n" << width << ' ' << height << "\n255\n";
      size_t plane = size_t(width) * height;
      std::vector<uint8_t> bytes(plane * 3);
      for (size_t i = 0; i < plane; ++i)
        for (int c = 0; c < 3; ++c)
          bytes[i * 3 + c] = uint8_t(std::lround(pixels[c * plane + i] * 255));
      file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
      file.close();
      if (!file)
        throw std::runtime_error("cannot write Fibo image");
      result.outputs.push_back(request.out_path);
    }
    result.seconds_output = elapsed(start);
    notify(RunStage::kFinished);
    result.ok = true;
  } catch (const Cancelled&) {
    result.cancelled = true;
    result.message = "Fibo generation cancelled";
  } catch (const std::exception& e) {
    result.message = e.what();
  }
  return result;
}
}
