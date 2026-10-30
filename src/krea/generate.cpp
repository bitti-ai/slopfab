#include "runtime.cuh"
#include "slopfab/generate.h"
#include "slopfab/krea.h"
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

namespace slopfab::krea {
namespace {
struct Cancelled {};

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

float bf(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

void parent(const std::string& path) {
  auto p = std::filesystem::path(path).parent_path();
  if (!p.empty())
    std::filesystem::create_directories(p);
}

Tensor load_context(const std::string& path) {
  SafeTensors st;
  st.open(path);
  const auto& t = st.at("prompt_embedding");
  if (t.shape.size() != 3 || t.shape[0] < 1 || t.shape[0] > 512 || t.shape[1] != 12 ||
      t.shape[2] != 2560)
    throw std::invalid_argument(
        "Krea 2 prompt_embedding must have shape [1..512,12,2560], valid tokens only");
  return from_host(to_f32(t), int(t.shape[0]) * 12, 2560);
}
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
      throw std::invalid_argument("Krea 2 file output requires .ppm; the C API returns RGB pixels");
    notify(RunStage::kStarting);
    const int h = plan.canvas_height / 16, w = plan.canvas_width / 16, n = h * w;
    notify(RunStage::kConditioning);
    auto start = Clock::now();
    std::vector<float> latent(size_t(n) * 64);
    sampler::fill_normal(request.seed, sampler::NoiseStream::kVideoLatents, latent.data(),
                         latent.size());
    for (auto& v : latent)
      v = bf(v);
    {
      auto context = options.prompt_embedding_path.empty()
                         ? fibo::encode_krea_prompt(request.text_encoder_path,
                                                    request.tokenizer_path, request.prompt,
                                                    [&] {
                                                      notify(RunStage::kConditioning);
                                                    })
                         : load_context(options.prompt_embedding_path);
      Tensor negative;
      const float guidance = krea_guidance(request);
      if (guidance > 0)
        negative = fibo::encode_krea_prompt(request.text_encoder_path, request.tokenizer_path,
                                            request.negative_prompt, [&] {
                                              notify(RunStage::kConditioning);
                                            });
      result.conditioner_executed = options.prompt_embedding_path.empty() || guidance > 0;
      result.seconds_conditioning = elapsed(start);
      notify(RunStage::kTransformerLoad);
      start = Clock::now();
      size_t free = 0, total = 0;
      SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free, &total));
      const bool streaming = free < std::filesystem::file_size(request.transformer_path) +
                                        size_t(4) * 1024 * 1024 * 1024;
      Runtime transformer(request.transformer_path);
      auto text = fuse_text(transformer, context);
      context = Tensor();
      Tensor uncond;
      if (guidance > 0)
        uncond = fuse_text(transformer, negative);
      negative = Tensor();
      const int steps = int(plan.video_timesteps.size());
      notify(RunStage::kDenoising, -1, steps);
      for (int step = 0; step < steps; ++step) {
        const auto check = [&] {
          notify(RunStage::kDenoising, step - 1, steps);
        };
        auto prediction = to_host(predict(transformer, from_host(latent, n, 64), text, h, w,
                                          plan.video_sigmas[step], streaming, check));
        if (guidance > 0) {
          auto u = to_host(predict(transformer, from_host(latent, n, 64), uncond, h, w,
                                   plan.video_sigmas[step], streaming, check));
          // Upstream CFG is cond + guidance * (cond - uncond).
          for (size_t i = 0; i < prediction.size(); ++i)
            prediction[i] = bf(prediction[i] + bf(guidance * bf(prediction[i] - u[i])));
        }
        float dt = plan.video_sigmas[step + 1] - plan.video_sigmas[step];
        for (size_t i = 0; i < latent.size(); ++i) {
          latent[i] = bf(latent[i] + bf(dt * prediction[i]));
          if (!std::isfinite(latent[i]))
            throw std::runtime_error("non-finite Krea 2 latent");
        }
        ++result.steps_computed;
        notify(RunStage::kDenoising, step, steps);
        if (options.verbose)
          std::printf("Krea 2 step %d/%d\n", step + 1, steps);
      }
      result.seconds_denoise = result.seconds_denoise_loop = elapsed(start);
    }
    if (!options.dump_latents_path.empty()) {
      parent(options.dump_latents_path);
      write_safetensors(options.dump_latents_path, {{"image_latents", {h, w, 64}, latent}},
                        {{"slopfab.model_family", "krea2"}});
    }
    notify(RunStage::kVideoDecode);
    start = Clock::now();
    auto decoded = decode_image(request.video_vae_path, latent, h, w);
    PixelBuffer pixels(decoded.begin(), decoded.end());
    int width = plan.canvas_width, height = plan.canvas_height;
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
        throw std::invalid_argument("Krea 2 file output requires .ppm");
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
        throw std::runtime_error("cannot write Krea 2 image");
      result.outputs.push_back(request.out_path);
    }
    result.seconds_output = elapsed(start);
    notify(RunStage::kFinished);
    result.ok = true;
  } catch (const Cancelled&) {
    result.cancelled = true;
    result.message = "Krea 2 generation cancelled";
  } catch (const std::exception& e) {
    result.message = e.what();
  }
  return result;
}
}
