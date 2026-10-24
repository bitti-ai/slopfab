#include "runtime.cuh"
#include "slopfab/generate.h"
#include "slopfab/sampler/noise.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/tokenizer.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace slopfab::moss {
namespace {
struct Cancelled {};

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

Tensor embedding_file(const SafeTensors& st, const std::string& key) {
  const auto& t = st.at(key);
  if (t.shape.size() != 2 || t.shape[0] < 1 || t.shape[0] > 4096 || t.shape[1] != 4096)
    throw std::invalid_argument("MOSS " + key + " must have shape [1..4096,4096]");
  return from_host(to_f32(t), int(t.shape[0]), 4096);
}

std::vector<int32_t> prompt_tokens(text::Tokenizer& tokenizer, const std::string& prompt) {
  // Single user turn, generation prefix and thinking enabled, as shipped with MOSS.
  auto ids = tokenizer.encode("<|im_start|>user\n" + prompt +
                              "<|im_end|>\n<|im_start|>assistant\n<think>\n");
  if (ids.size() > 4096)
    ids.resize(4096);
  return ids;
}

void make_parent(const std::string& path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty())
    std::filesystem::create_directories(parent);
}
}

RunResult run(const GenerateRequest& request, const GeneratePlan& plan, const RunOptions& options) {
  RunResult result;
  const auto notify = [&](RunStage stage, int step = -1, int steps = 0) {
    if (options.on_progress && !options.on_progress(stage, step, steps, options.hook_userdata))
      throw Cancelled{};
  };
  try {
    validate_generation_options(request, plan, options);
    if (!options.on_samples && std::filesystem::path(request.out_path).extension() != ".ppm")
      throw std::invalid_argument("MOSS file output requires .ppm; the C API returns RGB pixels");
    notify(RunStage::kStarting);
    const int h = plan.canvas_height / 16, w = plan.canvas_width / 16, rows = h * w;
    std::vector<float> latent(size_t(rows) * 128);
    if (options.init_latents_path.empty())
      sampler::fill_normal(request.seed, sampler::NoiseStream::kVideoLatents, latent.data(),
                           latent.size());
    else {
      SafeTensors initial;
      initial.open(options.init_latents_path);
      const auto& tensor = initial.at("image_latents");
      if (tensor.shape != std::vector<int64_t>{h, w, 128})
        throw std::invalid_argument("MOSS initial image_latents must be [H/16,W/16,128]");
      latent = to_f32(tensor);
    }
    if (options.source == LatentSource::kDenoise) {
      notify(RunStage::kConditioning);
      auto start = Clock::now();
      Tensor positive, negative;
      if (!options.prompt_embedding_path.empty()) {
        SafeTensors st;
        st.open(options.prompt_embedding_path);
        positive = embedding_file(st, "prompt_embedding");
        if (request.guidance_scale != 1.f)
          negative = embedding_file(st, "negative_prompt_embedding");
      } else {
        if (request.tokenizer_path.empty())
          throw std::invalid_argument(
              "MOSS requires the Qwen3.5 tokenizer: --tokenizer FILE (or weights/text_encoder/qwen3.5_tokenizer.json)");
        text::Tokenizer tokenizer;
        tokenizer.load(request.tokenizer_path);
        if (tokenizer.encode("<|im_start|>") != std::vector<int32_t>{248045})
          throw std::invalid_argument("MOSS requires the Qwen3.5 tokenizer, not the H3 tokenizer");
        Runtime conditioner(request.text_encoder_path);
        positive = encode_text(conditioner, prompt_tokens(tokenizer, request.prompt));
        if (request.guidance_scale != 1.f)
          negative = encode_text(conditioner, prompt_tokens(tokenizer, request.negative_prompt));
        result.conditioner_executed = true;
      }
      result.seconds_conditioning = elapsed(start);
      if (options.verbose)
        std::printf("MOSS text   %d positive, %d negative tokens in %.2f s\n", positive.rows,
                    negative.rows, result.seconds_conditioning);
      notify(RunStage::kTransformerLoad);
      start = Clock::now();
      Runtime transformer(request.transformer_path);
      const int steps = int(plan.video_timesteps.size());
      for (int step = 0; step < steps; ++step) {
        notify(RunStage::kDenoising, step, steps);
        auto input = from_host(latent, rows, 128);
        auto prediction = to_host(
            image_prediction(transformer, input, positive, h, w, plan.video_timesteps[step], {}));
        if (request.guidance_scale != 1.f) {
          // Default SLG interval is [0.01,1); the first step does not skip.
          const auto skip = double(step) / steps >= 0.01 ? request.uncond_skip_layers
                                                        : std::vector<int>{};
          const auto uncond = to_host(image_prediction(transformer, input, negative, h, w,
                                                       plan.video_timesteps[step], skip));
          for (size_t i = 0; i < latent.size(); ++i)
            prediction[i] = uncond[i] + request.guidance_scale * (prediction[i] - uncond[i]);
        }
        const float dt = plan.video_sigmas[step + 1] - plan.video_sigmas[step];
        for (size_t i = 0; i < latent.size(); ++i) {
          latent[i] += dt * prediction[i];
          if (!std::isfinite(latent[i]))
            throw std::runtime_error("non-finite MOSS latent");
        }
        ++result.steps_computed;
        if (options.verbose)
          std::printf("MOSS step   %d/%d\n", step + 1, steps);
      }
      notify(RunStage::kDenoising, steps, steps);
      result.seconds_denoise = result.seconds_denoise_loop = elapsed(start);
    }
    if (!options.dump_latents_path.empty()) {
      make_parent(options.dump_latents_path);
      write_safetensors(options.dump_latents_path, {{"image_latents", {h, w, 128}, latent}},
                        {{"slopfab.model_family", "moss"}});
    }
    notify(RunStage::kVideoDecode);
    auto start = Clock::now();
    auto decoded = decode_image(request.video_vae_path, latent, h, w);
    PixelBuffer pixels(decoded.begin(), decoded.end());
    result.seconds_video_decode = elapsed(start);
    notify(RunStage::kDelivering);
    start = Clock::now();
    RunSamples samples;
    samples.frames = 1;
    samples.height = plan.canvas_height;
    samples.width = plan.canvas_width;
    samples.video = &pixels;
    const bool taken = options.on_samples && options.on_samples(samples, options.hook_userdata);
    if (!taken) {
      if (std::filesystem::path(request.out_path).extension() != ".ppm")
        throw std::invalid_argument(
            "MOSS file output currently requires .ppm; the C API returns RGB pixels");
      make_parent(request.out_path);
      std::ofstream file(request.out_path, std::ios::binary);
      file << "P6\n" << samples.width << ' ' << samples.height << "\n255\n";
      const size_t plane = size_t(samples.width) * samples.height;
      std::vector<uint8_t> bytes(plane * 3);
      for (size_t i = 0; i < plane; ++i)
        for (size_t c = 0; c < 3; ++c)
          bytes[i * 3 + c] = uint8_t(std::lround(pixels[c * plane + i] * 255.f));
      file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
      file.close();
      if (!file)
        throw std::runtime_error("cannot write MOSS image: " + request.out_path);
      result.outputs.push_back(request.out_path);
    }
    result.seconds_output = elapsed(start);
    notify(RunStage::kFinished);
    result.ok = true;
  } catch (const Cancelled&) {
    result.cancelled = true;
    result.message = "MOSS generation cancelled";
  } catch (const std::exception& e) {
    result.message = e.what();
  }
  return result;
}
} // namespace slopfab::moss
