#include "vidfab/generate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "vidfab/audio/wav.h"
#include "vidfab/image.h"
#include "vidfab/cuda/profile.h"
#include "vidfab/dit/denoise.h"
#include "vidfab/dit/checkpoint.h"
#include "vidfab/dit/packing.h"
#include "vidfab/dit/ref2va.h"
#include "vidfab/dit/transformer.h"
#include "vidfab/text/encoder.h"
#include "vidfab/text/tokenizer.h"
#include "vidfab/sampler/scheduler.h"
#include "vidfab/safetensors.h"
#include "vidfab/safetensors_write.h"
#include "vidfab/sampler/noise.h"
#include "vidfab/tensor_convert.h"
#include "vidfab/vae/audio_decoder.h"
#include "vidfab/vae/vit_decoder.h"
#include "vidfab/vae/keyframe_encoder.h"
#include "vidfab/video/mux.h"
#include "vidfab/video/y4m.h"

namespace vidfab {
namespace {

std::vector<uint8_t> resize_rgb_bilinear(const RGBImage& in, int width, int height) {
  std::vector<uint8_t> out(static_cast<size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
    const float sy = (y + .5f) * in.height / height - .5f;
    const float sx = (x + .5f) * in.width / width - .5f;
    const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, in.height - 1);
    const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, in.width - 1);
    const int y1 = std::min(y0 + 1, in.height - 1), x1 = std::min(x0 + 1, in.width - 1);
    const float fy = std::clamp(sy - std::floor(sy), 0.0f, 1.0f);
    const float fx = std::clamp(sx - std::floor(sx), 0.0f, 1.0f);
    for (int c = 0; c < 3; ++c) {
      auto at=[&](int yy,int xx){return in.pixels[(static_cast<size_t>(yy)*in.width+xx)*3+c];};
      const float v=(1-fy)*((1-fx)*at(y0,x0)+fx*at(y0,x1))+fy*((1-fx)*at(y1,x0)+fx*at(y1,x1));
      out[(static_cast<size_t>(y)*width+x)*3+c]=static_cast<uint8_t>(std::clamp(std::lround(v),0l,255l));
    }
  } return out;
}

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string strip_extension(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  const size_t dot = path.find_last_of('.');
  if (dot == std::string::npos) return path;
  if (slash != std::string::npos && dot < slash) return path;
  return path.substr(0, dot);
}

// The two VAEs both ship per-channel latent statistics as tensors. Prefer them
// over the copies in the config JSON: the tensors are what the checkpoint
// actually carries, and a config file can drift from the weights beside it.
std::vector<float> read_stat(const SafeTensors& st, const char* name, int expect) {
  const TensorView* found = st.find(name);
  if (found == nullptr && expect == 24) {
    if (std::strcmp(name, "latents_mean") == 0) return vae::default_video_latents_mean();
    if (std::strcmp(name, "latents_std") == 0) return vae::default_video_latents_std();
  }
  const TensorView& view = found ? *found : st.at(name);
  std::vector<float> out = to_f32(view);
  if (static_cast<int>(out.size()) != expect) {
    throw std::runtime_error(std::string("vae: ") + name + " has " + std::to_string(out.size()) +
                             " entries, expected " + std::to_string(expect));
  }
  return out;
}

struct ReusedGenerationModels {
  std::string conditioning_key;
  text::PromptEmbedding prompt;

  void clear() {
    conditioning_key.clear();
    prompt = {};
  }
};

ReusedGenerationModels& reused_models() {
  static ReusedGenerationModels models;
  return models;
}

std::string conditioning_cache_key(const GenerateRequest& request) {
  std::string key = request.text_encoder_path;
  key.push_back('\0');
  key += request.tokenizer_path;
  key.push_back('\0');
  key += request.prompt;
  for (const std::string& path : request.reference_image_paths) {
    key.push_back('\0');
    key += path;
  }
  return key;
}

}  // namespace

RunResult run_generate(const GenerateRequest& request, const GeneratePlan& plan,
                       const RunOptions& options) {
  struct ReuseReleaseGuard {
    bool release = false;
    ~ReuseReleaseGuard() {
      if (release) reused_models().clear();
    }
  } release_guard{options.release_reused_models};
  RunResult result;
  const dit::SequenceLayout& layout = plan.layout;

  // Decode all references before opening a multi-gigabyte checkpoint. Besides
  // giving file errors promptly, this validates the Ref2VA aspect contract at
  // the dimensions actually presented by the decoder.
  std::vector<RGBImage> reference_images;
  reference_images.reserve(request.reference_image_paths.size());
  try {
    for (const std::string& path : request.reference_image_paths) {
      RGBImage image = load_reference_image(path);
      if (static_cast<int64_t>(image.width) > 4LL * image.height ||
          static_cast<int64_t>(image.height) > 4LL * image.width) {
        result.message = "reference image '" + path + "' must be within 1:4 and 4:1, got " +
                         std::to_string(image.width) + "x" + std::to_string(image.height);
        return result;
      }
      if (options.verbose) {
        std::printf("reference   %s (%dx%d)\n", path.c_str(), image.width, image.height);
      }
      int resized_h = 0, resized_w = 0;
      dit::resolve_reference_image_size(image.width, image.height, &resized_h, &resized_w);
      image = resize_reference_lanczos(image, resized_w, resized_h);
      reference_images.push_back(std::move(image));
    }
  } catch (const std::exception& e) {
    result.message = e.what();
    return result;
  }

  // --- latents ---------------------------------------------------------------

  std::vector<float> video_rows;  // [V, 96]
  std::vector<float> audio_rows;  // [Sa, 32]

  // Supplied initial latents, if any. Read before anything expensive happens so
  // a wrong shape fails in a second rather than after 25 GB of conditioner.
  std::vector<float> init_video;
  std::vector<float> init_audio;
  if (!options.init_latents_path.empty()) {
    SafeTensors file;
    file.open(options.init_latents_path);
    init_video = to_f32(file.at("video_rows"));
    init_audio = to_f32(file.at("audio_rows"));
    const size_t want_video = static_cast<size_t>(layout.num_video_rows) * 96;
    const size_t want_audio = static_cast<size_t>(layout.num_audio_rows) * 32;
    if (init_video.size() != want_video || init_audio.size() != want_audio) {
      result.message = "--init-latents " + options.init_latents_path + " holds " +
                       std::to_string(init_video.size()) + " video and " +
                       std::to_string(init_audio.size()) + " audio floats; this geometry wants " +
                       std::to_string(want_video) + " and " + std::to_string(want_audio);
      return result;
    }
    if (options.verbose) {
      std::printf("latents     %s replaces the seeded draw (%d video rows, %d audio rows)\n",
                  options.init_latents_path.c_str(), layout.num_video_rows, layout.num_audio_rows);
    }
  }

  if (options.source == LatentSource::kDenoise) {
    if (request.text_encoder_path.empty() || request.transformer_path.empty()) {
      result.message =
          "generate needs --text-encoder and --transformer (or pass "
          "--synthetic-latents to skip conditioning and denoising)";
      return result;
    }

    // Ref2VA and the pruned T2VA/FL2VA transformer share most tensor names but
    // have incompatible timestep/AdaLN graphs. Check the cheap header contract
    // before loading the 15+ GiB conditioner so a wrong --transformer fails in
    // milliseconds rather than after an otherwise successful text encode.
    if (!reference_images.empty()) {
      try {
        SafeTensors transformer_header;
        transformer_header.open(request.transformer_path);
        dit::require_ref2va_transformer(transformer_header, reference_images.size());
      } catch (const std::exception& e) {
        result.message = e.what();
        return result;
      }
    }

    // --- fixed image anchors ------------------------------------------------
    std::vector<float> condition_video_rows;
    std::vector<dit::ReferenceGeometry> reference_geometry;
    if (!reference_images.empty()) {
      if (request.video_vae_path.empty()) {
        result.message = "--reference-image requires --video-vae for H3 image encoding";
        return result;
      }
      const Clock::time_point t0 = Clock::now();
      SafeTensors vae_file;
      vae_file.open(request.video_vae_path);
      const std::vector<float> mean = read_stat(vae_file, "latents_mean", 24);
      const std::vector<float> stddev = read_stat(vae_file, "latents_std", 24);
      vae::KeyframeEncoder image_encoder(vae_file);
      for (size_t reference_index = 0; reference_index < reference_images.size();
           ++reference_index) {
        const RGBImage& image = reference_images[reference_index];
        std::vector<float> rows = image_encoder.encode_reference_image(image, mean, stddev);
        // Released Ref2VA anchors are almost clean, but not quite: the fixed
        // timestep is 0.999 and the request generator contributes the other
        // 0.001. Keep each ordered reference on its own deterministic stream.
        const int latent_h = image.height / 16;
        const int latent_w = image.width / 16;
        const uint64_t reference_seed =
            request.seed ^ (0x9e3779b97f4a7c15ULL * (reference_index + 1));
        const std::vector<float> noise_latents =
            sampler::video_noise(reference_seed, 1, latent_h, latent_w);
        const std::vector<float> noise_rows =
            vae::patchify_keyframe_latents(noise_latents.data(), latent_h, latent_w);
        sampler::FlowScheduler::scale_noise(rows.data(), noise_rows.data(), 0.999f,
                                            rows.size(), rows.data());
        condition_video_rows.insert(condition_video_rows.end(), rows.begin(), rows.end());
        reference_geometry.push_back({dit::ReferenceKind::kImage, 1, image.height / 16,
                                      image.width / 16, 0});
      }
      if (options.verbose)
        std::printf("references  %zu images -> %zu fixed video rows in %.2f s\n",
                    reference_images.size(), condition_video_rows.size() / 96,
                    seconds_since(t0));
    }

    // --- conditioning -------------------------------------------------------
    //
    // The encoder is loaded, used and freed before the transformer is touched.
    // The scoping below is the enforcement: `encoder` and its checkpoint
    // mapping both die at the closing brace, and the transformer is not
    // constructed until after it.
    //
    // The int8 conditioner and the fp8 transformer cannot co-exist on a 32 GB
    // card — measured, 23.1 GB peak and 19.3 GB. The nvfp4 pair can (13.1 and
    // 12.5), so the sequencing is no longer forced for that combination, but it
    // stays: a resident encode is 0.12 s against a whole denoising run, the
    // saving would be nothing, and dropping it would make three of the four
    // checkpoint combinations fail at the worst possible moment.
    text::PromptEmbedding prompt;
    ReusedGenerationModels& reuse = reused_models();
    const std::string prompt_key = conditioning_cache_key(request);
    if (options.reuse_models && reuse.conditioning_key == prompt_key &&
        !reuse.prompt.data.empty()) {
      prompt = reuse.prompt;
      if (options.verbose) {
        std::printf("prompt      reused cached [%d, %d] conditioning\n", prompt.num_tokens,
                    prompt.hidden_size);
      }
    } else {
      const Clock::time_point t0 = Clock::now();
      text::Tokenizer tokenizer;
      if (request.tokenizer_path.empty()) tokenizer.load_embedded();
      else tokenizer.load(request.tokenizer_path);

      // No chat template, no BOS, no EOS: `hidden_states[50]` of a raw prompt
      // is the conditioning H3 expects, and a special token here would shift
      // every rotary position downstream (spec 1.2).
      std::vector<int32_t> ids;
      std::vector<text::QwenPixelValues> qwen_images;
      for (size_t i = 0; i < reference_images.size(); ++i) {
        const auto grid = text::qwen3vl_image_grid(reference_images[i].width,
                                                    reference_images[i].height);
        const int rw = grid.width * 16, rh = grid.height * 16;
        auto rgb = resize_rgb_bilinear(reference_images[i], rw, rh);
        qwen_images.push_back(text::qwen3vl_patchify_resized_rgb(rgb, rw, rh));
        const auto label = tokenizer.encode("<Picture " + std::to_string(i + 1) + ">: ");
        const auto block = text::qwen3vl_image_block(label, grid.merged_token_count());
        ids.insert(ids.end(), block.begin(), block.end());
      }
      const auto prompt_ids = tokenizer.encode(request.prompt);
      ids.insert(ids.end(), prompt_ids.begin(), prompt_ids.end());
      if (ids.empty()) {
        result.message = "the prompt tokenised to zero tokens";
        return result;
      }

      SafeTensors encoder_file;
      encoder_file.open(request.text_encoder_path);
      try {
        text::require_reference_vision_support(encoder_file, reference_images.size());
      } catch (const std::exception& e) {
        result.message = e.what();
        return result;
      }
      text::Encoder encoder;
      text::EncoderConfig ecfg;
      ecfg.residency = text::Residency::kStreaming;
      encoder.load(encoder_file, ecfg);
      prompt = qwen_images.empty() ? encoder.encode(ids) : encoder.encode(ids, qwen_images);
      encoder.unload();
      if (options.reuse_models) {
        reuse.conditioning_key = prompt_key;
        reuse.prompt = prompt;
      }
      result.seconds_conditioning = seconds_since(t0);
      if (options.verbose) {
        std::printf("prompt      %d tokens -> [%d, %d] in %.2f s (%s residency)\n",
                    static_cast<int>(ids.size()), prompt.num_tokens, prompt.hidden_size,
                    result.seconds_conditioning,
                    encoder.residency() == text::Residency::kStreaming ? "streaming" : "resident");
      }
    }

    // --- denoise ------------------------------------------------------------
    //
    // The layout only now knows its text length, so the packed sequence and its
    // rotary coordinates are built here rather than in `resolve_plan`.
    dit::SequenceLayout live = layout;
    dit::PackedIndices idx;
    std::vector<double> pos;
    if (reference_geometry.empty()) {
      live.num_text = prompt.num_tokens;
      idx = dit::build_indices(live);
      pos = dit::build_position_ids(live);
    } else {
      dit::Ref2VAPackedSequence packed = dit::build_ref2va_packed_sequence(
          prompt.modality_tags, reference_geometry, live.num_latent_frames, live.latent_height,
          live.latent_width, live.num_audio_latents);
      live = std::move(packed.layout);
      idx = std::move(packed.indices);
      pos = std::move(packed.position_ids);
    }

    {
      const Clock::time_point t0 = Clock::now();
      SafeTensors dit_file;
      dit_file.open(request.transformer_path);
      dit::Transformer model;
      model.load(dit_file);
      result.seconds_transformer_load = seconds_since(t0);
      if (options.verbose) {
        std::printf("transformer %.2f GiB on device, %d packed rows, loaded in %.2f s\n",
                    static_cast<double>(model.weight_bytes()) / (1024.0 * 1024.0 * 1024.0),
                    live.total_rows(), result.seconds_transformer_load);
      }
      const Clock::time_point t_prep = Clock::now();
      // Before prepare_sequence: that is where the per-query-tile key ranges are
      // built, and they depend on the band.
      model.set_attention_band(options.attention_band);
      model.set_attention_mode(options.attention_mode);
      model.set_sol_schedule(options.sol_schedule);
      if (options.attention_band > 0 && options.verbose) {
        std::printf("attention  frame band +/-%d latent frames (lossy, changes the sample)\n",
                    options.attention_band);
      }
      if (options.verbose) {
        std::printf("attention  backend %s\n", attention_mode_name(options.attention_mode));
      }
      model.prepare_text(prompt.data.data(), prompt.num_tokens);
      model.prepare_sequence(live, idx, pos);
      result.seconds_prepare = seconds_since(t_prep);

      sampler::FlowScheduler video_sched(12.0f);
      sampler::FlowScheduler audio_sched(3.0f);
      video_sched.set_timesteps(request.num_inference_steps);
      audio_sched.set_timesteps(request.num_inference_steps);
      video_sched.set_sampler(options.sampler);
      audio_sched.set_sampler(options.sampler);

      dit::DenoiseInputs in;
      in.layout = &live;
      in.indices = &idx;
      in.video_timesteps = &plan.video_timesteps;
      in.audio_timesteps = &plan.audio_timesteps;
      in.video_scheduler = &video_sched;
      in.audio_scheduler = &audio_sched;
      in.seed = request.seed;
      if (!condition_video_rows.empty()) in.condition_video_rows = &condition_video_rows;
      if (!options.init_latents_path.empty()) {
        in.init_video_rows = &init_video;
        in.init_audio_rows = &init_audio;
      }
      in.cache.threshold = request.cache_threshold;
      in.cache.warmup = request.cache_warmup;
      in.cache.skip_every = request.skip_every;

      const int total_steps = plan.num_model_evaluations();
      // Say something before the first step rather than after it. At the
      // default geometry a step is ~19 s, so a silent half-minute is otherwise the
      // user's first impression and it reads as a hang.
      if (options.verbose) {
        std::printf("denoising   %d steps over %d rows; the first step sets the pace\n",
                    total_steps, live.total_rows());
        // Said only when it is not the default, so a run that looks like every
        // other run is one, and neither an ab2 run nor a cached one is ever
        // mistaken for a baseline.
        if (options.sampler == sampler::SamplerKind::kAb2) {
          std::printf("sampler     ab2 (Adams-Bashforth 2; step 1 is Euler)\n");
        }
        if (in.cache.enabled()) {
          if (in.cache.skip_every > 0) {
            std::printf("step cache  fixed interval: every %d-th step evaluated, warmup %d\n",
                        in.cache.skip_every, std::max(dit::kMinWarmup, in.cache.warmup));
          } else {
            std::printf(
                "step cache  threshold %.4g (accumulated relative-L1 of c(t)), warmup %d\n",
                static_cast<double>(in.cache.threshold),
                std::max(dit::kMinWarmup, in.cache.warmup));
          }
        }
        std::fflush(stdout);
      }
      const Clock::time_point loop_start = Clock::now();
      const dit::DenoiseOutputs out = dit::denoise(model, in, [&](int step, int steps) {
        if (options.verbose) {
          const double elapsed = seconds_since(loop_start);
          const double per_step = elapsed / static_cast<double>(step + 1);
          std::printf("\rstep %d/%d  %.1f s/step  eta %.0f s      ", step + 1, steps, per_step,
                      per_step * (steps - step - 1));
          std::fflush(stdout);
        }
        return true;
      });
      if (options.verbose) std::printf("\n");
      result.seconds_denoise_loop = seconds_since(loop_start);
      result.steps_computed = out.steps_computed;
      result.steps_skipped = out.steps_skipped;
      cuda::StepProfiler::instance().report(stdout);
      // Always printed when anything was reused, verbose or not: a run whose
      // skip count is invisible cannot be compared against another one.
      if (out.steps_skipped != 0) {
        std::printf("step cache  %d of %d evaluations skipped (%d computed), %.1f%%\n",
                    out.steps_skipped, out.steps_computed + out.steps_skipped, out.steps_computed,
                    100.0 * out.steps_skipped /
                        static_cast<double>(std::max(1, out.steps_computed + out.steps_skipped)));
      }

      video_rows = out.video_rows;
      audio_rows = out.audio_rows;
      // The video VAE follows immediately and can be ~9 GiB. Keeping the
      // transformer resident here makes counted runs page GPU memory and can
      // turn the next generation dramatically slower than the first.
      model.unload();

      // Latent statistics, because a wrong level downstream is ambiguous
      // between "the decoder's gain is off" and "the latents never got
      // denoised". Both VAEs were trained on normalised latents, so a
      // converged sample should land near mean 0, std 1 here; anything far
      // from that says the problem is upstream of the decoder.
      if (options.verbose) {
        auto stats = [](const std::vector<float>& v, const char* name) {
          if (v.empty()) return;
          double sum = 0.0;
          for (float x : v) sum += x;
          const double mean = sum / static_cast<double>(v.size());
          double var = 0.0;
          for (float x : v) var += (x - mean) * (x - mean);
          var /= static_cast<double>(v.size());
          std::printf("latents     %-5s mean %+.4f  std %.4f  (expect ~0, ~1)\n", name, mean,
                      std::sqrt(var));
        };
        stats(video_rows, "video");
        stats(audio_rows, "audio");
      }
      result.seconds_denoise = seconds_since(t0);
      if (options.verbose) {
        std::printf("denoised    %d steps in %.1f s (%.2f s/step); +%.1f s load, +%.1f s prepare\n",
                    total_steps, result.seconds_denoise_loop,
                    result.seconds_denoise_loop / std::max(1, total_steps),
                    result.seconds_transformer_load, result.seconds_prepare);
      }
    }
  } else if (!options.init_latents_path.empty()) {
    // Decode a latent that already exists. The whole back half — unpatchify,
    // both VAEs, the colour transform, the muxer — is the same code the
    // denoising path uses, which is the reason this is a flag on `generate`
    // rather than a second decoder that could drift from it.
    video_rows = std::move(init_video);
    audio_rows = std::move(init_audio);
  } else {
    // Seeded noise in exactly the shapes the denoiser would have produced, so
    // nothing downstream can tell the difference.
    const Clock::time_point t0 = Clock::now();
    const std::vector<float> video_latents =
        sampler::video_noise(request.seed, layout.num_latent_frames, layout.latent_height,
                             layout.latent_width);
    video_rows.resize(static_cast<size_t>(layout.num_video_rows) * 96);
    dit::patchify_video(video_latents.data(), layout, video_rows.data());
    audio_rows = sampler::audio_noise(request.seed, layout.num_audio_latents);
    result.seconds_denoise = seconds_since(t0);
  }

  // The denoiser's output, before either VAE. Written from both branches on
  // purpose: `--synthetic-latents` then dumps seeded noise, which is the
  // control that says the dump itself is deterministic.
  if (!options.dump_latents_path.empty()) {
    write_safetensors(options.dump_latents_path,
                      {{"video_rows", {layout.num_video_rows, 96}, video_rows},
                       {"audio_rows", {layout.num_audio_rows, 32}, audio_rows}});
    if (options.verbose) {
      std::printf("wrote       %s (denoiser output, fp32)\n", options.dump_latents_path.c_str());
    }
  }

  // --- video ----------------------------------------------------------------

  vae::DecodedVideo video;
  {
    const Clock::time_point t0 = Clock::now();
    if (request.video_vae_path.empty()) {
      result.message = "generate needs --vae <video_vae.safetensors>";
      return result;
    }

    // Rows back to a latent volume, then de-normalise per channel. The
    // multiply-then-add order is the reference's (decoders.py:107).
    std::vector<float> latents(static_cast<size_t>(24) * layout.num_latent_frames *
                               layout.latent_height * layout.latent_width);
    dit::unpatchify_video(video_rows.data(), layout, latents.data());

    SafeTensors vae_file;
    vae_file.open(request.video_vae_path);
    const std::vector<float> mean = read_stat(vae_file, "latents_mean", 24);
    const std::vector<float> std_dev = read_stat(vae_file, "latents_std", 24);

    vae::ViTDecoder decoder;
    decoder.load(vae_file);
    if (options.verbose) {
      std::printf("video vae   %.2f GiB on device\n",
                  static_cast<double>(decoder.weight_bytes()) / (1024.0 * 1024.0 * 1024.0));
    }
    video = decoder.decode(latents.data(), layout.num_latent_frames, layout.latent_height,
                           layout.latent_width, mean, std_dev);
    result.seconds_video_decode = seconds_since(t0);
    if (options.verbose) {
      std::printf("video       %d frames of %dx%d in %.2f s\n", video.frames, video.width,
                  video.height, result.seconds_video_decode);
    }
  }

  // --- audio ----------------------------------------------------------------

  vae::DecodedAudio audio;
  if (!request.audio_vae_path.empty()) {
    const Clock::time_point t0 = Clock::now();

    // (Sa, 32) rows -> (2, 32, A), then de-normalise per channel.
    std::vector<float> audio_latents(audio_rows.size());
    dit::unpack_audio(audio_rows.data(), layout.num_audio_latents, audio_latents.data());

    SafeTensors audio_file;
    audio_file.open(request.audio_vae_path);
    vae::AudioDecoder decoder;
    decoder.load(audio_file);

    const std::vector<float>& mean = decoder.latents_mean();
    const std::vector<float>& std_dev = decoder.latents_std();
    const int A = layout.num_audio_latents;
    for (int c = 0; c < 2; ++c) {
      for (int ch = 0; ch < 32; ++ch) {
        const float m = mean[static_cast<size_t>(ch)];
        const float s = std_dev[static_cast<size_t>(ch)];
        float* row = audio_latents.data() + (static_cast<size_t>(c) * 32 + ch) * A;
        for (int a = 0; a < A; ++a) row[a] = row[a] * s + m;
      }
    }

    audio = decoder.decode(audio_latents.data(), A);
    result.seconds_audio_decode = seconds_since(t0);
    if (options.verbose) {
      std::printf("audio       %lld frames at %d Hz in %.2f s\n",
                  static_cast<long long>(audio.num_frames()), audio.sample_rate,
                  result.seconds_audio_decode);
    }
  } else if (options.verbose) {
    std::printf("audio       skipped (no --audio-vae)\n");
  }

  // --- output ---------------------------------------------------------------

  {
    const Clock::time_point t0 = Clock::now();
    const bool have_audio = !audio.samples.empty();

    bool muxed = false;
    if (!request.raw_output) {
      std::string detail;
      if (video::ffmpeg_available(&detail)) {
        video::MuxRequest mux;
        mux.path = request.out_path;
        mux.video = &video.data;
        mux.frames = video.frames;
        mux.height = video.height;
        mux.width = video.width;
        mux.audio = have_audio ? &audio.samples : nullptr;
        mux.audio_channels = audio.channels;
        mux.audio_sample_rate = audio.sample_rate;

        const video::MuxStatus status = video::write_mp4(mux);
        if (status == video::MuxStatus::kOk) {
          muxed = true;
          result.outputs.push_back(request.out_path);
          if (options.verbose) {
            std::printf("muxed       %s (ffmpeg %s)\n", request.out_path.c_str(),
                        video::ffmpeg_version().c_str());
          }
        } else if (options.verbose) {
          std::printf("mux failed  %s; falling back to raw output\n",
                      video::mux_status_message(status));
        }
      } else if (options.verbose) {
        std::printf("no ffmpeg   %s; writing raw output\n", detail.c_str());
      }
    }

    // The fallback is not a degraded mode so much as the honest one: .y4m and
    // .wav put the samples on disk with nothing between them and the eye or
    // ear, so a wrong decode looks and sounds wrong instead of being masked by
    // a codec. mpv, VLC and ffmpeg all read both directly.
    if (!muxed) {
      const std::string base = strip_extension(request.out_path);
      const std::string y4m = base + ".y4m";
      video::write_y4m(y4m, video.data, video.frames, video.height, video.width);
      result.outputs.push_back(y4m);
      if (have_audio) {
        const std::string wav = base + ".wav";
        audio::write_wav(wav, audio.samples, audio.channels, audio.sample_rate);
        result.outputs.push_back(wav);
      }
      if (options.verbose) {
        for (const std::string& p : result.outputs) std::printf("wrote       %s\n", p.c_str());
      }
    }
    result.seconds_output = seconds_since(t0);
    if (options.verbose) std::printf("output      %.2f s\n", result.seconds_output);
  }

  result.ok = true;
  return result;
}

}  // namespace vidfab
