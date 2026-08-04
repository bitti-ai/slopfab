#include "vidfab/generate.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "vidfab/audio/wav.h"
#include "vidfab/dit/denoise.h"
#include "vidfab/dit/packing.h"
#include "vidfab/dit/transformer.h"
#include "vidfab/text/encoder.h"
#include "vidfab/text/tokenizer.h"
#include "vidfab/sampler/scheduler.h"
#include "vidfab/safetensors.h"
#include "vidfab/sampler/noise.h"
#include "vidfab/tensor_convert.h"
#include "vidfab/vae/audio_decoder.h"
#include "vidfab/vae/vit_decoder.h"
#include "vidfab/video/mux.h"
#include "vidfab/video/y4m.h"

namespace vidfab {
namespace {

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
  const TensorView& view = st.at(name);
  std::vector<float> out = to_f32(view);
  if (static_cast<int>(out.size()) != expect) {
    throw std::runtime_error(std::string("vae: ") + name + " has " + std::to_string(out.size()) +
                             " entries, expected " + std::to_string(expect));
  }
  return out;
}

}  // namespace

RunResult run_generate(const GenerateRequest& request, const GeneratePlan& plan,
                       const RunOptions& options) {
  RunResult result;
  const dit::SequenceLayout& layout = plan.layout;

  // --- latents ---------------------------------------------------------------

  std::vector<float> video_rows;  // [V, 96]
  std::vector<float> audio_rows;  // [Sa, 32]

  if (options.source == LatentSource::kDenoise) {
    if (request.text_encoder_path.empty() || request.tokenizer_path.empty() ||
        request.transformer_path.empty()) {
      result.message =
          "generate needs --text-encoder, --tokenizer and --transformer (or pass "
          "--synthetic-latents to skip conditioning and denoising)";
      return result;
    }

    // --- conditioning -------------------------------------------------------
    //
    // The conditioner and the transformer cannot co-exist on a 32 GB card
    // (24.4 GB and 19.3 GB of weights), so the encoder is loaded, used and
    // freed before the transformer is touched. The scoping below is the
    // enforcement: `encoder` and its checkpoint mapping both die at the closing
    // brace, and the transformer is not constructed until after it.
    text::PromptEmbedding prompt;
    {
      const Clock::time_point t0 = Clock::now();
      text::Tokenizer tokenizer;
      tokenizer.load(request.tokenizer_path);

      // No chat template, no BOS, no EOS: `hidden_states[50]` of a raw prompt
      // is the conditioning H3 expects, and a special token here would shift
      // every rotary position downstream (spec 1.2).
      const std::vector<int32_t> ids = tokenizer.encode(request.prompt);
      if (ids.empty()) {
        result.message = "the prompt tokenised to zero tokens";
        return result;
      }

      SafeTensors encoder_file;
      encoder_file.open(request.text_encoder_path);
      text::Encoder encoder;
      text::EncoderConfig ecfg;
      ecfg.residency = text::Residency::kStreaming;
      encoder.load(encoder_file, ecfg);
      prompt = encoder.encode(ids);
      encoder.unload();
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
    live.num_text = prompt.num_tokens;
    const dit::PackedIndices idx = dit::build_indices(live);
    const std::vector<double> pos = dit::build_position_ids(live);

    {
      const Clock::time_point t0 = Clock::now();
      SafeTensors dit_file;
      dit_file.open(request.transformer_path);
      dit::Transformer model;
      model.load(dit_file);
      if (options.verbose) {
        std::printf("transformer %.2f GiB on device, %d packed rows\n",
                    static_cast<double>(model.weight_bytes()) / (1024.0 * 1024.0 * 1024.0),
                    live.total_rows());
      }
      model.prepare_text(prompt.data.data(), prompt.num_tokens);
      model.prepare_sequence(live, idx, pos);

      sampler::FlowScheduler video_sched(12.0f);
      sampler::FlowScheduler audio_sched(3.0f);
      video_sched.set_timesteps(request.num_inference_steps);
      audio_sched.set_timesteps(request.num_inference_steps);

      dit::DenoiseInputs in;
      in.layout = &live;
      in.indices = &idx;
      in.video_timesteps = &plan.video_timesteps;
      in.audio_timesteps = &plan.audio_timesteps;
      in.video_scheduler = &video_sched;
      in.audio_scheduler = &audio_sched;
      in.seed = request.seed;

      const int total_steps = plan.num_model_evaluations();
      // Say something before the first step rather than after it. At the
      // default geometry a step is ~39 s, so a silent minute is otherwise the
      // user's first impression and it reads as a hang.
      if (options.verbose) {
        std::printf("denoising   %d steps over %d rows; the first step sets the pace\n",
                    total_steps, live.total_rows());
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

      video_rows = out.video_rows;
      audio_rows = out.audio_rows;
      model.unload();
      result.seconds_denoise = seconds_since(t0);
      if (options.verbose) {
        std::printf("denoised    %d steps in %.1f s (%.2f s/step)\n", total_steps,
                    result.seconds_denoise,
                    result.seconds_denoise / std::max(1, total_steps));
      }
    }
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
  }

  result.ok = true;
  return result;
}

}  // namespace vidfab
