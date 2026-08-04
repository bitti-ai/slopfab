#include "vidfab/generate.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "vidfab/audio/wav.h"
#include "vidfab/dit/packing.h"
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
    result.ok = false;
    result.message =
        "the conditioner and the denoising loop are not wired in yet. Everything downstream of "
        "them works: run with --synthetic-latents to exercise unpatchify, both VAEs, the colour "
        "transform and the muxer against the real weights.";
    return result;
  }

  {
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
