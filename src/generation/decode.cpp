#include "slopfab/text/prompt_embedding.h"
#include "slopfab/generate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "slopfab/audio/wav.h"
#include "slopfab/image.h"
#include "slopfab/cuda/profile.h"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/dit/denoise.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/dit/packing.h"
#include "slopfab/dit/ref2va.h"
#include "slopfab/dit/transformer.h"
#include "slopfab/text/encoder.h"
#include "slopfab/text/tokenizer.h"
#include "slopfab/sampler/scheduler.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sampler/noise.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vae/audio_decoder.h"
#include "slopfab/vae/vit_decoder.h"
#include "slopfab/vae/keyframe_encoder.h"
#include "slopfab/vae/audio_encoder.h"
#include "slopfab/reference_conditioning.h"
#include "slopfab/video/mux.h"
#include "slopfab/video/y4m.h"
#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/audio_decoder.h"
#include "slopfab/vulkan/dit_denoise.h"
#include "slopfab/vulkan/keyframe_encoder.h"
#include "slopfab/vulkan/reference_encoder.h"
#include "slopfab/vulkan/text_encoder.h"
#include "slopfab/vulkan/vae_decoder.h"
#endif

#include "helpers.h"
#include "decode.h"

namespace slopfab::generation {
RunResult decode_and_deliver(const GenerateRequest& request, const RunOptions& options,
                             const std::shared_ptr<const LatentClip>& completed, RunResult result) {
  const auto layout = completed->layout();
  auto notify = [&options](RunStage stage, int step, int steps) {
    return !options.on_progress || options.on_progress(stage, step, steps, options.hook_userdata);
  };
  auto stop = [&result](const char* where) {
    result.ok = false;
    result.cancelled = true;
    result.message = std::string("cancelled during ") + where;
    return result;
  };
  // --- video ----------------------------------------------------------------

  if (!notify(RunStage::kVideoDecode, -1, 0))
    return stop("video decode");
  vae::DecodedVideo video;
  {
    const Clock::time_point t0 = Clock::now();
    if (request.video_vae_path.empty()) {
      result.message = "generate needs --vae <video_vae.safetensors>";
      return result;
    }

    // Rows back to a latent volume, then de-normalise per channel. The
    // multiply-then-add order is the reference's (decoders.py:107).
    cuda::PhaseSpan s_unpatch("unpatchify latents");
    std::vector<float> latents(static_cast<size_t>(24) * layout.num_latent_frames *
                               layout.latent_height * layout.latent_width);
    dit::unpatchify_video(completed->video_rows.data(), layout, latents.data());
    s_unpatch.stop();

    // Before the span, so the readahead started under the loop is accounted to
    // the loop and this span keeps measuring the load it names. Joining is
    // required, not tidy: the mapping the worker holds is dropped here, and
    // nothing may outlive this function still holding one.

    cuda::PhaseSpan s_load("vae weight load");
    SafeTensors vae_file;
    vae_file.open(request.video_vae_path);
    const std::vector<float> mean = read_stat(vae_file, "latents_mean", 24);
    const std::vector<float> std_dev = read_stat(vae_file, "latents_std", 24);

    if (options.inference_backend == DeviceBackend::kCuda) {
      vae::ViTDecoder decoder;
      vae::ViTConfig config;
      if (options.attention_mode == AttentionMode::kExact)
        config.transformer_mode = vae::ViTTransformerMode::kExact;
      decoder.load(vae_file, config);
      s_load.stop();
      if (options.verbose) {
        std::printf("video vae   CUDA %.2f GiB on device\n",
                    static_cast<double>(decoder.weight_bytes()) / (1024.0 * 1024.0 * 1024.0));
      }
      video = request.still_image
                  ? vae::decode_still_image(decoder, latents.data(), layout.latent_height,
                                            layout.latent_width, mean, std_dev)
                  : decoder.decode(latents.data(), layout.num_latent_frames, layout.latent_height,
                                   layout.latent_width, mean, std_dev);
    } else {
#if SLOPFAB_WITH_VULKAN
      vulkan::Device device = create_vulkan_inference_device();
      vae::ViTConfig config;
      config.transformer_mode = vae::ViTTransformerMode::kExact;
      vulkan::VideoVaeDecoder decoder = vulkan::VideoVaeDecoder::create(device, config);
      decoder.load(vae_file);
      s_load.stop();
      if (options.verbose) {
        std::printf("video vae   Vulkan %.2f GiB on device\n",
                    static_cast<double>(decoder.persistent_bytes()) / (1024.0 * 1024.0 * 1024.0));
      }
      video = request.still_image
                  ? vae::decode_still_image(decoder, latents.data(), layout.latent_height,
                                            layout.latent_width, mean, std_dev)
                  : decoder.decode(latents.data(), layout.num_latent_frames, layout.latent_height,
                                   layout.latent_width, mean, std_dev);
#else
      throw std::logic_error("Vulkan inference compiled out after validation");
#endif
    }
    result.seconds_video_decode = seconds_since(t0);
    if (options.verbose) {
      std::printf("video       %d frames of %dx%d in %.2f s\n", video.frames, video.width,
                  video.height, result.seconds_video_decode);
    }
    // The spans above tile this block, so the elapsed time is their denominator.
    cuda::PhaseProfiler::instance().add_total("video vae stage",
                                              result.seconds_video_decode * 1000.0);
    cuda::PhaseProfiler::instance().report(stdout);
  }

  if (request.image_edit.image) {
    if (video.frames != 1 || video.channels != 3)
      throw std::logic_error("image editing requires a single decoded RGB frame");
    video.data = composite_image_edit(request.image_edit, video.data, video.width, video.height);
    video.width = request.image_edit.image->width;
    video.height = request.image_edit.image->height;
  }

  // --- audio ----------------------------------------------------------------

  vae::DecodedAudio audio;
  if (!request.still_image && !request.audio_vae_path.empty()) {
    if (!notify(RunStage::kAudioDecode, -1, 0))
      return stop("audio decode");
    const Clock::time_point t0 = Clock::now();

    // (Sa, 32) rows -> (2, 32, A), then de-normalise per channel.
    std::vector<float> audio_latents(completed->audio_rows.size());
    dit::unpack_audio(completed->audio_rows.data(), layout.num_audio_latents, audio_latents.data());

    SafeTensors audio_file;
    audio_file.open(request.audio_vae_path);
    const int A = layout.num_audio_latents;
    auto denormalize = [&](const std::vector<float>& mean, const std::vector<float>& std_dev) {
      for (int c = 0; c < 2; ++c) {
        for (int ch = 0; ch < 32; ++ch) {
          const float m = mean[static_cast<size_t>(ch)];
          const float s = std_dev[static_cast<size_t>(ch)];
          float* row = audio_latents.data() + (static_cast<size_t>(c) * 32 + ch) * A;
          for (int a = 0; a < A; ++a)
            row[a] = row[a] * s + m;
        }
      }
    };
    if (options.inference_backend == DeviceBackend::kCuda) {
      vae::AudioDecoder decoder;
      decoder.load(audio_file);
      denormalize(decoder.latents_mean(), decoder.latents_std());
      audio = decoder.decode(audio_latents.data(), A);
    } else {
#if SLOPFAB_WITH_VULKAN
      vulkan::Device device = create_vulkan_inference_device();
      vulkan::AudioDecoder decoder = vulkan::AudioDecoder::create(device);
      decoder.load(audio_file);
      denormalize(decoder.latents_mean(), decoder.latents_std());
      audio = decoder.decode(audio_latents.data(), A);
#else
      throw std::logic_error("Vulkan inference compiled out after validation");
#endif
    }
    result.seconds_audio_decode = seconds_since(t0);
    if (options.verbose) {
      std::printf("audio       %lld frames at %d Hz in %.2f s\n",
                  static_cast<long long>(audio.num_frames()), audio.sample_rate,
                  result.seconds_audio_decode);
    }
  } else if (options.verbose) {
    std::printf("audio       skipped (%s)\n",
                request.still_image ? "still-image mode" : "no --audio-vae");
  }

  // --- output ---------------------------------------------------------------

  {
    const Clock::time_point t0 = Clock::now();
    const bool have_audio = !audio.samples.empty();

    // An in-process host takes the samples here and there is nothing left to
    // write — no MP4, no .y4m, and in particular no call into the muxer,
    // which is the only thing in this project that loads FFmpeg. That is the
    // whole reason this hook is before the branch below rather than after it.
    if (options.on_samples != nullptr) {
      if (!notify(RunStage::kDelivering, -1, 0))
        return stop("delivery");
      RunSamples samples;
      samples.channels = video.channels;
      samples.frames = video.frames;
      samples.height = video.height;
      samples.width = video.width;
      samples.video = &video.data;
      samples.audio_channels = audio.channels;
      samples.audio_sample_rate = audio.sample_rate;
      samples.audio = have_audio ? &audio.samples : nullptr;
      // Read before the hook, which is entitled to move both buffers out — and
      // is expected to, since the video plane alone is gigabytes. Afterwards
      // `audio.num_frames()` is derived from a vector the caller now owns, so
      // reporting it here would print 0 for every run that delivered audio
      // perfectly well. The video counts survive only because they are
      // scalars, which is what made this look right.
      const long long delivered_audio_frames = static_cast<long long>(audio.num_frames());
      if (options.on_samples(samples, options.hook_userdata)) {
        result.seconds_output = seconds_since(t0);
        if (options.verbose) {
          std::printf("delivered   %d frames of %dx%d and %lld audio frames in-process\n",
                      video.frames, video.width, video.height, delivered_audio_frames);
        }
        result.ok = true;
        notify(RunStage::kFinished, -1, 0);
        return result;
      }
    }

    if (request.image_edit.image) {
      std::ofstream file(request.out_path, std::ios::binary);
      file << "P6\n" << video.width << ' ' << video.height << "\n255\n";
      const size_t plane = size_t(video.width) * video.height;
      std::vector<uint8_t> pixels(plane * 3);
      for (size_t p = 0; p < plane; ++p)
        for (size_t c = 0; c < 3; ++c) {
          const float value = video.data[c * plane + p];
          if (!std::isfinite(value))
            throw std::runtime_error("non-finite image edit output");
          pixels[p * 3 + c] =
              static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255));
        }
      file.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size()));
      file.close();
      if (!file)
        throw std::runtime_error("cannot write edited image: " + request.out_path);
      result.outputs.push_back(request.out_path);
      result.seconds_output = seconds_since(t0);
      result.ok = true;
      notify(RunStage::kFinished, -1, 0);
      return result;
    }

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
        mux.frame_converter = options.output_frame_converter;

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
      video::write_y4m(y4m, video.data, video.frames, video.height, video.width, {},
                       options.output_frame_converter);
      result.outputs.push_back(y4m);
      if (have_audio) {
        const std::string wav = base + ".wav";
        audio::write_wav(wav, audio.samples, audio.channels, audio.sample_rate);
        result.outputs.push_back(wav);
      }
      if (options.verbose) {
        for (const std::string& p : result.outputs)
          std::printf("wrote       %s\n", p.c_str());
      }
    }
    result.seconds_output = seconds_since(t0);
    if (options.verbose)
      std::printf("output      %.2f s\n", result.seconds_output);
  }

  result.ok = true;
  notify(RunStage::kFinished, -1, 0);
  return result;
}

}
