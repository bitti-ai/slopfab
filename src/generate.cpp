#include "slopfab/text/prompt_embedding.h"
#include "slopfab/generate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#include "generation/session_state.h"
#include "generation/helpers.h"
#include "generation/prefetch.h"
#include "generation/decode.h"
#include "generation/prompt.h"

namespace slopfab {
using namespace generation;

RunResult generation::run_generate_impl(const GenerateRequest& request, const GeneratePlan& plan,
                                        const RunOptions& options, ReusedGenerationModels& reuse) {
  struct ReuseReleaseGuard {
    ReusedGenerationModels& reuse;
    bool release = false;

    ~ReuseReleaseGuard() {
      if (release)
        reuse.clear();
    }
  } release_guard{reuse, options.release_reused_models};

  RunResult result;
  try {
    validate_generation_options(request, plan, options);
  } catch (const std::exception& e) {
    result.message = e.what();
    return result;
  }
  dit::SequenceLayout layout = plan.layout;
  LoraAdapters loras;
  validate_refmods(request.refmods);
#if !SLOPFAB_WITH_VULKAN
  if (options.inference_backend == DeviceBackend::kVulkan) {
    result.message = "Vulkan inference requested, but this build disabled Vulkan";
    return result;
  }
#endif
  // Validate the explicitly selected exact CUDA artifact before touching any
  // prompt/checkpoint. Synthetic-latent runs never execute a transformer and
  // therefore do not require this tuple.
  if (options.inference_backend == DeviceBackend::kCuda &&
      options.source == LatentSource::kDenoise && options.attention_mode == AttentionMode::kExact &&
      !cuda::deterministic_h3_attention_available()) {
    result.message = "exact attention is unavailable on this CUDA device/runtime tuple";
    return result;
  }

  // Host hooks (generate.h). `notify` is the run's only cancellation point:
  // it says where the run is and returns false when the host wants it
  // stopped, and every call site below turns that into an early return with
  // `cancelled` set. With no hooks installed this is one never-taken branch
  // per stage, and the run is the run it was.
  auto notify = [&options](RunStage stage, int step, int steps) {
    if (options.on_progress == nullptr)
      return true;
    return options.on_progress(stage, step, steps, options.hook_userdata);
  };
  auto stop = [&result](const char* where) {
    result.ok = false;
    result.cancelled = true;
    result.message = std::string("cancelled during ") + where;
    return result;
  };
  if (!notify(RunStage::kStarting, -1, 0))
    return stop("startup");

  if (!request.loras.empty()) {
    if (options.source != LatentSource::kDenoise) {
      result.message = "LoRAs require denoising";
      return result;
    }
    try {
      SafeTensors base;
      base.open(request.transformer_path);
      loras.load(request.loras, base);
    } catch (const std::exception& e) {
      result.message = e.what();
      return result;
    }
  }

  // Declared out here, not inside the denoise block, because it is started
  // under the loop and joined at the VAE load that follows the block. Its
  // destructor joins, so every exit path from this function — the early
  // `return result` cases below, and any exception out of the loop — leaves no
  // running thread behind.
  CheckpointPrefetch vae_prefetch;

  text::Tokenizer owned_conditioning_tokenizer;
  text::Tokenizer* conditioning_tokenizer_instance = nullptr;
  auto conditioning_tokenizer = [&]() -> text::Tokenizer& {
    if (conditioning_tokenizer_instance)
      return *conditioning_tokenizer_instance;
    if (!options.reuse_models) {
      if (request.tokenizer_path.empty())
        owned_conditioning_tokenizer.load_embedded();
      else
        owned_conditioning_tokenizer.load(request.tokenizer_path);
      conditioning_tokenizer_instance = &owned_conditioning_tokenizer;
      return *conditioning_tokenizer_instance;
    }
    const std::string key = tokenizer_cache_key(request);
    if (!reuse.tokenizer_valid || reuse.tokenizer_key != key) {
      reuse.tokenizer_valid = false;
      reuse.tokenizer_key.clear();
      reuse.tokenizer = text::Tokenizer();
      if (request.tokenizer_path.empty())
        reuse.tokenizer.load_embedded();
      else
        reuse.tokenizer.load(request.tokenizer_path);
      reuse.tokenizer_key = key;
      reuse.tokenizer_valid = true;
    } else if (options.verbose) {
      std::printf("tokenizer   reused (%zu tokens in vocabulary)\n", reuse.tokenizer.vocab_size());
    }
    conditioning_tokenizer_instance = &reuse.tokenizer;
    return *conditioning_tokenizer_instance;
  };

  text::PromptEmbedding fixed_prompt;
  if (options.source == LatentSource::kDenoise && !options.prompt_embedding_path.empty())
    fixed_prompt =
        text::read_prompt_embedding(options.prompt_embedding_path, request.has_native_references());

  if (plan.conditioning.fixed_prompt_tokens > 0 &&
      fixed_prompt.num_tokens != plan.conditioning.fixed_prompt_tokens)
    throw std::runtime_error(
        "fixed prompt embedding token count does not match conditioning settings");

  // Decode all references before opening a multi-gigabyte checkpoint. Besides
  // giving file errors promptly, this validates the Ref2VA aspect contract at
  // the dimensions actually presented by the decoder.
  //
  // `resolve_reference_image_size` targets a 2048-pixel short edge, so this
  // Lanczos resize runs on up to 8192x2048x3 in double precision on one
  // thread. Every generation of a counted run fed it byte-identical input, so
  // it is cached under the reference key and the storage below is either the
  // cache's or this call's, never a copy of one into the other.
  //
  // The reference hashes are taken once and shared by both keys, and only when
  // reuse is on at all: a cold `--count 1` run consults neither key, and would
  // otherwise pay to hash every reference twice for nothing.
  const bool cache_references = options.reuse_models;
  const std::vector<std::string> reference_identities =
      options.reuse_models ? reference_image_identities(request) : std::vector<std::string>();
  const std::string reference_key =
      options.reuse_models ? reference_cache_key(request, reference_identities) : std::string();
  std::vector<RGBImage> owned_reference_images;
  std::vector<RGBImage>& reference_images =
      cache_references ? reuse.reference_images : owned_reference_images;
  const bool reference_cache_hit =
      cache_references && reuse.reference_valid && reuse.reference_key == reference_key &&
      reuse.reference_images.size() == request.reference_image_paths.size();
  if (!reference_cache_hit) {
    if (!request.reference_image_paths.empty() && !notify(RunStage::kReferences, -1, 0)) {
      return stop("reference decode");
    }
    if (cache_references) {
      reuse.reference_valid = false;
      reuse.reference_key.clear();
      reuse.clean_reference_rows.clear();
      reuse.reference_geometry.clear();
    }
    reference_images.clear();
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
        dit::resolve_reference_image_size(image.width, image.height, &resized_h, &resized_w,
                                          plan.conditioning.references_at_target_canvas
                                              ? std::min(plan.canvas_width, plan.canvas_height)
                                              : plan.conditioning.image_short_edge);
        image = resize_reference_lanczos(image, resized_w, resized_h);
        reference_images.push_back(std::move(image));
      }
    } catch (const std::exception& e) {
      result.message = e.what();
      return result;
    }
  } else if (options.verbose && !reference_images.empty()) {
    std::printf("references  reusing %zu decoded images\n", reference_images.size());
  }

  // The keyframe VAE intentionally keeps the 2048-short-edge geometry. Qwen
  // has a separate 16,384-patch exact capacity, so derive its bounded
  // presentation and the complete decoder token stream before opening either
  // neural checkpoint or allocating a keyframe arena. This also makes an
  // impossible multi-reference/prompt aggregate a cheap transactional error.
  std::vector<text::QwenImageGrid> reference_conditioning_grids;
  std::vector<int32_t> reference_conditioning_ids;
  std::vector<PreparedReference> prepared_media;
  const bool mixed_reference_video =
      options.inference_backend == DeviceBackend::kCuda && !env_flag("SLOPFAB_REFERENCE_FP32");
  const auto media_authority = options.inference_backend == DeviceBackend::kCuda
                                   ? (mixed_reference_video ? ReferenceEncoderAuthority::kCudaFp16
                                                            : ReferenceEncoderAuthority::kCudaFp32)
                                   : ReferenceEncoderAuthority::kVulkanFp32;
  const std::string media_key = cache_references && !request.reference_media.empty()
                                    ? media_encoding_cache_key(request, media_authority)
                                    : std::string();
  const auto* cached_media = cache_references ? reuse.media_cache.find(media_key) : nullptr;
  for (const auto& media : request.reference_media) {
    if (!notify(RunStage::kReferences, -1, 0))
      return stop("reference preprocessing");
    const auto t0 = Clock::now();
    const auto reference_options =
        request.video_transition
            ? transition_reference_options(plan.canvas_width, plan.canvas_height,
                                           prepared_media.empty() ? -1 : 1)
            : plan.conditioning.reference_options(plan.canvas_width, plan.canvas_height);
    prepared_media.push_back(prepare_reference_condition(*media, double(plan.sampling_frames) / 24,
                                                         !cached_media, reference_options));
    if (plan.conditioning.pin_target_audio) {
      auto& prepared = prepared_media.back();
      prepared.plan.audio_samples =
          static_cast<int>(std::floor(double(plan.sampling_frames + 1) / 24 * 32000 + .5));
      if (!cached_media)
        prepared.audio = prepare_target_audio(*media->soundtrack(), plan.sampling_frames);
    }
    if (options.verbose) {
      const auto& p = prepared_media.back().plan;
      std::printf("references  preprocess %zu: %dx%d, %d VAE frames, %d audio samples in %.3f s\n",
                  prepared_media.size(), p.width, p.height, p.encoding_frames, p.audio_samples,
                  seconds_since(t0));
    }
  }
  std::vector<text::QwenPixelValues> media_qwen_pairs;
  PromptInputs prompt_inputs{fixed_prompt,
                             reference_identities,
                             reference_images,
                             reference_conditioning_grids,
                             reference_conditioning_ids,
                             media_qwen_pairs,
                             conditioning_tokenizer};
  if (!prepare_multimodal_prompt(request, options, prepared_media, prompt_inputs, result))
    return result;

  // --- latents ---------------------------------------------------------------

  std::vector<float> video_rows; // [V, 96]
  std::vector<float> audio_rows; // [Sa, 32]

  // Supplied initial latents, if any. Read before anything expensive happens so
  // a wrong shape fails in a second rather than after 25 GB of conditioner.
  std::vector<float> init_video;
  std::vector<float> init_audio;
  std::shared_ptr<InpaintConstraint> inpaint;
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
    if (request.transformer_path.empty() ||
        (options.prompt_embedding_path.empty() && request.text_encoder_path.empty())) {
      result.message = "generate needs --transformer and either --text-encoder or "
                       "--prompt-embedding (or pass "
                       "--synthetic-latents to skip conditioning and denoising)";
      return result;
    }
    // Ref2VA and the pruned T2VA/FL2VA transformer share most tensor names but
    // have incompatible timestep/AdaLN graphs. Check the cheap header contract
    // before loading the 15+ GiB conditioner so a wrong --transformer fails in
    // milliseconds rather than after an otherwise successful text encode.
    if (request.has_references()) {
      try {
        SafeTensors transformer_header;
        transformer_header.open(request.transformer_path);
        dit::require_ref2va_transformer(transformer_header, 1);

      } catch (const std::exception& e) {
        result.message = e.what();
        return result;
      }
    }

    if (request.image_edit.image) {
      if (!notify(RunStage::kReferences, -1, 0))
        return stop("image edit encoding");
      const RGBImage image =
          pad_edit_image(request.image_edit, plan.canvas_width, plan.canvas_height);
      SafeTensors vae_file;
      vae_file.open(request.video_vae_path);
      const auto mean = read_stat(vae_file, "latents_mean", 24);
      const auto stddev = read_stat(vae_file, "latents_std", 24);
      inpaint = std::make_shared<InpaintConstraint>();
      if (options.inference_backend == DeviceBackend::kCuda) {
        vae::KeyframeEncoder encoder(vae_file);
        inpaint->original = encoder.encode_reference_image(image, mean, stddev);
      } else {
#if SLOPFAB_WITH_VULKAN
        vulkan::Device device = create_vulkan_inference_device();
        auto encoder = vulkan::KeyframeEncoder::create(device);
        encoder.load(vae_file);
        inpaint->original = encoder.encode_reference_image(image, mean, stddev);
#endif
      }
      inpaint->mask = edit_mask_rows(request.image_edit, plan.canvas_width, plan.canvas_height);
      const auto noise =
          sampler::video_noise(request.seed, 1, layout.latent_height, layout.latent_width);
      inpaint->noise.resize(noise.size());
      dit::patchify_video(noise.data(), layout, inpaint->noise.data());
      inpaint->validate(static_cast<size_t>(layout.num_video_rows) * 96);
    }

    // --- fixed image anchors ------------------------------------------------
    std::vector<float> condition_video_rows;
    std::vector<float> condition_audio_rows;
    std::vector<dit::ReferenceGeometry> reference_geometry;
    if (!reference_images.empty()) {
      if (request.video_vae_path.empty()) {
        result.message = "--reference-image requires --video-vae for H3 image encoding";
        return result;
      }
      const Clock::time_point t0 = Clock::now();

      // The clean anchors: the six-level Conv3D keyframe encode at the resized
      // resolution, which reads nothing seed-dependent and so is cached whole.
      std::vector<std::vector<float>> owned_clean_rows;
      std::vector<dit::ReferenceGeometry> owned_geometry;
      std::vector<std::vector<float>>& clean_rows =
          cache_references ? reuse.clean_reference_rows : owned_clean_rows;
      std::vector<dit::ReferenceGeometry>& geometry =
          cache_references ? reuse.reference_geometry : owned_geometry;

      const bool encode_cache_hit =
          reference_cache_hit && cache_references && clean_rows.size() == reference_images.size();
      if (!encode_cache_hit) {
        if (cache_references)
          reuse.reference_valid = false;
        clean_rows.clear();
        geometry.clear();
        SafeTensors vae_file;
        vae_file.open(request.video_vae_path);
        const std::vector<float> mean = read_stat(vae_file, "latents_mean", 24);
        const std::vector<float> stddev = read_stat(vae_file, "latents_std", 24);
        if (options.inference_backend == DeviceBackend::kCuda) {
          vae::KeyframeEncoder image_encoder(vae_file);
          for (const RGBImage& image : reference_images) {
            clean_rows.push_back(image_encoder.encode_reference_image(image, mean, stddev));
            geometry.push_back(
                {dit::ReferenceKind::kImage, 1, image.height / 16, image.width / 16, 0});
          }
        } else {
#if SLOPFAB_WITH_VULKAN
          vulkan::Device keyframe_device = create_vulkan_inference_device();
          vulkan::KeyframeEncoder image_encoder = vulkan::KeyframeEncoder::create(keyframe_device);
          image_encoder.load(vae_file);
          for (const RGBImage& image : reference_images) {
            clean_rows.push_back(image_encoder.encode_reference_image(image, mean, stddev));
            geometry.push_back(
                {dit::ReferenceKind::kImage, 1, image.height / 16, image.width / 16, 0});
          }
          if (options.verbose) {
            const auto& stats = image_encoder.stats();
            std::printf(
                "keyframes   Vulkan exact peak/reserved %.2f/%.2f GiB, %llu descriptors\n",
                static_cast<double>(stats.allocator_peak_used_bytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(stats.allocator_reserved_bytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<unsigned long long>(stats.descriptor_set_allocations));
          }
          image_encoder.unload();
#else
          throw std::logic_error("Vulkan keyframe encoder compiled out after validation");
#endif
        }
        if (cache_references) {
          reuse.reference_key = reference_key;
          reuse.reference_valid = true;
        }
      }

      // Released Ref2VA anchors are almost clean, but not quite: the fixed
      // timestep is 0.999 and the request generator contributes the other
      // 0.001. That last 0.001 is the only seed-dependent thing here, so it is
      // reapplied every generation onto a fresh copy of the cached rows. Keep
      // each ordered reference on its own deterministic stream.
      reference_geometry = geometry;
      for (size_t reference_index = 0; reference_index < reference_images.size();
           ++reference_index) {
        const RGBImage& image = reference_images[reference_index];
        std::vector<float> rows = clean_rows[reference_index];
        const int latent_h = image.height / 16;
        const int latent_w = image.width / 16;
        const uint64_t reference_seed =
            request.seed ^ (0x9e3779b97f4a7c15ULL *
                            (reference_index + 1 +
                             (plan.conditioning.video_first ? request.reference_media.size() : 0)));
        const std::vector<float> noise_latents =
            sampler::video_noise(reference_seed, 1, latent_h, latent_w);
        const std::vector<float> noise_rows =
            vae::patchify_keyframe_latents(noise_latents.data(), latent_h, latent_w);
        sampler::FlowScheduler::scale_noise(rows.data(), noise_rows.data(), 0.999f, rows.size(),
                                            rows.data());
        condition_video_rows.insert(condition_video_rows.end(), rows.begin(), rows.end());
      }
      if (options.verbose)
        std::printf("references  %zu images -> %zu fixed video rows in %.2f s%s\n",
                    reference_images.size(), condition_video_rows.size() / 96, seconds_since(t0),
                    encode_cache_hit ? " (cached encode)" : "");
    }

    if (!prepared_media.empty()) {
      std::vector<EncodedReferenceCondition> encoded_media(prepared_media.size());
      for (size_t i = 0; i < prepared_media.size(); ++i)
        encoded_media[i].geometry = prepared_media[i].plan.geometry;
      bool has_video = false, has_audio = false;
      for (const auto& media : prepared_media) {
        has_video |= !media.frames.empty();
        has_audio |= media.plan.audio_samples > 0;
      }
      if (has_video && request.video_vae_path.empty())
        throw std::runtime_error("reference video requires --video-vae");
      if (has_audio && request.audio_vae_path.empty())
        throw std::runtime_error("reference audio requires --audio-vae");
      if (has_video && !cached_media) {
        const auto load_start = Clock::now();
        SafeTensors checkpoint;
        checkpoint.open(request.video_vae_path);
        std::unique_ptr<vae::KeyframeEncoder> cuda_encoder;
#if SLOPFAB_WITH_VULKAN
        vulkan::Device device;
        std::unique_ptr<vulkan::ReferenceEncoder> vk_encoder;
        if (options.inference_backend == DeviceBackend::kVulkan) {
          device = create_vulkan_inference_device();
          vk_encoder = std::make_unique<vulkan::ReferenceEncoder>(device, checkpoint, false);
        } else
#endif
          cuda_encoder = std::make_unique<vae::KeyframeEncoder>(checkpoint);
        auto mean = read_stat(checkpoint, "latents_mean", 24),
             stddev = read_stat(checkpoint, "latents_std", 24);
        if (options.verbose)
          std::printf("references  video VAE load in %.3f s (%s activations)\n",
                      seconds_since(load_start), mixed_reference_video ? "fp16" : "fp32");
        for (size_t i = 0; i < prepared_media.size(); ++i) {
          const auto& media = prepared_media[i];
          if (media.frames.empty())
            continue;
          if (!notify(RunStage::kReferences, -1, 0))
            return stop("reference video encode");
          const auto encode_start = Clock::now();
          std::vector<float> rows;
#if SLOPFAB_WITH_VULKAN
          if (vk_encoder)
            rows = vk_encoder->encode_reference_video(media.frames, media.plan.encoding_frames,
                                                      mean, stddev);
          else
#endif
            rows = cuda_encoder->encode_reference_video(media.frames, media.plan.encoding_frames,
                                                        mean, stddev, mixed_reference_video);
          if (options.verbose)
            std::printf("references  video %zu: %dx%d, %d frames -> %zu rows in %.3f s\n", i + 1,
                        media.plan.width, media.plan.height, media.plan.encoding_frames,
                        rows.size() / 96, seconds_since(encode_start));
          encoded_media[i].video_rows = std::move(rows);
        }
#if SLOPFAB_WITH_VULKAN
        if (vk_encoder && options.verbose)
          vk_encoder->report_memory();
#endif
      }
      if (has_audio && !cached_media) {
        const auto load_start = Clock::now();
        SafeTensors checkpoint;
        checkpoint.open(request.audio_vae_path);
        std::unique_ptr<vae::AudioEncoder> cuda_encoder;
#if SLOPFAB_WITH_VULKAN
        vulkan::Device device;
        std::unique_ptr<vulkan::ReferenceEncoder> vk_encoder;
        if (options.inference_backend == DeviceBackend::kVulkan) {
          device = create_vulkan_inference_device();
          vk_encoder = std::make_unique<vulkan::ReferenceEncoder>(device, checkpoint, true);
        } else
#endif
          cuda_encoder = std::make_unique<vae::AudioEncoder>(checkpoint);
        if (options.verbose)
          std::printf("references  audio VAE load in %.3f s\n", seconds_since(load_start));
        for (size_t i = 0; i < prepared_media.size(); ++i) {
          const auto& media = prepared_media[i];
          if (media.audio.empty())
            continue;
          if (!notify(RunStage::kReferences, -1, 0))
            return stop("reference audio encode");
          const auto encode_start = Clock::now();
          std::vector<float> rows;
#if SLOPFAB_WITH_VULKAN
          if (vk_encoder)
            rows = vk_encoder->encode_reference(media.audio.data(), media.plan.audio_samples);
          else
#endif
            rows = cuda_encoder->encode_reference(media.audio.data(), media.plan.audio_samples);
          if (options.verbose)
            std::printf("references  audio: %d samples -> %zu rows in %.3f s\n",
                        media.plan.audio_samples, rows.size() / 32, seconds_since(encode_start));
          encoded_media[i].audio_rows = std::move(rows);
        }
#if SLOPFAB_WITH_VULKAN
        if (vk_encoder && options.verbose)
          vk_encoder->report_memory();
#endif
      }
      const auto& clean_media = cached_media ? *cached_media : encoded_media;
      for (size_t i = 0; i < clean_media.size(); ++i) {
        append_encoded_reference_condition(
            clean_media[i], request.seed,
            (plan.conditioning.video_first ? 0 : reference_images.size()) + i, condition_video_rows,
            condition_audio_rows, plan.conditioning.include_reference_audio);
        reference_geometry.push_back(clean_media[i].geometry);
      }
      if (plan.conditioning.pin_target_audio)
        init_audio = target_audio_rows(clean_media.front().audio_rows, layout.num_audio_latents);
      if (cached_media && options.verbose)
        std::printf("references  reusing %zu encoded media references (host cache)\n",
                    clean_media.size());
      if (!cached_media && cache_references)
        reuse.media_cache.store(media_key, std::move(encoded_media));
    }

    if (plan.conditioning.video_first)
      order_animate_references(reference_geometry, condition_video_rows);

    if (request.has_refmods()) {
      if (!notify(RunStage::kReferences, -1, 0))
        return stop("refmod conditioning");
      append_refmod_conditions(request.refmods, request.seed, reference_geometry,
                               condition_video_rows, condition_audio_rows);
    }

    if (request.continuation) {
      append_continuation_guide(*request.continuation, plan.continuation, request.seed,
                                reference_geometry, condition_video_rows, condition_audio_rows);
    }
    if (request.video_transition)
      align_transition_guides(reference_geometry, layout.num_latent_frames);

    if (!notify(RunStage::kConditioning, -1, 0))
      return stop("conditioning");
    text::PromptEmbedding prompt;
    if (!encode_h3_prompt(request, options, reuse, prompt_inputs, prompt, result))
      return result;

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
      pos = dit::build_position_ids(live, plan.geometry);
    } else {
      dit::Ref2VAPackedSequence packed = dit::build_ref2va_packed_sequence(
          prompt.modality_tags, reference_geometry, live.num_latent_frames, live.latent_height,
          live.latent_width, live.num_audio_latents);
      live = std::move(packed.layout);
      idx = std::move(packed.indices);
      pos = std::move(packed.position_ids);
    }

    if (!notify(RunStage::kTransformerLoad, -1, 0))
      return stop("transformer load");
    if (options.inference_backend == DeviceBackend::kCuda) {
      const Clock::time_point t0 = Clock::now();
      SafeTensors dit_file;
      dit_file.open(request.transformer_path);
      dit::Transformer model;
      model.set_attention_mode(options.attention_mode);
      dit::TransformerLoadOptions load_options;
      load_options.layout = &live;
      load_options.block_cache = request.block_cache_span > 0;
      model.load(dit_file, {}, &loras, load_options);
      loras = LoraAdapters();
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
        std::printf("attention  backend %s\n", plan.model.compressed_attention
                                                   ? "vsa-h3 (dense text refiner)"
                                                   : attention_mode_name(options.attention_mode));
      }
      // Also before prepare_sequence, and for the same kind of reason: that is
      // where the two residual-stream-sized buffers are allocated, so a cache
      // configured afterwards would run against unallocated memory.
      {
        dit::BlockCacheConfig bc;
        bc.span = request.block_cache_span;
        bc.start = request.block_cache_start;
        bc.interval = request.block_cache_interval;
        bc.warmup = request.block_cache_warmup;
        model.set_block_cache(bc, plan.num_model_evaluations());
        if (bc.enabled() && options.verbose) {
          const dit::BlockSpan span = model.block_cache_span();
          std::printf("block cache blocks [%d,%d) of %d reused every %d-th step, warmup %d "
                      "(lossy, changes the sample)\n",
                      span.begin, span.end, model.num_blocks(), bc.interval,
                      std::max(dit::kMinBlockWarmup, bc.warmup));
        }
      }
      model.prepare_text(prompt.data.data(), prompt.num_tokens);
      {
        cuda::StageMemorySpan memory("transformer.sequence");
        model.prepare_sequence(live, idx, pos);
      }
      result.seconds_prepare = seconds_since(t_prep);

      // Rebuilt from the plan rather than from literals, so the loop integrates
      // on exactly the grid `describe_plan` printed and `plan.video_timesteps`
      // conditioned on. Same numbers as before; the point is that there is now
      // only one place they can be changed.
      sampler::FlowScheduler video_sched(plan.video_sigma_shift);
      sampler::FlowScheduler audio_sched(plan.audio_sigma_shift);
      video_sched.set_sigmas(plan.video_sigmas);
      audio_sched.set_sigmas(plan.audio_sigmas);
      video_sched.set_sampler(options.sampler);
      audio_sched.set_sampler(options.sampler);

      dit::DenoiseInputs in;
      in.inpaint = inpaint.get();
      in.layout = &live;
      in.indices = &idx;
      in.video_timesteps = &plan.video_timesteps;
      in.audio_timesteps = &plan.audio_timesteps;
      in.video_scheduler = &video_sched;
      in.audio_scheduler = &audio_sched;
      in.seed = request.seed;
      if (!condition_video_rows.empty())
        in.condition_video_rows = &condition_video_rows;
      if (!condition_audio_rows.empty())
        in.condition_audio_rows = &condition_audio_rows;
      if (!options.init_latents_path.empty()) {
        in.init_video_rows = &init_video;
        in.init_audio_rows = &init_audio;
      }
      if (plan.conditioning.pin_target_audio) {
        in.init_audio_rows = &init_audio;
        in.pin_target_audio = true;
      }
      in.cache.threshold = request.cache_threshold;
      in.motion_cache = request.motion_cache;
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
            std::printf("step cache  threshold %.4g (accumulated relative-L1 of c(t)), warmup %d\n",
                        static_cast<double>(in.cache.threshold),
                        std::max(dit::kMinWarmup, in.cache.warmup));
          }
        }
        std::fflush(stdout);
      }
      // Started here rather than at the top of the block on purpose: the
      // transformer load just above is itself a multi-gigabyte read, and
      // overlapping the two would only make them queue behind each other on the
      // same drive. From this line to the end of the loop the pipeline touches
      // no disk at all, which is the window this is trying to fill.
      vae_prefetch.start(request.still_image ? std::vector<std::string>{request.video_vae_path}
                                             : std::vector<std::string>{request.video_vae_path,
                                                                        request.audio_vae_path},
                         options.verbose);

      const Clock::time_point loop_start = Clock::now();
      // `denoise` breaks out of the loop when the callback returns false and
      // returns the rows it had reached, which are a half-denoised latent —
      // so the decision has to be remembered here rather than inferred from
      // the output, and the run stopped before either VAE sees it.
      bool cancel_requested = false;
      if (!notify(RunStage::kDenoising, -1, plan.num_model_evaluations()))
        return stop("denoising");
      const dit::DenoiseOutputs out = dit::denoise(model, in, [&](int step, int steps) {
        if (options.verbose) {
          const double elapsed = seconds_since(loop_start);
          const double per_step = elapsed / static_cast<double>(step + 1);
          std::printf("\rstep %d/%d  %.1f s/step  eta %.0f s      ", step + 1, steps, per_step,
                      per_step * (steps - step - 1));
          std::fflush(stdout);
        }
        if (notify(RunStage::kDenoising, step, steps))
          return true;
        cancel_requested = true;
        return false;
      });
      if (options.verbose)
        std::printf("\n");
      if (cancel_requested)
        return stop("denoising");
      result.seconds_denoise_loop = seconds_since(loop_start);
      result.steps_computed = out.steps_computed;
      result.steps_skipped = out.steps_skipped;
      cuda::StepProfiler::instance().report(stdout);
      // Always printed when anything was reused, verbose or not: a run whose
      // skip count is invisible cannot be compared against another one.
      if (out.steps_skipped != 0 || (request.motion_cache.active() && options.verbose)) {
        std::printf("%s  %d of %d evaluations skipped (%d computed), %.1f%%\n",
                    request.motion_cache.active() ? "MotionCache" : "step cache", out.steps_skipped,
                    out.steps_computed + out.steps_skipped, out.steps_computed,
                    100.0 * out.steps_skipped /
                        static_cast<double>(std::max(1, out.steps_computed + out.steps_skipped)));
      }
      // Same rule as above: printed whenever anything was reused, so the two
      // caches are equally visible in a log that is being compared against
      // another log. The span is repeated here because the reuse count means
      // nothing without knowing how many blocks each reuse covered.
      if (model.block_cache_reused() != 0) {
        const dit::BlockSpan span = model.block_cache_span();
        const int spans = model.block_cache_computed() + model.block_cache_reused();
        std::printf("block cache %d of %d spans reused (%d blocks each of %d), %.1f%%\n",
                    model.block_cache_reused(), spans, span.size(), model.num_blocks(),
                    100.0 * model.block_cache_reused() / static_cast<double>(std::max(1, spans)));
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
          if (v.empty())
            return;
          double sum = 0.0;
          for (float x : v)
            sum += x;
          const double mean = sum / static_cast<double>(v.size());
          double var = 0.0;
          for (float x : v)
            var += (x - mean) * (x - mean);
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
    } else {
#if SLOPFAB_WITH_VULKAN
      const Clock::time_point t0 = Clock::now();
      SafeTensors dit_file;
      dit_file.open(request.transformer_path);
      vulkan::Device device =
          create_vulkan_inference_device(true, options.attention_mode == AttentionMode::kSage2);
      vulkan::TensorContextOptions context_options;
      context_options.max_batch_operators =
          request.loras.empty() && !plan.model.compressed_attention ? 2048 : 4096;
      context_options.sage_extra_workspace_bytes = options.vulkan_sage_extra_workspace_bytes;
      vulkan::TensorContext context(device, context_options);
      context.require_h3_attention(options.attention_mode);
      vulkan::ExactH3DenoiseConfig config;
      config.inpaint = inpaint;
      config.motion_cache = request.motion_cache;
      config.transformer.main.layers = 50;
      config.transformer.main.block.attention_mode = options.attention_mode;
      config.transformer.main.block.sequence = static_cast<uint32_t>(live.total_rows());
      const bool conditioned = live.num_condition_video != 0 || live.num_condition_audio != 0;
      config.transformer.main.block.timesteps = conditioned ? 4u : 2u;
      config.transformer.main.block.loras = &loras;
      if (plan.model.compressed_attention)
        config.transformer.main.block.vsa_tiles =
            std::make_shared<dit::VsaTiles>(dit::build_vsa_tiles(live));
      config.transformer.text_rows = static_cast<uint32_t>(live.num_text);
      config.transformer.video_rows =
          static_cast<uint32_t>(live.num_condition_video + live.num_video_rows);
      config.transformer.audio_rows =
          static_cast<uint32_t>(live.num_condition_audio + live.num_audio_rows);
      if (conditioned) {
        config.transformer.video_output_rows = static_cast<uint32_t>(live.num_video_rows);
        config.transformer.audio_output_rows = static_cast<uint32_t>(live.num_audio_rows);
        config.transformer.video_output_start = static_cast<uint32_t>(live.video_start());
        config.transformer.audio_output_start = static_cast<uint32_t>(live.audio_start());
      }
      config.pin_target_audio = plan.conditioning.pin_target_audio;
      config.layout = live;
      config.indices = idx;
      config.position_ids = pos;
      config.attention_band = options.attention_band;
      vulkan::ExactH3Denoiser model = vulkan::ExactH3Denoiser::create(context, config);
      model.load(dit_file);
      loras = LoraAdapters();
      result.seconds_transformer_load = seconds_since(t0);

      std::vector<float> initial_video;
      std::vector<float> initial_audio;
      if (!options.init_latents_path.empty()) {
        initial_video = init_video;
        initial_audio = init_audio;
      } else {
        const std::vector<float> noise = sampler::video_noise(
            request.seed, live.num_latent_frames, live.latent_height, live.latent_width);
        initial_video.resize(static_cast<size_t>(live.num_video_rows) * 96);
        dit::patchify_video(noise.data(), live, initial_video.data());
        if (live.num_audio_latents > 0) {
          initial_audio = sampler::audio_noise(request.seed, live.num_audio_latents);
        }
      }
      if (plan.conditioning.pin_target_audio)
        initial_audio = init_audio;
      if (conditioned) {
        if (condition_video_rows.size() != static_cast<size_t>(live.num_condition_video) * 96u ||
            condition_audio_rows.size() != static_cast<size_t>(live.num_condition_audio) * 32u) {
          throw std::runtime_error(
              "Vulkan Ref2VA: condition row payload does not match packed layout");
        }
        initial_video.insert(initial_video.begin(), condition_video_rows.begin(),
                             condition_video_rows.end());
        initial_audio.insert(initial_audio.begin(), condition_audio_rows.begin(),
                             condition_audio_rows.end());
      }
      const Clock::time_point t_prep = Clock::now();
      model.prepare(prompt.data.data(), prompt.data.size(), initial_video.data(),
                    initial_video.size(), initial_audio.data(), initial_audio.size());
      result.seconds_prepare = seconds_since(t_prep);
      if (options.verbose) {
        std::printf(
            "transformer Vulkan %.2f GiB persistent, %.2f GiB peak, %d packed rows, loaded in %.2f s\n",
            static_cast<double>(model.persistent_bytes()) / (1024.0 * 1024.0 * 1024.0),
            static_cast<double>(model.peak_device_bytes()) / (1024.0 * 1024.0 * 1024.0),
            live.total_rows(), result.seconds_transformer_load);
      }

      sampler::FlowScheduler video_sched(plan.video_sigma_shift);
      sampler::FlowScheduler audio_sched(plan.audio_sigma_shift);
      video_sched.set_sigmas(plan.video_sigmas);
      audio_sched.set_sigmas(plan.audio_sigmas);
      const int total_steps = plan.num_model_evaluations();
      vae_prefetch.start(request.still_image ? std::vector<std::string>{request.video_vae_path}
                                             : std::vector<std::string>{request.video_vae_path,
                                                                        request.audio_vae_path},
                         options.verbose);
      const Clock::time_point loop_start = Clock::now();
      bool cancel_requested = false;
      if (!notify(RunStage::kDenoising, -1, total_steps))
        return stop("denoising");
      const vulkan::ExactH3DenoiseResult out =
          model.run(video_sched, audio_sched, [&](uint32_t step, uint32_t steps) {
            if (options.verbose) {
              const double elapsed = seconds_since(loop_start);
              const double per_step = elapsed / static_cast<double>(step + 1);
              std::printf("\rstep %u/%u  %.1f s/step  eta %.0f s      ", step + 1, steps, per_step,
                          per_step * (steps - step - 1));
              std::fflush(stdout);
            }
            if (notify(RunStage::kDenoising, static_cast<int>(step), static_cast<int>(steps)))
              return true;
            cancel_requested = true;
            return false;
          });
      if (options.verbose)
        std::printf("\n");
      if (cancel_requested || out.cancelled)
        return stop("denoising");
      result.seconds_denoise_loop = seconds_since(loop_start);
      result.steps_computed = static_cast<int>(out.steps_computed);
      result.steps_skipped = static_cast<int>(out.steps_skipped);
      video_rows = out.video_rows;
      audio_rows = out.audio_rows;
      model.unload();
      result.seconds_denoise = seconds_since(t0);
      if (options.verbose) {
        std::printf(
            "denoised    %d Vulkan %s steps in %.1f s (%.2f s/step); +%.1f s load, +%.1f s prepare\n",
            total_steps,
            plan.model.compressed_attention ? "vsa-h3"
                                            : attention_mode_name(options.attention_mode),
            result.seconds_denoise_loop, result.seconds_denoise_loop / std::max(1, total_steps),
            result.seconds_transformer_load, result.seconds_prepare);
        if (request.motion_cache.active())
          std::printf("MotionCache: reused %d/%d model calls (%d computed)\n", result.steps_skipped,
                      result.steps_skipped + result.steps_computed, result.steps_computed);
      }
#else
      throw std::logic_error("Vulkan inference compiled out after validation");
#endif
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
    const std::vector<float> video_latents = sampler::video_noise(
        request.seed, layout.num_latent_frames, layout.latent_height, layout.latent_width);
    video_rows.resize(static_cast<size_t>(layout.num_video_rows) * 96);
    dit::patchify_video(video_latents.data(), layout, video_rows.data());
    if (layout.num_audio_latents > 0) {
      audio_rows = sampler::audio_noise(request.seed, layout.num_audio_latents);
    }
    result.seconds_denoise = seconds_since(t0);
  }

  if (plan.conditioning.pin_target_audio) {
    if (audio_rows.size() != init_audio.size() ||
        std::memcmp(audio_rows.data(), init_audio.data(), audio_rows.size() * sizeof(float)) != 0)
      throw std::logic_error("pinned target audio changed during denoising");
    if (options.verbose)
      std::printf("audio       pinned target rows unchanged through denoising\n");
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

  // Preserve normalized sampler output, join in latent space, then decode the
  // cumulative stream so the VAE sees context on both sides of the join.
  auto completed = std::make_shared<LatentClip>();
  if (request.continuation) {
    *completed =
        join_continuation(*request.continuation, plan.continuation, video_rows, audio_rows);
    video_rows.clear();
    audio_rows.clear();
  } else {
    completed->width = plan.canvas_width;
    completed->height = plan.canvas_height;
    completed->frames = plan.aligned_frames;
    completed->video_rows = std::move(video_rows);
    completed->audio_rows = std::move(audio_rows);
  }
  completed->geometry = plan.geometry;
  completed->sampled = options.source == LatentSource::kDenoise;
  completed->transformer = request.transformer_path;
  completed->video_vae = request.video_vae_path;
  completed->audio_vae = request.audio_vae_path;
  completed->validate();
  layout = completed->layout();
  if (!options.save_latents_path.empty()) {
    completed->save(options.save_latents_path);
    result.outputs.push_back(options.save_latents_path);
    if (options.verbose)
      std::printf("saved       %s (%d frames, normalized fp32 AV latents)\n",
                  options.save_latents_path.c_str(), completed->frames);
  }
  if (options.on_latents)
    options.on_latents(completed, options.hook_userdata);

  vae_prefetch.join();
  return decode_and_deliver(request, options, completed, std::move(result));
}

} // namespace slopfab
