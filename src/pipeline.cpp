#include "slopfab/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>

#include "slopfab/sampler/scheduler.h"
#include "slopfab/reference_conditioning.h"
#include "sampling_plan.h"

namespace slopfab {
namespace {


// The video decoder consumes 7-token temporal windows (`tokens_chunk_size` 5 +
// `token_overlap` 2), so a latent shorter than that cannot be decoded at all.
// `F = 5k + 2` for `17k + 5` pixel frames, so `F >= 7` means `k >= 1` means at
// least 22 pixel frames.
constexpr int kMinLatentFrames = 7;
constexpr int kMinFrames = 22;

void append_media_identity(std::string& key, const GenerateRequest& request) {
  const auto settings = resolve_conditioning_settings(request);
  key += settings.cache_identity();
  if (settings.references_at_target_canvas) {
    LatentGeometry geometry;
    if (!request.transformer_path.empty() && std::filesystem::exists(request.transformer_path)) {
      SafeTensors checkpoint;
      checkpoint.open(request.transformer_path);
      geometry = read_model_geometry(checkpoint);
    }
    key += ":geometry:" + geometry.fingerprint();
    key += ":" + std::to_string(request.canvas_width) + "x" + std::to_string(request.canvas_height);
    key += ":aspect:" + std::to_string(request.aspect_w) + ":" + std::to_string(request.aspect_h);
  }
  if (request.reference_media.empty()) return;
  key.push_back('\0');
  const int frames = request.continuation
      ? plan_continuation(*request.continuation, request.continuation_overlap_frames, request.num_frames).window_frames
      : (request.still_image ? 1 : dit::align_num_frames(request.num_frames));
  key += "decoded-media-preprocessing-v2:" + std::to_string(frames);
  key.push_back('\0');
  for (const auto& reference : request.reference_media) {
    if (!reference) throw std::invalid_argument("reference media: null reference");
    key += reference_media_identity(*reference);
  }
}

}  // namespace

GeneratePlan resolve_plan(const GenerateRequest& request) {
  request.motion_cache.validate();
  if (request.motion_cache.active() &&
      (request.cache_threshold > 0 || request.skip_every > 0 || request.block_cache_span > 0 ||
       request.animate || request.schedule != sampler::ScheduleKind::kDefault))
    throw std::invalid_argument("MotionCache requires the default schedule without other caches or Animate");
  if (request.continuation && request.still_image)
    throw std::invalid_argument("continuation is unavailable in still-image mode");
  validate_refmods(request.refmods);
  validate_reference_media(request.reference_image_paths.size(), request.reference_media);
  GeneratePlan plan;
  if (!request.transformer_path.empty() && std::filesystem::exists(request.transformer_path)) {
    SafeTensors checkpoint;
    checkpoint.open(request.transformer_path);
    plan.checkpoint_available = true;
    plan.model = dit::resolve_model_descriptor(checkpoint);
    plan.geometry = read_model_geometry(checkpoint);
    require_h3_latent_geometry(plan.geometry);
    if ((request.has_references() || request.continuation) && !plan.model.supports_references)
      throw std::invalid_argument("selected model does not support reference conditioning");
    if (plan.model.compressed_attention && (request.has_references() || request.continuation))
      throw std::invalid_argument("compressed attention currently supports unconditioned media layouts only");
  }
  plan.conditioning = resolve_conditioning_settings(request);
  validate_conditioning_request(request, plan.conditioning);
  if (!request.reference_media.empty() && request.reference_image_paths.empty()) {
    bool has_video = bool(request.continuation);
    for (const auto& media : request.reference_media) has_video |= media->is_video();
    for (const auto& ref : request.refmods)
      has_video |= ref.enabled() && ref.mod->geometry().kind != dit::ReferenceKind::kAudio;
    if (!has_video) throw std::invalid_argument("reference audio requires an image or video reference");
  }
  if (request.reference_image_paths.size() > 9) {
    throw std::runtime_error("MiniMax-H3 Ref2VA accepts at most 9 reference images, got " +
                             std::to_string(request.reference_image_paths.size()));
  }
  if (request.has_explicit_canvas()) {
    dit::validate_canvas_size(request.canvas_height, request.canvas_width, plan.geometry);
    plan.canvas_height = request.canvas_height;
    plan.canvas_width = request.canvas_width;
  } else if (request.continuation) {
    plan.canvas_height = request.continuation->height;
    plan.canvas_width = request.continuation->width;
  } else if (plan.conditioning.canvas_from_reference_video) {
    const auto& frame = request.reference_media.front()->frames().front()->image;
    const int multiple = plan.geometry.canvas_multiple;
    const int short_edge = std::max(multiple, std::min(frame.width, frame.height) / multiple * multiple);
    dit::resolve_canvas_size(frame.width, frame.height, &plan.canvas_height, &plan.canvas_width,
                             short_edge, INT32_MAX, plan.geometry);
  } else {
    dit::resolve_canvas_size(static_cast<double>(request.aspect_w),
                             static_cast<double>(request.aspect_h), &plan.canvas_height,
                             &plan.canvas_width, 768, plan.geometry.trained_max_pixels, plan.geometry);
  }

  dit::validate_canvas_size(plan.canvas_height, plan.canvas_width, plan.geometry);

  if (request.continuation) {
    if (plan.canvas_width != request.continuation->width ||
        plan.canvas_height != request.continuation->height)
      throw std::invalid_argument("continuation canvas must match the saved latents");
    plan.continuation = plan_continuation(*request.continuation,
        request.continuation_overlap_frames, request.num_frames);
    plan.aligned_frames = plan.continuation.output_frames;
    plan.sampling_frames = plan.continuation.window_frames;
    plan.duration_seconds = double(plan.aligned_frames) / plan.geometry.fps;
  } else if (request.still_image) {
    plan.aligned_frames = 1;
    plan.duration_seconds = 1.0 / plan.geometry.fps;
  } else {
    plan.aligned_frames = dit::align_num_frames(request.num_frames, plan.geometry);
    plan.duration_seconds = static_cast<double>(plan.aligned_frames) / plan.geometry.fps;

    // Checked here rather than in the decoder so the run fails in milliseconds
    // instead of after uploading 9 GB of VAE weights. The decoder does keep its
    // own guard — this one is about where the user finds out.
    if (dit::video_latent_num_frames(plan.aligned_frames, plan.geometry) < kMinLatentFrames) {
      throw std::runtime_error(
          "num_frames = " + std::to_string(request.num_frames) + " aligns to " +
          std::to_string(plan.aligned_frames) + " frames, which is " +
          std::to_string(dit::video_latent_num_frames(plan.aligned_frames, plan.geometry)) +
          " latent frames; the video decoder needs at least " +
          std::to_string(kMinLatentFrames) + ". Ask for at least 6 frames, which aligns up to " +
          std::to_string(kMinFrames) + ".");
    }
  }

  if (!request.continuation) plan.sampling_frames = plan.aligned_frames;
  if (plan.conditioning.max_frames > 0 && plan.aligned_frames > plan.conditioning.max_frames)
    throw std::invalid_argument("aligned frame count exceeds conditioning max_frames");
  plan.layout.num_text = 0;  // filled in after tokenisation
  plan.layout.num_condition_video = 0;  // t2va has no conditioning rows
  plan.layout.num_latent_frames =
      request.still_image ? 1 : dit::video_latent_num_frames(plan.sampling_frames, plan.geometry);
  plan.layout.latent_height = plan.canvas_height / plan.geometry.spatial_compression;
  plan.layout.latent_width = plan.canvas_width / plan.geometry.spatial_compression;
  plan.layout.num_audio_latents =
      request.continuation ? plan.continuation.window_audio_latents :
      (request.still_image ? 0 : dit::audio_latents_for_frames(plan.sampling_frames, plan.geometry));
  plan.layout.num_audio_rows = plan.geometry.audio_channels * plan.layout.num_audio_latents;
  plan.layout.num_video_rows = plan.layout.num_latent_frames * plan.layout.rows_per_frame(plan.geometry);
  if (!request.reference_media.empty()) {
    plan.layout.condition_audio_is_explicit = true;
    for (const auto& media : request.reference_media) {
      const auto options = plan.conditioning.reference_options(plan.canvas_width, plan.canvas_height);
      const auto geometry = reference_condition_plan(*media, double(plan.sampling_frames) / plan.geometry.fps, options).geometry;
      plan.layout.num_condition_video += geometry.video_rows();
      plan.layout.num_condition_audio += geometry.audio_rows();
    }
  }

  for (const auto& ref : request.refmods) {
    if (!ref.enabled()) continue;
    plan.layout.condition_audio_is_explicit = true;
    plan.layout.num_condition_video += ref.mod->geometry().video_rows() * ref.copies;
    plan.layout.num_condition_audio += ref.mod->geometry().audio_rows() * ref.copies;
  }

  if (request.continuation) {
    plan.layout.condition_audio_is_explicit = true;
    plan.layout.num_condition_video += plan.continuation.overlap_video_latents * plan.layout.rows_per_frame(plan.geometry);
    plan.layout.num_condition_audio += plan.geometry.audio_channels * plan.continuation.overlap_audio_latents;
  }

  resolve_sampling_plan(request, plan);

  // The reference zips the two timestep lists to build its row-timestep plan
  // while iterating the video one. If `unique_consecutive` collapsed the two
  // shifted grids to different lengths, that zip would silently truncate. It
  // does not happen at practical step counts, but it is cheap to be certain.
  if (plan.video_timesteps.size() != plan.audio_timesteps.size()) {
    throw std::runtime_error(
        "video and audio schedules collapsed to different lengths (" +
        std::to_string(plan.video_timesteps.size()) + " vs " +
        std::to_string(plan.audio_timesteps.size()) +
        "); reduce num_inference_steps so the shifted sigma grids stay distinct in float32");
  }
  if (plan.video_timesteps.empty()) {
    throw std::runtime_error("schedule collapsed to zero model evaluations");
  }

  return plan;
}

void append_file_identity(std::string& key, const std::string& path) {
  key.push_back('\0');
  key += path;
  key.push_back('\0');
  if (path.empty()) return;

  std::error_code ec;
  const std::uintmax_t bytes = std::filesystem::file_size(path, ec);
  if (ec) {
    // Absent or unreadable. The path is already in the key, so this only has to
    // be distinguishable from any real (size, time) pair — and it must not
    // compare equal to the same path once the file exists.
    key += "missing";
    return;
  }
  key += std::to_string(static_cast<unsigned long long>(bytes));
  key.push_back('@');
  const std::filesystem::file_time_type mtime = std::filesystem::last_write_time(path, ec);
  if (ec) {
    key += "unknown";
    return;
  }
  key += std::to_string(static_cast<long long>(mtime.time_since_epoch().count()));
}

void append_file_content_identity(std::string& key, const std::string& path) {
  key.push_back('\0');
  key += path;
  key.push_back('\0');
  if (path.empty()) return;

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    key += "unreadable";
    return;
  }
  // FNV-1a, 64-bit. Not a cryptographic hash and does not need to be: the
  // threat is an honest overwrite the filesystem failed to distinguish, not a
  // crafted collision. Read in chunks so a large reference costs no more
  // resident memory than the buffer.
  uint64_t hash = 14695981039346656037ULL;
  uint64_t bytes = 0;
  char buffer[64 * 1024];
  while (in.read(buffer, sizeof(buffer)) || in.gcount() > 0) {
    const std::streamsize got = in.gcount();
    bytes += static_cast<uint64_t>(got);
    for (std::streamsize i = 0; i < got; ++i) {
      hash ^= static_cast<unsigned char>(buffer[i]);
      hash *= 1099511628211ULL;
    }
  }
  // The length goes in alongside the hash so a collision has to match both.
  key += std::to_string(static_cast<unsigned long long>(bytes));
  key.push_back('#');
  key += std::to_string(static_cast<unsigned long long>(hash));
}

std::vector<std::string> reference_image_identities(const GenerateRequest& request) {
  std::vector<std::string> identities;
  identities.reserve(request.reference_image_paths.size());
  for (const std::string& path : request.reference_image_paths) {
    std::string entry;
    append_file_content_identity(entry, path);
    identities.push_back(std::move(entry));
  }
  return identities;
}

std::string conditioning_cache_key(const GenerateRequest& request,
                                   const std::vector<std::string>& reference_identities) {
  if (reference_identities.size() != request.reference_image_paths.size()) {
    return conditioning_cache_key(request);
  }
  std::string key = "conditioning";
  append_file_identity(key, request.text_encoder_path);
  append_file_identity(key, request.tokenizer_path);
  key.push_back('\0');
  key += request.prompt;
  for (const std::string& identity : reference_identities) key += identity;
  append_media_identity(key, request);
  return key;
}

namespace {

void append_conditioner_authority(std::string& key, ConditionerAuthority authority) {
  key.push_back('\0');
  key += "conditioner-authority:";
  key.push_back(static_cast<char>(authority));
}

}  // namespace

std::string conditioning_cache_key_for_authority(
    const GenerateRequest& request,
    const std::vector<std::string>& reference_identities,
    ConditionerAuthority authority) {
  std::string key = conditioning_cache_key(request, reference_identities);
  append_conditioner_authority(key, authority);
  return key;
}

std::string conditioning_cache_key(const GenerateRequest& request) {
  std::string key = "conditioning";
  append_file_identity(key, request.text_encoder_path);
  append_file_identity(key, request.tokenizer_path);
  key.push_back('\0');
  key += request.prompt;
  for (const std::string& path : request.reference_image_paths) {
    append_file_content_identity(key, path);
  }
  append_media_identity(key, request);
  return key;
}

std::string conditioning_cache_key_for_authority(
    const GenerateRequest& request, ConditionerAuthority authority) {
  std::string key = conditioning_cache_key(request);
  append_conditioner_authority(key, authority);
  return key;
}

std::string reference_cache_key(const GenerateRequest& request,
                                const std::vector<std::string>& reference_identities) {
  if (reference_identities.size() != request.reference_image_paths.size()) {
    return reference_cache_key(request);
  }
  std::string key = "reference";
  append_file_identity(key, request.video_vae_path);
  for (const std::string& identity : reference_identities) key += identity;
  if (!request.reference_media.empty()) append_file_identity(key, request.audio_vae_path);
  append_media_identity(key, request);
  return key;
}

std::string reference_cache_key(const GenerateRequest& request) {
  std::string key = "reference";
  append_file_identity(key, request.video_vae_path);
  for (const std::string& path : request.reference_image_paths) {
    append_file_content_identity(key, path);
  }
  if (!request.reference_media.empty()) append_file_identity(key, request.audio_vae_path);
  append_media_identity(key, request);
  return key;
}

std::string tokenizer_cache_key(const GenerateRequest& request) {
  std::string key = "tokenizer";
  append_file_identity(key, request.tokenizer_path);
  return key;
}

std::string media_encoding_cache_key(const GenerateRequest& request,
                                     ReferenceEncoderAuthority authority) {
  std::string key = "encoded-media-v1";
  key.push_back(static_cast<char>(authority));
  append_file_identity(key, request.video_vae_path);
  append_file_identity(key, request.audio_vae_path);
  append_media_identity(key, request);
  return key;
}

std::string describe_plan(const GenerateRequest& request, const GeneratePlan& plan) {
  const dit::SequenceLayout& l = plan.layout;
  // The canvas line says where the number came from, because "1344 x 768" from
  // an explicit --resolution and the same figure derived from 16:9 are the same
  // canvas reached two ways, and only one of them was capped to the trained
  // area on the way.
  char provenance[64];
  if (request.has_explicit_canvas()) {
    std::snprintf(provenance, sizeof(provenance), "as given%s",
                  dit::canvas_exceeds_trained_area(plan.canvas_height, plan.canvas_width, plan.geometry)
                      ? ", above the trained area"
                      : "");
  } else if (plan.conditioning.canvas_from_reference_video) {
    std::snprintf(provenance, sizeof(provenance), "from driving video, aligned to 32");
  } else {
    std::snprintf(provenance, sizeof(provenance), "from %d:%d", request.aspect_w,
                  request.aspect_h);
  }
  char buf[2048];
  std::snprintf(
      buf, sizeof(buf),
      "request\n"
      "  prompt              %zu characters\n"
      "  reference images    %zu%s\n"
      "  canvas              %d x %d  (%s)\n"
      "  mode                %s\n"
      "  frames              %d requested -> %d %s (%.2f s at %d fps)\n"
      "  latent grid         %d frames of %d x %d  -> %d rows per frame\n"
      "  audio latents       %d per channel -> %d rows\n"
      "  packed sequence     %d rows + prompt length\n"
      "  steps               %d grid points -> %d model evaluations\n"
      "  sigma range         video %.6f .. %.6f (shift %.6g)\n"
      "                      audio %.6f .. %.6f (shift %.6g)\n"
      "  seed                %llu\n"
      "  output              %s\n",
      request.prompt.size(), request.reference_image_paths.size(),
      !request.has_references()
          ? (request.still_image ? " (text-to-image)" : " (text-to-video)")
          : " (Ref2VA, ordered)",
      plan.canvas_height, plan.canvas_width, provenance,
      request.still_image ? "still image (one video latent, no target audio)" : "video",
      request.num_frames, plan.aligned_frames, request.still_image ? "output" : "aligned",
      plan.duration_seconds, plan.geometry.fps,
      l.num_latent_frames, l.latent_height, l.latent_width, l.rows_per_frame(),
      l.num_audio_latents, l.num_audio_rows, l.total_rows(), plan.num_inference_steps,
      plan.num_model_evaluations(), static_cast<double>(plan.video_sigmas.front()),
      static_cast<double>(plan.video_sigmas[plan.video_sigmas.size() - 2]),
      static_cast<double>(plan.video_sigma_shift), static_cast<double>(plan.audio_sigmas.front()),
      static_cast<double>(plan.audio_sigmas[plan.audio_sigmas.size() - 2]),
      static_cast<double>(plan.audio_sigma_shift),
      static_cast<unsigned long long>(request.seed), request.out_path.c_str());
  std::string description = buf;
  description += "  conditioning        " + plan.conditioning.cache_identity() + "\n";
  description += "  model contract      " + plan.model.family + " (" + plan.model.origin + ")\n";
  if (plan.conditioning.fixed_prompt_tokens > 0)
    description += "  prompt              fixed " + std::to_string(plan.conditioning.fixed_prompt_tokens) + "-token embedding\n";
  if (plan.conditioning.pin_target_audio)
    description += "  target audio        pinned driving soundtrack (clean t=1)\n";
  if (request.continuation) {
    description += "  continuation        " + std::to_string(request.continuation->frames) +
        " source + " + std::to_string(plan.continuation.extension_frames) + " new frames\n" +
        "  sampling window     " + std::to_string(plan.sampling_frames) + " frames (" +
        std::to_string(plan.continuation.overlap_frames) + " hidden overlap); output is full joined clip\n";
  }
  for (const auto& ref : request.refmods) {
    description += "  refmod              " + ref.mod->path() + " (strength " +
        std::to_string(ref.strength) + ", copies " + std::to_string(ref.copies) +
        ", tokens " + std::to_string(ref.enabled() ? ref.mod->token_count() * ref.copies : 0) + ")\n";
  }
  if (plan.fixed_sampling_grid)
    description += "  schedule            fixed base grid (Euler, no approximate caches)\n";
  for (const auto& source : plan.sampling_sources)
    description += "  sampling source     " + source + "\n";
  if (plan.fasth3_v2)
    description += "  attention           VSA-H3 (64-token tiles, 80% sparsity, learned gates)\n";
  for (const auto& lora : request.loras) {
    char strength[64];
    std::snprintf(strength, sizeof(strength), "%.6g", static_cast<double>(lora.strength));
    description += "  LoRA                " + lora.path + " (strength " + strength + ")\n";
  }
  if (!request.reference_media.empty()) {
    size_t videos = 0, audios = 0, soundtracks = 0;
    for (const auto& media : request.reference_media) {
      if (media->is_video()) { ++videos; soundtracks += bool(media->soundtrack()); }
      else ++audios;
    }
    description += "  reference videos    " + std::to_string(videos) + " (" +
        std::to_string(soundtracks) + " with audio)\n  reference audios    " + std::to_string(audios) +
        "\n  media conditioning  CUDA/Vulkan (packed row count includes video/audio references)\n";
  }
  if (request.motion_cache.active()) {
    const auto& m = request.motion_cache;
    char detail[256];
    std::snprintf(detail, sizeof(detail),
        "  MotionCache         threshold %.3f, strength %.2f, warmup %d, max skips %d, range %.2f..%.2f, subsample %d\n",
        m.reuse_threshold, m.motion_strength, m.warmup_steps, m.max_consecutive_skips,
        m.start_percent, m.end_percent, m.subsample_factor);
    description += detail;
  }
  return description;
}

}  // namespace slopfab
