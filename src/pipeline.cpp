#include "vidfab/pipeline.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <system_error>

#include "vidfab/sampler/scheduler.h"

namespace vidfab {
namespace {

// ref/FL2VA/model_index.json -> _minimax_h3.sigma_shift_scales, consumed by
// convert.py:630-653.
constexpr float kVideoSigmaShift = 12.0f;
constexpr float kAudioSigmaShift = 3.0f;

constexpr int kSpatialCompression = 16;
constexpr int kFps = 24;

// The video decoder consumes 7-token temporal windows (`tokens_chunk_size` 5 +
// `token_overlap` 2), so a latent shorter than that cannot be decoded at all.
// `F = 5k + 2` for `17k + 5` pixel frames, so `F >= 7` means `k >= 1` means at
// least 22 pixel frames.
constexpr int kMinLatentFrames = 7;
constexpr int kMinFrames = 22;

}  // namespace

GeneratePlan resolve_plan(const GenerateRequest& request) {
  if (request.reference_image_paths.size() > 9) {
    throw std::runtime_error("MiniMax-H3 Ref2VA accepts at most 9 reference images, got " +
                             std::to_string(request.reference_image_paths.size()));
  }
  if (request.num_inference_steps < 2) {
    throw std::runtime_error("num_inference_steps must be at least 2: the grid includes a "
                             "terminal sigma of zero that gets no model evaluation");
  }

  GeneratePlan plan;
  if (request.has_explicit_canvas()) {
    dit::validate_canvas_size(request.canvas_height, request.canvas_width);
    plan.canvas_height = request.canvas_height;
    plan.canvas_width = request.canvas_width;
  } else {
    dit::resolve_canvas_size(static_cast<double>(request.aspect_w),
                             static_cast<double>(request.aspect_h), &plan.canvas_height,
                             &plan.canvas_width);
  }

  plan.aligned_frames = dit::align_num_frames(request.num_frames);
  plan.duration_seconds = static_cast<double>(plan.aligned_frames) / kFps;

  // Checked here rather than in the decoder so the run fails in milliseconds
  // instead of after uploading 9 GB of VAE weights. The decoder does keep its
  // own guard — this one is about where the user finds out.
  if (dit::video_latent_num_frames(plan.aligned_frames) < kMinLatentFrames) {
    throw std::runtime_error(
        "num_frames = " + std::to_string(request.num_frames) + " aligns to " +
        std::to_string(plan.aligned_frames) + " frames, which is " +
        std::to_string(dit::video_latent_num_frames(plan.aligned_frames)) +
        " latent frames; the video decoder needs at least " +
        std::to_string(kMinLatentFrames) + ". Ask for at least 6 frames, which aligns up to " +
        std::to_string(kMinFrames) + ".");
  }

  plan.layout.num_text = 0;  // filled in after tokenisation
  plan.layout.num_condition_video = 0;  // t2va has no conditioning rows
  plan.layout.num_latent_frames = dit::video_latent_num_frames(plan.aligned_frames);
  plan.layout.latent_height = plan.canvas_height / kSpatialCompression;
  plan.layout.latent_width = plan.canvas_width / kSpatialCompression;
  plan.layout.num_audio_latents = dit::audio_latents_for_frames(plan.aligned_frames);
  plan.layout.num_audio_rows = 2 * plan.layout.num_audio_latents;
  plan.layout.num_video_rows = plan.layout.num_latent_frames * plan.layout.rows_per_frame();

  sampler::FlowScheduler video(kVideoSigmaShift);
  sampler::FlowScheduler audio(kAudioSigmaShift);
  video.set_timesteps(request.num_inference_steps);
  audio.set_timesteps(request.num_inference_steps);

  plan.video_sigmas = video.sigmas();
  plan.audio_sigmas = audio.sigmas();
  plan.video_timesteps = video.timesteps();
  plan.audio_timesteps = audio.timesteps();

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

std::string conditioning_cache_key(const GenerateRequest& request) {
  std::string key = "conditioning";
  append_file_identity(key, request.text_encoder_path);
  append_file_identity(key, request.tokenizer_path);
  key.push_back('\0');
  key += request.prompt;
  for (const std::string& path : request.reference_image_paths) {
    append_file_identity(key, path);
  }
  return key;
}

std::string reference_cache_key(const GenerateRequest& request) {
  std::string key = "reference";
  append_file_identity(key, request.video_vae_path);
  for (const std::string& path : request.reference_image_paths) {
    append_file_identity(key, path);
  }
  return key;
}

std::string tokenizer_cache_key(const GenerateRequest& request) {
  std::string key = "tokenizer";
  append_file_identity(key, request.tokenizer_path);
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
                  dit::canvas_exceeds_trained_area(plan.canvas_height, plan.canvas_width)
                      ? ", above the trained area"
                      : "");
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
      "  frames              %d requested -> %d aligned (%.2f s at %d fps)\n"
      "  latent grid         %d frames of %d x %d  -> %d rows per frame\n"
      "  audio latents       %d per channel -> %d rows\n"
      "  packed sequence     %d rows + prompt length\n"
      "  steps               %d grid points -> %d model evaluations\n"
      "  sigma range         video %.6f .. %.6f (shift %.1f)\n"
      "                      audio %.6f .. %.6f (shift %.1f)\n"
      "  seed                %llu\n"
      "  output              %s\n",
      request.prompt.size(), request.reference_image_paths.size(),
      request.reference_image_paths.empty() ? " (text-to-video)" : " (Ref2VA, ordered)",
      plan.canvas_height, plan.canvas_width, provenance,
      request.num_frames, plan.aligned_frames, plan.duration_seconds, kFps,
      l.num_latent_frames, l.latent_height, l.latent_width, l.rows_per_frame(),
      l.num_audio_latents, l.num_audio_rows, l.total_rows(), request.num_inference_steps,
      plan.num_model_evaluations(), static_cast<double>(plan.video_sigmas.front()),
      static_cast<double>(plan.video_sigmas[plan.video_sigmas.size() - 2]),
      static_cast<double>(kVideoSigmaShift), static_cast<double>(plan.audio_sigmas.front()),
      static_cast<double>(plan.audio_sigmas[plan.audio_sigmas.size() - 2]),
      static_cast<double>(kAudioSigmaShift),
      static_cast<unsigned long long>(request.seed), request.out_path.c_str());
  return buf;
}

}  // namespace vidfab
