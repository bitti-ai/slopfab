#include "slopfab/refmod_export.h"
#include "slopfab/refmod.h"
#include "slopfab/reference_conditioning.h"
#include "slopfab/dit/packing.h"
#include "slopfab/pipeline.h"
#include "helpers.h"
#include <filesystem>
#include <stdexcept>
#if SLOPFAB_WITH_CUDA
#include "slopfab/vae/keyframe_encoder.h"
#include "slopfab/vae/audio_encoder.h"
#endif
#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/keyframe_encoder.h"
#include "slopfab/vulkan/reference_encoder.h"
#endif

namespace slopfab {
namespace {
std::string media_identity_hex(const ReferenceMedia& media) {
  const auto bytes = reference_media_identity(media);
  std::string result;
  for (unsigned char byte : bytes) {
    result += "0123456789abcdef"[byte >> 4];
    result += "0123456789abcdef"[byte & 15];
  }
  return result;
}
}

int export_refmod(const RefModExportRequest& request, const std::string& output) {
  if (output.empty() || (request.image_paths.empty() && request.media.empty()))
    throw std::invalid_argument("refmod export: raw references and output are required");
  if (request.short_edge < 32 || request.short_edge > 768 || request.short_edge % 32)
    throw std::invalid_argument("refmod export: short edge must be a multiple of 32 in 32..768");
  if (request.backend != DeviceBackend::kCuda && request.backend != DeviceBackend::kVulkan)
    throw std::invalid_argument("refmod export: invalid backend");
#if !SLOPFAB_WITH_CUDA
  if (request.backend == DeviceBackend::kCuda)
    throw std::runtime_error("refmod export: CUDA backend is not compiled in");
#endif
#if !SLOPFAB_WITH_VULKAN
  if (request.backend == DeviceBackend::kVulkan)
    throw std::runtime_error("refmod export: Vulkan backend is not compiled in");
#endif
  validate_reference_media(request.image_paths.size(), request.media);
  const auto destination = std::filesystem::u8path(output);
  const auto parent =
      destination.has_parent_path() ? destination.parent_path() : std::filesystem::path(".");
  if (!std::filesystem::is_directory(parent))
    throw std::invalid_argument("refmod export: output directory does not exist");
  const std::string name = request.name.empty() ? destination.stem().u8string() : request.name;
  ReferenceConditionOptions options;
  options.short_edge = request.short_edge;
  std::vector<ReferenceConditionPlan> plans;
  bool visual = !request.image_paths.empty(), audio = false, video = false;
  for (const auto& media : request.media) {
    if (!media)
      throw std::invalid_argument("refmod export: null media");
    plans.push_back(reference_condition_plan(*media, media->duration_seconds(), options));
    video |= media->is_video();
    audio |= plans.back().audio_samples > 0;
  }
  visual |= video;
  if ((visual && request.video_vae_path.empty()) || (audio && request.audio_vae_path.empty()))
    throw std::invalid_argument("refmod export: required video/audio VAE path is missing");
  std::vector<std::shared_ptr<const RefMod>> images;
  std::vector<std::shared_ptr<const RefMod>> videos(request.media.size()),
      sounds(request.media.size());
  std::vector<std::vector<float>> prepared_audio(request.media.size());
  const auto metadata = [&](const std::string& source, const std::string& vae) {
    std::string identity;
    append_file_identity(identity, vae);
    return json::Object{
        {"name", json::Value(name)},
        {"description", json::Value(request.description)},
        {"source", json::Value(source)},
        {"vae", json::Value(vae)},
        {"vae_identity", json::Value(identity)},
        {"backend",
         json::Value(std::string(request.backend == DeviceBackend::kCuda ? "cuda" : "vulkan"))}};
  };
  if (visual) {
    SafeTensors checkpoint;
    checkpoint.open(request.video_vae_path);
    const auto mean = generation::read_stat(checkpoint, "latents_mean", 24);
    const auto stddev = generation::read_stat(checkpoint, "latents_std", 24);
    const auto encode_images = [&](auto& encoder) {
      for (const auto& path : request.image_paths) {
        auto image = load_reference_image(path);
        int h, w;
        dit::resolve_canvas_size(image.width, image.height, &h, &w, options.short_edge,
                                 options.max_pixels);
        if (image.width != w || image.height != h)
          image = resize_reference_lanczos(image, w, h);
        auto meta = metadata(path, request.video_vae_path);
        meta["width"] = json::Value(double(w));
        meta["height"] = json::Value(double(h));
        meta["posterior_seed"] = json::Value(42.0);
        images.push_back(RefMod::from_rows({dit::ReferenceKind::kImage, 1, h / 16, w / 16, 0},
                                           encoder.encode_reference_image(image, mean, stddev),
                                           std::move(meta)));
      }
    };
    const auto encode_videos = [&](auto& encoder) {
      for (size_t i = 0; i < request.media.size(); ++i) {
        const auto& media = *request.media[i];
        if (!media.is_video())
          continue;
        auto prepared = prepare_reference_condition(media, media.duration_seconds(), true, options);
        auto g = prepared.plan.geometry;
        g.num_audio_latents = 0;
        auto meta = metadata("video", request.video_vae_path);
        meta["source_identity"] = json::Value(media_identity_hex(media));
        meta["source_seconds"] = json::Value(media.duration_seconds());
        meta["encoding_frames"] = json::Value(double(prepared.plan.encoding_frames));
        meta["fps"] = json::Value(24.0);
        meta["posterior_seed"] = json::Value(42.0);
        videos[i] =
            RefMod::from_rows(g,
                              encoder.encode_reference_video(
                                  prepared.frames, prepared.plan.encoding_frames, mean, stddev),
                              std::move(meta));
        prepared_audio[i] = std::move(prepared.audio);
      }
    };
    if (request.backend == DeviceBackend::kCuda) {
#if SLOPFAB_WITH_CUDA
      vae::KeyframeEncoder encoder(checkpoint);
      encode_images(encoder);
      encode_videos(encoder);
#endif
    } else {
#if SLOPFAB_WITH_VULKAN
      auto device = generation::create_vulkan_inference_device(request.vulkan_portable_arithmetic);
      if (!request.image_paths.empty()) {
        auto encoder = vulkan::KeyframeEncoder::create(device);
        encoder.load(checkpoint);
        encode_images(encoder);
      }
      if (video) {
        vulkan::ReferenceEncoder encoder(device, checkpoint, false);
        encode_videos(encoder);
      }
#endif
    }
  }
  if (audio) {
    SafeTensors checkpoint;
    checkpoint.open(request.audio_vae_path);
    const auto encode_audio = [&](auto& encoder) {
      for (size_t i = 0; i < request.media.size(); ++i) {
        if (!plans[i].audio_samples)
          continue;
        const auto& media = *request.media[i];
        if (!media.is_video())
          prepared_audio[i] =
              prepare_reference_condition(media, media.duration_seconds(), true, options).audio;
        auto meta =
            metadata(media.is_video() ? "video_soundtrack" : "audio", request.audio_vae_path);
        meta["source_identity"] = json::Value(media_identity_hex(media));
        meta["source_seconds"] = json::Value(media.duration_seconds());
        meta["audio_samples"] = json::Value(double(plans[i].audio_samples));
        sounds[i] = RefMod::from_rows(
            {dit::ReferenceKind::kAudio, 0, 0, 0, plans[i].geometry.num_audio_latents},
            encoder.encode_reference(prepared_audio[i].data(), plans[i].audio_samples),
            std::move(meta));
        prepared_audio[i].clear();
      }
    };
    if (request.backend == DeviceBackend::kCuda) {
#if SLOPFAB_WITH_CUDA
      vae::AudioEncoder encoder(checkpoint);
      encode_audio(encoder);
#endif
    } else {
#if SLOPFAB_WITH_VULKAN
      auto device = generation::create_vulkan_inference_device(request.vulkan_portable_arithmetic);
      vulkan::ReferenceEncoder encoder(device, checkpoint, true);
      encode_audio(encoder);
#endif
    }
  }
  RefModBundle bundle;
  bundle.members = std::move(images);
  for (size_t i = 0; i < request.media.size(); ++i) {
    if (videos[i])
      bundle.members.push_back(videos[i]);
    if (sounds[i])
      bundle.members.push_back(sounds[i]);
  }
  bundle.metadata = {{"name", json::Value(name)},
                     {"description", json::Value(request.description)}};
  if (bundle.members.size() == 1)
    bundle.members.front()->save(output);
  else
    bundle.save(output);
  return static_cast<int>(bundle.members.size());
}
} // namespace slopfab
