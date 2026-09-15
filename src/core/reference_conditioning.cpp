#include "slopfab/reference_conditioning.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "slopfab/sampler/noise.h"
#include "slopfab/sampler/scheduler.h"

namespace slopfab {
ReferenceConditionPlan reference_condition_plan(const ReferenceMedia& reference,
                                                double target_seconds) {
  reference.validate();
  if (!std::isfinite(target_seconds) || target_seconds <= 0)
    throw std::invalid_argument(
        "reference conditioning: invalid target duration");
  ReferenceConditionPlan p;
  p.geometry.kind = reference.is_video() ? dit::ReferenceKind::kVideo
                                         : dit::ReferenceKind::kAudio;
  const double duration =
      std::min(reference.duration_seconds(), target_seconds);
  if (reference.is_video()) {
    const auto& image = reference.frames().front()->image;
    dit::resolve_canvas_size(image.width, image.height, &p.height, &p.width);
    p.frames = static_cast<int>(std::floor(duration * 24 + .5));
    if (p.frames < 22)
      throw std::invalid_argument(
          "reference video needs at least 22 target frames; still-image "
          "generation is unsupported");
    p.encoding_frames = ((p.frames - 5) / 17) * 17 + 5;
    p.geometry.num_latent_frames = ((p.encoding_frames - 5) / 17) * 5 + 2;
    p.geometry.latent_height = p.height / 16;
    p.geometry.latent_width = p.width / 16;
  }
  if (const auto& audio = reference.soundtrack()) {
    const double end =
        std::min(duration, audio->start_seconds + audio->duration_seconds());
    if (end <= audio->start_seconds)
      throw std::invalid_argument(
          "reference soundtrack starts beyond the generated duration");
    p.audio_samples = static_cast<int>(std::ceil(end * 32000 - 1e-9));
    p.geometry.num_audio_latents = (p.audio_samples + 799) / 800;
  }
  return p;
}

PreparedReference prepare_reference_condition(const ReferenceMedia& reference,
                                              double target_seconds,
                                              bool prepare_vae) {
  PreparedReference out;
  out.plan = reference_condition_plan(reference, target_seconds);
  if (reference.is_video()) {
    size_t source = 0;
    out.frames.reserve(out.plan.frames);
    for (int i = 0; i < out.plan.frames; ++i) {
      // On an encoded-media hit only Qwen's 2 fps presentation is needed.
      // Preserve indexing so its paired-frame construction stays identical.
      if (!prepare_vae && i % 12 != 0) {
        out.frames.emplace_back();
        continue;
      }
      // FFmpeg fps uses rounded PTS boundaries: hold each source frame until
      // the next one's destination slot, and the last until clip end.
      while (source + 1 < reference.frames().size() &&
             std::floor(reference.frames()[source + 1]->timestamp_seconds * 24 +
                        .5) <= i)
        ++source;
      const auto& image = reference.frames()[source]->image;
      out.frames.push_back(image.width == out.plan.width &&
                                   image.height == out.plan.height
                               ? image
                               : resize_reference_lanczos(image, out.plan.width,
                                                          out.plan.height));
    }
  }
  if (const auto& audio = reference.soundtrack(); audio && prepare_vae) {
    const int n = out.plan.audio_samples;
    out.audio.assign(size_t(2) * n, 0);
    const int offset =
        static_cast<int>(std::floor(audio->start_seconds * 32000 + .5));
    const int native_count = std::min<int>(
        static_cast<int>(audio->interleaved.size() / audio->channels),
        static_cast<int>(
            std::floor((std::min(reference.duration_seconds(), target_seconds) -
                        audio->start_seconds) *
                           audio->sample_rate +
                       1e-7)));
    constexpr double pi = 3.14159265358979323846;
    const double base = std::min(audio->sample_rate, 32000) * .99;
    const int radius =
        static_cast<int>(std::ceil(6 * audio->sample_rate / base));
    for (int i = offset; i < n; ++i) {
      const double position = double(i - offset) * audio->sample_rate / 32000;
      for (int channel = 0; channel < 2; ++channel) {
        const int src_channel = std::min(channel, audio->channels - 1);
        float sum = 0;
        if (audio->sample_rate == 32000) {
          const int j = i - offset;
          if (j < native_count)
            sum = audio->interleaved[size_t(j) * audio->channels + src_channel];
        } else {
          const int center = static_cast<int>(std::floor(position));
          for (int j = std::max(0, center - radius);
               j <= std::min(native_count - 1, center + radius); ++j) {
            const double t = (j - position) * base / audio->sample_rate;
            if (std::abs(t) >= 6) continue;
            const double window = std::cos(t * pi / 12);
            const double sinc = t == 0 ? 1 : std::sin(pi * t) / (pi * t);
            const float weight = static_cast<float>(sinc * window * window *
                                                    base / audio->sample_rate);
            sum +=
                audio->interleaved[size_t(j) * audio->channels + src_channel] *
                weight;
          }
        }
        out.audio[size_t(channel) * n + i] = sum;
      }
    }
  }
  return out;
}

void append_encoded_reference_condition(const EncodedReferenceCondition& encoded,
    uint64_t seed, size_t reference_index, std::vector<float>& video_rows,
    std::vector<float>& audio_rows) {
  if (!encoded.video_rows.empty()) {
    const auto& g = encoded.geometry;
    auto noise = sampler::video_noise(seed ^ (0x9e3779b97f4a7c15ULL * (reference_index + 1)),
        g.num_latent_frames, g.latent_height, g.latent_width);
    auto noise_rows = patchify_reference_video(noise.data(), g.num_latent_frames,
                                               g.latent_height, g.latent_width);
    if (noise_rows.size() != encoded.video_rows.size())
      throw std::invalid_argument("reference cache: video row geometry mismatch");
    auto rows = encoded.video_rows;
    sampler::FlowScheduler::scale_noise(rows.data(), noise_rows.data(), .999f,
                                        rows.size(), rows.data());
    video_rows.insert(video_rows.end(), rows.begin(), rows.end());
  }
  audio_rows.insert(audio_rows.end(), encoded.audio_rows.begin(), encoded.audio_rows.end());
}

std::vector<float> patchify_reference_video(const float* latents, int frames,
                                            int height, int width) {
  if (!latents || frames <= 0 || height <= 0 || width <= 0 || height % 2 ||
      width % 2)
    throw std::invalid_argument("reference video: invalid latent geometry");
  dit::SequenceLayout layout;
  layout.num_latent_frames = frames;
  layout.latent_height = height;
  layout.latent_width = width;
  layout.num_video_rows = frames * (height / 2) * (width / 2);
  std::vector<float> out(size_t(layout.num_video_rows) * 96);
  dit::patchify_video(latents, layout, out.data());
  return out;
}
}  // namespace slopfab
