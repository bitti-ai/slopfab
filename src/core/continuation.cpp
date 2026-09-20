#include "slopfab/continuation.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sampler/noise.h"

namespace slopfab {
namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(std::string("latents: ") + message);
}
int integer(const SafeTensors& file, const char* key) {
  auto it = file.metadata().find(key);
  require(it != file.metadata().end(), "missing geometry metadata; use --save-latents, not an old diagnostic dump");
  size_t used = 0;
  const int value = std::stoi(it->second, &used);
  require(used == it->second.size(), "invalid integer metadata");
  return value;
}
std::string meta(const SafeTensors& file, const char* key) {
  auto it = file.metadata().find(key);
  return it == file.metadata().end() ? std::string() : it->second;
}
void read_rows(const SafeTensors& file, const char* name, int rows, int columns,
               std::vector<float>& output) {
  const auto& tensor = file.at(name);
  require(tensor.dtype == DType::kF32 && tensor.shape == std::vector<int64_t>{rows, columns},
          "tensor dtype or shape disagrees with metadata");
  output.resize(static_cast<size_t>(rows) * columns);
  if (!output.empty()) std::memcpy(output.data(), tensor.data, output.size() * sizeof(float));
  require(std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x); }),
          "non-finite latent value");
}

void validate_plan(const LatentClip& source, const ContinuationPlan& p) {
  const auto expected = plan_continuation(source, p.overlap_frames, p.extension_frames);
  require(p.extension_frames == expected.extension_frames && p.window_frames == expected.window_frames &&
              p.output_frames == expected.output_frames && p.overlap_video_latents == expected.overlap_video_latents &&
              p.overlap_audio_latents == expected.overlap_audio_latents && p.window_audio_latents == expected.window_audio_latents,
          "continuation plan does not match its source timeline");
}
}  // namespace

dit::SequenceLayout LatentClip::layout() const {
  require_h3_latent_geometry(geometry);
  dit::validate_canvas_size(height, width);
  require(frames == 1 || (frames >= 22 && frames % 17 == 5), "invalid frame count");
  // Packed row indices and downstream tensor counts use signed 32-bit integers.
  const int64_t f = frames == 1 ? 1 : dit::video_latent_num_frames(frames);
  const int64_t rows_per_frame = int64_t(height / 32) * (width / 32);
  require(rows_per_frame <= std::numeric_limits<int>::max() / 96 &&
              f <= (std::numeric_limits<int>::max() / 96) / rows_per_frame,
          "video geometry exceeds supported tensor sizes");
  const int64_t v = f * rows_per_frame;
  const int64_t a = frames == 1 ? 0 : (int64_t(frames) * 5 + 1) / 3;
  require(v <= std::numeric_limits<int>::max() / 96 &&
              a <= std::numeric_limits<int>::max() / 64,
          "clip geometry exceeds supported tensor sizes");
  dit::SequenceLayout l;
  l.latent_height = height / 16;
  l.latent_width = width / 16;
  l.num_latent_frames = static_cast<int>(f);
  l.num_video_rows = static_cast<int>(v);
  l.num_audio_latents = static_cast<int>(a);
  l.num_audio_rows = 2 * l.num_audio_latents;
  return l;
}

void LatentClip::validate() const {
  const auto l = layout();
  require(video_rows.size() == size_t(l.num_video_rows) * 96 &&
              audio_rows.size() == size_t(l.num_audio_rows) * 32,
          "buffer sizes disagree with clip geometry");
}

void LatentClip::save(const std::string& path) const {
  validate();
  require(!path.empty(), "empty output path");
  const auto l = layout();
  static std::atomic<uint64_t> serial{0};
  const auto target = std::filesystem::u8path(path);
  auto temporary = target;
  temporary += ".pending-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
               "-" + std::to_string(serial.fetch_add(1));
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
  } cleanup{temporary};
  write_safetensors(temporary.u8string(),
      {{"video_rows", {l.num_video_rows, 96}, video_rows},
       {"audio_rows", {l.num_audio_rows, 32}, audio_rows}},
      {{"slopfab_latents", "h3-av-v1"}, {"width", std::to_string(width)},
       {"height", std::to_string(height)}, {"frames", std::to_string(frames)},
       {"fps", std::to_string(geometry.fps)}, {"sampled", sampled ? "1" : "0"},
       {"slopfab.geometry", geometry_json(geometry)},
       {"transformer", transformer}, {"video_vae", video_vae}, {"audio_vae", audio_vae}});
#ifdef _WIN32
  if (!MoveFileExW(temporary.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    throw std::runtime_error("latents: cannot replace archive (Windows error " + std::to_string(GetLastError()) + ")");
#else
  std::filesystem::rename(temporary, target);
#endif
}

std::shared_ptr<const LatentClip> LatentClip::load(const std::string& path) {
  SafeTensors file;
  file.open(path);
  require(meta(file, "slopfab_latents") == "h3-av-v1",
          "unsupported archive; continuation requires a file written by --save-latents");
  require(meta(file, "fps") == "24", "only 24 fps H3 latents are supported");
  auto clip = std::make_shared<LatentClip>();
  clip->geometry = read_model_geometry(file);
  clip->width = integer(file, "width");
  clip->height = integer(file, "height");
  clip->frames = integer(file, "frames");
  const auto sampled = meta(file, "sampled");
  require(sampled == "0" || sampled == "1", "invalid sampled flag");
  clip->sampled = sampled == "1";
  clip->transformer = meta(file, "transformer");
  clip->video_vae = meta(file, "video_vae");
  clip->audio_vae = meta(file, "audio_vae");
  const auto l = clip->layout();
  read_rows(file, "video_rows", l.num_video_rows, 96, clip->video_rows);
  read_rows(file, "audio_rows", l.num_audio_rows, 32, clip->audio_rows);
  return clip;
}

ContinuationPlan plan_continuation(const LatentClip& source, int overlap, int extension) {
  source.validate();
  require(source.sampled && source.frames >= 22, "continuation needs completed video latents");
  require(overlap >= 5 && overlap % 17 == 5 && overlap <= source.frames,
          "overlap must be 17*k+5 frames, at least 5, and fit in the source clip");
  require(extension > 0, "extension must be positive");
  const int64_t added = ((int64_t(extension) + 16) / 17) * 17;
  require(int64_t(source.frames) + added <= std::numeric_limits<int>::max(), "extension is too long");
  ContinuationPlan p;
  p.overlap_frames = overlap;
  p.extension_frames = static_cast<int>(added);
  p.window_frames = overlap + p.extension_frames;
  p.output_frames = source.frames + p.extension_frames;
  p.overlap_video_latents = dit::video_latent_num_frames(overlap);
  const int64_t boundary = (int64_t(source.frames - overlap) * 5 + 1) / 3;
  const int64_t output_audio = (int64_t(p.output_frames) * 5 + 1) / 3;
  require(output_audio <= std::numeric_limits<int>::max() / 64, "audio timeline is too long");
  p.overlap_audio_latents = source.layout().num_audio_latents - static_cast<int>(boundary);
  p.window_audio_latents = static_cast<int>(output_audio - boundary);
  LatentClip output_geometry;
  output_geometry.geometry = source.geometry;
  output_geometry.width = source.width;
  output_geometry.height = source.height;
  output_geometry.frames = p.output_frames;
  (void)output_geometry.layout();
  return p;
}

void append_continuation_guide(const LatentClip& source, const ContinuationPlan& p,
                               uint64_t seed, std::vector<dit::ReferenceGeometry>& geometry,
                               std::vector<float>& video, std::vector<float>& audio) {
  validate_plan(source, p);
  const auto l = source.layout();
  const size_t n = size_t(p.overlap_video_latents) * l.rows_per_frame() * 96;
  require(n <= source.video_rows.size() && p.overlap_audio_latents <= l.num_audio_latents,
          "guide exceeds source");
  // Match H3 visual condition augmentation (t=0.999); audio anchors are clean.
  const auto noise = sampler::video_noise(seed, p.overlap_video_latents, l.latent_height, l.latent_width);
  auto guide_layout = l;
  guide_layout.num_latent_frames = p.overlap_video_latents;
  guide_layout.num_video_rows = p.overlap_video_latents * l.rows_per_frame();
  std::vector<float> packed(n);
  dit::patchify_video(noise.data(), guide_layout, packed.data());
  const size_t start = source.video_rows.size() - n;
  for (size_t i = 0; i < n; ++i)
    video.push_back(0.999f * source.video_rows[start + i] + (1.0f - 0.999f) * packed[i]);
  for (int c = 0; c < 2; ++c) {
    const auto end = source.audio_rows.begin() + size_t(c + 1) * l.num_audio_latents * 32;
    audio.insert(audio.end(), end - size_t(p.overlap_audio_latents) * 32, end);
  }
  geometry.push_back({dit::ReferenceKind::kVideo, p.overlap_video_latents,
                     l.latent_height, l.latent_width, p.overlap_audio_latents, true});
}

LatentClip join_continuation(const LatentClip& source, const ContinuationPlan& p,
                            const std::vector<float>& video, const std::vector<float>& audio) {
  validate_plan(source, p);
  const auto l = source.layout();
  const size_t frame_size = size_t(l.rows_per_frame()) * 96;
  require(video.size() == size_t(dit::video_latent_num_frames(p.window_frames)) * frame_size &&
              audio.size() == size_t(p.window_audio_latents) * 64,
          "sampled continuation window has the wrong shape");
  LatentClip out;
  out.geometry = source.geometry;
  out.width = source.width; out.height = source.height; out.frames = p.output_frames;
  out.transformer = source.transformer; out.video_vae = source.video_vae; out.audio_vae = source.audio_vae;
  out.video_rows = source.video_rows;
  out.video_rows.insert(out.video_rows.end(), video.begin() + size_t(p.overlap_video_latents) * frame_size, video.end());
  for (int c = 0; c < 2; ++c) {
    out.audio_rows.insert(out.audio_rows.end(), source.audio_rows.begin() + size_t(c) * l.num_audio_latents * 32,
                          source.audio_rows.begin() + size_t(c + 1) * l.num_audio_latents * 32);
    out.audio_rows.insert(out.audio_rows.end(), audio.begin() + (size_t(c) * p.window_audio_latents + p.overlap_audio_latents) * 32,
                          audio.begin() + size_t(c + 1) * p.window_audio_latents * 32);
  }
  out.validate();
  return out;
}
}  // namespace slopfab
