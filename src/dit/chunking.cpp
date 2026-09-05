#include "slopfab/dit/chunking.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "slopfab/sampler/noise.h"

namespace slopfab::dit {
namespace {

// The rotary temporal grid repeats every 5 latent frames (spec 2.3), so an
// offset that is a multiple of 5 gives a chunk the same *internal* frame
// spacing as the frames it stands in for. Any other offset does not.
constexpr int kLatentsPerChunkPeriod = 5;
constexpr int kFramesPerLatent[5] = {1, 4, 4, 4, 4};

// Audio latents per pixel frame: 40 per second against 24 fps.
constexpr double kAudioLatentsPerFrame = 40.0 / 24.0;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("chunking: " + message);
}

// Pixel frames covered by latent frames [0, f), i.e. the cumulative sum of
// FRAMES_PER_LATENT. Used only to report how far the audio offsets sit from
// where the video offsets put them.
int pixel_frames_before(int f) {
  int total = 0;
  for (int j = 0; j < f; ++j) total += kFramesPerLatent[j % kLatentsPerChunkPeriod];
  return total;
}

// The trapezoidal window described in the header. `lead`/`tail` are the ramp
// lengths, zero at the ends of the chunk sequence.
double window_weight(int j, int extent, int lead, int tail) {
  double w = 1.0;
  if (lead > 0 && j < lead) w = std::min(w, (static_cast<double>(j) + 0.5) / lead);
  if (tail > 0 && j >= extent - tail) {
    w = std::min(w, (static_cast<double>(extent - j) - 0.5) / tail);
  }
  return std::max(w, 0.0);
}

// Accumulates `src` into `dst` with the chunk window, one "slice" being a
// contiguous run of `slice_width` floats.
void accumulate(const float* src, int extent, int lead, int tail, int slice_width, int dst_offset,
                std::vector<float>& dst, std::vector<double>& weight) {
  for (int j = 0; j < extent; ++j) {
    const double w = window_weight(j, extent, lead, tail);
    const size_t out_slice = static_cast<size_t>(dst_offset + j);
    weight[out_slice] += w;
    const float* in = src + static_cast<size_t>(j) * slice_width;
    float* out = dst.data() + out_slice * slice_width;
    for (int k = 0; k < slice_width; ++k) out[k] += static_cast<float>(w * in[k]);
  }
}

void normalise(std::vector<float>& data, const std::vector<double>& weight, int slice_width) {
  for (size_t s = 0; s < weight.size(); ++s) {
    require(weight[s] > 0.0, "a slice of the full request is covered by no chunk");
    const double inv = 1.0 / weight[s];
    float* out = data.data() + s * slice_width;
    for (int k = 0; k < slice_width; ++k) out[k] = static_cast<float>(out[k] * inv);
  }
}

}  // namespace

ChunkPlan resolve_chunk_plan(const SequenceLayout& full, const SequenceLayout& chunk,
                             int num_chunks) {
  require(num_chunks >= 2, "a chunked run needs at least two chunks");
  require(full.latent_height == chunk.latent_height && full.latent_width == chunk.latent_width,
          "the full request and the chunk must share a canvas");
  require(chunk.num_latent_frames < full.num_latent_frames,
          "the chunk must be shorter than the full request");
  require(chunk.num_audio_latents < full.num_audio_latents,
          "the chunk must carry fewer audio latents than the full request");

  ChunkPlan plan;
  plan.num_chunks = num_chunks;
  plan.full_latent_frames = full.num_latent_frames;
  plan.chunk_latent_frames = chunk.num_latent_frames;
  plan.full_audio_latents = full.num_audio_latents;
  plan.chunk_audio_latents = chunk.num_audio_latents;

  const int frame_span = full.num_latent_frames - chunk.num_latent_frames;
  const int divisor = num_chunks - 1;
  require(frame_span % divisor == 0,
          "chunks do not tile: " + std::to_string(full.num_latent_frames) + " latent frames minus " +
              std::to_string(chunk.num_latent_frames) + " is not divisible by " +
              std::to_string(divisor));
  plan.latent_stride = frame_span / divisor;
  require(plan.latent_stride > 0, "the chunk stride must be positive");
  require(plan.latent_stride % kLatentsPerChunkPeriod == 0,
          "the chunk stride is " + std::to_string(plan.latent_stride) +
              " latent frames, which is not a multiple of 5; the rotary temporal grid repeats "
              "every 5 latent frames, so any other stride would give each chunk a different "
              "internal frame spacing from the frames it stands in for");
  plan.frame_overlap = plan.chunk_latent_frames - plan.latent_stride;
  require(plan.frame_overlap > 0,
          "the chunks do not overlap; there would be nothing to cross-fade");

  const int audio_span = full.num_audio_latents - chunk.num_audio_latents;
  require(audio_span % divisor == 0,
          "the audio latents do not tile: " + std::to_string(full.num_audio_latents) + " minus " +
              std::to_string(chunk.num_audio_latents) + " is not divisible by " +
              std::to_string(divisor));
  plan.audio_stride = audio_span / divisor;
  plan.audio_overlap = plan.chunk_audio_latents - plan.audio_stride;
  require(plan.audio_overlap > 0, "the chunks' audio does not overlap");

  plan.frame_offset.resize(static_cast<size_t>(num_chunks));
  plan.audio_offset.resize(static_cast<size_t>(num_chunks));
  for (int k = 0; k < num_chunks; ++k) {
    plan.frame_offset[static_cast<size_t>(k)] = k * plan.latent_stride;
    plan.audio_offset[static_cast<size_t>(k)] = k * plan.audio_stride;

    // Where the video offset says the audio offset should be. The two clocks
    // are commensurate only when the pixel-frame count is a multiple of 3, so
    // this is a fraction of an audio latent (25 ms) and is reported, not fixed:
    // rounding it away would be silently wrong and adjusting the stride per
    // chunk would stop the audio tiling exactly.
    const double ideal =
        pixel_frames_before(plan.frame_offset[static_cast<size_t>(k)]) * kAudioLatentsPerFrame;
    plan.worst_audio_drift_latents =
        std::max(plan.worst_audio_drift_latents,
                 std::fabs(ideal - plan.audio_offset[static_cast<size_t>(k)]));
  }

  require(plan.frame_offset.back() + plan.chunk_latent_frames == plan.full_latent_frames,
          "the last chunk does not end where the full request does");
  require(plan.audio_offset.back() + plan.chunk_audio_latents == plan.full_audio_latents,
          "the last chunk's audio does not end where the full request's does");
  return plan;
}

void slice_chunk_noise(uint64_t seed, const SequenceLayout& full, const SequenceLayout& chunk,
                       const ChunkPlan& plan, int index, std::vector<float>* video_rows_out,
                       std::vector<float>* audio_rows_out) {
  require(index >= 0 && index < plan.num_chunks, "chunk index out of range");
  require(video_rows_out != nullptr && audio_rows_out != nullptr, "null output");

  constexpr int kChannels = 24;
  constexpr int kAudioDim = 32;
  const int Hl = full.latent_height;
  const int Wl = full.latent_width;
  const int Fc = chunk.num_latent_frames;
  const int f0 = plan.frame_offset[static_cast<size_t>(index)];

  // The full draw, in the `(24, F, Hl, Wl)` layout `patchify_video` consumes,
  // then a contiguous copy of frames [f0, f0 + Fc) per channel. Drawing the
  // full field and slicing it is the whole point: `video_noise(seed, Fc, ...)`
  // would be a *different* field, because the generator is indexed by flat
  // position and F changes the stride.
  const std::vector<float> field =
      sampler::video_noise(seed, full.num_latent_frames, Hl, Wl, kChannels);
  const size_t frame_stride = static_cast<size_t>(Hl) * Wl;
  std::vector<float> sub(static_cast<size_t>(kChannels) * Fc * frame_stride);
  for (int c = 0; c < kChannels; ++c) {
    const float* src = field.data() + (static_cast<size_t>(c) * full.num_latent_frames + f0) *
                                          frame_stride;
    float* dst = sub.data() + static_cast<size_t>(c) * Fc * frame_stride;
    std::copy(src, src + static_cast<size_t>(Fc) * frame_stride, dst);
  }
  video_rows_out->assign(static_cast<size_t>(chunk.num_video_rows) * kChannels * 4, 0.0f);
  patchify_video(sub.data(), chunk, video_rows_out->data());

  // Audio is drawn directly in row layout `(2A, 32)`, channel-major: rows
  // [0, A) are channel 0 and [A, 2A) channel 1 (spec 1.3). The slice is
  // therefore two runs, not one.
  const int A = full.num_audio_latents;
  const int Ac = chunk.num_audio_latents;
  const int a0 = plan.audio_offset[static_cast<size_t>(index)];
  const std::vector<float> audio_field = sampler::audio_noise(seed, A, kAudioDim);
  audio_rows_out->assign(static_cast<size_t>(2 * Ac) * kAudioDim, 0.0f);
  for (int c = 0; c < 2; ++c) {
    const float* src = audio_field.data() + (static_cast<size_t>(c) * A + a0) * kAudioDim;
    float* dst = audio_rows_out->data() + static_cast<size_t>(c) * Ac * kAudioDim;
    std::copy(src, src + static_cast<size_t>(Ac) * kAudioDim, dst);
  }
}

void blend_chunks(const SequenceLayout& full, const SequenceLayout& chunk, const ChunkPlan& plan,
                  const std::vector<std::vector<float>>& chunk_video,
                  const std::vector<std::vector<float>>& chunk_audio,
                  std::vector<float>* video_rows_out, std::vector<float>* audio_rows_out) {
  require(video_rows_out != nullptr && audio_rows_out != nullptr, "null output");
  require(static_cast<int>(chunk_video.size()) == plan.num_chunks &&
              static_cast<int>(chunk_audio.size()) == plan.num_chunks,
          "expected one video and one audio buffer per chunk");

  constexpr int kVideoFeature = 96;
  constexpr int kAudioDim = 32;
  const int R = full.rows_per_frame();
  require(R == chunk.rows_per_frame(), "the two geometries disagree on rows per latent frame");

  const int Fc = plan.chunk_latent_frames;
  const int Ac = plan.chunk_audio_latents;
  const int frame_slice = R * kVideoFeature;

  video_rows_out->assign(static_cast<size_t>(full.num_video_rows) * kVideoFeature, 0.0f);
  audio_rows_out->assign(static_cast<size_t>(full.num_audio_rows) * kAudioDim, 0.0f);
  std::vector<double> frame_weight(static_cast<size_t>(plan.full_latent_frames), 0.0);
  // Audio weight is per (channel, latent) slice; the two channels are blended
  // with the same window but live in different halves of the row buffer.
  std::vector<double> audio_weight(static_cast<size_t>(2 * plan.full_audio_latents), 0.0);

  for (int k = 0; k < plan.num_chunks; ++k) {
    const std::vector<float>& v = chunk_video[static_cast<size_t>(k)];
    const std::vector<float>& a = chunk_audio[static_cast<size_t>(k)];
    require(v.size() == static_cast<size_t>(Fc) * frame_slice,
            "chunk " + std::to_string(k) + " video buffer has the wrong length");
    require(a.size() == static_cast<size_t>(2 * Ac) * kAudioDim,
            "chunk " + std::to_string(k) + " audio buffer has the wrong length");

    const int lead = (k == 0) ? 0 : plan.frame_overlap;
    const int tail = (k == plan.num_chunks - 1) ? 0 : plan.frame_overlap;
    accumulate(v.data(), Fc, lead, tail, frame_slice, plan.frame_offset[static_cast<size_t>(k)],
               *video_rows_out, frame_weight);

    const int a_lead = (k == 0) ? 0 : plan.audio_overlap;
    const int a_tail = (k == plan.num_chunks - 1) ? 0 : plan.audio_overlap;
    for (int c = 0; c < 2; ++c) {
      const int dst = c * plan.full_audio_latents + plan.audio_offset[static_cast<size_t>(k)];
      accumulate(a.data() + static_cast<size_t>(c) * Ac * kAudioDim, Ac, a_lead, a_tail, kAudioDim,
                 dst, *audio_rows_out, audio_weight);
    }
  }

  normalise(*video_rows_out, frame_weight, frame_slice);
  normalise(*audio_rows_out, audio_weight, kAudioDim);
}

}  // namespace slopfab::dit
