#include "slopfab/dit/motion_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace slopfab::dit {
namespace {
constexpr float epsilon = 1e-6f;

float mean(const std::vector<float>& values, const std::vector<float>* previous,
           const std::vector<float>& weights = {}) {
  if (values.empty())
    return 0;
  double total = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    const float value = values[i] - (previous ? (*previous)[i] : 0.0f);
    total += std::fabs(value) * (weights.empty() ? 1.0f : weights[i % weights.size()]);
  }
  return static_cast<float>(total / values.size());
}

float sigma_at(float percent, float shift) {
  const float base = 1.0f - percent;
  return shift * base / (1.0f + (shift - 1.0f) * base);
}
} // namespace

void MotionCacheConfig::validate() const {
  if (!std::isfinite(reuse_threshold) || reuse_threshold < 0 || reuse_threshold > 1 ||
      !std::isfinite(motion_strength) || motion_strength < 0 || motion_strength > 4 ||
      warmup_steps < 2 || warmup_steps > 20 || max_consecutive_skips < 1 ||
      max_consecutive_skips > 10 || !std::isfinite(start_percent) || !std::isfinite(end_percent) ||
      start_percent < 0 || end_percent > 1 || start_percent >= end_percent ||
      subsample_factor < 1 || subsample_factor > 32)
    throw std::invalid_argument(
        "MotionCache: invalid threshold, motion strength, warmup, skip limit, range or subsampling");
}

MotionCache::MotionCache(const MotionCacheConfig& config, const SequenceLayout& l, int video_dim,
                         int audio_dim, int steps, float video_shift, bool pin_audio)
    : config_(config), steps_(steps), frames_(l.num_latent_frames), channels_(video_dim / 4),
      start_sigma_(0), end_sigma_(0) {
  config_.validate();
  if (!enabled())
    return;
  if (frames_ <= 0 || l.latent_height <= 0 || l.latent_width <= 0 || l.latent_height % 2 ||
      l.latent_width % 2 || video_dim <= 0 || video_dim % 4 || audio_dim <= 0 ||
      l.num_audio_latents < 0 ||
      int64_t(l.num_video_rows) !=
          int64_t(frames_) * (l.latent_height / 2) * (l.latent_width / 2) ||
      int64_t(l.num_audio_rows) != 2 * int64_t(l.num_audio_latents) || steps <= 0 ||
      !std::isfinite(video_shift) || video_shift <= 0)
    throw std::invalid_argument("MotionCache: invalid target geometry or schedule");
  start_sigma_ = sigma_at(config.start_percent, video_shift);
  end_sigma_ = sigma_at(config.end_percent, video_shift);
  video_size_ = size_t(l.num_video_rows) * video_dim;
  audio_size_ = pin_audio ? 0 : size_t(l.num_audio_rows) * audio_dim;
  const int stride = config.subsample_factor;
  spatial_ =
      size_t((l.latent_height + stride - 1) / stride) * ((l.latent_width + stride - 1) / stride);
  // Match video[..., ::factor, ::factor], including offsets inside each patch.
  for (int c = 0; c < channels_; ++c)
    for (int f = 0; f < frames_; ++f)
      for (int h = 0; h < l.latent_height; h += stride)
        for (int w = 0; w < l.latent_width; w += stride) {
          const size_t row =
              (size_t(f) * (l.latent_height / 2) + h / 2) * (l.latent_width / 2) + w / 2;
          video_indices_.push_back(row * video_dim + c * 4 + (h % 2) * 2 + w % 2);
        }
  if (audio_size_)
    for (int c = 0; c < audio_dim; ++c)
      for (int stereo = 0; stereo < 2; ++stereo)
        for (int t = 0; t < l.num_audio_latents; t += stride)
          audio_indices_.push_back((size_t(stereo) * l.num_audio_latents + t) * audio_dim + c);
}

std::vector<float> MotionCache::sample(const float* data,
                                       const std::vector<size_t>& indices) const {
  std::vector<float> result;
  result.reserve(indices.size());
  for (size_t i : indices)
    result.push_back(data[i]);
  return result;
}

bool MotionCache::should_compute(int step, float sigma, const float* video, const float* audio) {
  bool compute = true;
  score_ = 0;
  if (enabled() && step >= config_.warmup_steps && step < steps_ - 1 && sigma > end_sigma_ &&
      sigma <= start_sigma_ && !video_residual_.empty() && video_rate_ >= 0 &&
      (audio_size_ == 0 || audio_rate_ >= 0)) {
    const float vs =
        video_rate_ * mean(sample(video, video_indices_), &previous_video_, weights_) / video_norm_;
    const float as =
        audio_size_
            ? audio_rate_ * mean(sample(audio, audio_indices_), &previous_audio_) / audio_norm_
            : 0;
    score_ = std::isfinite(vs) && std::isfinite(as) ? std::max(vs, as)
                                                    : std::numeric_limits<float>::infinity();
    accumulated_ += score_;
    compute =
        !(accumulated_ < config_.reuse_threshold && consecutive_ < config_.max_consecutive_skips);
  }
  if (compute) {
    accumulated_ = 0;
    consecutive_ = 0;
    ++computed_;
  } else {
    ++consecutive_;
    ++skipped_;
  }
  if (enabled() && config_.verbose)
    std::fprintf(stderr, "MotionCache: step %d %s, score %.5f, accumulated %.5f\n", step + 1,
                 compute ? "computed" : "reused", score_, accumulated_);
  return compute;
}

void MotionCache::update(float sigma, const float* video, const float* audio, const float* vv,
                         const float* av) {
  if (!enabled())
    return;
  auto vi = sample(video, video_indices_);
  auto ai = sample(audio, audio_indices_);
  auto vo = sample(vv, video_indices_);
  auto ao = sample(av, audio_indices_);
  weights_.assign(size_t(frames_) * spatial_, 1.0f);
  if (frames_ > 1 && config_.motion_strength > 0) {
    double total = 0;
    for (int f = 0; f < frames_; ++f)
      for (size_t p = 0; p < spatial_; ++p) {
        const size_t later = size_t(std::max(1, f)) * spatial_ + p;
        double delta = 0;
        for (int ch = 0; ch < channels_; ++ch) {
          const size_t i = size_t(ch) * frames_ * spatial_ + later;
          delta +=
              std::fabs((vi[i] + sigma * vo[i]) - (vi[i - spatial_] + sigma * vo[i - spatial_]));
        }
        weights_[size_t(f) * spatial_ + p] = static_cast<float>(delta / channels_);
        total += delta / channels_;
      }
    const float scale = std::max(static_cast<float>(total / weights_.size()), epsilon);
    total = 0;
    for (float& w : weights_) {
      w = 1.0f + config_.motion_strength * w / scale;
      total += w;
    }
    const float normalization = static_cast<float>(total / weights_.size());
    for (float& w : weights_)
      w /= normalization;
  }
  if (!previous_video_.empty()) {
    const float dv = mean(vi, &previous_video_, weights_);
    const float da = mean(ai, &previous_audio_);
    if (dv > epsilon)
      video_rate_ = mean(vo, &previous_vv_, weights_) / dv;
    if (da > epsilon)
      audio_rate_ = mean(ao, &previous_av_) / da;
  }
  video_norm_ = std::max(mean(vo, nullptr, weights_), epsilon);
  audio_norm_ = std::max(mean(ao, nullptr), epsilon);
  previous_video_ = std::move(vi);
  previous_audio_ = std::move(ai);
  previous_vv_ = std::move(vo);
  previous_av_ = std::move(ao);
  video_residual_.resize(video_size_);
  audio_residual_.resize(audio_size_);
  for (size_t i = 0; i < video_size_; ++i)
    video_residual_[i] = vv[i] + video[i];
  for (size_t i = 0; i < audio_size_; ++i)
    audio_residual_[i] = av[i] + audio[i];
}

void MotionCache::reuse(const float* video, const float* audio, float* vv, float* av) const {
  if (video_residual_.empty())
    throw std::logic_error("MotionCache: no residual to reuse");
  for (size_t i = 0; i < video_size_; ++i)
    vv[i] = video_residual_[i] - video[i];
  for (size_t i = 0; i < audio_size_; ++i)
    av[i] = audio_residual_[i] - audio[i];
}
} // namespace slopfab::dit
