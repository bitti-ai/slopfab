#include "vidfab/cuda/profile.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {
namespace {

using Clock = std::chrono::steady_clock;

long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
      .count();
}

double to_gib(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

}  // namespace

StepProfiler::StepProfiler() {
  const char* v = std::getenv("VIDFAB_PROFILE");
  enabled_ = v != nullptr && v[0] == '1';
}

StepProfiler& StepProfiler::instance() {
  static StepProfiler p;
  return p;
}

StepProfiler::Total& StepProfiler::slot(const char* label, bool host) {
  for (Total& t : totals_) {
    if (t.label == label) return t;
  }
  Total fresh;
  fresh.label = label;
  fresh.host = host;
  totals_.push_back(std::move(fresh));
  return totals_.back();
}

void StepProfiler::begin_step(cudaStream_t stream) {
  if (!enabled_) return;
  marks_.clear();
  pool_used_ = 0;
  active_ = true;
  tick(nullptr, stream);  // the origin; its interval belongs to no label
}

void StepProfiler::tick(const char* label, cudaStream_t stream) {
  if (!enabled_ || !active_) return;
  if (pool_used_ == pool_.size()) {
    cudaEvent_t e = nullptr;
    VIDFAB_CUDA_CHECK(cudaEventCreate(&e));
    pool_.push_back(e);
  }
  cudaEvent_t e = pool_[pool_used_++];
  VIDFAB_CUDA_CHECK(cudaEventRecord(e, stream));
  marks_.push_back(Mark{label, e, now_ns()});
}

void StepProfiler::end_step() {
  if (!enabled_ || !active_) return;
  active_ = false;
  if (marks_.size() < 2) return;

  const long long h0 = marks_.front().host_ns;
  double gpu_at = 0.0;        // GPU arrival at mark k, ms since the origin
  double gpu_prev = 0.0;      // ... at mark k-1
  double min_lead = 1.0e30;
  for (size_t i = 1; i < marks_.size(); ++i) {
    float ms = 0.0f;
    VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&ms, marks_[i - 1].event, marks_[i].event));
    const double host_ms =
        static_cast<double>(marks_[i].host_ns - marks_[i - 1].host_ns) / 1.0e6;
    Total& t = slot(marks_[i].label, /*host=*/false);
    t.ms += ms;
    t.host_ms += host_ms;
    t.count += 1;

    gpu_prev = gpu_at;
    gpu_at += ms;
    const double host_at = static_cast<double>(marks_[i].host_ns - h0) / 1.0e6;
    // The GPU cannot start segment k before the host has issued it. If the host
    // arrives after the GPU has already drained everything up to k-1, the
    // difference is time the card spent with nothing to run.
    if (host_at > gpu_prev) idle_ms_ += host_at - gpu_prev;
    min_lead = std::min(min_lead, gpu_at - host_at);
  }
  float span = 0.0f;
  VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&span, marks_.front().event, marks_.back().event));
  span_ms_ += span;
  min_lead_ms_ += min_lead;
  steps_ += 1;
  marks_.clear();
  pool_used_ = 0;
}

void StepProfiler::add_host(const char* label, double ms) {
  if (!enabled_) return;
  Total& t = slot(label, /*host=*/true);
  t.ms += ms;
  t.count += 1;
}

void StepProfiler::add_step_wall(double wall_ms, double issue_ms, double wait_ms) {
  if (!enabled_) return;
  wall_ms_ += wall_ms;
  issue_ms_ += issue_ms;
  wait_ms_ += wait_ms;
}

void StepProfiler::sample_memory() {
  if (!enabled_) return;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;
  total_bytes_ = total_bytes;
  const size_t used = total_bytes - free_bytes;
  if (used > peak_used_) peak_used_ = used;
  if (min_free_ == 0 || free_bytes < min_free_) min_free_ = free_bytes;
}

void StepProfiler::report(std::FILE* out) const {
  if (!enabled_ || steps_ == 0) return;

  const double n = static_cast<double>(steps_);
  const double step_ms = wall_ms_ / n;

  std::fprintf(out, "\n--- VIDFAB_PROFILE: %d step%s ---\n", steps_, steps_ == 1 ? "" : "s");
  std::fprintf(out, "%-24s %12s %12s %8s %12s %10s\n", "phase", "ms/step", "total ms", "%step",
               "host ms/step", "calls/step");

  // Device segments first, in the order they were first seen, which is the
  // order of the step. Then host spans, which sit outside the stream timeline
  // and so are not part of the 100%.
  double device_sum = 0.0;
  for (const Total& t : totals_) {
    if (t.host) continue;
    device_sum += t.ms;
  }
  double host_sum = 0.0;
  for (const Total& t : totals_) {
    if (t.host) continue;
    host_sum += t.host_ms;
    std::fprintf(out, "%-24s %12.3f %12.1f %7.2f%% %12.3f %10.1f\n", t.label.c_str(), t.ms / n,
                 t.ms, 100.0 * t.ms / (step_ms * n), t.host_ms / n,
                 static_cast<double>(t.count) / n);
  }
  std::fprintf(out, "%-24s %12.3f %12.1f %7.2f%% %12.3f\n", "= device timeline", device_sum / n,
               device_sum, 100.0 * device_sum / (step_ms * n), host_sum / n);

  bool any_host = false;
  for (const Total& t : totals_) {
    if (!t.host) continue;
    if (!any_host) {
      std::fprintf(out, "%-24s\n", "-- host (outside the stream timeline) --");
      any_host = true;
    }
    std::fprintf(out, "%-24s %12.3f %12.1f %7.2f%% %12s %10.1f\n", t.label.c_str(), t.ms / n, t.ms,
                 100.0 * t.ms / (step_ms * n), "-", static_cast<double>(t.count) / n);
  }

  std::fprintf(out, "\n%-24s %12.3f\n", "step wall clock", step_ms);
  std::fprintf(out, "%-24s %12.3f  (%.2f%%)\n", "  device event span", span_ms_ / n,
               100.0 * span_ms_ / (step_ms * n));
  std::fprintf(out, "%-24s %12.3f  (%.2f%%)\n", "  not on the timeline", (wall_ms_ - span_ms_) / n,
               100.0 * (wall_ms_ - span_ms_) / (step_ms * n));
  std::fprintf(out, "%-24s %12.3f  (%.2f%%)  host issuing the step\n", "  forward: issue",
               issue_ms_ / n, 100.0 * issue_ms_ / (step_ms * n));
  std::fprintf(out, "%-24s %12.3f  (%.2f%%)  host blocked on the GPU\n", "  forward: sync wait",
               wait_ms_ / n, 100.0 * wait_ms_ / (step_ms * n));
  std::fprintf(out, "%-24s %12.3f  (%.2f%%)  card with nothing to run\n", "  GPU idle in step",
               idle_ms_ / n, 100.0 * idle_ms_ / (step_ms * n));
  std::fprintf(out, "%-24s %12.3f  smallest backlog; <=0 would mean the host fell behind\n",
               "  min GPU backlog", min_lead_ms_ / n);

  if (total_bytes_ != 0) {
    std::fprintf(out, "\n%-24s %.3f GiB of %.3f GiB (device-wide, all processes)\n", "peak VRAM",
                 to_gib(peak_used_), to_gib(total_bytes_));
    std::fprintf(out, "%-24s %.3f GiB\n", "min free VRAM", to_gib(min_free_));
  }
  std::fprintf(out, "--- end VIDFAB_PROFILE ---\n");
  std::fflush(out);
}

HostSpan::HostSpan(const char* label) : label_(label) {
  if (!StepProfiler::instance().enabled()) return;
  t0_ = now_ns();
}

HostSpan::~HostSpan() {
  StepProfiler& p = StepProfiler::instance();
  if (!p.enabled()) return;
  p.add_host(label_, static_cast<double>(now_ns() - t0_) / 1.0e6);
}

}  // namespace vidfab::cuda
