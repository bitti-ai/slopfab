// Per-phase timing for the denoising loop, off unless VIDFAB_PROFILE=1.
//
// The loop issues everything onto one stream and synchronises once, at the end
// of `Transformer::forward`. That shape is what makes this cheap: a marker
// event between two phases costs a stream marker and nothing else, because the
// kernels either side were already serialised. So the profiler is a *timeline*
// rather than a set of nested timers — `tick` records one event and attributes
// the interval since the previous event to the label it is given. Segments
// therefore tile the step exactly and cannot double-count.
//
// Two consequences worth stating, because they bound what the numbers mean:
//
//   - A segment includes any GPU idle inside it. If the host stops feeding the
//     stream mid-phase, the stall lands in whichever segment was open. That is
//     why `forward` also records host clocks either side of its final
//     synchronise: if the host reaches the sync early, it was never the
//     bottleneck and no segment can be hiding a host stall.
//   - `cudaEventElapsedTime` resolves to about half a microsecond, so a label
//     summed over hundreds of tiny launches carries that much error per launch.
//     The counts are printed for exactly this reason.
//
// When disabled every entry point is a load of a bool and a return, and no
// event is ever created.
#pragma once

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace vidfab::cuda {

class StepProfiler {
 public:
  // Reads VIDFAB_PROFILE once. There is one profiler per process because there
  // is one denoising loop per process, and threading a handle through
  // `Transformer::forward` into `run_block` would put a parameter in a hot
  // signature for the sole benefit of a diagnostic.
  static StepProfiler& instance();

  bool enabled() const { return enabled_; }

  // Opens a step. `tick` before this, or after `end_step`, does nothing — the
  // token refiner shares `run_block` with the denoising loop and must not
  // contribute to its per-step totals.
  void begin_step(cudaStream_t stream);

  // Closes the interval that began at the previous event and attributes it to
  // `label`. `label` must outlive the process; a literal always does.
  void tick(const char* label, cudaStream_t stream);

  // Drains the timeline into the accumulator. The caller must already have
  // synchronised the stream — `forward` does, as its last act.
  void end_step();

  // Host-side spans, which have no place on the stream timeline: the scheduler
  // update, the row-timestep build, the parts of a step outside `forward`.
  void add_host(const char* label, double ms);

  // Wall clock and the host/device split for one call to `forward`.
  //   issue_ms: entry to the last launch, i.e. how long the host took to feed
  //             the step.
  //   wait_ms:  the final synchronise, i.e. how much GPU work outlived the
  //             host's issue.
  void add_step_wall(double wall_ms, double issue_ms, double wait_ms);

  // Samples device-wide free memory. Device-wide is the honest number here: it
  // includes the CUDA context, cuBLAS's private allocations and anything else
  // on the card, none of which our own allocation counters would see.
  void sample_memory();

  void report(std::FILE* out) const;

  int steps() const { return steps_; }

 private:
  StepProfiler();

  struct Mark {
    const char* label;
    cudaEvent_t event;
  };
  struct Total {
    std::string label;
    double ms = 0.0;
    long long count = 0;
    bool host = false;
  };

  Total& slot(const char* label, bool host);

  bool enabled_ = false;
  bool active_ = false;
  int steps_ = 0;

  std::vector<cudaEvent_t> pool_;
  size_t pool_used_ = 0;
  std::vector<Mark> marks_;      // marks_[0] is the step origin, label unused
  std::vector<Total> totals_;

  double wall_ms_ = 0.0;
  double issue_ms_ = 0.0;
  double wait_ms_ = 0.0;
  double span_ms_ = 0.0;

  size_t total_bytes_ = 0;
  size_t peak_used_ = 0;   // total - free, at its worst
  size_t min_free_ = 0;
};

// Scoped host timer. Costs a `steady_clock::now()` pair when profiling is on
// and a branch when it is not.
class HostSpan {
 public:
  explicit HostSpan(const char* label);
  ~HostSpan();
  HostSpan(const HostSpan&) = delete;
  HostSpan& operator=(const HostSpan&) = delete;

 private:
  const char* label_;
  long long t0_ = 0;
};

}  // namespace vidfab::cuda
