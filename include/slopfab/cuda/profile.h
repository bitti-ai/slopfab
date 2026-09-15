// Per-phase timing for the denoising loop, off unless SLOPFAB_PROFILE=1.
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

namespace slopfab::cuda {

// Reference encoders sample while convolution inputs, outputs and scratch are
// simultaneously live. Enabled by SLOPFAB_PROFILE=1; these are sampled,
// device-wide figures (including other processes), not allocator-exact peaks.
class ReferenceMemoryProfiler {
 public:
  ReferenceMemoryProfiler();
  void sample();
  void report(const char* label) const;
 private:
  bool enabled_ = false;
  size_t baseline_ = 0, peak_ = 0, minimum_free_ = 0;
  size_t samples_ = 0;
};

class StepProfiler {
 public:
  // Reads SLOPFAB_PROFILE once. There is one profiler per process because there
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
    long long host_ns;  // when the host finished issuing everything up to here
  };
  struct Total {
    std::string label;
    double ms = 0.0;        // GPU: the interval between two stream markers
    double host_ms = 0.0;   // host: how long the host spent issuing that interval
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

  // Whether the GPU ever ran out of work. The origin event is recorded onto an
  // empty stream — the previous step synchronised — so the GPU and host clocks
  // start together and the two timelines are directly comparable from there.
  // `lead` is how far the GPU is *behind* the host at a marker: positive means
  // a backlog, and a backlog is the only thing that keeps the GPU fed.
  double idle_ms_ = 0.0;      // sum of max(0, host reached mark k - GPU reached k-1)
  double min_lead_ms_ = 0.0;  // worst backlog over the step; <= 0 means a stall

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

// Phase accounting for a stage that is not the denoising loop.
//
// The video VAE decode has the opposite shape to a denoising step: it is host
// code punctuated by GPU work that it immediately waits on, because
// `forward_windows` synchronises before it can hand a tile back. Nothing
// overlaps, so a stream timeline would have nothing to interleave and the
// honest instrument is wall clock. Spans tile the stage; a handful of event
// pairs nested inside them say how much of a span the card was busy for, which
// is what separates "the decoder is slow" from "the host around it is slow".
//
// Unlike StepProfiler this is not normalised per step — a decode happens once.
class PhaseProfiler {
 public:
  static PhaseProfiler& instance();

  bool enabled() const { return enabled_; }

  // Host wall time. Spans at the same nesting level tile their stage.
  void add(const char* label, double ms);

  // Device time between two events, always nested inside some host span, so
  // these are reported separately and are not part of the host 100%.
  void add_gpu(const char* label, double ms);

  // Wall clock of the whole stage: the denominator for the percentages.
  void add_total(const char* stage, double ms);

  // Leases a pooled event, and banks a recorded pair. `flush_gpu` reads every
  // pair banked so far and returns the events to the pool; the caller must
  // already have synchronised the stream, which is why nothing here ever adds
  // a synchronise of its own and profiling cannot change the decode's shape.
  cudaEvent_t lease_event();
  void bank_pair(const char* label, cudaEvent_t begin, cudaEvent_t end);
  void flush_gpu();

  // Device-wide free memory, sampled inside the timed region. Device-wide is
  // the point: on a shared card a peer's allocation shows up here and nowhere
  // else, so a run whose peak is far above the steady-state figure was
  // contended and its timings are not comparable. Cheaper than the phase it
  // sits in by three orders of magnitude.
  void sample_memory();

  void report(std::FILE* out) const;

 private:
  PhaseProfiler();

  struct Total {
    std::string label;
    double ms = 0.0;
    long long count = 0;
    bool gpu = false;
  };
  struct Pair {
    const char* label;
    cudaEvent_t begin;
    cudaEvent_t end;
  };
  Total& slot(const char* label, bool gpu);

  bool enabled_ = false;
  std::string stage_;
  double total_ms_ = 0.0;
  std::vector<Total> totals_;
  std::vector<cudaEvent_t> pool_;
  size_t pool_used_ = 0;
  std::vector<Pair> pending_;

  size_t total_bytes_ = 0;
  size_t peak_used_ = 0;
  size_t min_free_ = 0;
};

// Scoped phase timer, the PhaseProfiler counterpart of HostSpan.
class PhaseSpan {
 public:
  explicit PhaseSpan(const char* label);
  ~PhaseSpan();
  PhaseSpan(const PhaseSpan&) = delete;
  PhaseSpan& operator=(const PhaseSpan&) = delete;

  // Ends the span early, so a span can cover part of a scope.
  void stop();

 private:
  const char* label_;
  long long t0_ = 0;
  bool running_ = false;
};

// Scoped device timer for a phase. Records an event at construction and one at
// destruction and hands the pair to the profiler; the elapsed time is read at
// the next `flush_gpu`, after a synchronise the code was doing anyway. So the
// span costs two stream markers and never blocks the host.
class PhaseGpuSpan {
 public:
  PhaseGpuSpan(const char* label, cudaStream_t stream);
  ~PhaseGpuSpan();
  PhaseGpuSpan(const PhaseGpuSpan&) = delete;
  PhaseGpuSpan& operator=(const PhaseGpuSpan&) = delete;

 private:
  const char* label_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t end_ = nullptr;
};

}  // namespace slopfab::cuda
