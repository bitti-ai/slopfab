#pragma once
#include "slopfab/generate.h"
#include <stdexcept>

namespace slopfab::music {
struct ProgressCancelled {};

// Match video: step is the zero-based index of the last COMPLETED unit,
// or -1 before any work completes. Block heartbeats retain that index so
// cancellation stays responsive without claiming unfinished work is done.
class Progress {
public:
  explicit Progress(const RunOptions& options) : options_(options) {}

  void begin(RunStage stage, int total = 0) {
    stage_ = stage;
    completed_ = 0;
    total_ = total;
    poll();
  }

  void poll() const {
    if (options_.on_progress &&
        !options_.on_progress(stage_, completed_ - 1, total_, options_.hook_userdata))
      throw ProgressCancelled{};
  }

  void advance() {
    if (completed_ >= total_)
      throw std::logic_error("music progress exceeded its work budget");
    ++completed_;
    poll();
  }

  // Autoregressive phases can stop at EOS before their requested limit.
  // Preserve completed work while revising the remaining upper bound.
  void remaining(int count) {
    if (count < 0)
      throw std::logic_error("negative music progress budget");
    total_ = completed_ + count;
    poll();
  }

private:
  const RunOptions& options_;
  RunStage stage_ = RunStage::kStarting;
  int completed_ = 0, total_ = 0;
};
} // namespace slopfab::music
