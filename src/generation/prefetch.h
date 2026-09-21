#pragma once
#include "helpers.h"
#include <algorithm>
#include <cstdio>
#include <system_error>
#include <thread>
#include "slopfab/safetensors.h"

namespace slopfab::generation {
class CheckpointPrefetch {
public:
  CheckpointPrefetch() = default;
  CheckpointPrefetch(const CheckpointPrefetch&) = delete;
  CheckpointPrefetch& operator=(const CheckpointPrefetch&) = delete;

  ~CheckpointPrefetch() {
    join();
  }

  // Off under the same idiom `prefetch()` itself honours, so the same binary
  // can be run both ways. Checked here as well so the flag also skips the
  // thread and the header reads, not just the hint.
  void start(std::vector<std::string> paths, bool verbose) {
    join();
    verbose_ = verbose;
    reported_ = false;
    skipped_ = false;
    spawn_failed_ = false;
    requested_ = 0;
    opened_ = 0;
    accepted_ = 0;
    bytes_ = 0;
    if (env_flag("SLOPFAB_NO_PREFETCH")) {
      skipped_ = true;
      return;
    }
    paths.erase(std::remove_if(paths.begin(), paths.end(),
                               [](const std::string& p) {
                                 return p.empty();
                               }),
                paths.end());
    if (paths.empty())
      return;
    requested_ = paths.size();
    // `std::thread`'s constructor throws `std::system_error` when the process
    // cannot spawn one. Letting that escape would kill a generation that was
    // about to denoise perfectly well, for the sake of an optimisation whose
    // whole contract is that losing it costs only time. Degrade to demand
    // faulting instead — which is exactly what the run did before this class
    // existed.
    try {
      worker_ = std::thread([this, paths = std::move(paths)] {
        for (const std::string& path : paths) {
          try {
            // Held open rather than closed here: PrefetchVirtualMemory returns
            // as soon as the read is *initiated*, so unmapping immediately after
            // it would race the readahead it just asked for. The mappings are
            // dropped in `join()`, by which point the loop has had minutes.
            SafeTensors file;
            file.open(path);
            ++opened_;
            if (file.prefetch()) {
              ++accepted_;
              bytes_ += file.file_size();
            }
            files_.push_back(std::move(file));
          } catch (...) {
            // A missing or malformed checkpoint fails on the main path in a
            // moment, with the message and the exit code the user needs. There
            // is nothing this thread can usefully add, and throwing out of it
            // would call std::terminate. The counters above are what makes the
            // swallow visible rather than silent.
          }
        }
      });
    } catch (const std::system_error&) {
      // Reported on its own line rather than folded into "nothing to do":
      // a machine that cannot spawn a thread is a real condition worth seeing,
      // and it must not look like a run that was given no VAE paths.
      spawn_failed_ = true;
    }
  }

  // Reports as well as joins, because the whole value of this class has to be
  // established by an A/B against `SLOPFAB_NO_PREFETCH=1` — and without a line
  // in the log, "the hint was refused", "the file would not open", "the thread
  // would not start", "the flag was set" and "it all worked" are five different
  // runs that look identical.
  void join() {
    if (worker_.joinable())
      worker_.join();
    if (verbose_ && !reported_) {
      reported_ = true;
      if (skipped_) {
        std::printf("prefetch    off (SLOPFAB_NO_PREFETCH=1); the vae load demand faults\n");
      } else if (spawn_failed_) {
        std::printf("prefetch    no worker thread available; the vae load demand faults\n");
      } else if (requested_ != 0) {
        std::printf("prefetch    %zu of %zu vae checkpoints hinted, %.2f GiB, %zu accepted\n",
                    opened_, requested_, static_cast<double>(bytes_) / (1024.0 * 1024.0 * 1024.0),
                    accepted_);
      }
    }
    files_.clear();
  }

private:
  std::thread worker_;
  // Written by the worker, read by the main thread only after `join()`, which
  // is the happens-before edge that makes them safe without atomics.
  std::vector<SafeTensors> files_;
  size_t requested_ = 0;
  size_t opened_ = 0;
  size_t accepted_ = 0;
  uint64_t bytes_ = 0;
  bool verbose_ = false;
  bool skipped_ = false;
  bool spawn_failed_ = false;
  bool reported_ = false;
};

}
