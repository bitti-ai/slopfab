#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace slopfab::cli {

// CLI-only media overlap. Each queue holds at most capacity items, plus one
// active item per worker. The caller still consumes and produces synchronously.
// stop_io must unblock source/sink I/O, be noexcept, and tolerate repeated calls.
template <typename T> class BoundedPipeline {
public:
  using Source = std::function<bool(T&)>;
  using Sink = std::function<void(const T&)>;
  using Action = std::function<void()>;

  BoundedPipeline(Source source, Sink sink, Action finish_sink, Action stop_io, Action checkpoint,
                  size_t capacity = 2)
      : source_(std::move(source)), sink_(std::move(sink)), finish_sink_(std::move(finish_sink)),
        stop_io_(std::move(stop_io)), checkpoint_(std::move(checkpoint)), capacity_(capacity) {
    if (!capacity_)
      throw std::invalid_argument("media queue capacity must be positive");
    try {
      producer_ = std::thread([this] {
        produce();
      });
      consumer_ = std::thread([this] {
        consume();
      });
    } catch (...) {
      stop();
      join();
      throw;
    }
  }

  ~BoundedPipeline() {
    stop();
    join();
  }

  BoundedPipeline(const BoundedPipeline&) = delete;
  BoundedPipeline& operator=(const BoundedPipeline&) = delete;

  bool read(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    wait(lock, [&] {
      return !input_.empty() || source_done_;
    });
    if (input_.empty())
      return false;
    item = std::move(input_.front());
    input_.pop_front();
    changed_.notify_all();
    return true;
  }

  void write(const T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    wait(lock, [&] {
      return output_.size() < capacity_;
    });
    if (output_done_)
      throw std::logic_error("write after media pipeline finish");
    output_.push_back(item);
    changed_.notify_all();
  }

  void finish() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      check();
      if (!source_done_ || !input_.empty())
        throw std::logic_error("media input must be drained before finish");
      output_done_ = true;
      changed_.notify_all();
      wait(lock, [&] {
        return sink_done_;
      });
    }
    join();
  }

  void rethrow_failure() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failure_)
      std::rethrow_exception(failure_);
  }

private:
  Source source_;
  Sink sink_;
  Action finish_sink_, stop_io_, checkpoint_;
  size_t capacity_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<T> input_, output_;
  std::exception_ptr failure_;
  bool stopped_ = false, source_done_ = false, output_done_ = false, sink_done_ = false;
  std::thread producer_, consumer_;

  void check() {
    if (failure_)
      std::rethrow_exception(failure_);
    checkpoint_();
    if (stopped_)
      throw std::runtime_error("media pipeline stopped");
  }

  template <typename Ready> void wait(std::unique_lock<std::mutex>& lock, Ready ready) {
    for (;;) {
      check();
      if (ready())
        return;
      // A signal cannot safely notify a condition variable. Poll cancellation
      // while the main thread is waiting for either bounded queue.
      changed_.wait_for(lock, std::chrono::milliseconds(50));
    }
  }

  void stop() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
      changed_.notify_all();
    }
    try {
      stop_io_();
    } catch (...) {
    }
  }

  void fail() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!failure_)
        failure_ = std::current_exception();
    }
    stop();
  }

  void join() noexcept {
    if (producer_.joinable())
      producer_.join();
    if (consumer_.joinable())
      consumer_.join();
  }

  void produce() noexcept {
    try {
      for (;;) {
        T item;
        if (!source_(item)) {
          std::lock_guard<std::mutex> lock(mutex_);
          source_done_ = true;
          changed_.notify_all();
          return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] {
          return stopped_ || input_.size() < capacity_;
        });
        if (stopped_)
          return;
        input_.push_back(std::move(item));
        changed_.notify_all();
      }
    } catch (...) {
      fail();
    }
  }

  void consume() noexcept {
    try {
      for (;;) {
        T item;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          changed_.wait(lock, [&] {
            return stopped_ || !output_.empty() || output_done_;
          });
          if (stopped_)
            return;
          if (output_.empty())
            break;
          item = std::move(output_.front());
          output_.pop_front();
          changed_.notify_all();
        }
        sink_(item);
      }
      finish_sink_();
      std::lock_guard<std::mutex> lock(mutex_);
      sink_done_ = true;
      changed_.notify_all();
    } catch (...) {
      fail();
    }
  }
};

} // namespace slopfab::cli
