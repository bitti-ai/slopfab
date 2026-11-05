#pragma once
#include <algorithm>
#include <cstdint>
#include <exception>
#include <thread>
#include <vector>

namespace slopfab::demucs {
// Native threads keep packaged CLIs independent of OpenMP runtime libraries.
template <class Function> void parallel_for(int count, Function function) {
  unsigned workers =
      std::min(unsigned(count), std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(workers);
  auto work = [&](unsigned worker) {
    try {
      int first = int(int64_t(count) * worker / workers);
      int last = int(int64_t(count) * (worker + 1) / workers);
      for (int i = first; i < last; ++i)
        function(i);
    } catch (...) {
      errors[worker] = std::current_exception();
    }
  };
  try {
    for (unsigned worker = 1; worker < workers; ++worker)
      threads.emplace_back(work, worker);
  } catch (...) {
    for (auto& thread : threads)
      thread.join();
    throw;
  }
  work(0);
  for (auto& thread : threads)
    thread.join();
  for (const auto& error : errors)
    if (error)
      std::rethrow_exception(error);
}
} // namespace slopfab::demucs
