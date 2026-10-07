#include "harness.h"
#include "../src/cli/bounded_pipeline.h"
#include "../src/cli/pipe_process.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace std::chrono_literals;
using slopfab::cli::BoundedPipeline;
using slopfab::cli::PipeProcess;
std::string executable;

template <typename Fn> bool fails_with(Fn fn, const char* message) {
  try {
    fn();
  } catch (const std::exception& e) {
    return std::string(e.what()) == message;
  }
  return false;
}
} // namespace

SLOPFAB_TEST(media_pipeline_order_and_overlap) {
  constexpr int count = 12;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < count; ++i) {
    std::this_thread::sleep_for(15ms); // decode
    std::this_thread::sleep_for(15ms); // inference
    std::this_thread::sleep_for(15ms); // encode
  }
  const double serial =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  int next = 0;
  bool finished = false;
  std::vector<int> output;
  start = std::chrono::steady_clock::now();
  {
    BoundedPipeline<int> pipeline(
        [&](int& value) {
          if (next == count)
            return false;
          std::this_thread::sleep_for(15ms);
          value = next++;
          return true;
        },
        [&](const int& value) {
          std::this_thread::sleep_for(15ms);
          output.push_back(value);
        },
        [&] {
          finished = true;
        },
        [] {
        },
        [] {
        });
    int value;
    while (pipeline.read(value)) {
      std::this_thread::sleep_for(15ms);
      pipeline.write(value * 2);
    }
    pipeline.finish();
  }
  const double overlapped =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  CHECK(finished);
  CHECK(output.size() == count);
  for (int i = 0; i < int(output.size()); ++i)
    CHECK(output[i] == i * 2);
  // Timing is reported, not asserted: shared CI hosts can deschedule workers.
  std::printf("  media overlap: serial %.3fs, pipelined %.3fs (%.2fx)\n", serial, overlapped,
              serial / overlapped);
}

SLOPFAB_TEST(media_pipeline_bounded_backpressure) {
  std::atomic<int> produced{0}, consumed{0};
  std::atomic<bool> release{false};
  {
    BoundedPipeline<int> pipeline(
        [&](int& value) {
          value = ++produced;
          return true;
        },
        [&](const int&) {
          ++consumed;
          while (!release.load())
            std::this_thread::sleep_for(1ms);
        },
        [] {
        },
        [&] {
          release.store(true);
        },
        [] {
        },
        2);
    std::this_thread::sleep_for(50ms);
    CHECK(produced.load() <= 3); // two queued, one pending producer item
    pipeline.write(1);
    pipeline.write(2);
    // Destruction wakes the producer blocked on a full input queue and the
    // active sink. Remaining queued output is deliberately discarded.
  }
  CHECK(produced.load() <= 3);
}

SLOPFAB_TEST(media_pipeline_source_failure_wakes_full_output) {
  std::atomic<bool> fail_source{false}, stopped{false}, sink_active{false};
  CHECK(fails_with(
      [&] {
        BoundedPipeline<int> pipeline(
            [&](int&) -> bool {
              while (!fail_source.load())
                std::this_thread::sleep_for(1ms);
              throw std::runtime_error("decoder failed");
            },
            [&](const int&) {
              sink_active.store(true);
              while (!stopped.load())
                std::this_thread::sleep_for(1ms);
            },
            [] {
            },
            [&] {
              stopped.store(true);
            },
            [] {
            },
            1);
        pipeline.write(1);
        while (!sink_active.load())
          std::this_thread::sleep_for(1ms);
        pipeline.write(2); // one in sink, one filling the queue
        fail_source.store(true);
        pipeline.write(3);
      },
      "decoder failed"));
  CHECK(stopped.load());
}

SLOPFAB_TEST(media_pipeline_sink_failure_wakes_empty_input) {
  std::atomic<bool> stopped{false};
  CHECK(fails_with(
      [&] {
        BoundedPipeline<int> pipeline(
            [&](int&) {
              while (!stopped.load())
                std::this_thread::sleep_for(1ms);
              return false;
            },
            [](const int&) {
              throw std::runtime_error("encoder failed");
            },
            [] {
            },
            [&] {
              stopped.store(true);
            },
            [] {
            });
        pipeline.write(1);
        int value;
        pipeline.read(value);
      },
      "encoder failed"));
}

SLOPFAB_TEST(media_pipeline_encoder_exit_failure) {
  CHECK(fails_with(
      [] {
        BoundedPipeline<int> pipeline(
            [](int&) {
              return false;
            },
            [](const int&) {
            },
            [] {
              throw std::runtime_error("encoder finish failed");
            },
            [] {
            },
            [] {
            });
        int value;
        CHECK(!pipeline.read(value));
        pipeline.finish();
      },
      "encoder finish failed"));
}

SLOPFAB_TEST(media_pipeline_cancel_unblocks_process_io) {
  PipeProcess reader({executable, "--blocked-child"}, false);
  PipeProcess writer({executable, "--blocked-child"}, true);
  const auto start = std::chrono::steady_clock::now();
  CHECK(fails_with(
      [&] {
        BoundedPipeline<std::vector<char>> pipeline(
            [&](std::vector<char>& item) {
              item.resize(1);
              return reader.read(item.data(), item.size()) != 0;
            },
            [&](const std::vector<char>& item) {
              writer.write(item.data(), item.size());
            },
            [&] {
              writer.finish();
            },
            [&] {
              reader.cancel();
              writer.cancel();
            },
            [&] {
              if (std::chrono::steady_clock::now() - start > 150ms)
                throw std::runtime_error("cancelled");
            });
        pipeline.write(std::vector<char>(1 << 20)); // child never reads, so this pipe fills
        std::vector<char> item;
        pipeline.read(item); // decoder child never writes
      },
      "cancelled"));
  CHECK(std::chrono::steady_clock::now() - start < 5s);
}

SLOPFAB_TEST(media_pipeline_cancel_unblocks_process_finish) {
  PipeProcess writer({executable, "--blocked-child"}, true);
  const auto start = std::chrono::steady_clock::now();
  CHECK(fails_with(
      [&] {
        BoundedPipeline<int> pipeline(
            [](int&) {
              return false;
            },
            [](const int&) {
            },
            [&] {
              writer.finish();
            },
            [&] {
              writer.cancel();
            },
            [&] {
              if (std::chrono::steady_clock::now() - start > 150ms)
                throw std::runtime_error("cancelled");
            });
        int item;
        CHECK(!pipeline.read(item));
        pipeline.finish();
      },
      "cancelled"));
  CHECK(std::chrono::steady_clock::now() - start < 5s);
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--blocked-child") {
    std::this_thread::sleep_for(30s);
    return 0;
  }
  executable = std::filesystem::absolute(std::filesystem::u8path(argv[0])).u8string();
  return slopfab::test::run_all();
}
