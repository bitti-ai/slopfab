// vidfab — MiniMax H3 video generation in C++/CUDA.

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "vidfab/dtype.h"
#include "vidfab/safetensors.h"

#if VIDFAB_WITH_CUDA
#include "vidfab/cuda/device.h"
#endif

namespace {

constexpr const char* kVersion = "0.1.0";

void print_usage() {
  std::printf(
      "vidfab %s — MiniMax H3 video generation\n"
      "\n"
      "usage: vidfab <command> [options]\n"
      "\n"
      "commands:\n"
      "  inspect <file.safetensors>   summarise a checkpoint's tensors\n"
      "  devices                      list visible CUDA devices\n"
      "  version                      print the version and exit\n"
      "\n"
      "inspect options:\n"
      "  --list                       print every tensor, not just a summary\n"
      "  --prefix <str>               only tensors whose name starts with <str>\n"
      "  --limit <n>                  cap listed tensors (default 40, 0 = all)\n",
      kVersion);
}

std::string format_shape(const std::vector<int64_t>& shape) {
  if (shape.empty()) return "scalar";
  std::string out;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out += "x";
    out += std::to_string(shape[i]);
  }
  return out;
}

std::string format_bytes(uint64_t n) {
  constexpr uint64_t kKiB = 1024;
  constexpr uint64_t kMiB = kKiB * 1024;
  constexpr uint64_t kGiB = kMiB * 1024;
  char buf[64];
  if (n >= kGiB) {
    std::snprintf(buf, sizeof(buf), "%.2f GiB", static_cast<double>(n) / kGiB);
  } else if (n >= kMiB) {
    std::snprintf(buf, sizeof(buf), "%.2f MiB", static_cast<double>(n) / kMiB);
  } else if (n >= kKiB) {
    std::snprintf(buf, sizeof(buf), "%.2f KiB", static_cast<double>(n) / kKiB);
  } else {
    std::snprintf(buf, sizeof(buf), "%" PRIu64 " B", n);
  }
  return buf;
}

int cmd_inspect(int argc, char** argv) {
  std::string path;
  std::string prefix;
  bool list = false;
  size_t limit = 40;

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--list") {
      list = true;
    } else if (arg == "--prefix" && i + 1 < argc) {
      prefix = argv[++i];
      list = true;
    } else if (arg == "--limit" && i + 1 < argc) {
      limit = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (!arg.empty() && arg.front() == '-') {
      std::fprintf(stderr, "vidfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    } else if (path.empty()) {
      path = argv[i];
    } else {
      std::fprintf(stderr, "vidfab: unexpected argument '%s'\n", argv[i]);
      return 2;
    }
  }

  if (path.empty()) {
    std::fprintf(stderr, "vidfab: inspect needs a .safetensors path\n");
    return 2;
  }

  vidfab::SafeTensors st;
  st.open(path);

  std::printf("file       %s\n", st.path().c_str());
  std::printf("size       %s\n", format_bytes(st.file_size()).c_str());
  std::printf("tensors    %zu\n", st.tensor_count());

  if (!st.metadata().empty()) {
    std::printf("metadata\n");
    for (const auto& [k, v] : st.metadata()) {
      // Metadata values can be long JSON blobs; keep the summary readable.
      std::string shown = v.size() > 120 ? v.substr(0, 117) + "..." : v;
      std::printf("  %-20s %s\n", k.c_str(), shown.c_str());
    }
  }

  // Breakdown by dtype tells us at a glance which quantisation scheme a file
  // uses, which is the first thing we need when wiring up a new checkpoint.
  std::map<std::string, std::pair<uint64_t, uint64_t>> by_dtype;  // count, bytes
  uint64_t total_bytes = 0;
  for (const auto& [name, view] : st.tensors()) {
    auto& slot = by_dtype[vidfab::dtype_name(view.dtype)];
    slot.first += 1;
    slot.second += view.nbytes;
    total_bytes += view.nbytes;
  }

  std::printf("\ndtype breakdown\n");
  for (const auto& [name, stats] : by_dtype) {
    std::printf("  %-10s %6" PRIu64 " tensors  %12s\n", name.c_str(), stats.first,
                format_bytes(stats.second).c_str());
  }
  std::printf("  %-10s %6zu tensors  %12s\n", "total", st.tensor_count(),
              format_bytes(total_bytes).c_str());

  if (list) {
    std::printf("\ntensors\n");
    size_t shown = 0;
    size_t matched = 0;
    for (const auto& [name, view] : st.tensors()) {
      if (!prefix.empty() && name.rfind(prefix, 0) != 0) continue;
      ++matched;
      if (limit != 0 && shown >= limit) continue;
      std::printf("  %-58s %-8s %-20s %10s\n", name.c_str(), vidfab::dtype_name(view.dtype),
                  format_shape(view.shape).c_str(), format_bytes(view.nbytes).c_str());
      ++shown;
    }
    if (matched > shown) {
      std::printf("  ... %zu more (raise --limit to see them)\n", matched - shown);
    }
    if (matched == 0) {
      std::printf("  no tensors matched prefix '%s'\n", prefix.c_str());
    }
  }

  return 0;
}

int cmd_devices() {
#if !VIDFAB_WITH_CUDA
  std::fprintf(stderr, "vidfab: built without CUDA support\n");
  return 1;
#else
  const int count = vidfab::cuda::device_count();
  if (count == 0) {
    std::fprintf(stderr, "vidfab: no CUDA device is visible\n");
    return 1;
  }
  for (int i = 0; i < count; ++i) {
    const vidfab::cuda::DeviceInfo d = vidfab::cuda::query_device(i);
    std::printf("device %d  %s\n", d.index, d.name.c_str());
    std::printf("  compute capability  %d.%d\n", d.major, d.minor);
    std::printf("  memory              %s free of %s\n", format_bytes(d.free_memory).c_str(),
                format_bytes(d.total_memory).c_str());
    std::printf("  multiprocessors     %d\n", d.multiprocessors);
    std::printf("  shared mem / block  %s\n", format_bytes(d.shared_memory_per_block).c_str());
    std::printf("  numeric support     bf16=%s fp8=%s fp4=%s\n", d.supports_bf16 ? "yes" : "no",
                d.supports_fp8 ? "yes" : "no", d.supports_fp4 ? "yes" : "no");
  }
  return 0;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 2;
  }

  const std::string_view command = argv[1];
  try {
    if (command == "inspect") return cmd_inspect(argc - 2, argv + 2);
    if (command == "devices") return cmd_devices();
    if (command == "version") {
      std::printf("vidfab %s\n", kVersion);
      return 0;
    }
    if (command == "help" || command == "--help" || command == "-h") {
      print_usage();
      return 0;
    }
    std::fprintf(stderr, "vidfab: unknown command '%s'\n\n", argv[1]);
    print_usage();
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vidfab: %s\n", e.what());
    return 1;
  }
}
