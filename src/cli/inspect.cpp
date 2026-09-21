// slopfab - MiniMax H3 video generation in C++/CUDA.

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Included directly rather than picked up from slopfab/generate.h, which is
// behind the CUDA guard below: the flag parsing that names an attention mode
// is not, so a build with SLOPFAB_ENABLE_CUDA=OFF could not see this type at
// all. It is a core header and costs a CPU-only build nothing.
#include "slopfab/attention_mode.h"
#include "slopfab/dtype.h"
#include "slopfab/json.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/dit/step_cache.h"
#include "slopfab/pipeline.h"
#include "slopfab/generate.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sampler/scheduler.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/tokenizer.h"
#include "reference_decode.h"

#include "slopfab/video/y4m.h"
#include "slopfab/video/y4m_compare.h"

#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/runtime.h"
#include "slopfab/vulkan/yuv_converter.h"
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#endif

#if SLOPFAB_WITH_CUDA
#include <chrono>

#include "slopfab/cuda/device.h"
#include "slopfab/cuda/cublas_dispatch.h"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/cuda/profile.h"
#include "slopfab/dit/transformer.h"
#include "slopfab/generate.h"
#include "slopfab/vae/vit_decoder.h"
#endif

#include "commands.h"

namespace slopfab::cli {
std::string format_shape(const std::vector<int64_t>& shape) {
  if (shape.empty())
    return "scalar";
  std::string out;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0)
      out += "x";
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
  if (wants_help(argc, argv))
    return print_command_help(*find_command("inspect"));

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
      std::fprintf(stderr, "slopfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    } else if (path.empty()) {
      path = argv[i];
    } else {
      std::fprintf(stderr, "slopfab: unexpected argument '%s'\n", argv[i]);
      return 2;
    }
  }

  if (path.empty()) {
    std::fprintf(stderr, "slopfab: inspect needs a .safetensors path\n");
    return 2;
  }

  slopfab::SafeTensors st;
  st.open(path);

  std::printf("file       %s\n", st.path().c_str());
  std::printf("size       %s\n", format_bytes(st.file_size()).c_str());
  std::printf("tensors    %zu\n", st.tensor_count());
  const auto architecture = slopfab::dit::detect_transformer_architecture(st);
  if (architecture != slopfab::dit::TransformerArchitecture::kUnknown) {
    std::printf("model      %s\n", slopfab::dit::transformer_architecture_name(architecture));
    std::printf("quant      %s\n", slopfab::dit::transformer_quantization_name(
                                       slopfab::dit::detect_transformer_quantization(st)));
    std::printf("qkv        %s\n", slopfab::dit::transformer_qkv_is_interleaved(st)
                                       ? "interleaved (reordered on load)"
                                       : "contiguous");
  }

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
  std::map<std::string, std::pair<uint64_t, uint64_t>> by_dtype; // count, bytes
  uint64_t total_bytes = 0;
  for (const auto& [name, view] : st.tensors()) {
    auto& slot = by_dtype[slopfab::dtype_name(view.dtype)];
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
      if (!prefix.empty() && name.rfind(prefix, 0) != 0)
        continue;
      ++matched;
      if (limit != 0 && shown >= limit)
        continue;
      std::printf("  %-58s %-8s %-20s %10s\n", name.c_str(), slopfab::dtype_name(view.dtype),
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

// Compares every tensor shared by two checkpoints. This is how a ported stage
// is validated: dump reference activations from the Python implementation, run
// ours, and diff. Exit code is non-zero when any tensor exceeds tolerance, so
// it can be used directly as a test.
int cmd_compare(int argc, char** argv) {
  if (wants_help(argc, argv))
    return print_command_help(*find_command("compare"));

  std::string ref_path;
  std::string act_path;
  double abs_tol = 1e-3;
  double rel_tol = 1e-2;
  bool verbose = false;

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--abs-tol" && i + 1 < argc) {
      abs_tol = std::strtod(argv[++i], nullptr);
    } else if (arg == "--rel-tol" && i + 1 < argc) {
      rel_tol = std::strtod(argv[++i], nullptr);
    } else if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (!arg.empty() && arg.front() == '-') {
      std::fprintf(stderr, "slopfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    } else if (ref_path.empty()) {
      ref_path = argv[i];
    } else if (act_path.empty()) {
      act_path = argv[i];
    } else {
      std::fprintf(stderr, "slopfab: unexpected argument '%s'\n", argv[i]);
      return 2;
    }
  }

  if (ref_path.empty() || act_path.empty()) {
    std::fprintf(stderr, "slopfab: compare needs a reference and an actual .safetensors path\n");
    return 2;
  }

  slopfab::SafeTensors ref;
  slopfab::SafeTensors act;
  ref.open(ref_path);
  act.open(act_path);

  std::printf("reference  %s (%zu tensors)\n", ref.path().c_str(), ref.tensor_count());
  std::printf("actual     %s (%zu tensors)\n", act.path().c_str(), act.tensor_count());
  std::printf("tolerance  abs %g, rel %g  (a tensor passes on either)\n\n", abs_tol, rel_tol);

  size_t compared = 0;
  size_t failed = 0;
  size_t missing = 0;
  double worst_abs = 0.0;
  std::string worst_name;

  std::vector<float> ref_values;
  std::vector<float> act_values;

  for (const auto& [name, ref_view] : ref.tensors()) {
    const slopfab::TensorView* act_view = act.find(name);
    if (act_view == nullptr) {
      ++missing;
      if (verbose)
        std::printf("  MISSING  %s\n", name.c_str());
      continue;
    }

    slopfab::to_f32(ref_view, ref_values);
    slopfab::to_f32(*act_view, act_values);
    const slopfab::CompareStats stats = slopfab::compare(ref_values, act_values);
    ++compared;

    if (stats.max_abs_err > worst_abs) {
      worst_abs = stats.max_abs_err;
      worst_name = name;
    }

    const bool ok = stats.passes(abs_tol, rel_tol);
    if (!ok)
      ++failed;

    if (!ok || verbose) {
      std::printf("  %-7s %-52s max_abs %.3e  max_rel %.3e  rms %.3e\n", ok ? "ok" : "FAIL",
                  name.c_str(), stats.max_abs_err, stats.max_rel_err, stats.rms_err);
      // On a continuation line rather than widened into the one above: that
      // line is already 120 columns and anything reading it would break.
      // Suppressed for a shape mismatch, where neither metric was computed and
      // printing 0.000e+00 / +0.0000 would read as agreement.
      if (stats.shape_match) {
        std::printf("          rel_L2 %.4e  correlation %+.4f\n", stats.rel_l2, stats.correlation);
      }
      if (!stats.shape_match) {
        std::printf("           shape/element-count mismatch: %lld vs %lld\n",
                    static_cast<long long>(ref_view.numel()),
                    static_cast<long long>(act_view->numel()));
      } else if (!ok && stats.argmax_abs >= 0) {
        std::printf("           worst at index %lld: reference %.6g, actual %.6g\n",
                    static_cast<long long>(stats.argmax_abs), stats.lhs_at_argmax,
                    stats.rhs_at_argmax);
      }
      if (stats.nan_mismatches != 0) {
        std::printf("           %lld non-finite mismatches\n",
                    static_cast<long long>(stats.nan_mismatches));
      }
    }
  }

  std::printf("\n%zu compared, %zu failed, %zu missing from actual\n", compared, failed, missing);
  if (!worst_name.empty()) {
    std::printf("worst absolute error %.3e in %s\n", worst_abs, worst_name.c_str());
  }
  return (failed == 0 && missing == 0) ? 0 : 1;
}

int cmd_compare_y4m(int argc, char** argv) {
  if (wants_help(argc, argv))
    return print_command_help(*find_command("compare-y4m"));
  if (argc != 2) {
    std::fprintf(stderr, "slopfab: compare-y4m needs expected and actual paths\n");
    return 2;
  }
  const slopfab::video::ExactY4mComparison result =
      slopfab::video::compare_y4m_exact(argv[0], argv[1]);
  std::printf("expected   %s (%llu bytes)\n", argv[0],
              static_cast<unsigned long long>(result.expected_size));
  std::printf("           %s\n", result.expected_header.c_str());
  std::printf("actual     %s (%llu bytes)\n", argv[1],
              static_cast<unsigned long long>(result.actual_size));
  std::printf("           %s\n", result.actual_header.c_str());
  if (result.equal()) {
    std::printf("exact      yes (all bytes identical)\n");
    return 0;
  }
  std::printf("exact      no; first difference at byte %llu: expected ",
              static_cast<unsigned long long>(result.first_difference));
  if (result.expected_byte < 0)
    std::printf("<EOF>");
  else
    std::printf("0x%02x", result.expected_byte);
  std::printf(", actual ");
  if (result.actual_byte < 0)
    std::printf("<EOF>\n");
  else
    std::printf("0x%02x\n", result.actual_byte);
  return 1;
}

}
