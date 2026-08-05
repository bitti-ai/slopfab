// vidfab - MiniMax H3 video generation in C++/CUDA.

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "vidfab/dtype.h"
#include "vidfab/json.h"
#include "vidfab/pipeline.h"
#include "vidfab/safetensors.h"
#include "vidfab/safetensors_write.h"
#include "vidfab/sampler/scheduler.h"
#include "vidfab/tensor_convert.h"
#include "vidfab/text/tokenizer.h"

#include "vidfab/video/y4m.h"

#if VIDFAB_WITH_CUDA
#include <chrono>

#include "vidfab/cuda/device.h"
#include "vidfab/dit/transformer.h"
#include "vidfab/generate.h"
#include "vidfab/vae/vit_decoder.h"
#endif

namespace {

constexpr const char* kVersion = "0.1.0";

// Per-command help. Keeping the detail beside the summary in one table is what
// stops `vidfab <cmd> --help` from drifting out of step with the top-level
// usage, which is the usual way CLI help rots.
struct CommandHelp {
  const char* name;
  const char* usage;
  const char* summary;
  const char* detail;
};

const CommandHelp kCommands[] = {
    {"generate", "vidfab generate --prompt <text> [options]", "text to video and audio",
     "  --prompt <text>              the prompt (MiniMax Context-IR structure)\n"
     "  --out <file>                 output path (default video.mp4)\n"
     "  --aspect <W:H>               display aspect, 1:4 to 4:1 (default 16:9)\n"
     "  --frames <n>                 snapped up to 17k+5 (default 124, minimum 6)\n"
     "  --steps <n>                  sigma grid points, n-1 evaluations (default 50)\n"
     "  --sampler euler|ab2          integrator (default euler)\n"
     "  --seed <n>                   noise seed\n"
     "  --raw                        write .y4m + .wav instead of muxing MP4\n"
     "  --dry-run                    resolve and print the plan, touch no weights\n"
     "  --synthetic-latents          skip conditioning and denoising and decode seeded\n"
     "                               noise, to exercise the VAEs and the muxer\n"
     "  --attn-band <frames>         frame-banded attention: a video row attends to\n"
     "                               +/- this many latent frames instead of the whole\n"
     "                               sequence. 0 (default) is off, and saves more the\n"
     "                               longer the request.\n"
     "                               Lossy. At +/-9 on the default geometry: ~1.4x\n"
     "                               faster per step; video latents land 1.6-2.0x the\n"
     "                               sampler noise floor away from unbanded, measured\n"
     "                               at 6, 24 and 40 steps with no trend in between.\n"
     "                               The audio cost is NOT characterised -- one seed\n"
     "                               per point leaves its noise floor moving as much\n"
     "                               as the effect. Judge output before relying on it\n"
     "  --dump-latents <f>           the denoiser's own output as fp32 safetensors,\n"
     "                               before either VAE; the diff point for a change\n"
     "                               to the transformer\n"
     "  --init-latents <f>           start the loop from these latents instead of the\n"
     "                               seeded draw; with --synthetic-latents, decode\n"
     "                               them straight to video and audio. Same shape as\n"
     "                               --dump-latents writes. Off by default.\n"
     "\n"
     "checkpoints (all required unless --dry-run or --synthetic-latents):\n"
     "  --tokenizer <f>              tokenizer.json\n"
     "  --text-encoder <f>           Qwen3-VL conditioner, int8 ConvRot or nvfp4 AWQ\n"
     "  --transformer <f>            H3 omni transformer, fp8 or nvfp4\n"
     "  --vae <f>                    video VAE decoder\n"
     "  --audio-vae <f>              audio VAE decoder\n"
     "\n"
     "  --bench-load <n>             load --transformer n times and exit, timing each.\n"
     "                               The first pays for reading the file, the rest do\n"
     "                               not; the gap is the cold-start cost.\n"
     "\n"
     "The quantisation of each checkpoint is read out of the file, so there is\n"
     "no flag for it and the two need not match.\n"
     "\n"
     "--sampler ab2 is Adams-Bashforth 2: second order at the same one forward\n"
     "pass per step, so a step costs what it always did and the reason to use it\n"
     "is to lower --steps. Its first step has no velocity history and is Euler.\n"
     "\n"
     "The conditioner and the transformer are loaded and freed in sequence\n"
     "rather than together: at 23.1 GB and 19.3 GB the int8 and fp8 pair cannot\n"
     "co-exist on a 32 GB card. The nvfp4 pair would (13.1 GB and 12.5 GB), but\n"
     "the sequencing costs nothing measurable and is what makes every mixture of\n"
     "the four safe. Expect the first output well after the progress line starts\n"
     "moving.\n"},
    {"inspect", "vidfab inspect <file.safetensors> [options]",
     "summarise a checkpoint's tensors",
     "  --list                       print every tensor, not just a summary\n"
     "  --prefix <str>               only tensors whose name starts with <str>\n"
     "  --limit <n>                  cap listed tensors (default 40, 0 = all)\n"},
    {"compare", "vidfab compare <reference> <actual> [options]",
     "diff two checkpoints tensor by tensor",
     "  --abs-tol <x>                absolute tolerance (default 1e-3)\n"
     "  --rel-tol <x>                relative tolerance (default 1e-2)\n"
     "  --verbose                    report passing tensors too\n"
     "\n"
     "Pass/fail is elementwise: a tensor passes on either tolerance. Beside\n"
     "that, two whole-tensor quality metrics are reported, because \"is any\n"
     "element wrong\" and \"is this the same tensor\" are different questions and\n"
     "the second is the one a quality comparison asks.\n"
     "\n"
     "  rel_L2       ||reference - actual|| / ||reference||\n"
     "               = sqrt(sum (r-a)^2) / sqrt(sum r^2)\n"
     "               Normalised by the REFERENCE -- not by actual, and not by\n"
     "               the mean of the two norms. So it is asymmetric on purpose:\n"
     "               compare(a,b) and compare(b,a) ask different questions.\n"
     "               Zero reference gives 0 if the difference is zero, else inf.\n"
     "\n"
     "  correlation  Pearson over the flattened tensor, MEANS SUBTRACTED:\n"
     "               S_ra / sqrt(S_rr * S_aa), S_xy = sum (x-mean_x)(y-mean_y).\n"
     "               Not cosine similarity. Identical tensors give exactly +1,\n"
     "               a tensor against its own negation exactly -1, and a\n"
     "               constant tensor (zero variance) gives 0, undefined.\n"
     "\n"
     "Both are computed over the flattened tensor across element pairs where\n"
     "both sides are finite. Matching global statistics with falling\n"
     "correlation is this project's signature of a different sample rather\n"
     "than a degraded one -- see the README.\n"},
    {"decode", "vidfab decode --vae <f> [--latent <f>] [options]",
     "run the video VAE decoder",
     "  --vae <f>                    video VAE checkpoint\n"
     "  --latent <f>                 latent safetensors; omit for a synthetic one\n"
     "  --shape <T> <H> <W>          synthetic latent shape\n"
     "  --out <f>                    .y4m output\n"
     "  --ppm <f>                    also write frame 0 as a PPM\n"
     "  --dump <f>                   raw fp32 pixels as safetensors\n"},
    {"tokenize", "vidfab tokenize --tokenizer <f> <text>",
     "encode text and round-trip it",
     "  --tokenizer <f>              tokenizer.json\n"
     "  --pieces                     also print the pre-tokenizer split\n"},
    {"devices", "vidfab devices", "list visible CUDA devices", ""},
    {"version", "vidfab version", "print the version and exit", ""},
};

const CommandHelp* find_command(std::string_view name) {
  for (const CommandHelp& c : kCommands) {
    if (name == c.name) return &c;
  }
  return nullptr;
}

// True when the argument list asks for help. Checked before any other option,
// so `--help` works even alongside otherwise invalid arguments.
bool wants_help(int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--help" || a == "-h" || a == "help") return true;
  }
  return false;
}

int print_command_help(const CommandHelp& c) {
  std::printf("usage: %s\n\n%s\n", c.usage, c.summary);
  if (c.detail[0] != '\0') std::printf("\noptions:\n%s", c.detail);
  return 0;
}

void print_usage() {
  std::printf(
      "vidfab %s - MiniMax H3 video generation\n"
      "\n"
      "usage: vidfab <command> [options]\n"
      "       vidfab <command> --help\n"
      "\n"
      "commands:\n",
      kVersion);
  for (const CommandHelp& c : kCommands) {
    std::printf("  %-9s %s\n", c.name, c.summary);
  }
  std::printf("\nRun `vidfab <command> --help` for that command's options.\n");
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
  if (wants_help(argc, argv)) return print_command_help(*find_command("inspect"));

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

// Compares every tensor shared by two checkpoints. This is how a ported stage
// is validated: dump reference activations from the Python implementation, run
// ours, and diff. Exit code is non-zero when any tensor exceeds tolerance, so
// it can be used directly as a test.
int cmd_compare(int argc, char** argv) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("compare"));

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
      std::fprintf(stderr, "vidfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    } else if (ref_path.empty()) {
      ref_path = argv[i];
    } else if (act_path.empty()) {
      act_path = argv[i];
    } else {
      std::fprintf(stderr, "vidfab: unexpected argument '%s'\n", argv[i]);
      return 2;
    }
  }

  if (ref_path.empty() || act_path.empty()) {
    std::fprintf(stderr, "vidfab: compare needs a reference and an actual .safetensors path\n");
    return 2;
  }

  vidfab::SafeTensors ref;
  vidfab::SafeTensors act;
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
    const vidfab::TensorView* act_view = act.find(name);
    if (act_view == nullptr) {
      ++missing;
      if (verbose) std::printf("  MISSING  %s\n", name.c_str());
      continue;
    }

    vidfab::to_f32(ref_view, ref_values);
    vidfab::to_f32(*act_view, act_values);
    const vidfab::CompareStats stats = vidfab::compare(ref_values, act_values);
    ++compared;

    if (stats.max_abs_err > worst_abs) {
      worst_abs = stats.max_abs_err;
      worst_name = name;
    }

    const bool ok = stats.passes(abs_tol, rel_tol);
    if (!ok) ++failed;

    if (!ok || verbose) {
      std::printf("  %-7s %-52s max_abs %.3e  max_rel %.3e  rms %.3e\n", ok ? "ok" : "FAIL",
                  name.c_str(), stats.max_abs_err, stats.max_rel_err, stats.rms_err);
      // On a continuation line rather than widened into the one above: that
      // line is already 120 columns and anything reading it would break.
      // Suppressed for a shape mismatch, where neither metric was computed and
      // printing 0.000e+00 / +0.0000 would read as agreement.
      if (stats.shape_match) {
        std::printf("          rel_L2 %.4e  correlation %+.4f\n", stats.rel_l2,
                    stats.correlation);
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

#if VIDFAB_WITH_CUDA
// Reads latents_mean / latents_std from the checkpoint's __metadata__ JSON,
// which carries full fp32 text. The F16 tensors of the same name lose
// precision, and the FL2VA copy of config.json has corrupted digits.
bool latent_stats_from_metadata(const vidfab::SafeTensors& ckpt, std::vector<float>& mean,
                                std::vector<float>& std_dev) {
  auto it = ckpt.metadata().find("minimax_h3_video_vae");
  if (it == ckpt.metadata().end()) return false;
  try {
    const vidfab::json::Value meta = vidfab::json::parse(it->second);
    const vidfab::json::Value* m = meta.find("latents_mean");
    const vidfab::json::Value* s = meta.find("latents_std");
    if (m == nullptr || s == nullptr) return false;
    mean.clear();
    std_dev.clear();
    for (const auto& v : m->as_array()) mean.push_back(static_cast<float>(v.as_number()));
    for (const auto& v : s->as_array()) std_dev.push_back(static_cast<float>(v.as_number()));
    return mean.size() == 24 && std_dev.size() == 24;
  } catch (const std::exception&) {
    return false;
  }
}

// Deterministic pseudo-random latent, so a run without a real latent file
// still exercises the whole path reproducibly.
std::vector<float> synthetic_latent(int T, int H, int W, uint32_t seed) {
  std::vector<float> z(static_cast<size_t>(24) * T * H * W);
  uint32_t state = seed != 0 ? seed : 1u;
  for (float& v : z) {
    // xorshift32, then map to roughly standard normal via Box-Muller-free
    // approximation: sum of uniforms is adequate for a smoke test.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    const float u = static_cast<float>(state & 0xFFFFFFu) / static_cast<float>(0x1000000u);
    v = (u - 0.5f) * 2.0f;
  }
  return z;
}

int cmd_decode(int argc, char** argv) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("decode"));

  std::string vae_path;
  std::string latent_path;
  std::string out_path = "out.y4m";
  std::string ppm_path;
  std::string dump_path;
  int T = 7;
  int H = 16;
  int W = 16;
  uint32_t seed = 1234;
  bool no_tiling = false;
  bool bench_load = false;
  int fps = 24;
  int repeat = 1;

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--vae" && i + 1 < argc) {
      vae_path = argv[++i];
    } else if (arg == "--latent" && i + 1 < argc) {
      latent_path = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--ppm" && i + 1 < argc) {
      ppm_path = argv[++i];
    } else if (arg == "--dump" && i + 1 < argc) {
      dump_path = argv[++i];
    } else if (arg == "--shape" && i + 3 < argc) {
      T = std::atoi(argv[++i]);
      H = std::atoi(argv[++i]);
      W = std::atoi(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--fps" && i + 1 < argc) {
      fps = std::atoi(argv[++i]);
    } else if (arg == "--repeat" && i + 1 < argc) {
      repeat = std::max(1, std::atoi(argv[++i]));
    } else if (arg == "--no-tiling") {
      no_tiling = true;
    } else if (arg == "--bench-load") {
      bench_load = true;
    } else {
      std::fprintf(stderr, "vidfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    }
  }

  if (vae_path.empty()) {
    std::fprintf(stderr, "vidfab: decode needs --vae <video_vae.safetensors>\n");
    return 2;
  }

  vidfab::SafeTensors ckpt;
  ckpt.open(vae_path);

  std::vector<float> mean;
  std::vector<float> std_dev;
  if (!latent_stats_from_metadata(ckpt, mean, std_dev)) {
    std::fprintf(stderr, "vidfab: could not read latents_mean/std from checkpoint metadata\n");
    return 1;
  }

  std::vector<float> z;
  if (!latent_path.empty()) {
    vidfab::SafeTensors latent_file;
    latent_file.open(latent_path);
    const vidfab::TensorView& lv = latent_file.at("latent");
    if (lv.shape.size() != 4 || lv.shape[0] != 24) {
      std::fprintf(stderr, "vidfab: latent tensor must be [24, T, H, W]\n");
      return 1;
    }
    T = static_cast<int>(lv.shape[1]);
    H = static_cast<int>(lv.shape[2]);
    W = static_cast<int>(lv.shape[3]);
    z = vidfab::to_f32(lv);
  } else {
    std::printf("no --latent given; decoding a deterministic synthetic latent (seed %u)\n", seed);
    z = synthetic_latent(T, H, W, seed);
  }

  std::printf("latent     [24, %d, %d, %d] -> %d x %d px\n", T, H, W, W * 16, H * 16);

  // Loading twice in one process distinguishes two very different causes of a
  // slow load: if the second is much faster, the first was paying page faults
  // to pull the mapping in from storage; if they match, the cost is the host
  // memcpy and PCIe, and only then is parallelising or double-buffering it
  // worth building.
  if (bench_load) {
    auto time_load = [&](const char* label) {
      vidfab::vae::ViTDecoder probe;
      const auto s0 = std::chrono::steady_clock::now();
      probe.load(ckpt);
      const auto s1 = std::chrono::steady_clock::now();
      const double sec = std::chrono::duration<double>(s1 - s0).count();
      std::printf("load %-8s %s in %.3f s (%.2f GB/s of fp16 across PCIe)\n", label,
                  format_bytes(probe.weight_bytes()).c_str(), sec,
                  (static_cast<double>(probe.weight_bytes()) / 2.0) / sec / 1e9);
    };
    time_load("first");
    time_load("second");
    time_load("third");
    return 0;
  }

  vidfab::vae::ViTDecoder decoder;
  const auto load_start = std::chrono::steady_clock::now();
  decoder.load(ckpt);
  const auto load_end = std::chrono::steady_clock::now();
  std::printf("weights    %s on device in %.2f s\n",
              format_bytes(decoder.weight_bytes()).c_str(),
              std::chrono::duration<double>(load_end - load_start).count());

  vidfab::vae::DecodeSchedule schedule;
  schedule.tiling_enabled = !no_tiling;

  // The first decode pays one-time costs the steady state does not: scratch
  // allocation, the first RoPE build, and cuBLAS heuristic selection for each
  // GEMM shape. Reporting it as the decode time overstates the cost by a
  // noticeable margin, so timings are reported per run and `--repeat` exists to
  // expose the warm number.
  vidfab::vae::DecodedVideo video;
  for (int run = 0; run < repeat; ++run) {
    const auto t0 = std::chrono::steady_clock::now();
    video = decoder.decode(z.data(), T, H, W, mean, std_dev, schedule);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const char* label = (run == 0) ? "decoded   " : "  (warm)  ";
    std::printf("%s %d frames of %dx%d in %.3f s (%.2f fps)\n", label, video.frames, video.width,
                video.height, seconds,
                seconds > 0 ? static_cast<double>(video.frames) / seconds : 0.0);
  }

  // Report basic statistics: a decode that silently produced NaN or a constant
  // image should be visible here without opening the file.
  double sum = 0.0;
  float lo = 1e30f;
  float hi = -1e30f;
  size_t nonfinite = 0;
  for (float v : video.data) {
    if (!std::isfinite(v)) {
      ++nonfinite;
      continue;
    }
    sum += v;
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  std::printf("pixels     min %.4f  max %.4f  mean %.4f  non-finite %zu\n", lo, hi,
              sum / static_cast<double>(video.data.size()), nonfinite);
  if (nonfinite != 0) {
    std::fprintf(stderr, "vidfab: decode produced non-finite pixels\n");
    return 1;
  }

  if (!dump_path.empty()) {
    // Raw fp32 pixels, so two runs can be diffed with `vidfab compare` at
    // float precision rather than after 8-bit quantisation.
    vidfab::write_safetensors(
        dump_path, {{"pixels",
                     {3, video.frames, video.height, video.width},
                     video.data}});
    std::printf("wrote      %s\n", dump_path.c_str());
  }

  vidfab::video::write_y4m(out_path, video.data, video.frames, video.height, video.width,
                           {fps, 1});
  std::printf("wrote      %s\n", out_path.c_str());
  if (!ppm_path.empty()) {
    vidfab::video::write_ppm(ppm_path, video.data, video.frames, video.height, video.width, 0);
    std::printf("wrote      %s\n", ppm_path.c_str());
  }
  return 0;
}
#endif  // VIDFAB_WITH_CUDA

int cmd_tokenize(int argc, char** argv) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("tokenize"));

  std::string tok_path;
  std::string text;
  bool show_pieces = false;

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--tokenizer" && i + 1 < argc) {
      tok_path = argv[++i];
    } else if (arg == "--pieces") {
      show_pieces = true;
    } else if (!arg.empty() && arg.front() == '-') {
      std::fprintf(stderr, "vidfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    } else {
      if (!text.empty()) text += " ";
      text += argv[i];
    }
  }
  if (tok_path.empty()) {
    std::fprintf(stderr, "vidfab: tokenize needs --tokenizer <tokenizer.json>\n");
    return 2;
  }

  vidfab::text::Tokenizer tok;
  tok.load(tok_path);
  std::printf("vocab      %zu tokens\n", tok.vocab_size());

  if (show_pieces) {
    std::printf("pieces     ");
    for (const std::string& p : tok.pre_tokenize(text)) std::printf("[%s]", p.c_str());
    std::printf("\n");
  }

  const std::vector<int32_t> ids = tok.encode(text);
  std::printf("ids (%zu)   ", ids.size());
  for (int32_t id : ids) std::printf("%d ", id);
  std::printf("\n");
  std::printf("tokens     ");
  for (int32_t id : ids) std::printf("[%s]", tok.id_to_token(id).c_str());
  std::printf("\n");

  const std::string round = tok.decode(ids);
  std::printf("decoded    %s\n", round.c_str());
  std::printf("round trip %s\n", round == text ? "OK" : "MISMATCH");
  return round == text ? 0 : 1;
}

// Resolves a request all the way to a plan, then runs it. The stages are added
// one at a time; until they all exist this reports precisely which one is
// missing rather than pretending. `--dry-run` stops after the plan, which
// costs no I/O and is the fastest way to check geometry and schedule.
int cmd_generate(int argc, char** argv) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("generate"));

  vidfab::GenerateRequest req;
  bool dry_run = false;
  bool synthetic = false;
  vidfab::sampler::SamplerKind sampler_kind = vidfab::sampler::SamplerKind::kEuler;
  std::string dump_latents;
  int attn_band = 0;
  std::string init_latents;
  int bench_load = 0;

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("generate: ") + what + " needs a value");
      }
      return argv[++i];
    };
    if (arg == "--prompt") {
      req.prompt = next("--prompt");
    } else if (arg == "--out") {
      req.out_path = next("--out");
    } else if (arg == "--frames") {
      req.num_frames = std::atoi(next("--frames"));
    } else if (arg == "--steps") {
      req.num_inference_steps = std::atoi(next("--steps"));
    } else if (arg == "--seed") {
      req.seed = std::strtoull(next("--seed"), nullptr, 10);
    } else if (arg == "--sampler") {
      const std::string v = next("--sampler");
      if (v == "euler") {
        sampler_kind = vidfab::sampler::SamplerKind::kEuler;
      } else if (v == "ab2") {
        sampler_kind = vidfab::sampler::SamplerKind::kAb2;
      } else {
        std::fprintf(stderr, "vidfab: --sampler wants euler or ab2, got '%s'\n", v.c_str());
        return 2;
      }
    } else if (arg == "--aspect") {
      const std::string v = next("--aspect");
      const size_t colon = v.find(':');
      if (colon == std::string::npos) {
        std::fprintf(stderr, "vidfab: --aspect wants W:H, e.g. 16:9\n");
        return 2;
      }
      req.aspect_w = std::atoi(v.substr(0, colon).c_str());
      req.aspect_h = std::atoi(v.substr(colon + 1).c_str());
    } else if (arg == "--transformer") {
      req.transformer_path = next("--transformer");
    } else if (arg == "--text-encoder") {
      req.text_encoder_path = next("--text-encoder");
    } else if (arg == "--tokenizer") {
      req.tokenizer_path = next("--tokenizer");
    } else if (arg == "--vae") {
      req.video_vae_path = next("--vae");
    } else if (arg == "--audio-vae") {
      req.audio_vae_path = next("--audio-vae");
    } else if (arg == "--raw") {
      req.raw_output = true;
    } else if (arg == "--dry-run") {
      dry_run = true;
    } else if (arg == "--synthetic-latents") {
      synthetic = true;
    } else if (arg == "--dump-latents") {
      dump_latents = next("--dump-latents");
    } else if (arg == "--attn-band") {
      attn_band = std::atoi(next("--attn-band"));
    } else if (arg == "--init-latents") {
      init_latents = next("--init-latents");
    } else if (arg == "--bench-load") {
      bench_load = std::atoi(next("--bench-load"));
    } else {
      std::fprintf(stderr, "vidfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    }
  }

  // `--synthetic-latents --init-latents <f>` is the decode-an-existing-latent
  // path and needs no prompt; the seeded-noise form still does not either.
  if (req.prompt.empty() && !dry_run && !synthetic) {
    std::fprintf(stderr, "vidfab: generate needs --prompt \"...\"\n");
    return 2;
  }

  const vidfab::GeneratePlan plan = vidfab::resolve_plan(req);
  std::fputs(vidfab::describe_plan(req, plan).c_str(), stdout);
  if (dry_run) return 0;

#if VIDFAB_WITH_CUDA
  // Times the transformer load on its own, the same way `decode --bench-load`
  // times the VAE's and for the same reason: the first load in a process pays
  // for pulling the mapping in from storage and the later ones do not, and the
  // gap between them is the entire subject of cold-start work. Loading through
  // `generate` proper would first stream 25 GB of conditioner, which both costs
  // a minute and evicts the very file being measured.
  //
  // Nothing here evicts the cache, so the first number is only a *cold* number
  // if the caller made it one.
  if (bench_load > 0) {
    if (req.transformer_path.empty()) {
      std::fprintf(stderr, "vidfab: --bench-load needs --transformer <f>\n");
      return 2;
    }
    vidfab::SafeTensors ckpt;
    ckpt.open(req.transformer_path);
    std::printf("\n%s\n%.3f GB on disk, %zu tensors\n", req.transformer_path.c_str(),
                ckpt.file_size() / 1e9, ckpt.tensor_count());
    for (int i = 0; i < bench_load; ++i) {
      vidfab::dit::Transformer probe;
      const auto s0 = std::chrono::steady_clock::now();
      probe.load(ckpt);
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
      std::printf("load %d: %s on device in %6.3f s  (%.2f GB/s off disk)\n", i + 1,
                  format_bytes(probe.weight_bytes()).c_str(), sec,
                  static_cast<double>(ckpt.file_size()) / sec / 1e9);
      std::fflush(stdout);
    }
    return 0;
  }
#else
  if (bench_load > 0) {
    std::fprintf(stderr, "vidfab: --bench-load needs a GPU build\n");
    return 1;
  }
#endif

#if !VIDFAB_WITH_CUDA
  (void)sampler_kind;
  (void)dump_latents;
  (void)init_latents;
  std::fprintf(stderr, "vidfab: built without CUDA support; generate needs a GPU\n");
  return 1;
#else
  vidfab::RunOptions options;
  options.source =
      synthetic ? vidfab::LatentSource::kSyntheticNoise : vidfab::LatentSource::kDenoise;
  options.sampler = sampler_kind;
  options.dump_latents_path = dump_latents;
  options.attention_band = attn_band;
  options.init_latents_path = init_latents;

  std::printf("\n");
  const vidfab::RunResult run = vidfab::run_generate(req, plan, options);
  if (!run.ok) {
    std::fprintf(stderr, "\nvidfab: %s\n", run.message.c_str());
    return 1;
  }
  std::printf("\ndone in %.2f s\n", run.seconds_denoise + run.seconds_video_decode +
                                        run.seconds_audio_decode + run.seconds_output);
  return 0;
#endif
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

  // `--help` is honoured for every command here rather than inside each one,
  // so a command that takes no arguments at all still answers it. Checked
  // before dispatch, so it never runs the command by accident.
  if (const CommandHelp* c = find_command(command); c != nullptr && wants_help(argc - 2, argv + 2)) {
    return print_command_help(*c);
  }

  try {
    if (command == "generate") return cmd_generate(argc - 2, argv + 2);
    if (command == "inspect") return cmd_inspect(argc - 2, argv + 2);
    if (command == "compare") return cmd_compare(argc - 2, argv + 2);
    if (command == "devices") return cmd_devices();
    if (command == "tokenize") return cmd_tokenize(argc - 2, argv + 2);
#if VIDFAB_WITH_CUDA
    if (command == "decode") return cmd_decode(argc - 2, argv + 2);
#endif
    if (command == "version") {
      std::printf("vidfab %s\n", kVersion);
      return 0;
    }
    if (command == "help" || command == "--help" || command == "-h") {
      if (argc > 2) {
        const CommandHelp* c = find_command(argv[2]);
        if (c != nullptr) return print_command_help(*c);
        std::fprintf(stderr, "vidfab: unknown command '%s'\n\n", argv[2]);
        print_usage();
        return 2;
      }
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


