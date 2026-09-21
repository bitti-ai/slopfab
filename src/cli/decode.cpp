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
#if SLOPFAB_WITH_CUDA
bool latent_stats_from_metadata(const slopfab::SafeTensors& ckpt, std::vector<float>& mean,
                                std::vector<float>& std_dev) {
  auto it = ckpt.metadata().find("minimax_h3_video_vae");
  if (it == ckpt.metadata().end())
    return false;
  try {
    const slopfab::json::Value meta = slopfab::json::parse(it->second);
    const slopfab::json::Value* m = meta.find("latents_mean");
    const slopfab::json::Value* s = meta.find("latents_std");
    if (m == nullptr || s == nullptr)
      return false;
    mean.clear();
    std_dev.clear();
    for (const auto& v : m->as_array())
      mean.push_back(static_cast<float>(v.as_number()));
    for (const auto& v : s->as_array())
      std_dev.push_back(static_cast<float>(v.as_number()));
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
  if (wants_help(argc, argv))
    return print_command_help(*find_command("decode"));

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
      std::fprintf(stderr, "slopfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    }
  }

  if (vae_path.empty()) {
    std::fprintf(stderr, "slopfab: decode needs --vae <video_vae.safetensors>\n");
    return 2;
  }

  slopfab::SafeTensors ckpt;
  ckpt.open(vae_path);

  std::vector<float> mean;
  std::vector<float> std_dev;
  if (!latent_stats_from_metadata(ckpt, mean, std_dev)) {
    mean = slopfab::vae::default_video_latents_mean();
    std_dev = slopfab::vae::default_video_latents_std();
  }

  std::vector<float> z;
  if (!latent_path.empty()) {
    slopfab::SafeTensors latent_file;
    latent_file.open(latent_path);
    const slopfab::TensorView& lv = latent_file.at("latent");
    if (lv.shape.size() != 4 || lv.shape[0] != 24) {
      std::fprintf(stderr, "slopfab: latent tensor must be [24, T, H, W]\n");
      return 1;
    }
    T = static_cast<int>(lv.shape[1]);
    H = static_cast<int>(lv.shape[2]);
    W = static_cast<int>(lv.shape[3]);
    z = slopfab::to_f32(lv);
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
      slopfab::vae::ViTDecoder probe;
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

  slopfab::vae::ViTDecoder decoder;
  const auto load_start = std::chrono::steady_clock::now();
  decoder.load(ckpt);
  const auto load_end = std::chrono::steady_clock::now();
  std::printf("weights    %s on device in %.2f s\n", format_bytes(decoder.weight_bytes()).c_str(),
              std::chrono::duration<double>(load_end - load_start).count());

  slopfab::vae::DecodeSchedule schedule;
  schedule.tiling_enabled = !no_tiling;

  // The first decode pays one-time costs the steady state does not: scratch
  // allocation, the first RoPE build, and cuBLAS heuristic selection for each
  // GEMM shape. Reporting it as the decode time overstates the cost by a
  // noticeable margin, so timings are reported per run and `--repeat` exists to
  // expose the warm number.
  slopfab::vae::DecodedVideo video;
  for (int run = 0; run < repeat; ++run) {
    const auto t0 = std::chrono::steady_clock::now();
    video = decoder.decode(z.data(), T, H, W, mean, std_dev, schedule);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const char* label = (run == 0) ? "decoded   " : "  (warm)  ";
    std::printf("%s %d frames of %dx%d in %.3f s (%.2f fps)\n", label, video.frames, video.width,
                video.height, seconds,
                seconds > 0 ? static_cast<double>(video.frames) / seconds : 0.0);
    // The phase spans inside `decode` tile exactly this interval, so it is
    // their denominator. With --repeat both sides accumulate together.
    slopfab::cuda::PhaseProfiler::instance().add_total("video vae decode", seconds * 1000.0);
  }
  slopfab::cuda::PhaseProfiler::instance().report(stdout);

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
    std::fprintf(stderr, "slopfab: decode produced non-finite pixels\n");
    return 1;
  }

  if (!dump_path.empty()) {
    // Raw fp32 pixels, so two runs can be diffed with `slopfab compare` at
    // float precision rather than after 8-bit quantisation. The copy into the
    // writer's own vector type is what this diagnostic path already did.
    slopfab::write_safetensors(dump_path,
                               {{"pixels",
                                 {3, video.frames, video.height, video.width},
                                 std::vector<float>(video.data.begin(), video.data.end())}});
    std::printf("wrote      %s\n", dump_path.c_str());
  }

  slopfab::video::write_y4m(out_path, video.data, video.frames, video.height, video.width,
                            {fps, 1});
  std::printf("wrote      %s\n", out_path.c_str());
  if (!ppm_path.empty()) {
    slopfab::video::write_ppm(ppm_path, video.data, video.frames, video.height, video.width, 0);
    std::printf("wrote      %s\n", ppm_path.c_str());
  }
  return 0;
}
#endif // SLOPFAB_WITH_CUDA

}
