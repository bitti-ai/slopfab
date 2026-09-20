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
int cmd_generate(int argc, char** argv, const char* executable) {
  if (wants_help(argc, argv)) return print_command_help(*find_command("generate"));

  slopfab::GenerateRequest req;
  std::vector<std::pair<std::string, bool>> reference_files;
  req.canvas_width = 864;
  req.canvas_height = 480;
  req.num_frames = 124;
  // Step defaults are resolved from the model/task recipe.
  req.seed = 0;
  bool dry_run = false;
  bool synthetic = false;
  slopfab::sampler::SamplerKind sampler_kind = slopfab::sampler::SamplerKind::kEuler;
  std::string dump_latents;
  std::string save_latents;
  std::string continue_from;
  bool saw_overlap = false;
  std::string prompt_file;
  bool saw_prompt = false;
  int attn_band = 0;
  slopfab::AttentionMode attention_mode = slopfab::AttentionMode::kSage2;
  uint64_t vulkan_sage_workspace_mib = 64;
  slopfab::SolSchedule sol_schedule;
  std::string init_latents;
  std::string prompt_embedding;
  int bench_load = 0;
  bool saw_aspect = false;
  bool saw_resolution = false;
  bool saw_out = false;
  bool saw_seed = false;
  int count = 1;
  std::string inference_backend = "cuda";
  std::string output_accelerator = "cpu";

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
      saw_prompt = true;
    } else if (arg == "--prompt-file") {
      prompt_file = next("--prompt-file");
    } else if (arg == "--out") {
      req.out_path = next("--out");
      saw_out = true;
    } else if (arg == "--frames") {
      req.num_frames = std::atoi(next("--frames"));
    } else if (arg == "--steps") {
      req.num_inference_steps = std::atoi(next("--steps"));
    } else if (arg == "--seed") {
      // A negative seed asks for a random one, the same as passing no --seed at
      // all. Checked on the text rather than on the parsed value because
      // `strtoull` silently wraps "-1" to 2^64-1, which would read as an
      // ordinary explicit seed. Testing the sign first also keeps the whole
      // unsigned range usable for seeds that really are meant to be large.
      const std::string_view value = next("--seed");
      const size_t first = value.find_first_not_of(" \t");
      if (first != std::string_view::npos && value[first] == '-') {
        saw_seed = false;
      } else {
        req.seed = std::strtoull(value.data(), nullptr, 10);
        saw_seed = true;
      }
    } else if (arg == "--count") {
      count = std::atoi(next("--count"));
    } else if (arg == "--sampler") {
      const std::string v = next("--sampler");
      if (v == "euler") {
        sampler_kind = slopfab::sampler::SamplerKind::kEuler;
      } else if (v == "ab2") {
        sampler_kind = slopfab::sampler::SamplerKind::kAb2;
      } else {
        std::fprintf(stderr, "slopfab: --sampler wants euler or ab2, got '%s'\n", v.c_str());
        return 2;
      }
    } else if (arg == "--aspect") {
      const std::string v = next("--aspect");
      const size_t colon = v.find(':');
      if (colon == std::string::npos) {
        std::fprintf(stderr, "slopfab: --aspect wants W:H, e.g. 16:9\n");
        return 2;
      }
      req.aspect_w = std::atoi(v.substr(0, colon).c_str());
      req.aspect_h = std::atoi(v.substr(colon + 1).c_str());
      req.canvas_width = 0;
      req.canvas_height = 0;
      saw_aspect = true;
    } else if (arg == "--resolution") {
      const std::string v = next("--resolution");
      const size_t x = v.find_first_of("xX");
      if (x == std::string::npos) {
        std::fprintf(stderr, "slopfab: --resolution wants WxH, e.g. 1344x768\n");
        return 2;
      }
      req.canvas_width = std::atoi(v.substr(0, x).c_str());
      req.canvas_height = std::atoi(v.substr(x + 1).c_str());
      if (req.canvas_width <= 0 || req.canvas_height <= 0) {
        std::fprintf(stderr, "slopfab: --resolution wants two positive numbers, got '%s'\n",
                     v.c_str());
        return 2;
      }
      saw_resolution = true;
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
    } else if (arg == "--refmod") {
      req.refmods.push_back({slopfab::RefMod::load(next("--refmod")), 1.0f, 1});
    } else if (arg == "--refmod-strength" || arg == "--refmod-copies") {
      if (req.refmods.empty()) throw std::runtime_error(std::string(arg) + " must follow --refmod");
      const std::string value = next(arg == "--refmod-strength" ? "--refmod-strength" : "--refmod-copies");
      size_t consumed = 0;
      if (arg == "--refmod-strength") req.refmods.back().strength = std::stof(value, &consumed);
      else req.refmods.back().copies = std::stoi(value, &consumed);
      if (consumed != value.size()) throw std::runtime_error("invalid refmod numeric value: " + value);
      slopfab::validate_refmods(req.refmods);
    } else if (arg == "--reference-image") {
      req.reference_image_paths.emplace_back(next("--reference-image"));
    } else if (arg == "--reference-video" || arg == "--reference-audio") {
      const bool is_video = arg == "--reference-video";
      reference_files.emplace_back(next(is_video ? "--reference-video" : "--reference-audio"), is_video);
    } else if (arg == "--raw") {
      req.raw_output = true;
    } else if (arg == "--inference-backend") {
      inference_backend = next("--inference-backend");
      if (inference_backend != "cuda" && inference_backend != "vulkan") {
        std::fprintf(stderr,
                     "slopfab: --inference-backend wants cuda or vulkan, got '%s'\n",
                     inference_backend.c_str());
        return 2;
      }
    } else if (arg == "--output-accelerator") {
      output_accelerator = next("--output-accelerator");
      if (output_accelerator != "cpu" && output_accelerator != "vulkan") {
        std::fprintf(stderr,
                     "slopfab: --output-accelerator wants cpu or vulkan, got '%s'\n",
                     output_accelerator.c_str());
        return 2;
      }
    } else if (arg == "--motion-cache") {
      req.motion_cache.enabled = true;
    } else if (arg == "--motion-cache-verbose") {
      req.motion_cache.verbose = true;
    } else if (arg == "--motion-cache-threshold" || arg == "--motion-cache-strength" ||
               arg == "--motion-cache-start" || arg == "--motion-cache-end") {
      const char* value = next(arg.data());
      char* end = nullptr;
      const float parsed = std::strtof(value, &end);
      if (end == value || *end != '\0') throw std::runtime_error(std::string(arg) + " requires a number");
      if (arg == "--motion-cache-threshold") req.motion_cache.reuse_threshold = parsed;
      else if (arg == "--motion-cache-strength") req.motion_cache.motion_strength = parsed;
      else if (arg == "--motion-cache-start") req.motion_cache.start_percent = parsed;
      else req.motion_cache.end_percent = parsed;
    } else if (arg == "--motion-cache-warmup" || arg == "--motion-cache-max-skips" ||
               arg == "--motion-cache-subsample") {
      const char* value = next(arg.data());
      char* end = nullptr;
      const long parsed = std::strtol(value, &end, 10);
      if (end == value || *end != '\0' || parsed < 0 || parsed > 32)
        throw std::runtime_error(std::string(arg) + " requires a small positive integer");
      if (arg == "--motion-cache-warmup") req.motion_cache.warmup_steps = static_cast<int>(parsed);
      else if (arg == "--motion-cache-max-skips") req.motion_cache.max_consecutive_skips = static_cast<int>(parsed);
      else req.motion_cache.subsample_factor = static_cast<int>(parsed);
    } else if (arg == "--cache-threshold") {
      req.cache_threshold = static_cast<float>(std::strtod(next("--cache-threshold"), nullptr));
    } else if (arg == "--cache-warmup") {
      req.cache_warmup = std::atoi(next("--cache-warmup"));
    } else if (arg == "--skip-every") {
      req.skip_every = std::atoi(next("--skip-every"));
    } else if (arg == "--block-cache-span") {
      req.block_cache_span = std::atoi(next("--block-cache-span"));
    } else if (arg == "--block-cache-start") {
      req.block_cache_start = std::atoi(next("--block-cache-start"));
    } else if (arg == "--block-cache-interval") {
      req.block_cache_interval = std::atoi(next("--block-cache-interval"));
    } else if (arg == "--block-cache-warmup") {
      req.block_cache_warmup = std::atoi(next("--block-cache-warmup"));
    } else if (arg == "--dry-run") {
      dry_run = true;
    } else if (arg == "--synthetic-latents") {
      synthetic = true;
    } else if (arg == "--dump-latents") {
      dump_latents = next("--dump-latents");
    } else if (arg == "--save-latents") {
      save_latents = next("--save-latents");
      if (save_latents.empty()) throw std::runtime_error("--save-latents needs a nonempty path");
    } else if (arg == "--continue-from") {
      continue_from = next("--continue-from");
      if (continue_from.empty()) throw std::runtime_error("--continue-from needs a nonempty path");
    } else if (arg == "--overlap-frames") {
      const std::string value = next("--overlap-frames");
      size_t consumed = 0;
      req.continuation_overlap_frames = std::stoi(value, &consumed);
      if (consumed != value.size()) throw std::runtime_error("invalid --overlap-frames value");
      saw_overlap = true;
    } else if (arg == "--attn-band") {
      attn_band = std::atoi(next("--attn-band"));
    } else if (arg == "--attention") {
      const std::string v = next("--attention");
      if (!slopfab::parse_attention_mode(v, &attention_mode)) {
        std::fprintf(stderr,
                     "slopfab: --attention wants none, flash2, sage2, sol, "
                     "sol-experimental, or exact, got '%s'\n", v.c_str());
        return 2;
      }
    } else if (arg == "--vulkan-sage-workspace-mib") {
      const std::string value = next("--vulkan-sage-workspace-mib");
      if (value.empty() || value.size() > 5 || value.find_first_not_of("0123456789") != std::string::npos ||
          (vulkan_sage_workspace_mib = std::strtoull(value.c_str(), nullptr, 10)) > 65536) {
        std::fprintf(stderr, "slopfab: --vulkan-sage-workspace-mib wants an integer from 0 to 65536\n");
        return 2;
      }
    } else if (arg == "--sol-beta") {
      sol_schedule.beta = std::strtof(next("--sol-beta"), nullptr);
    } else if (arg == "--sol-error-k") {
      sol_schedule.error_k=std::strtof(next("--sol-error-k"),nullptr);
    } else if (arg == "--sol-error-v") {
      sol_schedule.error_v=std::strtof(next("--sol-error-v"),nullptr);
    } else if (arg == "--sol-step-start") {
      sol_schedule.step_begin = std::atoi(next("--sol-step-start"));
    } else if (arg == "--sol-step-end") {
      sol_schedule.step_end = std::atoi(next("--sol-step-end"));
    } else if (arg == "--sol-step-every") {
      sol_schedule.step_every = std::atoi(next("--sol-step-every"));
    } else if (arg == "--sol-layer-start") {
      sol_schedule.layer_begin = std::atoi(next("--sol-layer-start"));
    } else if (arg == "--sol-layer-end") {
      sol_schedule.layer_end = std::atoi(next("--sol-layer-end"));
    } else if (arg == "--sol-layer-every") {
      sol_schedule.layer_every = std::atoi(next("--sol-layer-every"));
    } else if (arg == "--init-latents") {
      init_latents = next("--init-latents");
    } else if (arg == "--lora") {
      req.loras.push_back({next("--lora"), 1.0f});
    } else if (arg == "--lora-strength") {
      if (req.loras.empty()) throw std::runtime_error("--lora-strength must follow --lora");
      const std::string value = next("--lora-strength");
      size_t used = 0;
      const float strength = std::stof(value, &used);
      if (used != value.size() || !std::isfinite(strength))
        throw std::runtime_error("--lora-strength requires a finite number");
      req.loras.back().strength = strength;
    } else if (arg == "--conditioning-settings") {
      const std::string path = next("--conditioning-settings");
      std::ifstream in(path, std::ios::binary);
      if (!in) throw std::runtime_error("cannot read conditioning settings: " + path);
      std::ostringstream contents;
      contents << in.rdbuf();
      req.conditioning = slopfab::parse_conditioning_settings(contents.str());
    } else if (arg == "--sampling-settings") {
      const std::string path = next("--sampling-settings");
      std::ifstream in(path, std::ios::binary);
      if (!in) throw std::runtime_error("cannot read sampling settings: " + path);
      std::ostringstream contents;
      contents << in.rdbuf();
      req.sampling = slopfab::parse_sampling_settings(contents.str());
    } else if (arg == "--schedule") {
      const std::string value = next("--schedule");
      if (value == "default") req.schedule = slopfab::sampler::ScheduleKind::kDefault;
      else if (value == "taomate-3step") req.schedule = slopfab::sampler::ScheduleKind::kTaoMate3Step;
      else throw std::runtime_error("--schedule wants default or taomate-3step");
    } else if (arg == "--animate") {
      req.animate = true;
    } else if (arg == "--preserve-driving-audio") {
      req.preserve_driving_audio = true;
    } else if (arg == "--prompt-embedding") {
      prompt_embedding = next("--prompt-embedding");
    } else if (arg == "--bench-load") {
      bench_load = std::atoi(next("--bench-load"));
    } else {
      std::fprintf(stderr, "slopfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    }
  }

  if (inference_backend == "vulkan") {
    if (!slopfab::attention_mode_supported(slopfab::DeviceBackend::kVulkan, attention_mode)) {
      std::fprintf(stderr,
                   "slopfab: Vulkan inference requires --attention exact, flash2 or sage2; mode '%s' "
                   "will not be remapped and no CUDA fallback was used\n",
                   slopfab::attention_mode_name(attention_mode));
      return 1;
    }
  }

  // Both write the same field, so accepting both would mean silently honouring
  // one of them and dropping the other.
  if (saw_prompt && !prompt_file.empty()) {
    std::fprintf(stderr, "slopfab: --prompt and --prompt-file cannot be combined\n");
    return 2;
  }
  if (!prompt_file.empty()) {
    // Read here rather than at run time: the prompt shapes the plan, so
    // `--dry-run` has to see it, and a bad path should fail before any weights
    // are touched. The text is used verbatim apart from a UTF-8 BOM, CRLF line
    // endings and surrounding blank space -- all things an editor adds and no
    // prompt wants in its token stream.
    std::ifstream in(prompt_file, std::ios::binary);
    if (!in) {
      std::fprintf(stderr, "slopfab: cannot read --prompt-file '%s'\n", prompt_file.c_str());
      return 2;
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    std::string text = contents.str();
    if (text.rfind("\xEF\xBB\xBF", 0) == 0) text.erase(0, 3);
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    const size_t first = text.find_first_not_of(" \t\n");
    if (first == std::string::npos) {
      std::fprintf(stderr, "slopfab: --prompt-file '%s' has no prompt in it\n",
                   prompt_file.c_str());
      return 2;
    }
    const size_t last = text.find_last_not_of(" \t\n");
    req.prompt = text.substr(first, last - first + 1);
  }

  // `--synthetic-latents --init-latents <f>` is the decode-an-existing-latent
  // path and needs no prompt; the seeded-noise form still does not either.
  if (req.prompt.empty() && prompt_embedding.empty() && !dry_run && !synthetic) {
    std::fprintf(stderr, "slopfab: generate needs --prompt \"...\" or --prompt-file <file>\n");
    return 2;
  }
  // Rejected rather than ranked. Silently letting one win would mean a run
  // whose canvas is not the one half the command line asked for, and the two
  // flags are close enough in intent that a user passing both has made a
  // mistake worth telling them about.
  if (saw_aspect && saw_resolution) {
    std::fprintf(stderr,
                 "slopfab: --aspect and --resolution set the same thing; pass one or the other\n");
    return 2;
  }
  if (count <= 0) {
    std::fprintf(stderr, "slopfab: --count must be a positive integer\n");
    return 2;
  }
  if (!saw_out) req.out_path = timestamped_output_path();
  if (saw_overlap && continue_from.empty())
    throw std::runtime_error("--overlap-frames requires --continue-from");
  if (!continue_from.empty()) {
    if (synthetic || !init_latents.empty())
      throw std::runtime_error("continuation requires denoising from fresh noise");
    if (saw_aspect && !saw_resolution)
      throw std::runtime_error("continuation inherits its canvas; omit --aspect");
    req.continuation = slopfab::LatentClip::load(continue_from);
  }
  for (const auto& entry : reference_files) {
    req.reference_media.push_back(std::make_shared<const slopfab::ReferenceMedia>(
        slopfab::cli::decode_reference_file(entry.first, entry.second, executable)));
    slopfab::validate_reference_media(req.reference_image_paths.size(), req.reference_media);
  }
  const std::string base_out_path = req.out_path;
  const uint64_t base_seed = req.seed;
  slopfab::RunOptions options;
  options.source =
      synthetic ? slopfab::LatentSource::kSyntheticNoise : slopfab::LatentSource::kDenoise;
  options.inference_backend = inference_backend == "vulkan"
      ? slopfab::DeviceBackend::kVulkan : slopfab::DeviceBackend::kCuda;
  options.sampler = sampler_kind;
  options.dump_latents_path = dump_latents;
  options.save_latents_path = save_latents;
  options.attention_band = attn_band;
  options.attention_mode = attention_mode;
  options.vulkan_sage_extra_workspace_bytes = vulkan_sage_workspace_mib << 20;
  options.sol_schedule = sol_schedule;
  options.init_latents_path = init_latents;
  options.prompt_embedding_path = prompt_embedding;
  discover_generate_checkpoints(req, executable);
  slopfab::GeneratePlan plan = slopfab::resolve_plan(req);
  slopfab::validate_generation_options(req, plan, options);

  // After `resolve_plan`, so a canvas that is going to be rejected outright is
  // not first warned about — an invalid request should produce one message
  // about what is wrong with it, not a size advisory followed by a refusal.
  if (saw_resolution && slopfab::dit::canvas_exceeds_trained_area(req.canvas_height,
                                                                req.canvas_width)) {
    // A warning, not a refusal: the caller named this canvas. But packed rows
    // grow with area and attention with their square, so an innocent-looking
    // doubling is roughly four times the attention cost.
    std::fprintf(stderr,
                 "slopfab: %dx%d is %.2fx the 1344x768 area the model was trained at; "
                 "attention cost grows with the square of that, and quality outside the "
                 "trained range is uncharacterised\n",
                 req.canvas_width, req.canvas_height,
                 static_cast<double>(req.canvas_width) * req.canvas_height / (1344.0 * 768.0));
  }
  if (dry_run) {
    for (int generation = 0; generation < count; ++generation) {
      req.seed = saw_seed ? base_seed + static_cast<uint64_t>(generation) : random_seed();
      req.out_path = counted_output_path(base_out_path, generation, count);
      if (generation > 0) std::printf("\n");
      std::fputs(slopfab::describe_plan(req, plan).c_str(), stdout);
    }
    return 0;
  }

#if SLOPFAB_WITH_CUDA
  // Dry-run above is deliberately device-free. Synthetic latents skip the
  // transformer, so only a real denoise run needs the pinned exact tuple.
  if (inference_backend == "cuda" && !synthetic &&
      attention_mode == slopfab::AttentionMode::kExact &&
      !slopfab::cuda::deterministic_h3_attention_available()) {
    std::fprintf(stderr,
                 "slopfab: --attention exact is unavailable on this CUDA device/runtime tuple\n");
    return 1;
  }
#endif

#if !SLOPFAB_WITH_CUDA
  (void)executable;
  std::fprintf(stderr,
               "slopfab: built without CUDA support; model inference requires CUDA. "
               "Vulkan accelerates output conversion only\n");
  return 1;
#else

#if SLOPFAB_WITH_VULKAN
  std::unique_ptr<slopfab::vulkan::Yuv420Converter> output_converter;
  if (output_accelerator == "vulkan") {
    try {
      output_converter = std::make_unique<slopfab::vulkan::Yuv420Converter>();
      std::printf("output      Vulkan RGB-to-YUV on %s (independent output backend)\n",
                  output_converter->device_name());
    } catch (const std::exception& error) {
      std::fprintf(stderr, "slopfab: Vulkan output accelerator unavailable: %s\n", error.what());
      return 1;
    }
  }
#else
  if (output_accelerator == "vulkan") {
    std::fprintf(stderr,
                 "slopfab: Vulkan output accelerator requested, but this build disabled Vulkan\n");
    return 1;
  }
#endif

  if (!saw_out) std::filesystem::create_directories(std::filesystem::path(req.out_path).parent_path());

  ensure_generate_models(req, executable, prompt_embedding.empty());
  plan = slopfab::resolve_plan(req);
  slopfab::validate_generation_options(req, plan, options);

#if SLOPFAB_WITH_CUDA
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
      std::fprintf(stderr, "slopfab: --bench-load needs --transformer <f>\n");
      return 2;
    }
    slopfab::SafeTensors ckpt;
    ckpt.open(req.transformer_path);
    slopfab::LoraAdapters loras;
    loras.load(req.loras, ckpt);
    std::printf("\n%s\n%.3f GB on disk, %zu tensors\n", req.transformer_path.c_str(),
                ckpt.file_size() / 1e9, ckpt.tensor_count());
    for (int i = 0; i < bench_load; ++i) {
      slopfab::dit::Transformer probe;
      const auto s0 = std::chrono::steady_clock::now();
      probe.load(ckpt, {}, &loras);
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
    std::fprintf(stderr, "slopfab: --bench-load needs a GPU build\n");
    return 1;
  }
#endif

#if !SLOPFAB_WITH_CUDA
  (void)sampler_kind;
  (void)dump_latents;
  (void)init_latents;
  (void)attn_band;
  std::fprintf(stderr, "slopfab: built without CUDA support; generate needs a GPU\n");
  return 1;
#else
#if SLOPFAB_WITH_VULKAN
  options.output_frame_converter = output_converter.get();
#endif

  slopfab::GenerationSession session;
  for (int generation = 0; generation < count; ++generation) {
    req.seed = saw_seed ? base_seed + static_cast<uint64_t>(generation) : random_seed();
    req.out_path = counted_output_path(base_out_path, generation, count);
    options.reuse_models = count > 1;
    options.save_latents_path = save_latents.empty() ? std::string() : counted_output_path(save_latents, generation, count);
    options.release_reused_models = generation + 1 == count;
    if (generation > 0) std::printf("\n");
    std::fputs(slopfab::describe_plan(req, plan).c_str(), stdout);
    std::printf("\n");
    const slopfab::RunResult run = slopfab::run_generate(session, req, plan, options);
    if (!run.ok) {
      std::fprintf(stderr, "\nslopfab: generation %d of %d: %s\n", generation + 1, count,
                   run.message.c_str());
      return 1;
    }
    std::printf("\ndone in %.2f s\n", run.seconds_denoise + run.seconds_video_decode +
                                          run.seconds_audio_decode + run.seconds_output);
  }
  return 0;
#endif
#endif  // SLOPFAB_WITH_CUDA
}


}
