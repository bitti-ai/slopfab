#include "harness.h"
#include "slopfab/generate.h"
#if SLOPFAB_WITH_VULKAN
#include "../src/generation/helpers.h"
#endif

#include <memory>
#include <stdexcept>
#include <string>

namespace {
slopfab::GenerateRequest small_request() {
  slopfab::GenerateRequest request;
  request.prompt = "backend smoke test";
  request.canvas_width = 256;
  request.canvas_height = 256;
  request.num_frames = 22;
  request.num_inference_steps = 4;
  return request;
}

struct Capture {
  std::shared_ptr<const slopfab::LatentClip> latents;
  bool started = false;
  bool reached_decode = false;
};
}

SLOPFAB_TEST(generation_compiled_backend_default) {
  const slopfab::RunOptions options;
#if SLOPFAB_WITH_CUDA
  CHECK(options.inference_backend == slopfab::DeviceBackend::kCuda);
  CHECK(options.attention_mode == slopfab::AttentionMode::kFlash2);
#else
  CHECK(options.inference_backend == slopfab::DeviceBackend::kVulkan);
  CHECK(options.attention_mode == slopfab::AttentionMode::kExact);
#endif
}

// This executes the shared session, sampler and latent handoff on each compiled
// backend. Cancellation before VAE loading needs neither a GPU nor model files.
SLOPFAB_TEST(generation_synthetic_pipeline_without_gpu) {
  using namespace slopfab;
  const auto request = small_request();
  const auto plan = resolve_plan(request);
  GenerationSession session;
  for (auto backend : {DeviceBackend::kCuda, DeviceBackend::kVulkan}) {
#if !SLOPFAB_WITH_CUDA
    if (backend == DeviceBackend::kCuda)
      continue;
#endif
#if !SLOPFAB_WITH_VULKAN
    if (backend == DeviceBackend::kVulkan)
      continue;
#endif
    Capture capture;
    RunOptions options;
    options.inference_backend = backend;
    options.attention_mode = AttentionMode::kExact;
    options.source = LatentSource::kSyntheticNoise;
    options.verbose = false;
    options.hook_userdata = &capture;
    options.on_latents = [](const std::shared_ptr<const LatentClip>& clip, void* userdata) {
      static_cast<Capture*>(userdata)->latents = clip;
    };
    options.on_progress = [](RunStage stage, int, int, void* userdata) {
      auto& capture = *static_cast<Capture*>(userdata);
      capture.started |= stage == RunStage::kStarting;
      capture.reached_decode |= stage == RunStage::kVideoDecode;
      return stage != RunStage::kVideoDecode;
    };
    const auto result = run_generate(session, request, plan, options);
    CHECK(!result.ok && result.cancelled);
    CHECK(result.message == "cancelled during video decode");
    CHECK(capture.started && capture.reached_decode);
    CHECK(capture.latents != nullptr);
    CHECK(capture.latents->width == 256 && capture.latents->height == 256);
    CHECK(!capture.latents->sampled);
    CHECK(!capture.latents->video_rows.empty());
    CHECK(!capture.latents->audio_rows.empty());
    session.clear();
  }
}

SLOPFAB_TEST(generation_unavailable_backend_rejected_before_start) {
  using namespace slopfab;
  const auto request = small_request();
  const auto plan = resolve_plan(request);
  for (auto backend : {DeviceBackend::kCuda, DeviceBackend::kVulkan}) {
#if SLOPFAB_WITH_CUDA
    if (backend == DeviceBackend::kCuda)
      continue;
#endif
#if SLOPFAB_WITH_VULKAN
    if (backend == DeviceBackend::kVulkan)
      continue;
#endif
    bool called = false;
    RunOptions options;
    options.inference_backend = backend;
    options.attention_mode = AttentionMode::kExact;
    options.source = LatentSource::kSyntheticNoise;
    options.verbose = false;
    options.hook_userdata = &called;
    options.on_progress = [](RunStage, int, int, void* userdata) {
      *static_cast<bool*>(userdata) = true;
      return true;
    };
    const auto result = run_generate(request, plan, options);
    CHECK(!result.ok && !result.cancelled);
    CHECK(result.message.find("this build disabled") != std::string::npos);
    CHECK(!called);
  }
}

#if SLOPFAB_WITH_VULKAN
SLOPFAB_TEST(generation_vulkan_device_selection) {
  using namespace slopfab;
  using generation::select_vulkan_inference_device;
  vulkan::DeviceInfo integrated;
  integrated.name = "integrated";
  integrated.timeline_semaphore = integrated.shader_int64 = true;
  integrated.fp32_signed_zero_inf_nan_preserve = integrated.fp32_rounding_rte = true;
  integrated.max_compute_workgroup_invocations = integrated.max_compute_workgroup_size[0] = 1024;
  integrated.max_compute_shared_memory_bytes = 65536;
  integrated.memory_heaps.push_back({4ull << 30, true});
  auto discrete = integrated;
  discrete.name = "discrete";
  discrete.discrete = true;
  discrete.memory_heaps.front().bytes = 16ull << 30;
  auto software = integrated;
  software.name = "software";
  software.software = true;
  software.memory_heaps.front().bytes = 64ull << 30;

  CHECK(select_vulkan_inference_device({software, integrated, discrete}, true, false, false) == 2);
  CHECK(select_vulkan_inference_device({discrete, integrated}, true, false, false) == 0);
  auto invalid = discrete;
  invalid.shader_int64 = false;
  CHECK(select_vulkan_inference_device({invalid, integrated}, true, false, false) == 1);

  // An unqualified discrete device cannot displace a qualified exact device.
  auto qualified = integrated;
  qualified.vendor_id = 0x10de;
  qualified.device_id = 0x2b85;
  qualified.driver_version = 0x98960000;
  CHECK(select_vulkan_inference_device({discrete, qualified}, false, false, false) == 1);

  discrete.cooperative_matrix = discrete.shader_float16 = discrete.storage_buffer_16bit = true;
  discrete.shader_bfloat16_type = discrete.shader_bfloat16_cooperative_matrix = true;
  discrete.cooperative_matrix_bf16_f32_16x16x16 = true;
  discrete.cooperative_matrix_f16_f32_16x16x16 = true;
  CHECK(select_vulkan_inference_device({integrated, discrete}, true, true, false) == 1);

  auto sage = discrete;
  sage.name = "sage";
  sage.shader_int8 = sage.cooperative_matrix_i8_i32_16x16x32 = true;
  sage.compute_subgroup_shuffle = sage.compute_subgroup_arithmetic = true;
  sage.subgroup_size = 32;
  CHECK(select_vulkan_inference_device({discrete, sage}, true, true, true) == 1);
  // A discrete GPU must not displace another adapter that can actually run
  // the requested attention graph. Conditioner GEMM can still use its scalar
  // fallback and Sage has its own subgroup64-capable shader.
  auto attention_gpu = discrete;
  attention_gpu.discrete = false;
  attention_gpu.subgroup_size = 32;
  auto wave64 = discrete;
  wave64.subgroup_size = 64;
  CHECK(select_vulkan_inference_device({wave64, attention_gpu}, true, true, false,
                                       AttentionMode::kExact) == 1);
  CHECK(select_vulkan_inference_device({wave64, attention_gpu}, true, true, false) == 0);
  auto small_shared = discrete;
  small_shared.subgroup_size = 32;
  small_shared.max_compute_shared_memory_bytes = 32768;
  CHECK(select_vulkan_inference_device({small_shared, attention_gpu}, true, true, false,
                                       AttentionMode::kExact) == 1);
  auto small_workgroup = discrete;
  small_workgroup.subgroup_size = 32;
  small_workgroup.max_compute_workgroup_invocations = 512;
  CHECK(select_vulkan_inference_device({small_workgroup, attention_gpu}, true, true, false,
                                       AttentionMode::kExact) == 1);
  // Flash2 fits smaller workgroups; it must not inherit exact-H3's 1024 gate.
  small_workgroup.compute_subgroup_shuffle = true;
  CHECK(select_vulkan_inference_device({small_workgroup, attention_gpu}, true, true, false,
                                       AttentionMode::kFlash2) == 0);
  sage.subgroup_size = 64;
  CHECK(select_vulkan_inference_device({sage}, true, true, true, AttentionMode::kSage2) == 0);
  bool diagnostic = false;
  try {
    (void)select_vulkan_inference_device({integrated}, false, false, false);
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    diagnostic = message.find("integrated") != std::string::npos &&
                 message.find("--vulkan-arithmetic portable") != std::string::npos;
  }
  CHECK(diagnostic);
  bool empty_rejected = false;
  try {
    (void)select_vulkan_inference_device({}, true, false, false);
  } catch (const std::runtime_error&) {
    empty_rejected = true;
  }
  CHECK(empty_rejected);
}
#endif
