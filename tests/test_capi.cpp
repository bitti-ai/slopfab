// The C ABI, exercised through the DLL.
//
// This binary links slopfab_c and nothing else, which is the point: it is the
// only test in the tree whose subject is the *module* rather than the code. A
// missing export, a mangled name, a calling convention the two sides disagree
// about, or a struct they lay out differently would all pass a test that
// compiled capi.cpp into itself, and fail here — which is where a Rust or C#
// consumer would meet them.
//
// Nothing here reads a checkpoint or touches the GPU. Everything up to the
// plan is arithmetic, and the generation entry points are called only on the
// paths that reject their arguments before any work starts: a test that
// actually generated would need 34 GB of weights and several minutes.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <limits>

#include "harness.h"
#include "slopfab/capi.h"
#include "refmod_fixture.h"
#include "latent_fixture.h"

SLOPFAB_TEST(capi_motion_cache_validation_and_atomic_setter) {
  auto* request = slopfab_request_create();
  CHECK(request != nullptr);
  CHECK(slopfab_request_set_motion_cache(nullptr, 1, .15f, 1, 4, 2, .15f, .95f, 8, 0) ==
        SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_motion_cache(request, 1, .15f, 1, 4, 2, .15f, .95f, 8, 0) == SLOPFAB_OK);
  CHECK(slopfab_request_set_motion_cache(request, 1,
        std::numeric_limits<float>::quiet_NaN(), 1, 4, 2, .15f, .95f, 8, 0) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_motion_cache(request, 1, .15f, 1, 4, 2, .95f, .15f, 8, 0) ==
        SLOPFAB_ERR_INVALID_ARGUMENT);
  char* description = nullptr;
  CHECK(slopfab_describe_plan(request, &description) == SLOPFAB_OK);
  CHECK(description && std::strstr(description, "MotionCache") && std::strstr(description, "threshold 0.150"));
  slopfab_free_string(description);
  CHECK(slopfab_request_set_motion_cache(request, 0, .15f, 1, 4, 2, .15f, .95f, 8, 0) == SLOPFAB_OK);
  description = nullptr;
  CHECK(slopfab_describe_plan(request, &description) == SLOPFAB_OK);
  CHECK(description && !std::strstr(description, "MotionCache"));
  slopfab_free_string(description);
  slopfab_request_destroy(request);
}

SLOPFAB_TEST(capi_continuation_snapshot_and_plan) {
  LatentFixture fixture;
  fixture.write();
  auto* request = slopfab_request_create();
  CHECK(request != nullptr);
  CHECK(slopfab_request_set_save_latents(request, "next.safetensors") == SLOPFAB_OK);
  CHECK(slopfab_request_set_save_latents(request, nullptr) == SLOPFAB_OK);
  CHECK(slopfab_request_set_retain_latents(request, 1) == SLOPFAB_OK);
  CHECK(slopfab_request_set_continuation_file(request, fixture.path.string().c_str(), 22) == SLOPFAB_OK);
  // The attached snapshot survives replacement by invalid bytes.
  fixture.write(39, true, "unsupported");
  CHECK(slopfab_request_set_continuation_file(request, fixture.path.string().c_str(), 22) != SLOPFAB_OK);
  std::filesystem::remove(fixture.path);
  CHECK(slopfab_request_set_frames(request, 18) == SLOPFAB_OK);
  slopfab_plan plan{};
  CHECK(slopfab_resolve_plan(request, &plan) == SLOPFAB_OK);
  CHECK(plan.aligned_frames == 73 && plan.canvas_width == 64 && plan.canvas_height == 32);
  CHECK(plan.latent_frames == 17);
  CHECK(plan.num_audio_latents == 94);  // round(73*5/3) - round(17*5/3)
  CHECK(plan.sequence_rows_without_text == 34 + 188 + 14 + 74);
  CHECK(slopfab_request_set_still_image(request, 1) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request, &plan) == SLOPFAB_ERR_INVALID_REQUEST);
  CHECK(slopfab_request_set_still_image(request, 0) == SLOPFAB_OK);
  CHECK(slopfab_request_set_synthetic_latents(request, 1) == SLOPFAB_OK);
  slopfab_generation* generation = nullptr;
  CHECK(slopfab_generation_start(request, nullptr, nullptr, &generation) == SLOPFAB_ERR_INVALID_REQUEST);
  CHECK(generation == nullptr);
  CHECK(slopfab_request_clear_continuation(request) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request, &plan) == SLOPFAB_OK);
  CHECK(plan.aligned_frames == 22);
  slopfab_request_destroy(request);
}

SLOPFAB_TEST(capi_continuation_null_arguments) {
  CHECK(slopfab_request_set_save_latents(nullptr, "x") == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_retain_latents(nullptr, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_clear_continuation(nullptr) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_continuation_file(nullptr, "x", 22) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_continuation_generation(nullptr, nullptr, 22) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_generation_save_latents(nullptr, "x") == SLOPFAB_ERR_INVALID_ARGUMENT);
}

SLOPFAB_TEST(capi_refmod_loading_ownership_and_validation) {
  RefModFixture fixture; fixture.write();
  slopfab_request* request = slopfab_request_create();
  CHECK(request != nullptr);
  if (!request) return;
  struct Guard { slopfab_request* p; ~Guard() { slopfab_request_destroy(p); } } guard{request};
  slopfab_plan base{}, loaded{}, cleared{};
  CHECK(slopfab_resolve_plan(request, &base) == SLOPFAB_OK);
  CHECK(slopfab_request_add_refmod(nullptr, "x", 1, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_add_refmod(request, nullptr, 1, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_add_refmod(request, "", 1, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_add_refmod(request, "x", NAN, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_add_refmod(request, "x", 1.1f, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_add_refmod(request, "x", 1, 0) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_add_refmod(request, "x", 1, 11) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_add_refmod(request, fixture.path.string().c_str(), 1, 2) == SLOPFAB_OK);
  std::filesystem::remove(fixture.path);  // Neither a retained mapping nor deferred I/O.
  CHECK(slopfab_resolve_plan(request, &loaded) == SLOPFAB_OK);
  CHECK(loaded.sequence_rows_without_text == base.sequence_rows_without_text + 8);
  CHECK(slopfab_request_add_refmod(request, fixture.path.string().c_str(), 1, 1) != SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request, &loaded) == SLOPFAB_OK);
  CHECK(loaded.sequence_rows_without_text == base.sequence_rows_without_text + 8);
  CHECK(slopfab_request_clear_refmods(request) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request, &cleared) == SLOPFAB_OK);
  CHECK(cleared.sequence_rows_without_text == base.sequence_rows_without_text);
  fixture.write();
  CHECK(slopfab_request_add_refmod(request, fixture.path.string().c_str(), 0, 2) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request, &loaded) == SLOPFAB_OK);
  CHECK(loaded.sequence_rows_without_text == base.sequence_rows_without_text);
  CHECK(slopfab_request_clear_refmods(nullptr) == SLOPFAB_ERR_INVALID_ARGUMENT);
}

namespace {

// The C API has no RAII of its own — that is what makes it a C API — so the
// tests bring their own, and a CHECK failure part-way through cannot leak a
// handle into the next case.
struct Request {
  slopfab_request* handle = slopfab_request_create();
  ~Request() { slopfab_request_destroy(handle); }
};

struct OwnedString {
  char* text = nullptr;
  ~OwnedString() { slopfab_free_string(text); }
};

std::filesystem::path scratch_file(const char* name, const std::string& contents) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path, std::ios::binary);
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  return path;
}

}  // namespace

SLOPFAB_TEST(capi_version) {
  const uint32_t packed = slopfab_capi_version();
  CHECK((packed >> 24) == SLOPFAB_CAPI_VERSION_MAJOR);
  CHECK(((packed >> 12) & 0xFFF) == SLOPFAB_CAPI_VERSION_MINOR);
  CHECK((packed & 0xFFF) == SLOPFAB_CAPI_VERSION_PATCH);

  // The DLL's number must match the header this test compiled against. If it
  // does not, every other case here is testing something other than what it
  // claims to — and that is precisely the mismatch the version exists to catch
  // in a consumer.
  const char* text = slopfab_capi_version_string();
  CHECK(text != nullptr);
  const std::string expected = std::to_string(SLOPFAB_CAPI_VERSION_MAJOR) + "." +
                               std::to_string(SLOPFAB_CAPI_VERSION_MINOR) + "." +
                               std::to_string(SLOPFAB_CAPI_VERSION_PATCH);
  CHECK(text == expected);
}

SLOPFAB_TEST(capi_reused_models_can_be_cleared_while_idle) {
  CHECK(slopfab_reused_models_clear() == SLOPFAB_OK);
  CHECK(slopfab_reused_models_clear() == SLOPFAB_OK);
}

// Every entry point takes a null handle without crashing and says so. This is
// the failure a binding hits first — an uninitialised or already-freed pointer
// — and the answer has to be a status code rather than an access violation
// inside the DLL.
SLOPFAB_TEST(capi_rejects_null_handles) {
  CHECK(slopfab_request_set_prompt(nullptr, "x") == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_seed(nullptr, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_frames(nullptr, 5) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_still_image(nullptr, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_reuse_models(nullptr, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_model_path(nullptr, SLOPFAB_MODEL_TRANSFORMER, "x") ==
        SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_prompt_embedding_path(nullptr, "x") ==
        SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_resolve_plan(nullptr, nullptr) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_describe_plan(nullptr, nullptr) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_generation_start(nullptr, nullptr, nullptr, nullptr) ==
        SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_generation_status(nullptr) == SLOPFAB_ERR_INVALID_ARGUMENT);

  // A null request must be refused rather than dereferenced, even with a
  // perfectly good out-parameter to write into.
  slopfab_plan plan;
  CHECK(slopfab_resolve_plan(nullptr, &plan) == SLOPFAB_ERR_INVALID_ARGUMENT);

  // The accessors with no status to return answer with a neutral value.
  CHECK(std::strcmp(slopfab_generation_error(nullptr), "") == 0);

  // Destroying and freeing null are no-ops, so a binding's Drop impl needs no
  // guard.
  slopfab_request_destroy(nullptr);
  slopfab_generation_destroy(nullptr);
  slopfab_generation_cancel(nullptr);
  slopfab_free_string(nullptr);

  // A failure always leaves a message behind.
  CHECK(std::strlen(slopfab_last_error()) > 0);
}

// A freshly created request resolves to the same geometry the C++ defaults do
// — 16:9 at the trained area, 124 frames, 50 grid points. If these ever
// diverge, the DLL is quietly a different product from the CLI.
SLOPFAB_TEST(capi_default_request_matches_cpp_defaults) {
  Request request;
  CHECK(request.handle != nullptr);
  CHECK(slopfab_request_set_prompt(request.handle, "a test prompt") == SLOPFAB_OK);

  slopfab_plan plan;
  std::memset(&plan, 0, sizeof(plan));
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);

  CHECK(plan.canvas_width == 1344);
  CHECK(plan.canvas_height == 768);
  CHECK(plan.aligned_frames == 124);
  CHECK(plan.duration_seconds > 0.0);
  CHECK(plan.num_model_evaluations == 49);
  CHECK(plan.sequence_rows_without_text == 37710);
  CHECK(plan.latent_frames == 37);
  CHECK(plan.latent_height == 48);
  CHECK(plan.latent_width == 84);
  CHECK(plan.num_video_rows == 37296);
  CHECK(plan.num_audio_rows == 414);
}

SLOPFAB_TEST(capi_sampling_settings) {
  Request request;
  CHECK(slopfab_request_set_sampling_settings(nullptr, nullptr) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_sampling_settings(request.handle,
      R"({"version":1,"video_sigma_shift":6,"base_sigmas":[1,0.5,0]})") == SLOPFAB_OK);
  slopfab_plan plan{};
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.num_model_evaluations == 2);
  CHECK(slopfab_request_set_sampling_settings(request.handle,
      R"({"version":1,"video_sigma_shift":0})") == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.num_model_evaluations == 2);
  CHECK(slopfab_request_set_sampling_settings(request.handle, nullptr) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.num_model_evaluations == 49);
}

SLOPFAB_TEST(capi_session_ownership_and_conditioning_setter) {
  Request request;
  slopfab_session* session = nullptr;
  CHECK(slopfab_session_create(nullptr) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_session_create(&session) == SLOPFAB_OK);
  CHECK(slopfab_request_set_session(request.handle, session) == SLOPFAB_OK);
  CHECK(slopfab_session_clear(session) == SLOPFAB_OK);
  slopfab_session_destroy(session);
  CHECK(slopfab_request_set_conditioning_settings(request.handle,
      R"({"version":1,"max_frames":22})") == SLOPFAB_OK);
  slopfab_plan plan{};
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_ERR_INVALID_REQUEST);
  CHECK(slopfab_request_set_conditioning_settings(request.handle, "{}") == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_ERR_INVALID_REQUEST);
  CHECK(slopfab_request_set_conditioning_settings(request.handle, nullptr) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(slopfab_request_set_session(request.handle, nullptr) == SLOPFAB_OK);
}

SLOPFAB_TEST(capi_request_geometry) {
  Request request;

  CHECK(slopfab_request_set_aspect(request.handle, 0, 9) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_aspect(request.handle, 16, -9) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_frames(request.handle, 0) == SLOPFAB_ERR_INVALID_ARGUMENT);
  // The grid includes its terminal zero, so one point is not a schedule.
  CHECK(slopfab_request_set_steps(request.handle, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_resolution(request.handle, -640, 384) == SLOPFAB_ERR_INVALID_ARGUMENT);

  // An explicit canvas wins over the aspect...
  CHECK(slopfab_request_set_resolution(request.handle, 640, 384) == SLOPFAB_OK);
  slopfab_plan plan;
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.canvas_width == 640);
  CHECK(plan.canvas_height == 384);

  // ...and setting an aspect afterwards takes it back, rather than leaving a
  // canvas the caller thought they had replaced.
  CHECK(slopfab_request_set_aspect(request.handle, 16, 9) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.canvas_width == 1344);

  // Frames snap up to the next 17*k + 5 the video VAE can encode.
  CHECK(slopfab_request_set_frames(request.handle, 100) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.aligned_frames >= 100);
  CHECK((plan.aligned_frames - 5) % 17 == 0);

  // Steps drive the evaluation count, one fewer than the grid points.
  CHECK(slopfab_request_set_steps(request.handle, 8) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.num_model_evaluations == 7);
}

SLOPFAB_TEST(capi_still_image_plan) {
  Request request;
  CHECK(slopfab_request_set_prompt(request.handle, "a still life") == SLOPFAB_OK);
  CHECK(slopfab_request_set_aspect(request.handle, 1, 1) == SLOPFAB_OK);
  CHECK(slopfab_request_set_frames(request.handle, 1) == SLOPFAB_OK);
  CHECK(slopfab_request_set_still_image(request.handle, 1) == SLOPFAB_OK);

  slopfab_plan plan{};
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.aligned_frames == 1);
  CHECK(plan.latent_frames == 1);
  CHECK(plan.num_video_rows == 576);
  CHECK(plan.num_audio_latents == 0);
  CHECK(plan.num_audio_rows == 0);
  CHECK(plan.sequence_rows_without_text == 576);

  CHECK(slopfab_request_set_still_image(request.handle, 0) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_ERR_INVALID_REQUEST);
}

// A canvas off the 32-pixel grid is well-formed as arguments and impossible as
// a request, which is exactly the distinction the two error codes carry.
SLOPFAB_TEST(capi_rejects_unresolvable_requests) {
  Request request;
  CHECK(slopfab_request_set_resolution(request.handle, 1000, 700) == SLOPFAB_OK);

  slopfab_plan plan;
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_ERR_INVALID_REQUEST);
  CHECK(std::strlen(slopfab_last_error()) > 0);

  // A ratio outside 1:4..4:1 is the other kind.
  CHECK(slopfab_request_set_aspect(request.handle, 100, 1) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_ERR_INVALID_REQUEST);
}

SLOPFAB_TEST(capi_model_paths_and_attention) {
  Request request;

  CHECK(slopfab_request_set_model_path(request.handle, SLOPFAB_MODEL_TRANSFORMER, "t.st") ==
        SLOPFAB_OK);
  CHECK(slopfab_request_set_model_path(request.handle, SLOPFAB_MODEL_AUDIO_VAE, "a.st") == SLOPFAB_OK);
  // An unknown id is rejected rather than landing on whichever member happens
  // to be next in the struct.
  CHECK(slopfab_request_set_model_path(request.handle, 42, "x") == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_model_path(request.handle, SLOPFAB_MODEL_TOKENIZER, nullptr) ==
        SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_prompt_embedding_path(request.handle, "prompt.st") ==
        SLOPFAB_OK);
  CHECK(slopfab_request_set_prompt_embedding_path(request.handle, nullptr) ==
        SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_reuse_models(request.handle, 1) == SLOPFAB_OK);
  CHECK(slopfab_request_set_reuse_models(request.handle, 0) == SLOPFAB_OK);

  for (const char* mode : {"none", "flash2", "sage2", "sol", "sol-experimental", "exact"}) {
    CHECK(slopfab_request_set_attention(request.handle, mode) == SLOPFAB_OK);
  }
  CHECK(slopfab_request_set_attention(request.handle, "flash3") == SLOPFAB_ERR_INVALID_ARGUMENT);
  // The message has to name what was wrong, since there is no enum to consult.
  CHECK(std::string(slopfab_last_error()).find("flash3") != std::string::npos);

  CHECK(slopfab_request_set_inference_backend(
            request.handle, SLOPFAB_INFERENCE_CUDA) == SLOPFAB_OK);
  CHECK(slopfab_request_set_inference_backend(
            request.handle, SLOPFAB_INFERENCE_VULKAN) == SLOPFAB_OK);
  CHECK(slopfab_request_set_attention(request.handle, "exact") == SLOPFAB_OK);
  CHECK(slopfab_request_set_inference_backend(request.handle, 42) ==
        SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(std::string(slopfab_last_error()).find("42") != std::string::npos);

  CHECK(slopfab_request_set_synthetic_latents(request.handle, 1) == SLOPFAB_OK);
  CHECK(slopfab_request_set_verbose(request.handle, 0) == SLOPFAB_OK);
}

SLOPFAB_TEST(capi_reference_image_limit) {
  Request request;
  for (int i = 0; i < 9; ++i) {
    CHECK(slopfab_request_add_reference_image(request.handle, "ref.png") == SLOPFAB_OK);
  }
  // Refused at the setter, so the caller learns which call was the tenth
  // rather than finding out when a checkpoint is already open.
  CHECK(slopfab_request_add_reference_image(request.handle, "ref.png") ==
        SLOPFAB_ERR_INVALID_REQUEST);
  CHECK(slopfab_request_add_reference_image(request.handle, nullptr) ==
        SLOPFAB_ERR_INVALID_ARGUMENT);
}

// The same normalisation the CLI applies, so a prompt file that works there
// conditions identically here.
SLOPFAB_TEST(capi_prompt_file) {
  Request request;
  CHECK(slopfab_request_set_prompt_file(request.handle, "definitely-not-here.txt") ==
        SLOPFAB_ERR_NOT_FOUND);

  const std::filesystem::path good =
      scratch_file("slopfab_capi_prompt.txt", "\xEF\xBB\xBF  a lit room\r\nwith rain\r\n  ");
  CHECK(slopfab_request_set_prompt_file(request.handle, good.string().c_str()) == SLOPFAB_OK);

  // `describe_plan` reports the prompt's length rather than its text, which
  // makes the count the assertion: "a lit room\nwith rain" is 20 characters
  // once the BOM, both CRs and the surrounding blank space are gone. The
  // interior newline stays, because it is part of the prompt. Any of those
  // four rules breaking moves this number.
  OwnedString described;
  CHECK(slopfab_describe_plan(request.handle, &described.text) == SLOPFAB_OK);
  const std::string text = described.text;
  CHECK(text.find("20 characters") != std::string::npos);

  const std::filesystem::path blank = scratch_file("slopfab_capi_blank.txt", "  \r\n\t ");
  CHECK(slopfab_request_set_prompt_file(request.handle, blank.string().c_str()) ==
        SLOPFAB_ERR_INVALID_REQUEST);

  std::filesystem::remove(good);
  std::filesystem::remove(blank);
}

SLOPFAB_TEST(capi_describe_plan) {
  Request request;
  CHECK(slopfab_request_set_prompt(request.handle, "a describable thing") == SLOPFAB_OK);

  OwnedString described;
  CHECK(slopfab_describe_plan(request.handle, &described.text) == SLOPFAB_OK);
  CHECK(described.text != nullptr);
  CHECK(std::strlen(described.text) > 0);

  // An unresolvable request leaves the out-parameter null rather than a
  // pointer the caller would have to know not to free.
  CHECK(slopfab_request_set_resolution(request.handle, 1000, 700) == SLOPFAB_OK);
  char* text = reinterpret_cast<char*>(0x1);
  CHECK(slopfab_describe_plan(request.handle, &text) == SLOPFAB_ERR_INVALID_REQUEST);
  CHECK(text == nullptr);
}

// Only the path that refuses before any work starts. Starting a real
// generation would begin loading tens of gigabytes from inside a unit test.
SLOPFAB_TEST(capi_generation_start_validates_first) {
  Request request;
  // No transformer, but that is not what is being checked: an unsatisfiable
  // *geometry* must be reported synchronously, as a return code, rather than
  // as a handle that fails a millisecond later on a worker thread.
  CHECK(slopfab_request_set_resolution(request.handle, 1000, 700) == SLOPFAB_OK);

  slopfab_generation* generation = reinterpret_cast<slopfab_generation*>(0x1);
  CHECK(slopfab_generation_start(request.handle, nullptr, nullptr, &generation) ==
        SLOPFAB_ERR_INVALID_REQUEST);
  CHECK(generation == nullptr);

  // A refused start must not have claimed the process-wide generation slot,
  // or every later run would come back SLOPFAB_ERR_BUSY forever.
  CHECK(slopfab_generation_start(request.handle, nullptr, nullptr, &generation) ==
        SLOPFAB_ERR_INVALID_REQUEST);
}

// Its own, rather than tests/test_main.cpp: that file registers C++ cases that
// would pull slopfab_core into a binary whose whole point is linking nothing
// but the DLL.
SLOPFAB_TEST(capi_reference_video_audio_ingestion) {
  Request request;
  slopfab_reference_video* video = reinterpret_cast<slopfab_reference_video*>(1);
  CHECK(slopfab_reference_video_create(1, &video) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(video == nullptr);
  CHECK(slopfab_reference_video_create(2, nullptr) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_reference_video_create(2, &video) == SLOPFAB_OK);
  CHECK(slopfab_request_add_reference_video(request.handle, video) == SLOPFAB_ERR_INVALID_ARGUMENT);
  uint8_t rgba[] = {10, 20, 30, 255};
  CHECK(slopfab_reference_video_append_rgba8(video, rgba, sizeof(rgba), 1, 1, 4, 0) == SLOPFAB_OK);
  CHECK(slopfab_reference_video_append_rgb24(video, rgba, 2, 1, 1, 3, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_reference_video_append_rgb24(video, rgba, 3, 1, 1, 3, 1) == SLOPFAB_OK);
  CHECK(slopfab_reference_video_append_rgb24(video, rgba, 3, 1, 1, 3, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_reference_video_append_rgb24(nullptr, rgba, 3, 1, 1, 3, 0) == SLOPFAB_ERR_INVALID_ARGUMENT);
  std::vector<float> pcm(64000 * 2, .25f);
  CHECK(slopfab_reference_video_set_audio_f32(video, pcm.data(), pcm.size(), 2, 32000, 0) == SLOPFAB_OK);
  CHECK(slopfab_reference_video_set_audio_f32(video, pcm.data(), pcm.size(), 2, 32000, 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_add_reference_video(request.handle, video) == SLOPFAB_OK);
  slopfab_reference_video_destroy(video);
  slopfab_reference_video_destroy(nullptr);
  rgba[0] = 99;
  CHECK(slopfab_request_add_reference_audio_f32(request.handle, pcm.data(), pcm.size(), 2, 32000) == SLOPFAB_OK);
  pcm.assign(pcm.size(), std::numeric_limits<float>::quiet_NaN());
  CHECK(slopfab_request_add_reference_audio_f32(request.handle, pcm.data(), pcm.size(), 2, 32000) == SLOPFAB_ERR_INVALID_ARGUMENT);
  slopfab_plan plan{};
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  OwnedString description;
  CHECK(slopfab_describe_plan(request.handle, &description.text) == SLOPFAB_OK);
  CHECK(std::strstr(description.text, "reference videos    1 (1 with audio)") != nullptr);
  CHECK(std::strstr(description.text, "reference audios    1") != nullptr);
  slopfab_generation* generation = reinterpret_cast<slopfab_generation*>(1);
  CHECK(slopfab_request_set_inference_backend(request.handle, SLOPFAB_INFERENCE_VULKAN) == SLOPFAB_OK);
  CHECK(slopfab_generation_start(request.handle, nullptr, nullptr, &generation) == SLOPFAB_OK);
  CHECK(generation != nullptr);
  // The request is accepted without loading models synchronously. This fixture
  // intentionally has no checkpoints; the worker reports that failure.
  CHECK(slopfab_generation_wait(generation, -1) != SLOPFAB_OK);
  slopfab_generation_destroy(generation);
}

int main() { return slopfab::test::run_all(); }

SLOPFAB_TEST(capi_lora_and_taomate_schedule) {
  Request request;
  CHECK(slopfab_request_add_lora(nullptr, "lora", 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_add_lora(request.handle, "", 1) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_add_lora(request.handle, "lora", std::numeric_limits<float>::infinity()) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_schedule(request.handle, 99) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_prompt(request.handle, "a video") == SLOPFAB_OK);
  CHECK(slopfab_request_set_schedule(request.handle, SLOPFAB_SCHEDULE_TAOMATE_3STEP) == SLOPFAB_OK);
  slopfab_plan plan{};
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_ERR_INVALID_REQUEST);
  CHECK(slopfab_request_add_lora(request.handle, "TaoMate.safetensors", 1) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.num_model_evaluations == 3);
  CHECK(slopfab_request_clear_loras(request.handle) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_ERR_INVALID_REQUEST);
  CHECK(slopfab_request_set_schedule(request.handle, SLOPFAB_SCHEDULE_DEFAULT) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.num_model_evaluations == 49);
}

SLOPFAB_TEST(capi_animate_plan_and_audio_mode) {
  Request request;
  CHECK(slopfab_request_set_animate(nullptr, 1, 0) == SLOPFAB_ERR_INVALID_ARGUMENT);
  CHECK(slopfab_request_set_animate(request.handle, 1, 1) == SLOPFAB_OK);
  CHECK(slopfab_request_set_frames(request.handle, 39) == SLOPFAB_OK);
  CHECK(slopfab_request_set_prompt_embedding_path(request.handle, "fixed.safetensors") == SLOPFAB_OK);
  slopfab_plan plan{};
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_ERR_INVALID_REQUEST);
  slopfab_reference_video* video = nullptr;
  CHECK(slopfab_reference_video_create(2, &video) == SLOPFAB_OK);
  std::vector<uint8_t> rgb(64 * 96 * 3, 127);
  CHECK(slopfab_reference_video_append_rgb24(video, rgb.data(), rgb.size(), 64, 96, 64 * 3, 0) == SLOPFAB_OK);
  std::vector<float> pcm(64000, .25f);
  CHECK(slopfab_reference_video_set_audio_f32(video, pcm.data(), pcm.size(), 1, 32000, 0) == SLOPFAB_OK);
  CHECK(slopfab_request_add_reference_video(request.handle, video) == SLOPFAB_OK);
  slopfab_reference_video_destroy(video);
  CHECK(slopfab_request_add_reference_image(request.handle, "repainted.png") == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.canvas_width == 64 && plan.canvas_height == 96);
  CHECK(plan.num_model_evaluations == 3);
  CHECK(plan.num_audio_rows == 130);
  OwnedString description;
  CHECK(slopfab_describe_plan(request.handle, &description.text) == SLOPFAB_OK);
  CHECK(std::strstr(description.text, "fixed 362-token embedding") != nullptr);
  CHECK(std::strstr(description.text, "pinned driving soundtrack") != nullptr);
  CHECK(std::strstr(description.text, "shift 3)") != nullptr);
  CHECK(slopfab_request_set_steps(request.handle, 8) == SLOPFAB_OK);
  CHECK(slopfab_request_set_animate(request.handle, 1, 1) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.num_model_evaluations == 7);
  CHECK(slopfab_request_set_animate(request.handle, 0, 0) == SLOPFAB_OK);
  CHECK(slopfab_resolve_plan(request.handle, &plan) == SLOPFAB_OK);
  CHECK(plan.canvas_width == 1344 && plan.canvas_height == 768);
}
