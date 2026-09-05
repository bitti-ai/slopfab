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

#include "harness.h"
#include "slopfab/capi.h"

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
int main() { return slopfab::test::run_all(); }
