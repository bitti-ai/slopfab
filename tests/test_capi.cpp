// The C ABI, exercised through the DLL.
//
// This binary links vidfab_c and nothing else, which is the point: it is the
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
#include "vidfab/capi.h"

namespace {

// The C API has no RAII of its own — that is what makes it a C API — so the
// tests bring their own, and a CHECK failure part-way through cannot leak a
// handle into the next case.
struct Request {
  vidfab_request* handle = vidfab_request_create();
  ~Request() { vidfab_request_destroy(handle); }
};

struct OwnedString {
  char* text = nullptr;
  ~OwnedString() { vidfab_free_string(text); }
};

std::filesystem::path scratch_file(const char* name, const std::string& contents) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path, std::ios::binary);
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  return path;
}

}  // namespace

VIDFAB_TEST(capi_version) {
  const uint32_t packed = vidfab_capi_version();
  CHECK((packed >> 24) == VIDFAB_CAPI_VERSION_MAJOR);
  CHECK(((packed >> 12) & 0xFFF) == VIDFAB_CAPI_VERSION_MINOR);
  CHECK((packed & 0xFFF) == VIDFAB_CAPI_VERSION_PATCH);

  // The DLL's number must match the header this test compiled against. If it
  // does not, every other case here is testing something other than what it
  // claims to — and that is precisely the mismatch the version exists to catch
  // in a consumer.
  const char* text = vidfab_capi_version_string();
  CHECK(text != nullptr);
  const std::string expected = std::to_string(VIDFAB_CAPI_VERSION_MAJOR) + "." +
                               std::to_string(VIDFAB_CAPI_VERSION_MINOR) + "." +
                               std::to_string(VIDFAB_CAPI_VERSION_PATCH);
  CHECK(text == expected);
}

// Every entry point takes a null handle without crashing and says so. This is
// the failure a binding hits first — an uninitialised or already-freed pointer
// — and the answer has to be a status code rather than an access violation
// inside the DLL.
VIDFAB_TEST(capi_rejects_null_handles) {
  CHECK(vidfab_request_set_prompt(nullptr, "x") == VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_request_set_seed(nullptr, 1) == VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_request_set_frames(nullptr, 5) == VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_request_set_model_path(nullptr, VIDFAB_MODEL_TRANSFORMER, "x") ==
        VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_resolve_plan(nullptr, nullptr) == VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_describe_plan(nullptr, nullptr) == VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_generation_start(nullptr, nullptr, nullptr, nullptr) ==
        VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_generation_status(nullptr) == VIDFAB_ERR_INVALID_ARGUMENT);

  // A null request must be refused rather than dereferenced, even with a
  // perfectly good out-parameter to write into.
  vidfab_plan plan;
  CHECK(vidfab_resolve_plan(nullptr, &plan) == VIDFAB_ERR_INVALID_ARGUMENT);

  // The accessors with no status to return answer with a neutral value.
  CHECK(std::strcmp(vidfab_generation_error(nullptr), "") == 0);

  // Destroying and freeing null are no-ops, so a binding's Drop impl needs no
  // guard.
  vidfab_request_destroy(nullptr);
  vidfab_generation_destroy(nullptr);
  vidfab_generation_cancel(nullptr);
  vidfab_free_string(nullptr);

  // A failure always leaves a message behind.
  CHECK(std::strlen(vidfab_last_error()) > 0);
}

// A freshly created request resolves to the same geometry the C++ defaults do
// — 16:9 at the trained area, 124 frames, 50 grid points. If these ever
// diverge, the DLL is quietly a different product from the CLI.
VIDFAB_TEST(capi_default_request_matches_cpp_defaults) {
  Request request;
  CHECK(request.handle != nullptr);
  CHECK(vidfab_request_set_prompt(request.handle, "a test prompt") == VIDFAB_OK);

  vidfab_plan plan;
  std::memset(&plan, 0, sizeof(plan));
  CHECK(vidfab_resolve_plan(request.handle, &plan) == VIDFAB_OK);

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

VIDFAB_TEST(capi_request_geometry) {
  Request request;

  CHECK(vidfab_request_set_aspect(request.handle, 0, 9) == VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_request_set_aspect(request.handle, 16, -9) == VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_request_set_frames(request.handle, 0) == VIDFAB_ERR_INVALID_ARGUMENT);
  // The grid includes its terminal zero, so one point is not a schedule.
  CHECK(vidfab_request_set_steps(request.handle, 1) == VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_request_set_resolution(request.handle, -640, 384) == VIDFAB_ERR_INVALID_ARGUMENT);

  // An explicit canvas wins over the aspect...
  CHECK(vidfab_request_set_resolution(request.handle, 640, 384) == VIDFAB_OK);
  vidfab_plan plan;
  CHECK(vidfab_resolve_plan(request.handle, &plan) == VIDFAB_OK);
  CHECK(plan.canvas_width == 640);
  CHECK(plan.canvas_height == 384);

  // ...and setting an aspect afterwards takes it back, rather than leaving a
  // canvas the caller thought they had replaced.
  CHECK(vidfab_request_set_aspect(request.handle, 16, 9) == VIDFAB_OK);
  CHECK(vidfab_resolve_plan(request.handle, &plan) == VIDFAB_OK);
  CHECK(plan.canvas_width == 1344);

  // Frames snap up to the next 17*k + 5 the video VAE can encode.
  CHECK(vidfab_request_set_frames(request.handle, 100) == VIDFAB_OK);
  CHECK(vidfab_resolve_plan(request.handle, &plan) == VIDFAB_OK);
  CHECK(plan.aligned_frames >= 100);
  CHECK((plan.aligned_frames - 5) % 17 == 0);

  // Steps drive the evaluation count, one fewer than the grid points.
  CHECK(vidfab_request_set_steps(request.handle, 8) == VIDFAB_OK);
  CHECK(vidfab_resolve_plan(request.handle, &plan) == VIDFAB_OK);
  CHECK(plan.num_model_evaluations == 7);
}

// A canvas off the 32-pixel grid is well-formed as arguments and impossible as
// a request, which is exactly the distinction the two error codes carry.
VIDFAB_TEST(capi_rejects_unresolvable_requests) {
  Request request;
  CHECK(vidfab_request_set_resolution(request.handle, 1000, 700) == VIDFAB_OK);

  vidfab_plan plan;
  CHECK(vidfab_resolve_plan(request.handle, &plan) == VIDFAB_ERR_INVALID_REQUEST);
  CHECK(std::strlen(vidfab_last_error()) > 0);

  // A ratio outside 1:4..4:1 is the other kind.
  CHECK(vidfab_request_set_aspect(request.handle, 100, 1) == VIDFAB_OK);
  CHECK(vidfab_resolve_plan(request.handle, &plan) == VIDFAB_ERR_INVALID_REQUEST);
}

VIDFAB_TEST(capi_model_paths_and_attention) {
  Request request;

  CHECK(vidfab_request_set_model_path(request.handle, VIDFAB_MODEL_TRANSFORMER, "t.st") ==
        VIDFAB_OK);
  CHECK(vidfab_request_set_model_path(request.handle, VIDFAB_MODEL_AUDIO_VAE, "a.st") == VIDFAB_OK);
  // An unknown id is rejected rather than landing on whichever member happens
  // to be next in the struct.
  CHECK(vidfab_request_set_model_path(request.handle, 42, "x") == VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(vidfab_request_set_model_path(request.handle, VIDFAB_MODEL_TOKENIZER, nullptr) ==
        VIDFAB_ERR_INVALID_ARGUMENT);

  for (const char* mode : {"none", "flash2", "sage2", "sol", "sol-experimental", "exact"}) {
    CHECK(vidfab_request_set_attention(request.handle, mode) == VIDFAB_OK);
  }
  CHECK(vidfab_request_set_attention(request.handle, "flash3") == VIDFAB_ERR_INVALID_ARGUMENT);
  // The message has to name what was wrong, since there is no enum to consult.
  CHECK(std::string(vidfab_last_error()).find("flash3") != std::string::npos);

  CHECK(vidfab_request_set_inference_backend(
            request.handle, VIDFAB_INFERENCE_CUDA) == VIDFAB_OK);
  CHECK(vidfab_request_set_inference_backend(
            request.handle, VIDFAB_INFERENCE_VULKAN) == VIDFAB_OK);
  CHECK(vidfab_request_set_inference_backend(request.handle, 42) ==
        VIDFAB_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vidfab_last_error()).find("42") != std::string::npos);

  CHECK(vidfab_request_set_synthetic_latents(request.handle, 1) == VIDFAB_OK);
  CHECK(vidfab_request_set_verbose(request.handle, 0) == VIDFAB_OK);
}

VIDFAB_TEST(capi_reference_image_limit) {
  Request request;
  for (int i = 0; i < 9; ++i) {
    CHECK(vidfab_request_add_reference_image(request.handle, "ref.png") == VIDFAB_OK);
  }
  // Refused at the setter, so the caller learns which call was the tenth
  // rather than finding out when a checkpoint is already open.
  CHECK(vidfab_request_add_reference_image(request.handle, "ref.png") ==
        VIDFAB_ERR_INVALID_REQUEST);
  CHECK(vidfab_request_add_reference_image(request.handle, nullptr) ==
        VIDFAB_ERR_INVALID_ARGUMENT);
}

// The same normalisation the CLI applies, so a prompt file that works there
// conditions identically here.
VIDFAB_TEST(capi_prompt_file) {
  Request request;
  CHECK(vidfab_request_set_prompt_file(request.handle, "definitely-not-here.txt") ==
        VIDFAB_ERR_NOT_FOUND);

  const std::filesystem::path good =
      scratch_file("vidfab_capi_prompt.txt", "\xEF\xBB\xBF  a lit room\r\nwith rain\r\n  ");
  CHECK(vidfab_request_set_prompt_file(request.handle, good.string().c_str()) == VIDFAB_OK);

  // `describe_plan` reports the prompt's length rather than its text, which
  // makes the count the assertion: "a lit room\nwith rain" is 20 characters
  // once the BOM, both CRs and the surrounding blank space are gone. The
  // interior newline stays, because it is part of the prompt. Any of those
  // four rules breaking moves this number.
  OwnedString described;
  CHECK(vidfab_describe_plan(request.handle, &described.text) == VIDFAB_OK);
  const std::string text = described.text;
  CHECK(text.find("20 characters") != std::string::npos);

  const std::filesystem::path blank = scratch_file("vidfab_capi_blank.txt", "  \r\n\t ");
  CHECK(vidfab_request_set_prompt_file(request.handle, blank.string().c_str()) ==
        VIDFAB_ERR_INVALID_REQUEST);

  std::filesystem::remove(good);
  std::filesystem::remove(blank);
}

VIDFAB_TEST(capi_describe_plan) {
  Request request;
  CHECK(vidfab_request_set_prompt(request.handle, "a describable thing") == VIDFAB_OK);

  OwnedString described;
  CHECK(vidfab_describe_plan(request.handle, &described.text) == VIDFAB_OK);
  CHECK(described.text != nullptr);
  CHECK(std::strlen(described.text) > 0);

  // An unresolvable request leaves the out-parameter null rather than a
  // pointer the caller would have to know not to free.
  CHECK(vidfab_request_set_resolution(request.handle, 1000, 700) == VIDFAB_OK);
  char* text = reinterpret_cast<char*>(0x1);
  CHECK(vidfab_describe_plan(request.handle, &text) == VIDFAB_ERR_INVALID_REQUEST);
  CHECK(text == nullptr);
}

// Only the path that refuses before any work starts. Starting a real
// generation would begin loading tens of gigabytes from inside a unit test.
VIDFAB_TEST(capi_generation_start_validates_first) {
  Request request;
  // No transformer, but that is not what is being checked: an unsatisfiable
  // *geometry* must be reported synchronously, as a return code, rather than
  // as a handle that fails a millisecond later on a worker thread.
  CHECK(vidfab_request_set_resolution(request.handle, 1000, 700) == VIDFAB_OK);

  vidfab_generation* generation = reinterpret_cast<vidfab_generation*>(0x1);
  CHECK(vidfab_generation_start(request.handle, nullptr, nullptr, &generation) ==
        VIDFAB_ERR_INVALID_REQUEST);
  CHECK(generation == nullptr);

  // A refused start must not have claimed the process-wide generation slot,
  // or every later run would come back VIDFAB_ERR_BUSY forever.
  CHECK(vidfab_generation_start(request.handle, nullptr, nullptr, &generation) ==
        VIDFAB_ERR_INVALID_REQUEST);
}

// Its own, rather than tests/test_main.cpp: that file registers C++ cases that
// would pull vidfab_core into a binary whose whole point is linking nothing
// but the DLL.
int main() { return vidfab::test::run_all(); }
