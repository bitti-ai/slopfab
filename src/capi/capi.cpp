#include "internal.h"
#include "slopfab/lora.h"
extern "C" {

SLOPFAB_C_API int SLOPFAB_CALL slopfab_prepare_lora_grid(const char* adapter_path, int32_t width,
                                                         int32_t allow_download) {
  if (!adapter_path || !*adapter_path || width <= 0)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "prepare_lora_grid: nonempty adapter path and positive width required");
  return guarded([&] {
    slopfab::prepare_lora_grid(adapter_path, width, allow_download != 0);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_session_create(slopfab_session** out_session) {
  if (!out_session)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "session_create: null output");
  *out_session = nullptr;
  return guarded([&] {
    auto handle = std::make_unique<slopfab_session>();
    handle->value = std::make_shared<slopfab::GenerationSession>();
    *out_session = handle.release();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API void SLOPFAB_CALL slopfab_session_destroy(slopfab_session* session) {
  delete session;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_session_clear(slopfab_session* session) {
  if (!session)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "session_clear: null session");
  return guarded([&] {
    session->value->clear();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_session(slopfab_request* request,
                                                           const slopfab_session* session) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "set_session: null request");
  return guarded([&] {
    request->session = session ? session->value : nullptr;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API uint32_t SLOPFAB_CALL slopfab_capi_version(void) {
  return (static_cast<uint32_t>(SLOPFAB_CAPI_VERSION_MAJOR) << 24) |
         (static_cast<uint32_t>(SLOPFAB_CAPI_VERSION_MINOR) << 12) |
         static_cast<uint32_t>(SLOPFAB_CAPI_VERSION_PATCH);
}

// Built from the same macros rather than written out, so the string and the
// packed number cannot disagree after someone bumps one of them.
#define SLOPFAB_CAPI_STRINGIFY_(x) #x
#define SLOPFAB_CAPI_STRINGIFY(x) SLOPFAB_CAPI_STRINGIFY_(x)

SLOPFAB_C_API const char* SLOPFAB_CALL slopfab_capi_version_string(void) {
  return SLOPFAB_CAPI_STRINGIFY(SLOPFAB_CAPI_VERSION_MAJOR) "." SLOPFAB_CAPI_STRINGIFY(
      SLOPFAB_CAPI_VERSION_MINOR) "." SLOPFAB_CAPI_STRINGIFY(SLOPFAB_CAPI_VERSION_PATCH);
}

SLOPFAB_C_API const char* SLOPFAB_CALL slopfab_last_error(void) {
  return g_last_error.c_str();
}

SLOPFAB_C_API void SLOPFAB_CALL slopfab_free_string(char* text) {
  std::free(text);
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_cuda_set_version(const char* version) {
  if (version == nullptr)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_cuda_set_version: null version");
  if (std::strcmp(version, "auto") != 0 && std::strcmp(version, "13") != 0 &&
      std::strcmp(version, "12") != 0)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_cuda_set_version: expected auto, 13, or 12");
  return guarded([&] {
    slopfab::cuda::set_cublas_version_request(version);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_cuda_loaded_major(int32_t* out_major) {
  if (out_major == nullptr)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_cuda_loaded_major: null out_major");
  *out_major = 0;
  return guarded([&] {
    *out_major = slopfab::cuda::cublas_loaded_major();
    return SLOPFAB_OK;
  });
}
}
