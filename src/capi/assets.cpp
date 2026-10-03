#include "internal.h"
#include "slopfab/text/export.h"

SLOPFAB_C_API int SLOPFAB_CALL slopfab_export_prompt_embedding(const slopfab_request* request,
                                                               const char* output_path) {
  if (!request || !output_path || !*output_path)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "text export: request and output path are required");
  return guarded([&] {
    if (request->request.prompt.empty() || request->request.text_encoder_path.empty())
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                  "text export: prompt and text encoder are required");
    bool expected = false;
    if (!g_generation_active.compare_exchange_strong(expected, true))
      return fail(SLOPFAB_ERR_BUSY, "generation or text export is already running");
    struct Claim {
      ~Claim() {
        g_generation_active.store(false);
      }
    } claim;
    slopfab::text::TextExportRequest options;
    options.prompt = request->request.prompt;
    options.text_encoder_path = request->request.text_encoder_path;
    options.tokenizer_path = request->request.tokenizer_path;
    options.backend = request->options.inference_backend;
    options.vulkan_portable_arithmetic = request->options.vulkan_portable_arithmetic;
    if (request->options.attention_mode == slopfab::AttentionMode::kExact)
      options.arithmetic = slopfab::text::EncoderArithmetic::kExact;
    slopfab::text::export_prompt_embedding(options, output_path);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_save_refmod_bundle(const char* const* paths,
                                                          int32_t path_count,
                                                          const char* output_path, const char* name,
                                                          const char* description) {
  if (!paths || path_count < 1 || path_count > 256 || !output_path || !*output_path)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "refmod bundle: paths and output are required (1..256 inputs)");
  return guarded([&] {
    for (int32_t i = 0; i < path_count; ++i)
      if (!paths[i] || !*paths[i])
        return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "refmod bundle: empty input path");
    slopfab::RefModBundle bundle;
    for (int32_t i = 0; i < path_count; ++i) {
      const auto input = slopfab::RefModBundle::load(paths[i]);
      if (i == 0)
        bundle.metadata = input.metadata;
      if (bundle.members.size() + input.members.size() > 256)
        return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "refmod bundle: maximum 256 members");
      bundle.members.insert(bundle.members.end(), input.members.begin(), input.members.end());
    }
    if (name)
      bundle.metadata["name"] = slopfab::json::Value(std::string(name));
    if (description)
      bundle.metadata["description"] = slopfab::json::Value(std::string(description));
    bundle.save(output_path);
    return SLOPFAB_OK;
  });
}
