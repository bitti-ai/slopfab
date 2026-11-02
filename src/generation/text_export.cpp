#include "slopfab/text/export.h"
#include "slopfab/text/prompt_embedding.h"
#include "slopfab/pipeline.h"
#include "slopfab/dit/packing.h"
#include "helpers.h"
#include "clip_student.h"
#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/text_encoder.h"
#include "slopfab/vulkan/tensor.h"
#endif

namespace slopfab::text {

int export_prompt_embedding(const TextExportRequest& request, const std::string& output) {
  if (request.prompt.empty() || request.text_encoder_path.empty() || output.empty())
    throw std::invalid_argument("text export: prompt, text encoder and output are required");
  if (request.backend != DeviceBackend::kCuda && request.backend != DeviceBackend::kVulkan)
    throw std::invalid_argument("text export: invalid backend");
#if !SLOPFAB_WITH_CUDA
  if (request.backend == DeviceBackend::kCuda)
    throw std::runtime_error("text export: CUDA backend is not compiled in");
#endif
#if !SLOPFAB_WITH_VULKAN
  if (request.backend == DeviceBackend::kVulkan)
    throw std::runtime_error("text export: Vulkan backend is not compiled in");
#endif
  if (!request.clip_projection_path.empty()) {
    if (request.residency == Residency::kResident ||
        request.arithmetic == EncoderArithmetic::kExact)
      throw std::invalid_argument(
          "ClipProj uses streaming student layers and shipped BF16 arithmetic; resident/exact modes are unsupported");
    if (request.backend == DeviceBackend::kVulkan && !request.vulkan_portable_arithmetic)
      throw std::invalid_argument("ClipProj Vulkan requires --vulkan-arithmetic portable");
  }
  Tokenizer tokenizer;
  if (request.tokenizer_path.empty())
    tokenizer.load_embedded();
  else
    tokenizer.load(request.tokenizer_path);
  tokenizer.enable_h3_special_tokens();
  const auto ids = tokenizer.encode(request.prompt);
  if (ids.empty() || ids.size() > kMaxPromptTokens)
    throw std::invalid_argument("text export: prompt token count is outside encoder capacity");

  SafeTensors checkpoint;
  checkpoint.open(request.text_encoder_path);
  EncoderConfig config;
  config.residency = request.residency;
  config.arithmetic = request.arithmetic;
  config.max_prompt_tokens = static_cast<int>(ids.size());
  const auto descriptor = request.clip_projection_path.empty()
                              ? resolve_conditioner_descriptor(checkpoint, config)
                              : ConditionerDescriptor{};
  PromptEmbedding embedding;
  if (!request.clip_projection_path.empty()) {
    ClipProjection projection(request.clip_projection_path);
    (void)validate_clip_student(checkpoint, projection);
    PromptEmbedding hidden;
    if (request.backend == DeviceBackend::kCuda) {
#if SLOPFAB_WITH_CUDA
      hidden = encode_clip_student_cuda(request.text_encoder_path, projection, ids, {});
#endif
    } else {
#if SLOPFAB_WITH_VULKAN
      hidden = encode_clip_student_vulkan(request.text_encoder_path, projection, ids, {});
#endif
    }
    embedding = projection.apply(hidden);
  } else if (request.backend == DeviceBackend::kCuda) {
#if SLOPFAB_WITH_CUDA
    Encoder encoder;
    encoder.load(checkpoint, config);
    embedding = encoder.encode(ids);
#endif
  } else {
#if SLOPFAB_WITH_VULKAN
    auto device =
        generation::create_vulkan_inference_device(request.vulkan_portable_arithmetic, true);
    vulkan::TensorContextOptions options;
    options.max_batch_operators = 64;
    vulkan::TensorContext context(device, options);
    auto encoder = vulkan::ExactQwenTextEncoder::create(context);
    encoder.load(checkpoint);
    embedding = encoder.encode(ids);
#endif
  }
  // Text-only archives always have explicit text tags, even if an encoder
  // implementation omitted the optional host vector.
  embedding.modality_tags.assign(embedding.num_tokens, dit::kTagText);
  std::string encoder_identity;
  append_file_identity(encoder_identity, request.text_encoder_path);
  std::map<std::string, std::string> metadata = {
      {"prompt", request.prompt},
      {"conditioner", descriptor.fingerprint()},
      {"text_encoder", request.text_encoder_path},
      {"text_encoder_identity", encoder_identity},
      {"tokenizer", request.tokenizer_path.empty() ? "embedded" : request.tokenizer_path},
      {"tokenizer_sha256", tokenizer.source_sha256()},
      {"tokenizer_profile", "minimax-h3-special-v1"},
      {"backend", request.backend == DeviceBackend::kCuda ? "cuda" : "vulkan"},
      {"arithmetic",
       request.backend == DeviceBackend::kVulkan || request.arithmetic == EncoderArithmetic::kExact
           ? "exact"
           : "shipped"}};
  if (!request.clip_projection_path.empty()) {
    metadata["clip_projection"] = request.clip_projection_path;
    metadata["conditioner"] = "clipproj-qwen3-vl-v1";
    metadata["arithmetic"] = "bf16-student-host-projection";
    append_file_identity(metadata["clip_projection_identity"], request.clip_projection_path);
  }
  write_prompt_embedding(output, embedding, metadata);
  return embedding.num_tokens;
}

} // namespace slopfab::text
