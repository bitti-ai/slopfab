#include "slopfab/text/export.h"
#include "slopfab/text/prompt_embedding.h"
#include "slopfab/pipeline.h"
#include "slopfab/dit/packing.h"
#include "helpers.h"
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
  Tokenizer tokenizer;
  if (request.tokenizer_path.empty())
    tokenizer.load_embedded();
  else
    tokenizer.load(request.tokenizer_path);
  const auto ids = tokenizer.encode(request.prompt);
  if (ids.empty() || ids.size() > kMaxPromptTokens)
    throw std::invalid_argument("text export: prompt token count is outside encoder capacity");

  SafeTensors checkpoint;
  checkpoint.open(request.text_encoder_path);
  EncoderConfig config;
  config.residency = request.residency;
  config.arithmetic = request.arithmetic;
  config.max_prompt_tokens = static_cast<int>(ids.size());
  const auto descriptor = resolve_conditioner_descriptor(checkpoint, config);
  PromptEmbedding embedding;
  if (request.backend == DeviceBackend::kCuda) {
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
      {"backend", request.backend == DeviceBackend::kCuda ? "cuda" : "vulkan"},
      {"arithmetic",
       request.backend == DeviceBackend::kVulkan || request.arithmetic == EncoderArithmetic::kExact
           ? "exact"
           : "shipped"}};
  write_prompt_embedding(output, embedding, metadata);
  return embedding.num_tokens;
}

} // namespace slopfab::text
