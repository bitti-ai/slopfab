#include "commands.h"
#include "slopfab/refmod.h"
#include "slopfab/text/export.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace slopfab::cli {

int cmd_bundle_refmods(int argc, char** argv) {
  RefModBundle bundle;
  std::string output, name, description;
  bool has_description = false;
  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      if (++i >= argc)
        throw std::invalid_argument("missing value for " + arg);
      return argv[i];
    };
    if (arg == "--refmod") {
      auto input = RefModBundle::load(next());
      if (bundle.members.empty())
        bundle.metadata = input.metadata;
      if (bundle.members.size() + input.members.size() > 256)
        throw std::invalid_argument("refmod bundle: maximum 256 members");
      bundle.members.insert(bundle.members.end(), input.members.begin(), input.members.end());
    } else if (arg == "--output")
      output = next();
    else if (arg == "--name")
      name = next();
    else if (arg == "--description") {
      description = next();
      has_description = true;
    } else
      throw std::invalid_argument("unknown bundle-refmods option: " + arg);
  }
  if (output.empty() || bundle.members.empty())
    throw std::invalid_argument("bundle-refmods requires --refmod and --output");
  if (!name.empty())
    bundle.metadata["name"] = json::Value(name);
  if (has_description)
    bundle.metadata["description"] = json::Value(description);
  bundle.save(output);
  int64_t tokens = 0;
  for (const auto& mod : bundle.members)
    tokens += mod->token_count();
  std::printf("saved %zu references (%lld tokens) to %s\n", bundle.members.size(),
              static_cast<long long>(tokens), output.c_str());
  return 0;
}

int cmd_encode_text(int argc, char** argv, const char* executable) {
  text::TextExportRequest request;
  std::string output, prompt_file;
  bool saw_prompt = false;
  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      if (++i >= argc)
        throw std::invalid_argument("missing value for " + arg);
      return argv[i];
    };
    if (arg == "--prompt") {
      request.prompt = next();
      saw_prompt = true;
    } else if (arg == "--prompt-file")
      prompt_file = next();
    else if (arg == "--output")
      output = next();
    else if (arg == "--text-encoder")
      request.text_encoder_path = next();
    else if (arg == "--tokenizer")
      request.tokenizer_path = next();
    else if (arg == "--inference-backend") {
      const auto value = next();
      if (value == "cuda")
        request.backend = DeviceBackend::kCuda;
      else if (value == "vulkan")
        request.backend = DeviceBackend::kVulkan;
      else
        throw std::invalid_argument("inference backend must be cuda or vulkan");
    } else if (arg == "--arithmetic") {
      const auto value = next();
      if (value == "exact")
        request.arithmetic = text::EncoderArithmetic::kExact;
      else if (value == "shipped")
        request.arithmetic = text::EncoderArithmetic::kShipped;
      else
        throw std::invalid_argument("arithmetic must be shipped or exact");
    } else if (arg == "--vulkan-arithmetic") {
      const auto value = next();
      if (value != "portable" && value != "exact")
        throw std::invalid_argument("Vulkan arithmetic must be portable or exact");
      request.vulkan_portable_arithmetic = value == "portable";
    } else if (arg == "--residency") {
      const auto value = next();
      if (value == "streaming")
        request.residency = text::Residency::kStreaming;
      else if (value == "resident")
        request.residency = text::Residency::kResident;
      else if (value == "auto")
        request.residency = text::Residency::kAuto;
      else
        throw std::invalid_argument("residency must be streaming, resident or auto");
    } else
      throw std::invalid_argument("unknown encode-text option: " + arg);
  }
  if (saw_prompt && !prompt_file.empty())
    throw std::invalid_argument("use either --prompt or --prompt-file");
  if (!prompt_file.empty()) {
    std::ifstream input(std::filesystem::u8path(prompt_file), std::ios::binary);
    if (!input)
      throw std::runtime_error("cannot read prompt file: " + prompt_file);
    request.prompt.assign(std::istreambuf_iterator<char>(input), {});
    if (request.prompt.compare(0, 3, "\xEF\xBB\xBF") == 0)
      request.prompt.erase(0, 3);
  }
  if (request.prompt.empty() || output.empty())
    throw std::invalid_argument("encode-text requires a nonempty prompt and --output");
  GenerateRequest discovery;
  discovery.text_encoder_path = request.text_encoder_path;
  discover_generate_checkpoints(discovery, executable);
  request.text_encoder_path = discovery.text_encoder_path;
#if SLOPFAB_WITH_CUDA || SLOPFAB_WITH_VULKAN
  std::printf("encoding text using %s\n", request.text_encoder_path.c_str());
  const int tokens = text::export_prompt_embedding(request, output);
  std::printf("saved prompt_embedding [%d,5120] and text_token_tags to %s\n", tokens,
              output.c_str());
  return 0;
#else
  throw std::runtime_error("encode-text requires a build with CUDA or Vulkan");
#endif
}

} // namespace slopfab::cli
