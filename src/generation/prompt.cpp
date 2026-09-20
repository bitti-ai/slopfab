#include "slopfab/text/prompt_embedding.h"
#include "slopfab/generate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "slopfab/audio/wav.h"
#include "slopfab/image.h"
#include "slopfab/cuda/profile.h"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/dit/denoise.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/dit/packing.h"
#include "slopfab/dit/ref2va.h"
#include "slopfab/dit/transformer.h"
#include "slopfab/text/encoder.h"
#include "slopfab/text/tokenizer.h"
#include "slopfab/sampler/scheduler.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sampler/noise.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vae/audio_decoder.h"
#include "slopfab/vae/vit_decoder.h"
#include "slopfab/vae/keyframe_encoder.h"
#include "slopfab/vae/audio_encoder.h"
#include "slopfab/reference_conditioning.h"
#include "slopfab/video/mux.h"
#include "slopfab/video/y4m.h"
#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/audio_decoder.h"
#include "slopfab/vulkan/dit_denoise.h"
#include "slopfab/vulkan/keyframe_encoder.h"
#include "slopfab/vulkan/reference_encoder.h"
#include "slopfab/vulkan/text_encoder.h"
#include "slopfab/vulkan/vae_decoder.h"
#endif

#include "session_state.h"
#include "helpers.h"
#include "prefetch.h"
#include "decode.h"
#include "prompt.h"
namespace slopfab::generation {
bool prepare_multimodal_prompt(const GenerateRequest& request, const RunOptions& options,
    const std::vector<PreparedReference>& prepared_media, PromptInputs& inputs, RunResult& result) {
  auto& fixed_prompt = inputs.fixed_prompt;
  const auto& reference_identities = inputs.reference_identities;
  const auto& reference_images = inputs.reference_images;
  auto& reference_conditioning_grids = inputs.reference_conditioning_grids;
  auto& reference_conditioning_ids = inputs.reference_conditioning_ids;
  auto& media_qwen_pairs = inputs.media_qwen_pairs;
  auto& conditioning_tokenizer = inputs.tokenizer;
  if (options.source == LatentSource::kDenoise && !reference_images.empty() &&
      options.prompt_embedding_path.empty()) {
    try {
      text::Tokenizer& tokenizer = conditioning_tokenizer();
      reference_conditioning_grids.reserve(reference_images.size());
      std::vector<std::vector<int32_t>> labels;
      labels.reserve(reference_images.size());
      size_t nonvision_tokens = 0;
      for (size_t i = 0; i < reference_images.size(); ++i) {
        reference_conditioning_grids.push_back(
            text::qwen3vl_conditioning_grid(reference_images[i].width,
                                            reference_images[i].height));
        labels.push_back(tokenizer.encode(
            "<Picture " + std::to_string(i + 1) + ">: "));
        if (labels.back().size() > std::numeric_limits<size_t>::max() -
                                     nonvision_tokens)
          throw std::overflow_error("reference conditioning token overflow");
        nonvision_tokens += labels.back().size();
      }
      const std::vector<int32_t> prompt_ids = tokenizer.encode(request.prompt);
      if (prompt_ids.size() > std::numeric_limits<size_t>::max() -
                                  nonvision_tokens)
        throw std::overflow_error("reference conditioning token overflow");
      nonvision_tokens += prompt_ids.size();
      const size_t total = text::qwen3vl_conditioning_token_count(
          reference_conditioning_grids, nonvision_tokens);
      reference_conditioning_ids.reserve(total);
      for (size_t i = 0; i < labels.size(); ++i) {
        const std::vector<int32_t> block = text::qwen3vl_image_block(
            labels[i], reference_conditioning_grids[i].merged_token_count());
        reference_conditioning_ids.insert(reference_conditioning_ids.end(),
                                          block.begin(), block.end());
      }
      reference_conditioning_ids.insert(reference_conditioning_ids.end(),
                                        prompt_ids.begin(), prompt_ids.end());
      if (reference_conditioning_ids.size() != total)
        throw std::logic_error("reference conditioning token count drift");
    } catch (const std::exception& e) {
      result.message = e.what();
      return false;
    }
  }

  if (!prepared_media.empty() && options.prompt_embedding_path.empty()) {
    auto& tokenizer = conditioning_tokenizer();
    const auto prompt_ids = tokenizer.encode(request.prompt);
    if (!reference_images.empty()) reference_conditioning_ids.resize(reference_conditioning_ids.size() - prompt_ids.size());
    auto emit_text = [&](const std::string& value) {
      auto ids = tokenizer.encode(value);
      reference_conditioning_ids.insert(reference_conditioning_ids.end(), ids.begin(), ids.end());
    };
    int audio_number = 0, video_number = 0;
    for (const auto& media : prepared_media) {
      if (media.plan.audio_samples) emit_text("<Audio " + std::to_string(++audio_number) + ">: ");
      if (media.frames.empty()) continue;
      emit_text("<Video " + std::to_string(++video_number) + ">: ");
      const auto grid = text::qwen3vl_conditioning_grid(media.plan.width, media.plan.height);
      const int rw = grid.width * 16, rh = grid.height * 16;
      // 2 fps presentation, paired into Qwen's temporal patch size of two.
      // Pair labels use decimal round-half-to-even, independently of locale.
      const int sampled = (media.plan.frames + 11) / 12;
      for (int pair = 0; pair < sampled; pair += 2) {
        const int second = std::min(pair + 1, sampled - 1);
        const int quarters = pair + second; // timestamp = quarters / 4
        const int scaled = quarters * 5; // tenths = scaled / 2
        const int tenths = scaled / 2 + ((scaled % 2) && ((scaled / 2) % 2));
        emit_text("<" + std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + " seconds>");
        auto ids = text::qwen3vl_image_block({}, grid.merged_token_count(), 151652, 151656, 151653);
        reference_conditioning_ids.insert(reference_conditioning_ids.end(), ids.begin(), ids.end());
        if (reference_conditioning_ids.size() + prompt_ids.size() > text::kMaxPromptTokens)
          throw std::runtime_error("reference video conditioning exceeds " +
                                   std::to_string(text::kMaxPromptTokens) + " prompt tokens");
        auto first_rgb = resize_rgb_bilinear(media.frames[pair * 12], rw, rh);
        auto second_rgb = resize_rgb_bilinear(media.frames[second * 12], rw, rh);
        media_qwen_pairs.push_back(text::qwen3vl_patchify_resized_rgb_pair(first_rgb, second_rgb, rw, rh));
      }
    }
    reference_conditioning_ids.insert(reference_conditioning_ids.end(), prompt_ids.begin(), prompt_ids.end());
    if (reference_conditioning_ids.size() > text::kMaxPromptTokens)
      throw std::runtime_error("reference conditioning exceeds " +
                               std::to_string(text::kMaxPromptTokens) + " prompt tokens");
  }

  return true;
}
bool encode_h3_prompt(const GenerateRequest& request, const RunOptions& options,
    ReusedGenerationModels& reuse, PromptInputs& inputs, text::PromptEmbedding& prompt, RunResult& result) {
  auto& fixed_prompt = inputs.fixed_prompt;
  const auto& reference_identities = inputs.reference_identities;
  const auto& reference_images = inputs.reference_images;
  auto& reference_conditioning_grids = inputs.reference_conditioning_grids;
  auto& reference_conditioning_ids = inputs.reference_conditioning_ids;
  auto& media_qwen_pairs = inputs.media_qwen_pairs;
  auto& conditioning_tokenizer = inputs.tokenizer;
    // Same snapshot of the references as the reference key above, and likewise
    // skipped outright when nothing will consult it.
    const ConditionerAuthority conditioner_authority =
        options.inference_backend == DeviceBackend::kVulkan
            ? ConditionerAuthority::kVulkanExact
            : (options.attention_mode == AttentionMode::kExact
                   ? ConditionerAuthority::kCudaExact
                   : ConditionerAuthority::kCudaShipped);
    const std::string prompt_key =
        options.reuse_models
            ? conditioning_cache_key_for_authority(
                  request, reference_identities, conditioner_authority)
            : std::string();
    if (!options.prompt_embedding_path.empty()) {
      const Clock::time_point t0 = Clock::now();
      try {
        prompt = std::move(fixed_prompt);
      } catch (const std::exception& e) {
        result.message = e.what();
        return false;
      }
      result.seconds_conditioning = seconds_since(t0);
      if (options.verbose)
        std::printf("prompt      captured [%d, %d] from %s (no conditioner)\n",
                    prompt.num_tokens, prompt.hidden_size,
                    options.prompt_embedding_path.c_str());
    } else if (options.reuse_models && reuse.conditioning_key == prompt_key &&
        !reuse.prompt.data.empty()) {
      prompt = reuse.prompt;
      if (options.verbose) {
        std::printf("prompt      reused cached [%d, %d] conditioning\n", prompt.num_tokens,
                    prompt.hidden_size);
      }
    } else {
      const Clock::time_point t0 = Clock::now();

      // Reaching here means the conditioning cache missed. The tokenizer does
      // not depend on the prompt, so it is kept across those misses and
      // reloaded only when its own file changes; `encode()` is const and
      // stateless, so one instance serves every caller.
      // The early reference preflight and the actual conditioner deliberately
      // share this instance. Besides avoiding a second tokenizer load, that
      // guarantees the IDs validated before the keyframe checkpoint opens are
      // exactly the IDs consumed here.
      text::Tokenizer& tokenizer = conditioning_tokenizer();

      // No chat template, no BOS, no EOS: `hidden_states[50]` of a raw prompt
      // is the conditioning H3 expects, and a special token here would shift
      // every rotary position downstream (spec 1.2).
      std::vector<int32_t> ids;
      std::vector<text::QwenPixelValues> qwen_images;
      if (request.has_native_references()) {
        if (reference_conditioning_grids.size() != reference_images.size() ||
            reference_conditioning_ids.empty())
          throw std::logic_error("reference conditioning preflight is absent");
        ids = reference_conditioning_ids;
      }
      for (size_t i = 0; i < reference_images.size(); ++i) {
        const auto& grid = reference_conditioning_grids[i];
        const int rw = grid.width * 16, rh = grid.height * 16;
        auto rgb = resize_rgb_bilinear(reference_images[i], rw, rh);
        qwen_images.push_back(text::qwen3vl_patchify_resized_rgb(rgb, rw, rh));
      }
      for (auto& pair : media_qwen_pairs) qwen_images.push_back(std::move(pair));
      if (!request.has_native_references()) {
        const auto prompt_ids = tokenizer.encode(request.prompt);
        ids.insert(ids.end(), prompt_ids.begin(), prompt_ids.end());
      }
      if (ids.empty()) {
        result.message = "the prompt tokenised to zero tokens";
        return false;
      }

      SafeTensors encoder_file;
      encoder_file.open(request.text_encoder_path);
      try {
        text::require_reference_vision_support(encoder_file, qwen_images.size());
      } catch (const std::exception& e) {
        result.message = e.what();
        return false;
      }
      const char* conditioner_mode = nullptr;
      if (options.inference_backend == DeviceBackend::kCuda) {
        text::Encoder encoder;
        text::EncoderConfig ecfg;
        ecfg.residency = text::Residency::kStreaming;
        if (options.attention_mode == AttentionMode::kExact)
          ecfg.arithmetic = text::EncoderArithmetic::kExact;
        encoder.load(encoder_file, ecfg);
        prompt = qwen_images.empty() ? encoder.encode(ids)
                                     : encoder.encode(ids, qwen_images);
        conditioner_mode = ecfg.arithmetic == text::EncoderArithmetic::kExact
            ? "CUDA streaming exact" : "CUDA streaming shipped";
        encoder.unload();
      } else {
#if SLOPFAB_WITH_VULKAN
        vulkan::Device device = create_vulkan_inference_device(true);
        vulkan::TensorContextOptions tensor_options;
        tensor_options.max_batch_operators = 64;
        vulkan::TensorContext context(device, tensor_options);
        vulkan::ExactQwenTextEncoder encoder =
            vulkan::ExactQwenTextEncoder::create(context);
        encoder.load(encoder_file);
        prompt = qwen_images.empty() ? encoder.encode(ids)
                                     : encoder.encode(ids, qwen_images);
        if (options.verbose) {
          const auto& stats = encoder.stats();
          std::printf(
              "conditioner Vulkan exact peak/reserved %.2f/%.2f GiB, %llu descriptors\n",
              static_cast<double>(stats.peak_device_bytes) /
                  (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(stats.allocator_reserved_bytes) /
                  (1024.0 * 1024.0 * 1024.0),
              static_cast<unsigned long long>(stats.descriptor_set_allocations));
        }
        encoder.unload();
        conditioner_mode = "Vulkan streaming exact";
#else
        throw std::logic_error("Vulkan conditioner compiled out after validation");
#endif
      }
      result.conditioner_executed = true;
      if (options.reuse_models) {
        reuse.conditioning_key = prompt_key;
        reuse.prompt = prompt;
      }
      result.seconds_conditioning = seconds_since(t0);
      if (options.verbose) {
        std::printf("prompt      %d tokens -> [%d, %d] in %.2f s (%s)\n",
                    static_cast<int>(ids.size()), prompt.num_tokens, prompt.hidden_size,
                    result.seconds_conditioning, conditioner_mode);
      }
    }

  return true;
}
}
