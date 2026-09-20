#pragma once
#include "slopfab/generate.h"
#include "slopfab/text/prompt_embedding.h"
#include "slopfab/text/tokenizer.h"
#include "slopfab/reference_conditioning.h"
namespace slopfab::generation {
struct ReusedGenerationModels {
  std::string conditioning_key;
  text::PromptEmbedding prompt;

  // The tokenizer is 147-166 ms to load and depends only on its own file, so it
  // survives the conditioning misses a prompt sweep is made of.
  std::string tokenizer_key;
  bool tokenizer_valid = false;
  text::Tokenizer tokenizer;

  // The seed-independent half of the reference-image path: decode, Lanczos
  // resize to the ~2048-pixel short edge, and the six-level Conv3D keyframe
  // encode at that resolution. The seed-dependent half — one noise draw and one
  // `scale_noise` at t = 0.999 — stays per generation, so two generations of a
  // counted run differ exactly where they are supposed to.
  std::string reference_key;
  bool reference_valid = false;
  std::vector<RGBImage> reference_images;
  std::vector<std::vector<float>> clean_reference_rows;
  std::vector<dit::ReferenceGeometry> reference_geometry;

  EncodedMediaCache media_cache;

  void clear() {
    conditioning_key.clear();
    prompt = {};
    tokenizer_key.clear();
    tokenizer_valid = false;
    tokenizer = text::Tokenizer();
    reference_key.clear();
    reference_valid = false;
    reference_images.clear();
    clean_reference_rows.clear();
    reference_geometry.clear();
    media_cache.clear();
  }
};


RunResult run_generate_impl(const GenerateRequest&, const GeneratePlan&, const RunOptions&, ReusedGenerationModels&);
}
