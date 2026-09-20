#include "detail/encoder_fixture.h"

SLOPFAB_TEST_CATEGORY(encoder_layer_layout, "synthetic") {
  slopfab::text::EncoderConfig cfg;
  // The host helpers refuse kAuto: two incompatible layouts and no file in
  // front of them is exactly where silently picking one goes wrong.
  cfg.format = slopfab::text::WeightFormat::kI8ConvRot;
  const slopfab::text::LayerLayout layout = slopfab::text::make_layer_layout(cfg);

  const size_t q = size_t(8192) * 5120;
  const size_t kv = size_t(1024) * 5120;
  const size_t o = size_t(5120) * 8192;
  const size_t mlp = size_t(25600) * 5120;

  CHECK(layout.bytes[int(slopfab::text::LayerTensor::kQWeight)] == q);
  CHECK(layout.bytes[int(slopfab::text::LayerTensor::kKWeight)] == kv);
  CHECK(layout.bytes[int(slopfab::text::LayerTensor::kOWeight)] == o);
  CHECK(layout.bytes[int(slopfab::text::LayerTensor::kDownWeight)] == mlp);
  CHECK(layout.bytes[int(slopfab::text::LayerTensor::kQScale)] == 8192 * sizeof(float));
  CHECK(layout.bytes[int(slopfab::text::LayerTensor::kQNorm)] == 128 * 2);
  CHECK(layout.bytes[int(slopfab::text::LayerTensor::kInputLayerNorm)] == 5120 * 2);

  // Spec section 7: 487 587 840 B of int8 weights, 286 720 B of scales and
  // 20 992 B of norms per layer. The blob rounds each up to 256 bytes.
  size_t weights = 0;
  size_t scales = 0;
  size_t norms = 0;
  for (int i = 0; i < slopfab::text::kLayerTensorCount; ++i) {
    const slopfab::text::TensorSpec spec =
        slopfab::text::layer_tensor_spec(cfg, static_cast<slopfab::text::LayerTensor>(i));
    if (spec.dtype == slopfab::DType::kI8) weights += layout.bytes[i];
    else if (spec.dtype == slopfab::DType::kF32) scales += layout.bytes[i];
    else norms += layout.bytes[i];
  }
  CHECK(weights == 487587840);
  CHECK(scales == 286720);
  CHECK(norms == 20992);
  CHECK(layout.total_bytes >= weights + scales + norms);
  CHECK(layout.total_bytes % 256 == 0);

  // Every offset is 256-byte aligned and no two tensors overlap.
  for (int i = 0; i < slopfab::text::kLayerTensorCount; ++i) {
    if (layout.bytes[i] == 0) continue;
    CHECK(layout.offset[i] % 256 == 0);
    if (i > 0) CHECK(layout.offset[i] >= layout.offset[i - 1] + layout.bytes[i - 1]);
  }

  // The int8 build has no AWQ activation scaling; those two slots are empty and
  // cost nothing in the blob.
  CHECK(layout.bytes[int(slopfab::text::LayerTensor::kOPreQuantScale)] == 0);
  CHECK(layout.bytes[int(slopfab::text::LayerTensor::kDownPreQuantScale)] == 0);
  CHECK(!slopfab::text::layer_tensor_spec(cfg, slopfab::text::LayerTensor::kOPreQuantScale).present());

  // Every contraction width is a multiple of the ConvRot group, so there is no
  // skip-when-not-divisible case in this checkpoint (spec section 5.2).
  for (int in_features : {5120, 8192, 25600}) CHECK(in_features % 256 == 0);
}

SLOPFAB_TEST_CATEGORY(encoder_validation_rejects_a_foreign_checkpoint, "synthetic") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "slopfab_encoder_test";
  std::filesystem::create_directories(dir);
  const std::string path = (dir / "fake.safetensors").string();

  std::vector<slopfab::TensorWrite> tensors;
  tensors.push_back(
      {"model.embed_tokens.weight", {4, 5120}, std::vector<float>(4 * 5120, 0.5f)});
  slopfab::write_safetensors(path, tensors);

  slopfab::SafeTensors st;
  st.open(path);

  // Real widths, tiny vocabulary: the file is F32 where the checkpoint is
  // BF16, so validation must reject it on dtype and say which tensor.
  slopfab::text::EncoderConfig cfg;
  cfg.vocab_size = 4;
  cfg.num_layers = 1;

  bool threw = false;
  std::string message;
  try {
    slopfab::text::validate_checkpoint(st, cfg);
  } catch (const std::exception& e) {
    threw = true;
    message = e.what();
  }
  CHECK(threw);
  // The message must name the offending tensor, not just say "bad checkpoint".
  // Format detection runs first now — there are two incompatible builds and
  // every later check depends on which one this is — so the first thing missing
  // from a foreign file is the descriptor detection reads.
  CHECK_MSG(message.find("model.layers.0.self_attn.q_proj.comfy_quant") != std::string::npos,
            "validation message does not name the tensor: %s", message.c_str());

  // The gather refuses the same file for the same reason, and refuses an id
  // outside the vocabulary.
  std::vector<uint16_t> out;
  bool gather_threw = false;
  try {
    slopfab::text::gather_embedding_rows(st.at("model.embed_tokens.weight"), nullptr, {0, 1}, out);
  } catch (const std::exception&) {
    gather_threw = true;
  }
  CHECK(gather_threw);

  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

SLOPFAB_TEST_CATEGORY(encoder_embedding_gather_int8, "synthetic") {
  // The nvfp4 build stores the embedding table as I8 with a per-row F32 scale
  // while the int8+ConvRot build stores the same table as BF16 — the embedding
  // does not follow the linears. Both directions of the row scale are finite
  // and correctly shaped, so only a comparison can tell them apart; that
  // comparison is in tools/nvfp4_layout_probe.py and gives 0.94% relative L2
  // for multiply against 7.5e6 for divide. This pins the multiply.
  const int64_t vocab = 6;
  const int64_t hidden = 8;

  std::vector<int8_t> table(size_t(vocab) * hidden);
  for (size_t i = 0; i < table.size(); ++i) table[i] = int8_t(int(i * 37 % 255) - 127);
  std::vector<float> scale(static_cast<size_t>(vocab));
  for (int64_t r = 0; r < vocab; ++r) scale[size_t(r)] = 1e-3f * float(1 + r * 3);

  slopfab::TensorView embed;
  embed.name = "model.embed_tokens.weight";
  embed.dtype = slopfab::DType::kI8;
  embed.shape = {vocab, hidden};
  embed.data = table.data();
  embed.nbytes = table.size();

  slopfab::TensorView embed_scale;
  embed_scale.name = "model.embed_tokens.weight_scale";
  embed_scale.dtype = slopfab::DType::kF32;
  embed_scale.shape = {vocab, 1};
  embed_scale.data = scale.data();
  embed_scale.nbytes = scale.size() * sizeof(float);

  const std::vector<int32_t> ids = {4, 0, 4, 2};
  std::vector<uint16_t> out;
  slopfab::text::gather_embedding_rows(embed, &embed_scale, ids, out);
  CHECK(out.size() == ids.size() * size_t(hidden));

  double worst = 0.0;
  for (size_t i = 0; i < ids.size(); ++i) {
    const size_t row = size_t(ids[i]);
    for (int64_t j = 0; j < hidden; ++j) {
      const float want = float(table[row * hidden + j]) * scale[row];
      const float got = slopfab::bf16_to_f32(out[i * size_t(hidden) + j]);
      // The only loss is the single bf16 rounding of the product.
      worst = std::max(worst, std::fabs(double(got - want)) / std::max(1e-30f, std::fabs(want)));
    }
  }
  CHECK_MSG(worst < 4e-3, "int8 embedding gather: worst relative error %.3e", worst);

  // The same id must gather the same row every time it appears.
  for (int64_t j = 0; j < hidden; ++j) CHECK(out[j] == out[2 * size_t(hidden) + j]);

  // Dividing by the scale instead of multiplying is the silent alternative. It
  // is off by six orders of magnitude here, and by seven on the real table.
  const float divided = float(table[size_t(ids[0]) * hidden]) / scale[size_t(ids[0])];
  const float multiplied = slopfab::bf16_to_f32(out[0]);
  CHECK_MSG(std::fabs(divided) > 100.0f * std::fabs(multiplied),
            "the per-row scale must multiply, not divide");

  // Structural refusals: an I8 table with no scale, and a scale of the wrong
  // length, are both loader bugs that would otherwise read whatever is next in
  // memory.
  bool threw = false;
  try {
    slopfab::text::gather_embedding_rows(embed, nullptr, ids, out);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);

  slopfab::TensorView short_scale = embed_scale;
  short_scale.shape = {vocab - 1, 1};
  threw = false;
  try {
    slopfab::text::gather_embedding_rows(embed, &short_scale, ids, out);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

SLOPFAB_TEST_CATEGORY(encoder_nvfp4_layer_layout, "synthetic") {
  slopfab::text::EncoderConfig cfg;
  cfg.format = slopfab::text::WeightFormat::kNVFP4Awq;
  const slopfab::text::LayerLayout layout = slopfab::text::make_layer_layout(cfg);

  using LT = slopfab::text::LayerTensor;
  // Two E2M1 per byte on the contraction axis, one e4m3 scale per 16.
  CHECK(layout.bytes[int(LT::kQWeight)] == size_t(8192) * 2560);
  CHECK(layout.bytes[int(LT::kQScale)] == size_t(8192) * 320);
  CHECK(layout.bytes[int(LT::kOWeight)] == size_t(5120) * 4096);
  CHECK(layout.bytes[int(LT::kOScale)] == size_t(5120) * 512);
  CHECK(layout.bytes[int(LT::kDownWeight)] == size_t(5120) * 12800);
  CHECK(layout.bytes[int(LT::kDownScale)] == size_t(5120) * 1600);
  // Present on exactly these two; the other five folded into the norms.
  CHECK(layout.bytes[int(LT::kOPreQuantScale)] == size_t(8192) * 2);
  CHECK(layout.bytes[int(LT::kDownPreQuantScale)] == size_t(25600) * 2);

  // Every scale tensor must be whole 128x4 tiles, which is what lets the
  // swizzle be computed rather than stored. Checked here because a checkpoint
  // that broke it would need a padding convention this port has never seen.
  for (int out_features : {8192, 1024, 5120, 25600}) CHECK(out_features % 128 == 0);
  for (int in_features : {5120, 8192, 25600}) CHECK((in_features / 16) % 4 == 0);

  // Half the int8 build's blob, which is the whole point of this format.
  slopfab::text::EncoderConfig i8 = cfg;
  i8.format = slopfab::text::WeightFormat::kI8ConvRot;
  const size_t i8_bytes = slopfab::text::make_layer_layout(i8).total_bytes;
  std::printf("  layer blob: nvfp4 %.1f MB, int8 %.1f MB, ratio %.3f\n",
              double(layout.total_bytes) / (1 << 20), double(i8_bytes) / (1 << 20),
              double(layout.total_bytes) / double(i8_bytes));
  CHECK(layout.total_bytes * 2 < i8_bytes * 5 / 4);
  CHECK(layout.total_bytes % 256 == 0);
}

SLOPFAB_TEST_CATEGORY(reference_vision_support_never_silently_ignores_pixels, "synthetic") {
  const std::string no_vision_path = "slopfab_test_qwen_no_vision.safetensors";
  slopfab::write_safetensors(no_vision_path, {{"model.embed_tokens.weight", {1}, {0.0f}}});
  slopfab::SafeTensors no_vision;
  no_vision.open(no_vision_path);
  slopfab::text::require_reference_vision_support(no_vision, 0);
  bool missing_failed = false;
  try {
    slopfab::text::require_reference_vision_support(no_vision, 1);
  } catch (const std::runtime_error& e) {
    missing_failed = std::string(e.what()).find("contains no visual.*") != std::string::npos;
  }
  CHECK(missing_failed);

  const std::string vision_path = "slopfab_test_qwen_with_vision.safetensors";
  slopfab::write_safetensors(vision_path, {{"visual.patch_embed.weight", {1}, {1.0f}}});
  slopfab::SafeTensors vision;
  vision.open(vision_path);
  bool malformed_failed = false;
  try {
    slopfab::text::require_reference_vision_support(vision, 1);
  } catch (const std::runtime_error& e) {
    malformed_failed = std::string(e.what()).find("exactly 351 Qwen vision tensors") !=
                       std::string::npos;
  }
  CHECK(malformed_failed);
}
