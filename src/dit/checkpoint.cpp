#include "slopfab/dit/checkpoint.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <string>
#include <set>
#include "slopfab/json.h"
#include "slopfab/w4a8.h"
#include "slopfab/dit/adaln.h"

namespace slopfab::dit {
namespace {

bool has(const SafeTensors& st, const char* name) {
  return st.find(name) != nullptr;
}

bool path_marks_ref2va(const SafeTensors& st) {
  std::string name = std::filesystem::path(st.path()).filename().string();
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (name.find("ref2va") != std::string::npos)
    return true;
  // The published PRISM merge is Ref2VA, but its renamed exports omit
  // "ref2va" and retain only {format:pt, pruned:curve_rank8}. Its model card
  // identifies both parents as Ref2VA. Limit this alias to the named release;
  // neither W6A8 storage nor generic Singularity filenames imply references.
  name.erase(std::remove_if(name.begin(), name.end(),
                            [](char c) { return c == '_' || c == '-'; }), name.end());
  const std::string prism = "minimaxh3singularityx10erosmaxprism";
  return name == prism + "w6a8.safetensors" || name == prism + "int8.safetensors" ||
         name == prism + "fp16.safetensors";
}

bool marks_viggle(const SafeTensors& st) {
  const auto source = st.metadata().find("source");
  if (source != st.metadata().end())
    return source->second == "Viggle/Viggle-Animate";
  std::string name = std::filesystem::path(st.path()).filename().string();
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return name.find("viggle-animate") != std::string::npos;
}

bool marks_fasth3_v2(const SafeTensors& st) {
  // V1 and V2 share tensor layouts but have different trained schedules.
  // Converted files often retain only {format: pt}, so recognize the release
  // filename as well as an explicit source/model_id in renamed archives.
  std::string identity = std::filesystem::path(st.path()).filename().string();
  for (const char* key : {"source", "model_id"}) {
    const auto it = st.metadata().find(key);
    if (it != st.metadata().end())
      identity += " " + it->second;
  }
  std::transform(identity.begin(), identity.end(), identity.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  identity.erase(std::remove_if(identity.begin(), identity.end(),
                                [](char c) {
                                  return c == '_' || c == '-';
                                }),
                 identity.end());
  return identity.find("fasth38stepv2") != std::string::npos;
}

int pdd_heads(const SafeTensors& st) {
  // OmniRef's pruned PDD export shares the H3 backbone and VSA gate names,
  // but predicts multiple fine-grid velocities per token. Detect the widened
  // heads before applying legacy filename aliases or metadata.
  const auto* video = st.find("final_layer.video_out.weight");
  const auto* audio = st.find("final_layer.audio_out.weight");
  if (!has(st, "blocks.0.attn.to_gate_compress.weight") || !video || !audio ||
      video->shape.size() != 2 || audio->shape.size() != 2)
    return 1;
  const int64_t heads = video->shape[0] / 96;
  if (heads > 1 && video->shape[0] % 96 == 0 && audio->shape[0] == heads * 32 &&
      video->shape[1] == audio->shape[1])
    return static_cast<int>(heads);
  return 1;
}

} // namespace

static TransformerArchitecture infer_legacy_architecture(const SafeTensors& checkpoint) {
  // The released pruned checkpoints replace the timestep MLP with a lookup
  // table and contract every AdaLN projection to rank eight.
  if (has(checkpoint, "adaln_t_table") && has(checkpoint, "blocks.0.adaln_proj.linear.weight")) {
    if (has(checkpoint, "blocks.0.attn.to_gate_compress.weight"))
      return marks_fasth3_v2(checkpoint) ? TransformerArchitecture::kFastH3V2PrunedTable
                                         : TransformerArchitecture::kUnknown;
    if (marks_viggle(checkpoint))
      return TransformerArchitecture::kViggleAnimatePrunedTable;
    // The released pruned Ref2VA FP8 file has the exact same tensor names,
    // shapes, dtypes and empty metadata as the FL2VA file. The distribution
    // filename is therefore the only identity signal available in the file.
    if (path_marks_ref2va(checkpoint))
      return TransformerArchitecture::kRef2VAPrunedTable;
    return TransformerArchitecture::kPrunedTable;
  }

  // The unpruned Ref2VA file retains the timestep MLP and full AdaLN
  // projections. Quantization is deliberately not part of architecture
  // detection; the released pruned FP8 variant is handled above.
  if (has(checkpoint, "time_embedder.proj_in.weight") &&
      has(checkpoint, "time_embedder.proj_out.weight") &&
      has(checkpoint, "blocks.0.adaln_proj.linear.weight")) {
    return TransformerArchitecture::kRef2VAFullAdaLN;
  }
  return TransformerArchitecture::kUnknown;
}

void validate_adaln_table_config(int rank, int rows) {
  if (rank != AdaLNTable::kRank || rows != AdaLNTable::kRows)
    throw std::runtime_error("transformer: table modulation requires AdaLN rank 8 and 1025 rows");
}

ModelDescriptor resolve_model_descriptor(const SafeTensors& checkpoint) {
  ModelDescriptor model;
  if (has(checkpoint, "cond_layer_logits") && has(checkpoint, "diffusion_transformer.transformer.project_in.weight")) {
    for (const auto& item : std::vector<std::pair<std::string, std::vector<int64_t>>>{
        {"cond_layer_logits", {8}}, {"latent_conditioners.0.weight", {2048,4096,3}},
        {"diffusion_transformer.transformer.project_in.weight", {2048,2304}},
        {"diffusion_transformer.transformer.layers.35.self_attn.to_qkv.weight", {6144,2048}},
        {"diffusion_transformer.transformer.project_out.weight", {128,2048}}})
      if (!checkpoint.find(item.first) || checkpoint.at(item.first).shape != item.second)
        throw std::invalid_argument("MiniMax Music3: incompatible tensor " + item.first);
    model.family = "minimax_music3";
    model.modulation = ModulationImplementation::kTimestepMlp;
    model.origin = "MiniMax Music3 fused flow/conditioner tensor contract";
    model.quantization = checkpoint.at("diffusion_transformer.transformer.layers.35.self_attn.to_qkv.weight").dtype == DType::kI8 ? TransformerQuantization::kInt8ConvRot : TransformerQuantization::kBFloat16;
    return model;
  }
  if (has(checkpoint, "text_encoders.yue2_tokenizer_json")) {
    for (const auto& item : std::vector<std::pair<std::string, std::vector<int64_t>>>{
        {"text_encoders.model.embed_tokens.weight", {184704,2048}},
        {"text_encoders.model.layers.27.self_attn.qkv_proj.weight", {4096,2048}},
        {"model.diffusion_model.model.layers.27.self_attn.qkv_proj.weight", {4096,2048}},
        {"model.diffusion_model.vae2llm.weight", {2048,64}},
        {"model.diffusion_model.llm2vae.weight", {64,2048}},
        {"vae.decoder.layers.8.weight_v", {2,64,7}}})
      if (!checkpoint.find(item.first) || checkpoint.at(item.first).shape != item.second)
        throw std::invalid_argument("YuE2: incompatible tensor " + item.first);
    model.family = "yue2";
    model.modulation = ModulationImplementation::kTimestepMlp;
    model.origin = "YuE2 bundled AR/NAR/VAE tensor contract";
    model.quantization = checkpoint.at("model.diffusion_model.llm2vae.weight").dtype == DType::kI8 ? TransformerQuantization::kInt8ConvRot : TransformerQuantization::kBFloat16;
    return model;
  }
  if (has(checkpoint, "perceiver.latent_queries") && has(checkpoint, "noise_refiner.0.attention.qkv.weight")) {
    for (const auto& pair : std::vector<std::pair<std::string, std::vector<int64_t>>>{
        {"x_embedder.weight", {3584,128}}, {"perceiver.latent_queries", {512,3584}},
        {"layers.43.attention.qkv.weight", {10752,3584}},
        {"final_layer.linear.weight", {128,3584}},
        {"perceiver.layers.1.cross_attention.to_kv.weight", {7168,7680}}}) {
      if (!checkpoint.find(pair.first) || checkpoint.at(pair.first).shape != pair.second)
        throw std::invalid_argument("Fibo 2.0: incompatible tensor " + pair.first);
    }
    model.family = "fibo2";
    model.compatibility_architecture = TransformerArchitecture::kFibo2EditTurbo;
    model.supports_references = true;
    model.modulation = ModulationImplementation::kTimestepMlp;
    model.quantization = detect_transformer_quantization(checkpoint);
    model.origin = "Fibo 2.0 Comfy fused tensor contract";
    return model;
  }
  model.pdd_steps = pdd_heads(checkpoint);
  if (model.pdd_steps > 1) {
    if (model.pdd_steps != 32 || !has(checkpoint, "adaln_t_table") ||
        checkpoint.at("adaln_t_table").shape != std::vector<int64_t>{1025, 8} ||
        !has(checkpoint, "blocks.0.adaln_proj.linear.weight"))
      throw std::invalid_argument("OmniRef requires 32 PDD heads and a rank-8 AdaLN table");
    if (checkpoint.metadata().count("slopfab.model"))
      throw std::invalid_argument("OmniRef PDD cannot use a legacy slopfab.model descriptor");
    model.compatibility_architecture = TransformerArchitecture::kOmniRefPrunedTable;
    model.supports_references = model.compressed_attention = true;
    model.quantization = detect_transformer_quantization(checkpoint);
    model.origin = "FastH3 OmniRef PDD32 tensor contract";
    const auto layout = checkpoint.metadata().find("qkv_layout");
    if (layout != checkpoint.metadata().end()) {
      if (layout->second != "contiguous" && layout->second != "interleaved")
        throw std::invalid_argument("OmniRef: unsupported qkv_layout");
      model.qkv_interleaved = layout->second == "interleaved";
    }
    return model;
  }
  if (has(checkpoint, "video_embedder.weight") && has(checkpoint, "shared_block_modulation.weight")) {
    const auto shape = [&](const std::string& key, std::vector<int64_t> expected) {
      if (!checkpoint.find(key) || checkpoint.at(key).shape != expected)
        throw std::invalid_argument("MOSS: incompatible tensor " + key);
    };
    shape("video_embedder.weight", {3072, 128});
    shape("text_embedder.proj.weight", {3072, 4096});
    shape("shared_block_modulation.weight", {18432, 3072});
    shape("final_linear_video.weight", {128, 3072});
    shape("final_norm_video.weight", {3072});
    shape("time_embedding.0.weight", {3072, 256});
    shape("time_embedding.2.weight", {3072, 3072});
    shape("video_head_modulation.weight", {6144, 3072});
    for (int i = 0; i < 42; ++i) {
      const auto prefix = "layers." + std::to_string(i) + ".";
      shape(prefix + "norm1.weight", {3072});
      shape(prefix + "norm2.weight", {3072});
      shape(prefix + "q_norm.weight", {128});
      shape(prefix + "k_norm.weight", {128});
      shape(prefix + "proj.o.weight", {3072, 3072});
      if (checkpoint.find(prefix + "proj.qkv.weight"))
        shape(prefix + "proj.qkv.weight", {9216, 3072});
      else
        for (const char* part : {"q", "k", "v"})
          shape(prefix + "proj." + part + ".weight", {3072, 3072});
      shape(prefix + "proj.ffn_up_gate.weight", {16384, 3072});
      shape(prefix + "proj.ffn_down.weight", {3072, 8192});
    }
    model.family = "moss";
    const auto metadata = checkpoint.metadata().find("slopfab.model");
    if (metadata != checkpoint.metadata().end()) {
      const auto root = json::parse(metadata->second);
      if (!root.is_object() || !root.find("version") || !root.find("family") ||
          root.find("version")->as_number() != 1 || root.find("family")->as_string() != "moss")
        throw std::invalid_argument("MOSS: incompatible slopfab.model descriptor");
      for (const auto& field : root.as_object())
        if (field.first != "version" && field.first != "family")
          throw std::invalid_argument("MOSS: unsupported descriptor field " + field.first);
      model.explicit_metadata = true;
    }
    model.modulation = ModulationImplementation::kTimestepMlp;
    model.compatibility_architecture = TransformerArchitecture::kMossImage;
    model.quantization = detect_transformer_quantization(checkpoint);
    model.origin = "MOSS image tensor contract";
    return model;
  }
  model.compatibility_architecture = infer_legacy_architecture(checkpoint);
  model.quantization = detect_transformer_quantization(checkpoint);
  const auto legacy = model.compatibility_architecture;
  model.modulation = legacy == TransformerArchitecture::kRef2VAFullAdaLN
                         ? ModulationImplementation::kTimestepMlp
                         : ModulationImplementation::kTable;
  model.supports_references = legacy == TransformerArchitecture::kRef2VAFullAdaLN ||
                              legacy == TransformerArchitecture::kRef2VAPrunedTable ||
                              legacy == TransformerArchitecture::kViggleAnimatePrunedTable;
  model.compressed_attention = has(checkpoint, "blocks.0.attn.to_gate_compress.weight");
  const auto layout = checkpoint.metadata().find("qkv_layout");
  if (layout != checkpoint.metadata().end()) {
    if (layout->second != "contiguous" && layout->second != "interleaved")
      throw std::runtime_error("transformer: unsupported qkv_layout '" + layout->second + "'");
    model.qkv_interleaved = layout->second == "interleaved";
  }
  const auto metadata = checkpoint.metadata().find("slopfab.model");
  if (metadata == checkpoint.metadata().end())
    return model;
  const auto root = json::parse(metadata->second);
  if (!root.is_object())
    throw std::runtime_error("slopfab.model: expected an object");
  const std::set<std::string> keys = {
      "version",    "family",         "modulation",   "supports_references", "compressed_attention",
      "qkv_layout", "legacy_profile", "adaln_grid_id"};
  for (const auto& item : root.as_object())
    if (!keys.count(item.first))
      throw std::runtime_error("slopfab.model: unknown field '" + item.first + "'");
  auto required = [&](const char* key) -> const json::Value& {
    const auto* value = root.find(key);
    if (!value)
      throw std::runtime_error(std::string("slopfab.model: missing '") + key + "'");
    return *value;
  };
  if (required("version").as_number() != 1)
    throw std::runtime_error("slopfab.model: unsupported version");
  if (required("family").as_string() != "h3")
    throw std::runtime_error("slopfab.model: unsupported transformer family");
  const std::string modulation = required("modulation").as_string();
  if (modulation != "table" && modulation != "timestep_mlp")
    throw std::runtime_error("slopfab.model: unsupported modulation implementation");
  model.modulation = modulation == "table" ? ModulationImplementation::kTable
                                           : ModulationImplementation::kTimestepMlp;
  model.supports_references = required("supports_references").as_bool();
  model.compressed_attention = required("compressed_attention").as_bool();
  const std::string qkv = required("qkv_layout").as_string();
  if (qkv != "contiguous" && qkv != "interleaved")
    throw std::runtime_error("slopfab.model: unsupported qkv_layout");
  model.qkv_interleaved = qkv == "interleaved";
  if (layout != checkpoint.metadata().end() && layout->second != qkv)
    throw std::runtime_error("slopfab.model: qkv_layout conflicts with legacy metadata");
  const bool table = model.modulation == ModulationImplementation::kTable;
  if (!has(checkpoint, "blocks.0.adaln_proj.linear.weight") ||
      (table && (!has(checkpoint, "adaln_t_table") ||
                 checkpoint.at("adaln_t_table").shape !=
                     std::vector<int64_t>{AdaLNTable::kRows, AdaLNTable::kRank})) ||
      (!table && (!has(checkpoint, "time_embedder.proj_in.weight") ||
                  !has(checkpoint, "time_embedder.proj_out.weight"))))
    throw std::runtime_error("slopfab.model: modulation tensor contract does not match metadata");
  if (model.compressed_attention != has(checkpoint, "blocks.0.attn.to_gate_compress.weight"))
    throw std::runtime_error("slopfab.model: compressed attention requires matching gate tensors");
  if (!table && (!model.supports_references || model.compressed_attention))
    throw std::runtime_error("slopfab.model: unsupported timestep MLP capability combination");
  model.compatibility_architecture = !table ? TransformerArchitecture::kRef2VAFullAdaLN
                                     : model.supports_references
                                         ? TransformerArchitecture::kRef2VAPrunedTable
                                         : TransformerArchitecture::kPrunedTable;
  if (const auto* profile = root.find("legacy_profile")) {
    const auto& name = profile->as_string();
    if (name == "animate" && table && model.supports_references && !model.compressed_attention)
      model.compatibility_architecture = TransformerArchitecture::kViggleAnimatePrunedTable;
    else if (name == "fast_h3_v2" && table && !model.supports_references &&
             model.compressed_attention)
      model.compatibility_architecture = TransformerArchitecture::kFastH3V2PrunedTable;
    else if (name != "none")
      throw std::runtime_error("slopfab.model: incompatible legacy_profile");
  }
  if (const auto* grid = root.find("adaln_grid_id")) {
    model.adaln_grid_id = grid->as_string();
    if (model.adaln_grid_id.empty() || !table)
      throw std::runtime_error("slopfab.model: adaln_grid_id requires a nonempty table identity");
  }
  model.explicit_metadata = true;
  model.origin = "slopfab.model v1 metadata";
  return model;
}

TransformerArchitecture detect_transformer_architecture(const SafeTensors& checkpoint) {
  return resolve_model_descriptor(checkpoint).compatibility_architecture;
}

TransformerQuantization detect_transformer_quantization(const SafeTensors& checkpoint) {
  std::string prefix = "blocks.0.attn.qkv_proj";
  if (checkpoint.find("perceiver.latent_queries"))
    prefix = "layers.0.attention.qkv";
  if (checkpoint.find("video_embedder.weight"))
    prefix = checkpoint.find("layers.0.proj.qkv.weight") ? "layers.0.proj.qkv" : "layers.0.proj.q";
  const TensorView* weight = checkpoint.find(prefix + ".weight");
  if (weight == nullptr)
    return TransformerQuantization::kUnknown;
  if (is_w6a8_weight(checkpoint, prefix + ".weight"))
    return TransformerQuantization::kW6A8ConvRot;
  if (is_w4a8_weight(checkpoint, prefix + ".weight"))
    return TransformerQuantization::kW4A8ConvRot;
  if (has(checkpoint, "blocks.0.attn.qkv_proj.weight.quant_state.bitsandbytes__nf4")) {
    return TransformerQuantization::kBitsAndBytesNF4;
  }
  if (weight->dtype == DType::kF8E4M3)
    return TransformerQuantization::kFloat8;
  if (weight->dtype == DType::kBF16)
    return TransformerQuantization::kBFloat16;
  if (weight->dtype == DType::kF32)
    return TransformerQuantization::kFloat32;
  const TensorView* scale = checkpoint.find(prefix + ".weight_scale");
  if (weight->dtype == DType::kI8 && weight->shape.size() == 2 && scale != nullptr &&
      scale->dtype == DType::kF32 && scale->shape.size() == 2 &&
      scale->shape[0] == weight->shape[0] && scale->shape[1] == 1) {
    return TransformerQuantization::kInt8ConvRot;
  }
  if (weight->dtype == DType::kU8 && scale != nullptr && scale->dtype == DType::kF8E4M3) {
    return TransformerQuantization::kNativeNVFP4;
  }
  return TransformerQuantization::kUnknown;
}

const char* transformer_architecture_name(TransformerArchitecture architecture) {
  switch (architecture) {
  case TransformerArchitecture::kFibo2EditTurbo:
    return "Fibo 2.0 Edit Turbo transformer";
  case TransformerArchitecture::kMossImage:
    return "MOSS Image transformer";
  case TransformerArchitecture::kPrunedTable:
    return "pruned AdaLN-table transformer";
  case TransformerArchitecture::kFastH3V2PrunedTable:
    return "FastH3 V2 (VSA-H3, 8-step) transformer";
  case TransformerArchitecture::kOmniRefPrunedTable:
    return "FastH3 OmniRef (PDD32, VSA-H3, 8-step) transformer";
  case TransformerArchitecture::kRef2VAPrunedTable:
    return "pruned AdaLN-table Ref2VA transformer";
  case TransformerArchitecture::kRef2VAFullAdaLN:
    return "full-AdaLN Ref2VA transformer";
  case TransformerArchitecture::kViggleAnimatePrunedTable:
    return "Viggle-Animate pruned AdaLN-table Ref2VA transformer";
  case TransformerArchitecture::kUnknown:
    return "unknown transformer";
  }
  return "unknown transformer";
}

const char* transformer_quantization_name(TransformerQuantization quantization) {
  switch (quantization) {
  case TransformerQuantization::kBFloat16:
    return "bfloat16";
  case TransformerQuantization::kW6A8ConvRot:
    return "W6A8 INT6 ConvRot";
  case TransformerQuantization::kW4A8ConvRot:
    return "W4A8 INT4 ConvRot";
  case TransformerQuantization::kFloat32:
    return "float32";
  case TransformerQuantization::kFloat8:
    return "float8";
  case TransformerQuantization::kInt8ConvRot:
    return "int8 ConvRot";
  case TransformerQuantization::kNativeNVFP4:
    return "native NVFP4";
  case TransformerQuantization::kBitsAndBytesNF4:
    return "bitsandbytes NF4";
  case TransformerQuantization::kUnknown:
    return "unknown";
  }
  return "unknown";
}

bool transformer_qkv_is_interleaved(const SafeTensors& checkpoint) {
  return resolve_model_descriptor(checkpoint).qkv_interleaved;
}

void validate_interleaved_qkv(const TensorView& tensor, int head_dim) {
  if (head_dim <= 0 || tensor.shape.size() != 2 || tensor.shape[0] <= 0 || tensor.shape[1] <= 0 ||
      tensor.shape[0] % (3LL * head_dim) != 0 ||
      (tensor.dtype != DType::kI8 && tensor.dtype != DType::kF32 && tensor.dtype != DType::kF16 &&
       tensor.dtype != DType::kBF16 && tensor.dtype != DType::kF8E4M3))
    throw std::runtime_error("transformer: unsupported interleaved QKV tensor '" + tensor.name +
                             "'");
}

std::vector<uint8_t> deinterleave_qkv_rows(const TensorView& tensor, int head_dim,
                                           size_t row_offset, size_t row_count) {
  validate_interleaved_qkv(tensor, head_dim);
  const size_t rows = static_cast<size_t>(tensor.shape[0]);
  if (row_count == 0)
    row_count = rows;
  if (row_offset > rows || row_count > rows - row_offset || tensor.nbytes % rows)
    throw std::runtime_error("transformer: invalid interleaved QKV row slice");
  const size_t row_bytes = tensor.nbytes / rows;
  const size_t inner = rows / 3;
  std::vector<uint8_t> result(row_count * row_bytes);
  for (size_t i = 0; i < row_count; ++i) {
    const size_t row = row_offset + i;
    const size_t channel = row % inner;
    const size_t source = (channel / head_dim * 3 + row / inner) * head_dim + channel % head_dim;
    std::memcpy(result.data() + i * row_bytes,
                static_cast<const uint8_t*>(tensor.data) + source * row_bytes, row_bytes);
  }
  return result;
}

void require_ref2va_transformer(const SafeTensors& checkpoint, size_t reference_count) {
  if (reference_count == 0)
    return;
  const auto model = resolve_model_descriptor(checkpoint);
  if (model.supports_references)
    return;
  const auto architecture = model.compatibility_architecture;
  throw std::runtime_error("reference conditioning requires a Ref2VA transformer, but '" +
                           checkpoint.path() + "' is a " +
                           transformer_architecture_name(architecture));
}

} // namespace slopfab::dit
